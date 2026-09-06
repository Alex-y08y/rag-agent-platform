"""SQL Query Tool: read-only SQL execution with safety checks."""
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from app.core.database import SessionLocal
from app.core.exceptions import SQLSafetyError
from app.core.logging import get_logger
from app.tools.base import BaseTool

logger = get_logger(__name__)

# Allowed SQL keywords (read-only)
ALLOWED_KEYWORDS = {"SELECT", "WITH", "FROM", "WHERE", "GROUP", "BY", "ORDER",
                    "LIMIT", "OFFSET", "HAVING", "JOIN", "LEFT", "RIGHT", "INNER",
                    "OUTER", "ON", "AS", "AND", "OR", "NOT", "IN", "BETWEEN",
                    "LIKE", "IS", "NULL", "DISTINCT", "UNION", "ALL", "CASE",
                    "WHEN", "THEN", "ELSE", "END", "EXISTS", "CAST", "COALESCE",
                    "COUNT", "SUM", "AVG", "MIN", "MAX", "ROUND", "DATE", "EXTRACT",
                    "INTERVAL", "ASC", "DESC", "LIMIT", "OVER", "PARTITION"}

# Forbidden operations
FORBIDDEN_PATTERNS = [
    r"\bDROP\b", r"\bDELETE\b", r"\bUPDATE\b", r"\bINSERT\b",
    r"\bALTER\b", r"\bTRUNCATE\b", r"\bCREATE\b", r"\bGRANT\b",
    r"\bREVOKE\b", r"\bEXEC\b", r"\bEXECUTE\b", r"\bMERGE\b",
    r"\bREPLACE\b", r"\bINTO\s+OUTFILE\b", r"\bLOAD_FILE\b",
    r";",  # No multiple statements
]


class SQLQueryInput(BaseModel):
    sql: str = Field(..., description="要执行的SQL查询语句（仅支持SELECT）")


class SQLQueryTool(BaseTool[SQLQueryInput, dict[str, Any]]):
    """Execute read-only SQL queries against the business database.

    Only SELECT statements are allowed. All other operations are blocked
    by a safety checker.
    """

    name = "sql_query"
    description = (
        "执行业务数据库的只读SQL查询，返回查询结果。"
        "可用表：products（产品）、sales_orders（销售订单）、customers（客户）、marketing_campaigns（营销活动）。"
        "当用户需要查询销售数据、产品信息、客户统计、订单趋势等结构化数据时使用。"
        "仅支持SELECT语句，禁止任何修改操作。"
    )
    input_schema = SQLQueryInput

    @staticmethod
    def safety_check(sql: str) -> None:
        """Validate SQL is read-only and safe."""
        sql_upper = sql.upper().strip()

        # Must start with SELECT or WITH
        if not (sql_upper.startswith("SELECT") or sql_upper.startswith("WITH")):
            raise SQLSafetyError("Only SELECT/WITH queries are allowed")

        # Check forbidden patterns
        for pattern in FORBIDDEN_PATTERNS:
            if re.search(pattern, sql_upper):
                raise SQLSafetyError(f"Forbidden operation detected: {pattern}")

        # Limit result rows
        if "LIMIT" not in sql_upper:
            raise SQLSafetyError("Query must include a LIMIT clause")

    async def execute(self, input_data: SQLQueryInput) -> dict[str, Any]:
        sql = input_data.sql.strip()

        try:
            self.safety_check(sql)
        except SQLSafetyError as exc:
            logger.warning("SQL safety check failed: %s", exc)
            return {"error": str(exc), "columns": [], "rows": []}

        try:
            from sqlalchemy import text

            db = SessionLocal()
            result = db.execute(text(sql))
            columns = list(result.keys())
            rows = [dict(row._mapping) for row in result.fetchall()]
            db.close()

            # Make rows JSON-serializable (Decimal -> float, datetime -> isoformat)
            import datetime as _dt
            from decimal import Decimal as _Decimal

            def _sanitize(value: Any) -> Any:
                if isinstance(value, _Decimal):
                    return float(value)
                if isinstance(value, (_dt.datetime, _dt.date, _dt.time)):
                    return value.isoformat()
                return value

            rows = [{k: _sanitize(v) for k, v in row.items()} for row in rows]

            # Generate summary
            summary = f"查询返回 {len(rows)} 行，{len(columns)} 列"
            if rows and len(rows) <= 10:
                summary += f"。列：{', '.join(columns)}"

            logger.info("SQL query executed: %d rows", len(rows))
            return {
                "columns": columns,
                "rows": rows,
                "row_count": len(rows),
                "summary": summary,
            }
        except Exception as exc:
            logger.error("SQL query failed: %s", exc)
            return {"error": f"SQL execution failed: {exc}", "columns": [], "rows": []}
