"""Tests for SQL safety checker."""
from __future__ import annotations

import pytest

from app.tools.sql_query import SQLQueryTool, SQLQueryInput


class TestSQLSafety:
    def test_select_allowed(self):
        tool = SQLQueryTool()
        tool.safety_check("SELECT * FROM products LIMIT 10")  # should not raise

    def test_with_allowed(self):
        tool = SQLQueryTool()
        tool.safety_check("WITH t AS (SELECT 1) SELECT * FROM t LIMIT 5")

    def test_drop_blocked(self):
        tool = SQLQueryTool()
        with pytest.raises(Exception):
            tool.safety_check("DROP TABLE products")

    def test_delete_blocked(self):
        tool = SQLQueryTool()
        with pytest.raises(Exception):
            tool.safety_check("DELETE FROM products WHERE id=1")

    def test_update_blocked(self):
        tool = SQLQueryTool()
        with pytest.raises(Exception):
            tool.safety_check("UPDATE products SET price=100")

    def test_insert_blocked(self):
        tool = SQLQueryTool()
        with pytest.raises(Exception):
            tool.safety_check("INSERT INTO products VALUES (1, 'test')")

    def test_alter_blocked(self):
        tool = SQLQueryTool()
        with pytest.raises(Exception):
            tool.safety_check("ALTER TABLE products ADD COLUMN x int")

    def test_truncate_blocked(self):
        tool = SQLQueryTool()
        with pytest.raises(Exception):
            tool.safety_check("TRUNCATE TABLE products")

    def test_multiple_statements_blocked(self):
        tool = SQLQueryTool()
        with pytest.raises(Exception):
            tool.safety_check("SELECT 1; SELECT 2")

    def test_missing_limit_blocked(self):
        tool = SQLQueryTool()
        with pytest.raises(Exception):
            tool.safety_check("SELECT * FROM products")

    def test_case_insensitive_block(self):
        tool = SQLQueryTool()
        with pytest.raises(Exception):
            tool.safety_check("select * from products; drop table x")

    def test_complex_select_allowed(self):
        tool = SQLQueryTool()
        sql = """
            SELECT p.name, SUM(s.amount) as total
            FROM products p
            JOIN sales_orders s ON p.id = s.product_id
            WHERE s.order_date > '2026-01-01'
            GROUP BY p.name
            ORDER BY total DESC
            LIMIT 10
        """
        tool.safety_check(sql)  # should not raise
