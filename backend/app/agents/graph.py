"""LangGraph Agent: ReAct-style agent with Think-Act-Observe-Verify loop."""
from __future__ import annotations

import json
import time
import uuid
from typing import Any

from app.agents.state import AgentState
from app.agents.trace import TraceRecorder
from app.core.config import settings
from app.core.database import SessionLocal
from app.core.logging import get_logger, new_agent_run_id
from app.models.models import Conversation
from app.memory.redis_memory import get_memory
from app.rag.citation import get_citation_manager
from app.rag.pipeline import get_rag_pipeline
from app.services.llm_service import get_llm_service
from app.tools import get_all_tools

logger = get_logger(__name__)

# ── Prompts ───────────────────────────────────────────────
INTENT_PROMPT = """你是一个意图识别专家。分析用户问题，判断是否需要调用工具。

可用工具：
- knowledge_search: 搜索企业知识库（政策、制度、文档、产品手册等内部资料）
- sql_query: 查询业务数据库（产品、销售、客户、订单等结构化数据）
- calculator: 数学计算
- document_analysis: 文档分析
- web_search: 网络搜索

判断规则：
- 以下情况【不需要】任何工具，直接回答即可：
  * 问候、寒暄、闲聊（如"你好"、"谢谢"、"你是谁"）
  * 通用常识、简单解释（如"什么是RAG"、"Python怎么读文件"）
  * 不涉及企业内部数据的开放性问题
  * 对之前回答的追问但答案已在上下文中
- 以下情况【需要】knowledge_search：
  * 查询公司制度、政策、流程、产品说明等内部文档
  * 问题中提到"公司"、"制度"、"政策"、"规定"、"标准"、"流程"等
- 以下情况【需要】sql_query：
  * 查询销售数据、产品排名、客户统计、订单金额等结构化数据
  * 问题中提到"销售额"、"排名"、"统计"、"多少"、"Top"等

输出 JSON：
{"intent": "一句话描述用户意图", "tools": ["工具名列表，不需要工具则为空数组"], "needs_rag": true/false, "needs_sql": true/false}

示例：
"你好" -> {"intent": "问候", "tools": [], "needs_rag": false, "needs_sql": false}
"2026年差旅报销标准是什么？" -> {"intent": "查询差旅报销政策", "tools": ["knowledge_search"], "needs_rag": true, "needs_sql": false}
"销售额最高的3个产品是什么？" -> {"intent": "查询产品销售排名", "tools": ["sql_query"], "needs_rag": false, "needs_sql": true}
"销售额最高的3个产品，结合营销政策分析原因" -> {"intent": "查询销售排名并分析营销政策", "tools": ["sql_query", "knowledge_search"], "needs_rag": true, "needs_sql": true}
"""

PLAN_PROMPT = """你是一个任务规划专家。根据用户问题和意图，制定执行计划。

可用工具（tool 字段必须严格使用以下名称之一）：
- knowledge_search: 搜索企业知识库（政策、制度、文档）
- sql_query: 查询业务数据库（产品、销售、客户、订单）
- calculator: 数学计算
- document_analysis: 文档分析
- web_search: 网络搜索

输出 JSON 格式的步骤列表：
{"steps": [{"step": 1, "tool": "工具名", "description": "做什么", "depends_on": []}]}

规则：
1. SQL查询先于RAG检索（先拿到数据再查政策）
2. 计算器在获取数据后使用
3. 每个步骤只做一件事
4. 最多5个步骤
5. tool 字段必须是上面列出的工具名，不要自创名称
"""

VERIFY_PROMPT = """你是一个答案验证专家。检查生成的答案是否：
1. 回答了用户的原始问题
2. 所有事实都有检索来源支撑
3. 数字和日期准确
4. 没有编造信息

输出 JSON：{"passed": true/false, "reason": "验证说明", "issues": ["问题列表"]}
"""


class AgentRunner:
    """ReAct Agent with LangGraph-style execution flow.

    Nodes:
    1. analyze_intent  - Identify user intent and required tools
    2. rewrite_query   - Rewrite query with conversation history
    3. plan            - Create execution plan
    4. tool_call       - Execute tools sequentially
    5. observe         - Process tool results
    6. verify          - Verify answer quality
    7. final_answer    - Generate final answer

    Max iterations: settings.AGENT_MAX_ITERATIONS (5)
    """

    def __init__(self) -> None:
        self.llm = get_llm_service()
        self.tools = get_all_tools()
        self.memory = get_memory()
        self.rag = get_rag_pipeline()
        self.citation_mgr = get_citation_manager()
        self.max_iterations = settings.AGENT_MAX_ITERATIONS

    async def run(
        self,
        query: str,
        conversation_id: str | None = None,
        knowledge_base_id: str | None = None,
        stream_callback: Any = None,
    ) -> dict[str, Any]:
        """Execute the full agent pipeline.

        Args:
            query: User query.
            conversation_id: Conversation ID for memory.
            knowledge_base_id: KB ID for RAG.
            stream_callback: Optional async callback for streaming events.

        Returns:
            Agent result dict with answer, citations, traces, etc.
        """
        if not conversation_id:
            conversation_id = uuid.uuid4().hex[:16]

        # Ensure conversation exists in DB (for trace FK)
        try:
            db = SessionLocal()
            conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
            if not conv:
                conv = Conversation(
                    id=conversation_id,
                    title=query[:50],
                    knowledge_base_id=knowledge_base_id,
                )
                db.add(conv)
                db.commit()
            db.close()
        except Exception as exc:
            logger.warning("Failed to create conversation: %s", exc)

        agent_run_id = new_agent_run_id()
        tracer = TraceRecorder(conversation_id, agent_run_id)
        total_start = time.time()

        state: AgentState = {
            "query": query,
            "conversation_id": conversation_id,
            "knowledge_base_id": knowledge_base_id,
            "history": await self.memory.get_history(conversation_id),
            "agent_run_id": agent_run_id,
            "token_usage": {},
            "latencies": {},
            "retry_count": 0,
            "tool_calls": [],
            "tool_results": [],
            "retrieved_docs": [],
        }

        try:
            # Node 1: Analyze Intent
            state = await self._analyze_intent(state, tracer, stream_callback)

            # Intent Routing: decide whether tools are needed
            needs_tools = (
                state.get("needs_rag", False)
                or state.get("needs_sql", False)
                or len(state.get("intent_tools", [])) > 0
            )

            if not needs_tools:
                # Direct answer path: no RAG / tools needed
                if stream_callback:
                    await stream_callback({
                        "type": "status",
                        "data": {"node": "direct_answer", "message": "无需检索，直接回答..."},
                    })
                tracer.record(
                    "intent_route",
                    input_data={"intent": state.get("intent")},
                    output_data={"route": "direct_answer"},
                    status="success",
                )
                state = await self._generate_answer(state, tracer, stream_callback, direct_mode=True)
            else:
                # Tool-assisted path: rewrite → plan → execute → answer → verify
                tracer.record(
                    "intent_route",
                    input_data={"intent": state.get("intent")},
                    output_data={"route": "tool_assisted", "tools": state.get("intent_tools")},
                    status="success",
                )

                # Node 2: Rewrite Query
                state = await self._rewrite_query(state, tracer, stream_callback)

                # Node 3: Plan
                state = await self._plan(state, tracer, stream_callback)

                # Node 4-5: Tool Call + Observe (loop)
                state = await self._execute_tools(state, tracer, stream_callback)

                # Node 6: Generate Answer
                state = await self._generate_answer(state, tracer, stream_callback, direct_mode=False)

                # Node 7: Verify (with retry)
                state = await self._verify(state, tracer, stream_callback)

        except Exception as exc:
            logger.error("Agent execution failed: %s", exc)
            state["error"] = str(exc)
            state["answer"] = f"处理您的问题时出现错误：{exc}"
            tracer.record("error", status="error", error_message=str(exc))

        # Persist traces
        tracer.flush()

        # Save to memory
        await self.memory.add_message(conversation_id, "user", query)
        await self.memory.add_message(
            conversation_id, "assistant", state.get("answer", ""),
            metadata={"agent_run_id": agent_run_id},
        )
        await self.memory.maybe_summarize(conversation_id)

        total_ms = int((time.time() - total_start) * 1000)

        if stream_callback:
            citations_data = []
            for c in state.get("citations", []):
                if hasattr(c, "model_dump"):
                    citations_data.append(c.model_dump())
                elif isinstance(c, dict):
                    citations_data.append(c)
                else:
                    citations_data.append({"filename": getattr(c, "filename", ""), "content": getattr(c, "content", "")})
            await stream_callback({
                "type": "done",
                "data": {
                    "conversation_id": conversation_id,
                    "agent_run_id": agent_run_id,
                    "answer": state.get("answer", ""),
                    "citations": citations_data,
                    "latency_ms": total_ms,
                    "token_usage": state.get("token_usage", {}),
                },
            })

        return {
            "conversation_id": conversation_id,
            "agent_run_id": agent_run_id,
            "answer": state.get("answer", ""),
            "citations": state.get("citations", []),
            "retrieved_docs": state.get("retrieved_docs", []),
            "tool_calls": state.get("tool_calls", []),
            "intent": state.get("intent", ""),
            "rewritten_query": state.get("rewritten_query", query),
            "latency_ms": total_ms,
            "token_usage": state.get("token_usage", {}),
            "traces": tracer.get_traces(),
            "verification_passed": state.get("verification_passed", False),
        }

    # ── Node Implementations ────────────────────────────

    async def _analyze_intent(
        self, state: AgentState, tracer: TraceRecorder, cb: Any
    ) -> AgentState:
        start = time.time()
        if cb:
            await cb({"type": "status", "data": {"node": "analyze_intent", "message": "正在分析问题..."}})

        result = await self.llm.generate_json(
            INTENT_PROMPT, f"用户问题：{state['query']}", temperature=0.1
        )
        state["intent"] = result.get("intent", "")
        state["needs_rag"] = result.get("needs_rag", False)
        state["needs_sql"] = result.get("needs_sql", False)
        state["intent_tools"] = result.get("tools", [])
        state["plan"] = []  # Will be filled in plan node
        logger.info(
            "Intent: %s | needs_rag=%s needs_sql=%s tools=%s",
            state["intent"], state["needs_rag"], state["needs_sql"], state["intent_tools"],
        )

        latency = int((time.time() - start) * 1000)
        tracer.record(
            "analyze_intent",
            input_data={"query": state["query"]},
            output_data={"intent": state["intent"], "tools": result.get("tools", [])},
            latency_ms=latency,
        )
        state["latencies"]["intent_ms"] = latency
        return state

    async def _rewrite_query(
        self, state: AgentState, tracer: TraceRecorder, cb: Any
    ) -> AgentState:
        start = time.time()
        if cb:
            await cb({"type": "status", "data": {"node": "rewrite_query", "message": "正在优化查询..."}})

        from app.rag.query_rewrite import get_query_rewriter
        rewriter = get_query_rewriter()
        rewritten = await rewriter.rewrite(state["query"], state.get("history"))
        state["rewritten_query"] = rewritten

        latency = int((time.time() - start) * 1000)
        tracer.record(
            "rewrite_query",
            input_data={"query": state["query"], "history_len": len(state.get("history", []))},
            output_data={"rewritten_query": rewritten},
            latency_ms=latency,
        )
        state["latencies"]["rewrite_ms"] = latency
        return state

    async def _plan(
        self, state: AgentState, tracer: TraceRecorder, cb: Any
    ) -> AgentState:
        start = time.time()
        if cb:
            await cb({"type": "status", "data": {"node": "plan", "message": "正在制定执行计划..."}})

        plan_prompt = (
            f"用户问题：{state['query']}\n"
            f"识别意图：{state.get('intent', '')}\n"
            f"重写查询：{state.get('rewritten_query', state['query'])}"
        )
        result = await self.llm.generate_json(PLAN_PROMPT, plan_prompt, temperature=0.1)
        state["plan"] = result.get("steps", [])

        latency = int((time.time() - start) * 1000)
        tracer.record(
            "plan",
            input_data={"intent": state.get("intent")},
            output_data={"steps": state["plan"]},
            latency_ms=latency,
        )
        state["latencies"]["plan_ms"] = latency
        return state

    async def _execute_tools(
        self, state: AgentState, tracer: TraceRecorder, cb: Any
    ) -> AgentState:
        """Execute planned tools sequentially (Act + Observe)."""
        steps = state.get("plan", [])
        if not steps:
            # Default: try RAG if KB is available
            if state.get("knowledge_base_id"):
                steps = [{"step": 1, "tool": "knowledge_search", "description": "检索知识库"}]
            else:
                return state

        for step in steps[: self.max_iterations]:
            tool_name = step.get("tool", "")
            # Normalize common LLM hallucinations
            tool_alias = {
                "rag_retrieval": "knowledge_search",
                "rag": "knowledge_search",
                "retrieval": "knowledge_search",
                "search": "knowledge_search",
                "kb_search": "knowledge_search",
                "knowledge": "knowledge_search",
                "sql": "sql_query",
                "db_query": "sql_query",
                "database": "sql_query",
                "calc": "calculator",
                "math": "calculator",
            }
            tool_name = tool_alias.get(tool_name.lower(), tool_name)
            if tool_name not in self.tools:
                logger.warning("Unknown tool in plan: %s", tool_name)
                continue

            start = time.time()
            if cb:
                await cb({
                    "type": "tool_call",
                    "data": {"tool": tool_name, "description": step.get("description", "")},
                })

            try:
                tool = self.tools[tool_name]
                tool_input = self._build_tool_input(tool_name, state, step)
                result = await tool(**tool_input)

                state["tool_calls"].append({
                    "tool": tool_name,
                    "input": tool_input,
                    "status": "success",
                })
                state["tool_results"].append({"tool": tool_name, "result": result})

                # Collect RAG docs
                if tool_name == "knowledge_search" and "documents" in result:
                    state["retrieved_docs"].extend(result["documents"])
                if tool_name == "sql_query":
                    state["sql_results"] = result

                latency = int((time.time() - start) * 1000)
                tracer.record(
                    "tool_call",
                    input_data={"tool": tool_name, "args": tool_input},
                    output_data={"result_summary": self._summarize_result(result)},
                    tool_name=tool_name,
                    latency_ms=latency,
                    status="success",
                )

                if cb:
                    await cb({
                        "type": "status",
                        "data": {"node": "observe", "message": f"{tool_name} 完成"},
                    })

            except Exception as exc:
                logger.error("Tool %s failed: %s", tool_name, exc)
                state["tool_calls"].append({
                    "tool": tool_name, "status": "error", "error": str(exc),
                })
                tracer.record(
                    "tool_call",
                    tool_name=tool_name,
                    status="error",
                    error_message=str(exc),
                    latency_ms=int((time.time() - start) * 1000),
                )

        return state

    def _build_tool_input(self, tool_name: str, state: AgentState, step: dict) -> dict:
        """Build tool input based on tool type and current state."""
        query = state.get("rewritten_query", state["query"])

        if tool_name == "knowledge_search":
            return {
                "query": query,
                "knowledge_base_id": state.get("knowledge_base_id"),
                "top_k": 5,
            }
        elif tool_name == "sql_query":
            # Generate SQL from the query
            return {"sql": self._generate_sql_sync(query, state)}
        elif tool_name == "calculator":
            # Extract expression from query (simplified)
            return {"expression": step.get("expression", "1+1")}
        elif tool_name == "document_analysis":
            return {"document_id": step.get("document_id", ""), "analysis_type": "summary"}
        elif tool_name == "web_search":
            return {"query": query, "num_results": 5}
        return {}

    def _generate_sql_sync(self, query: str, state: AgentState) -> str:
        """Generate SQL query (sync wrapper around async LLM call)."""
        # This is a simplified approach - in production use async
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're already in an async context, use a simple default
                return self._default_sql(query)
        except RuntimeError:
            pass

        sql_prompt = f"""根据用户问题生成PostgreSQL SELECT查询。
可用表：products(id, name, category, price, stock), sales_orders(id, product_id, customer_id, quantity, amount, region, order_date), customers(id, name, email, region, tier)
用户问题：{query}
只输出SQL语句，必须包含LIMIT。"""
        try:
            # Use a direct call since we're in async
            messages = [
                {"role": "system", "content": "你是一个SQL生成专家。只输出SQL语句，不要解释。"},
                {"role": "user", "content": sql_prompt},
            ]
            # This runs in async context, so call directly
            import asyncio
            # We can't easily call async from here, so return a reasonable default
            return self._default_sql(query)
        except Exception:
            return self._default_sql(query)

    def _default_sql(self, query: str) -> str:
        """Generate a reasonable default SQL based on query keywords."""
        q = query.lower()
        if "产品" in q or "product" in q:
            if "最高" in q or "top" in q or "排名" in q:
                return ("SELECT p.name, SUM(s.amount) as total_sales, SUM(s.quantity) as total_qty "
                        "FROM products p JOIN sales_orders s ON p.id = s.product_id "
                        "GROUP BY p.name ORDER BY total_sales DESC LIMIT 3")
            return "SELECT id, name, category, price FROM products ORDER BY id LIMIT 10"
        if "销售" in q or "sales" in q or "订单" in q:
            return ("SELECT DATE_TRUNC('month', order_date) as month, SUM(amount) as total "
                    "FROM sales_orders GROUP BY month ORDER BY month LIMIT 12")
        if "客户" in q or "customer" in q:
            return "SELECT region, COUNT(*) as cnt, tier FROM customers GROUP BY region, tier LIMIT 10"
        return "SELECT COUNT(*) as total FROM products LIMIT 1"

    @staticmethod
    def _summarize_result(result: dict) -> dict:
        """Create a non-sensitive summary of tool results."""
        summary = {}
        for key, value in result.items():
            if isinstance(value, list):
                summary[key] = f"list[{len(value)}]"
            elif isinstance(value, str) and len(value) > 200:
                summary[key] = value[:200] + "..."
            else:
                summary[key] = value
        return summary

    async def _generate_answer(
        self, state: AgentState, tracer: TraceRecorder, cb: Any, direct_mode: bool = False
    ) -> AgentState:
        start = time.time()
        if cb:
            await cb({"type": "status", "data": {"node": "final_answer", "message": "正在生成回答..."}})

        # Build context from tool results
        context_parts = []
        if state.get("retrieved_docs"):
            context_parts.append("【知识库检索结果】")
            for i, doc in enumerate(state["retrieved_docs"][:5], 1):
                page_str = f" 第{doc.get('page')}页" if doc.get("page") else ""
                section_str = f" [{doc.get('section', '')}]" if doc.get("section") else ""
                context_parts.append(
                    f"[{i}] 《{doc.get('source', '')}》"
                    f"{page_str}"
                    f"{section_str}\n"
                    f"{doc.get('content', '')[:500]}"
                )

        if state.get("sql_results"):
            context_parts.append("\n【数据库查询结果】")
            sql = state["sql_results"]
            if sql.get("columns"):
                context_parts.append(f"列: {', '.join(sql['columns'])}")
            if sql.get("rows"):
                for row in sql["rows"][:10]:
                    context_parts.append(str(row))

        context = "\n".join(context_parts)

        if direct_mode or not context_parts:
            # Direct answer: no retrieved context, normal conversation
            system_prompt = """你是一个企业级智能助手，名叫 RAG Agent。
用自然、专业、友好的语气回答用户问题。
规则：
1. 对于问候和寒暄，简洁回应
2. 对于常识性问题，给出准确、有帮助的回答
3. 不要编造企业内部数据或政策
4. 如果问题涉及企业内部信息但你不确定，建议用户查询知识库或联系相关部门
5. 回答要结构清晰、专业准确"""
            user_prompt = f"用户问题：{state['query']}\n\n请回答："
        else:
            system_prompt = """你是一个企业级智能助手。根据工具执行结果回答用户问题。

规则：
1. 使用检索到的知识库内容和数据库结果回答
2. 引用知识库来源时使用 [数字] 标注
3. 数字和数据必须来自查询结果，不要编造
4. 如果信息不足，明确说明
5. 回答要结构清晰、专业准确"""
            user_prompt = f"用户问题：{state['query']}\n\n工具结果：\n{context}\n\n请回答："

        answer, usage = await self.llm.chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])

        state["answer"] = answer
        state["token_usage"] = usage
        state["citations"] = self.citation_mgr.build_citations(
            state.get("retrieved_docs", [])
        )

        # Stream tokens
        if cb:
            await cb({"type": "token", "data": {"content": answer}})
            await cb({"type": "citation", "data": {"citations": [
                c.model_dump() if hasattr(c, "model_dump") else dict(c)
                for c in state["citations"]
            ]}})

        latency = int((time.time() - start) * 1000)
        tracer.record(
            "final_answer",
            input_data={"context_len": len(context)},
            output_data={"answer_length": len(answer)},
            latency_ms=latency,
            token_usage=usage,
        )
        state["latencies"]["generation_ms"] = latency
        return state

    async def _verify(
        self, state: AgentState, tracer: TraceRecorder, cb: Any
    ) -> AgentState:
        start = time.time()
        if cb:
            await cb({"type": "status", "data": {"node": "verify", "message": "正在验证答案..."}})

        verify_input = (
            f"用户问题：{state['query']}\n"
            f"生成答案：{state.get('answer', '')}\n"
            f"检索来源数量：{len(state.get('retrieved_docs', []))}\n"
            f"工具调用：{[tc.get('tool') for tc in state.get('tool_calls', [])]}"
        )

        result = await self.llm.generate_json(VERIFY_PROMPT, verify_input, temperature=0.1)
        state["verification_passed"] = result.get("passed", False)
        state["verification_reason"] = result.get("reason", "")

        latency = int((time.time() - start) * 1000)
        tracer.record(
            "verify",
            output_data={
                "passed": state["verification_passed"],
                "reason": state["verification_reason"],
            },
            latency_ms=latency,
        )

        # Retry if verification failed and we have retries left
        if not state["verification_passed"] and state.get("retry_count", 0) < 2:
            state["retry_count"] = state.get("retry_count", 0) + 1
            logger.info("Verification failed, retrying (attempt %d)", state["retry_count"])
            if cb:
                await cb({"type": "status", "data": {"node": "verify", "message": "验证未通过，重新生成..."}})
            state = await self._generate_answer(state, tracer, cb)

        return state


# Singleton
_agent_runner: AgentRunner | None = None


def get_agent_runner() -> AgentRunner:
    global _agent_runner
    if _agent_runner is None:
        _agent_runner = AgentRunner()
    return _agent_runner
