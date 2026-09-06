"""Chat API routes with streaming support, auth, and message persistence."""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.agents.graph import get_agent_runner
from app.core.database import SessionLocal
from app.core.deps import get_current_user_optional
from app.core.exceptions import to_http_error
from app.core.logging import get_logger, new_request_id
from app.models.models import Conversation, Message, User
from app.schemas.schemas import ChatRequest, ChatResponse

logger = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


def _save_chat_messages(
    conv_id: str,
    query: str,
    answer: str,
    citations: list,
    latency_ms: int,
    token_usage: dict,
    user: User | None,
    kb_id: str | None,
) -> None:
    """Save user + assistant messages and update conversation metadata."""
    db = SessionLocal()
    try:
        conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
        if not conv:
            conv = Conversation(
                id=conv_id,
                title=query[:50],
                knowledge_base_id=kb_id,
                user_id=user.id if user else None,
            )
            db.add(conv)
        else:
            if user and not conv.user_id:
                conv.user_id = user.id
            if conv.title == "新对话" or not conv.title:
                conv.title = query[:50]
        conv.message_count = (conv.message_count or 0) + 2
        db.commit()

        db.add(Message(conversation_id=conv_id, role="user", content=query))
        db.add(Message(
            conversation_id=conv_id,
            role="assistant",
            content=answer,
            citations=citations,
            latency_ms=latency_ms,
            token_usage=token_usage,
        ))
        db.commit()
    except Exception as exc:
        logger.warning("Failed to save chat messages: %s", exc)
        db.rollback()
    finally:
        db.close()


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    user: User | None = Depends(get_current_user_optional),
) -> ChatResponse:
    """Non-streaming chat endpoint using the Agent."""
    new_request_id()
    try:
        runner = get_agent_runner()
        result = await runner.run(
            query=request.query,
            conversation_id=request.conversation_id,
            knowledge_base_id=request.knowledge_base_id,
        )

        citations_data = [
            c.model_dump() if hasattr(c, "model_dump") else dict(c)
            for c in result.get("citations", [])
        ]

        _save_chat_messages(
            conv_id=result["conversation_id"],
            query=request.query,
            answer=result["answer"],
            citations=citations_data,
            latency_ms=result.get("latency_ms", 0),
            token_usage=result.get("token_usage", {}),
            user=user,
            kb_id=request.knowledge_base_id,
        )

        return ChatResponse(
            conversation_id=result["conversation_id"],
            message_id=str(uuid.uuid4())[:16],
            answer=result["answer"],
            citations=result.get("citations", []),
            retrieved_docs=len(result.get("retrieved_docs", [])),
            latency_ms=result.get("latency_ms", 0),
            token_usage=result.get("token_usage", {}),
            tool_calls=result.get("tool_calls", []),
        )
    except Exception as exc:
        logger.error("Chat failed: %s", exc)
        raise to_http_error(exc) if hasattr(exc, "code") else exc


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    user: User | None = Depends(get_current_user_optional),
):
    """Streaming chat endpoint using Server-Sent Events."""
    new_request_id()

    async def event_generator():
        queue: asyncio.Queue = asyncio.Queue()
        done_received = False
        final_result: dict[str, Any] | None = None

        async def stream_callback(event: dict[str, Any]):
            await queue.put(event)

        runner = get_agent_runner()
        task = asyncio.create_task(
            runner.run(
                query=request.query,
                conversation_id=request.conversation_id,
                knowledge_base_id=request.knowledge_base_id,
                stream_callback=stream_callback,
            )
        )

        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=2.0)
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    if event.get("type") in ("done", "error"):
                        done_received = True
                        if event.get("type") == "done":
                            final_result = event.get("data")
                        break
                except asyncio.TimeoutError:
                    if task.done():
                        break
                    continue

            if not task.done():
                result = await task
            else:
                result = task.result()

            if not done_received:
                citations_data = [
                    c.model_dump() if hasattr(c, "model_dump") else dict(c)
                    for c in result.get("citations", [])
                ]
                done_event = {
                    "type": "done",
                    "data": {
                        "conversation_id": result["conversation_id"],
                        "answer": result["answer"],
                        "citations": citations_data,
                        "latency_ms": result.get("latency_ms", 0),
                        "token_usage": result.get("token_usage", {}),
                    },
                }
                yield f"data: {json.dumps(done_event, ensure_ascii=False)}\n\n"
                final_result = done_event["data"]

            # Persist messages after stream completes
            if final_result:
                _save_chat_messages(
                    conv_id=final_result.get("conversation_id", ""),
                    query=request.query,
                    answer=final_result.get("answer", ""),
                    citations=final_result.get("citations", []),
                    latency_ms=final_result.get("latency_ms", 0),
                    token_usage=final_result.get("token_usage", {}),
                    user=user,
                    kb_id=request.knowledge_base_id,
                )

        except Exception as exc:
            logger.error("Chat stream failed: %s", exc)
            error_event = {"type": "error", "data": {"message": str(exc)}}
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
