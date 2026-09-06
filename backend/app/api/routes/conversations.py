"""Conversation API routes: list, get with messages, create, delete."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.core.database import SessionLocal
from app.core.deps import get_current_user
from app.models.models import Conversation, Message, User

router = APIRouter(prefix="/conversations", tags=["conversations"])


# ── Schemas ───────────────────────────────────────────────
class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    citations: list | None = None
    token_usage: dict | None = None

    model_config = {"from_attributes": True}


class ConversationOut(BaseModel):
    id: str
    title: str
    knowledge_base_id: str | None = None
    message_count: int

    model_config = {"from_attributes": True}


class ConversationDetail(ConversationOut):
    messages: list[MessageOut] = []


class CreateConversationRequest(BaseModel):
    title: str = "新对话"
    knowledge_base_id: str | None = None


# ── Routes ────────────────────────────────────────────────
@router.get("", response_model=list[ConversationOut])
def list_conversations(current_user: User = Depends(get_current_user)):
    """List all conversations for the current user, newest first."""
    db = SessionLocal()
    try:
        convs = (
            db.query(Conversation)
            .filter(Conversation.user_id == current_user.id)
            .order_by(Conversation.updated_at.desc())
            .all()
        )
        return [ConversationOut.model_validate(c) for c in convs]
    finally:
        db.close()


@router.get("/{conversation_id}", response_model=ConversationDetail)
def get_conversation(
    conversation_id: str, current_user: User = Depends(get_current_user)
):
    """Get a conversation with all its messages."""
    db = SessionLocal()
    try:
        conv = (
            db.query(Conversation)
            .filter(
                Conversation.id == conversation_id,
                Conversation.user_id == current_user.id,
            )
            .first()
        )
        if not conv:
            raise HTTPException(status_code=404, detail="对话不存在")
        messages = (
            db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
            .all()
        )
        detail = ConversationDetail.model_validate(conv)
        detail.messages = [MessageOut.model_validate(m) for m in messages]
        return detail
    finally:
        db.close()


@router.post("", response_model=ConversationOut)
def create_conversation(
    req: CreateConversationRequest, current_user: User = Depends(get_current_user)
):
    """Create a new conversation."""
    db = SessionLocal()
    try:
        conv = Conversation(
            user_id=current_user.id,
            title=req.title,
            knowledge_base_id=req.knowledge_base_id,
            message_count=0,
        )
        db.add(conv)
        db.commit()
        db.refresh(conv)
        return ConversationOut.model_validate(conv)
    finally:
        db.close()


@router.delete("/{conversation_id}")
def delete_conversation(
    conversation_id: str, current_user: User = Depends(get_current_user)
):
    """Delete a conversation and all its messages."""
    db = SessionLocal()
    try:
        conv = (
            db.query(Conversation)
            .filter(
                Conversation.id == conversation_id,
                Conversation.user_id == current_user.id,
            )
            .first()
        )
        if not conv:
            raise HTTPException(status_code=404, detail="对话不存在")
        db.delete(conv)
        db.commit()
        return {"ok": True, "message": "对话已删除"}
    finally:
        db.close()
