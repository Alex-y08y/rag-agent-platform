"""One-off rebuild seed: KB + SQL demo data + full real ingestion (embed/Milvus/ES).

Runs INSIDE the backend container:
    docker exec -it rag-agent-platform-backend-1 python /app/data/rebuild_seed.py
"""
from __future__ import annotations

import asyncio
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, "/app")

from app.core.database import SessionLocal, init_db
from app.models.models import (
    KnowledgeBase,
    Product,
    Customer,
    SalesOrder,
    MarketingCampaign,
)
from app.retrieval.es_store import get_es_store
from app.services.document_service import get_document_service

DOCUMENTS_DIR = Path("/app/data/documents")


def get_or_create_kb() -> str:
    db = SessionLocal()
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.name == "企业知识库").first()
    if not kb:
        kb = KnowledgeBase(
            name="企业知识库",
            description="员工手册、差旅制度、费用报销、销售政策、营销政策、产品手册、客服、IT制度、请假制度",
            status="active",
        )
        db.add(kb)
        db.commit()
        db.refresh(kb)
        print(f"Created knowledge base: {kb.id}")
    else:
        print(f"Knowledge base exists: {kb.id}")
    kb_id = kb.id
    db.close()

    # Make sure ES index exists
    try:
        get_es_store().create_index(kb_id)
        print("ES index ready")
    except Exception as exc:
        print(f"[WARN] ES create_index failed: {exc}")
    return kb_id


def seed_sql_data() -> None:
    db = SessionLocal()
    if db.query(Product).count() > 0:
        print("SQL data already seeded")
        db.close()
        return

    products = [
        Product(name="AI智能客服SaaS-基础版", category="软件", price=499, stock=9999, description="3坐席，1GB知识库"),
        Product(name="AI智能客服SaaS-标准版", category="软件", price=999, stock=9999, description="10坐席，10GB知识库"),
        Product(name="AI智能客服SaaS-企业版", category="软件", price=2999, stock=9999, description="50坐席，100GB知识库"),
        Product(name="AI智能客服私有化-企业版", category="软件", price=500000, stock=100, description="私有化部署，1年维护"),
        Product(name="AI智能客服私有化-旗舰版", category="软件", price=1000000, stock=50, description="含定制开发"),
        Product(name="智能硬件-语音网关", category="硬件", price=5000, stock=200, description="电话语音接入设备"),
        Product(name="AI培训服务", category="服务", price=5000, stock=999, description="现场培训1天"),
        Product(name="定制开发服务", category="服务", price=1000, stock=999, description="按人天计费"),
    ]
    db.add_all(products)
    db.commit()

    regions = ["华北", "华东", "华南", "西南", "华中"]
    tiers = ["standard", "premium", "enterprise"]
    customers = []
    for i in range(50):
        customers.append(Customer(
            name=f"客户{i+1:03d}",
            email=f"customer{i+1}@example.com",
            region=random.choice(regions),
            tier=random.choice(tiers),
        ))
    db.add_all(customers)
    db.commit()

    orders = []
    base_date = datetime(2026, 1, 1)
    for _ in range(500):
        product = random.choice(products)
        customer = random.choice(customers)
        qty = random.randint(1, 20)
        order_date = base_date + timedelta(days=random.randint(0, 365))
        orders.append(SalesOrder(
            product_id=product.id,
            customer_id=customer.id,
            quantity=qty,
            amount=qty * product.price * random.uniform(0.85, 1.0),
            region=customer.region,
            order_date=order_date,
        ))
    db.add_all(orders)

    campaigns = [
        MarketingCampaign(name="2026 Q1 AI新品发布会", product_id=2, start_date=datetime(2026, 3, 1), end_date=datetime(2026, 3, 31), budget=2000000, description="AI智能客服3.0发布"),
        MarketingCampaign(name="2026 Q2 行业峰会", product_id=3, start_date=datetime(2026, 6, 1), end_date=datetime(2026, 6, 30), budget=1500000, description="金融科技峰会"),
        MarketingCampaign(name="2026 Q3 客户答谢会", product_id=4, start_date=datetime(2026, 9, 1), end_date=datetime(2026, 9, 30), budget=1000000, description="S级客户答谢"),
        MarketingCampaign(name="2026 Q4 年终促销", product_id=1, start_date=datetime(2026, 11, 1), end_date=datetime(2026, 12, 31), budget=3000000, description="双11+年终大促"),
    ]
    db.add_all(campaigns)
    db.commit()
    db.close()
    print(f"Seeded SQL: {len(products)} products, {len(customers)} customers, {len(orders)} orders, {len(campaigns)} campaigns")


async def ingest_all(kb_id: str) -> None:
    service = get_document_service()
    md_files = sorted(DOCUMENTS_DIR.glob("*.md"))
    print(f"\nIngesting {len(md_files)} documents through full pipeline ...")
    for f in md_files:
        content = f.read_bytes()
        print(f"  -> {f.name} ({len(content)} bytes)")
        try:
            doc = await service.ingest_document(kb_id, f.name, content)
            print(f"     SUCCESS: {doc.chunk_count} chunks, status={doc.status}")
        except Exception as exc:
            print(f"     FAILED: {exc}")
            raise


async def main() -> None:
    print("Initializing database ...")
    init_db()
    kb_id = get_or_create_kb()
    seed_sql_data()
    await ingest_all(kb_id)

    db = SessionLocal()
    from app.models.models import Document, DocumentChunk
    dcount = db.query(Document).count()
    ccount = db.query(DocumentChunk).filter(DocumentChunk.embedding_status == "DONE").count()
    db.close()
    print(f"\n=== DONE === KB={kb_id} documents={dcount} indexed_chunks={ccount}")


if __name__ == "__main__":
    asyncio.run(main())
