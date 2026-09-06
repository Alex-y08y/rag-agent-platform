"""Seed demo data for Docker container."""
import sys
sys.path.insert(0, "/app")

from pathlib import Path
from app.core.database import SessionLocal, init_db
from app.models.models import (
    KnowledgeBase, Document, DocumentChunk,
    Product, Customer, SalesOrder, MarketingCampaign,
)
from app.services.parser import get_parser
from app.services.chunking import get_chunker
import random
from datetime import datetime, timedelta

DOCUMENTS_DIR = Path("/app/data/documents")


def seed_documents():
    db = SessionLocal()
    existing = db.query(KnowledgeBase).filter(KnowledgeBase.name == "企业知识库").first()
    if existing:
        print(f"Knowledge base already exists: {existing.id}")
        db.close()
        return existing.id

    kb = KnowledgeBase(
        name="企业知识库",
        description="包含员工手册、差旅制度、费用报销、销售政策、营销政策、产品手册、客户服务、IT制度、请假制度",
        status="active",
    )
    db.add(kb)
    db.commit()
    db.refresh(kb)
    print(f"Created knowledge base: {kb.id}")

    parser = get_parser()
    chunker = get_chunker()
    total_chunks = 0

    for doc_file in sorted(DOCUMENTS_DIR.glob("*.md")):
        print(f"  Processing: {doc_file.name}")
        try:
            pages = parser.parse(str(doc_file), "md")
            chunks = chunker.chunk_markdown(
                "\n".join(p.content for p in pages),
                source=doc_file.name,
            )
            doc = Document(
                knowledge_base_id=kb.id,
                filename=doc_file.name,
                file_type="md",
                file_size=doc_file.stat().st_size,
                file_hash=str(hash(doc_file.name)),
                storage_path=str(doc_file),
                status="SUCCESS",
                chunk_count=len(chunks),
            )
            db.add(doc)
            db.commit()
            db.refresh(doc)

            for chunk in chunks:
                db.add(DocumentChunk(
                    document_id=doc.id,
                    knowledge_base_id=kb.id,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    page=chunk.page,
                    section=chunk.section,
                    title=chunk.title,
                    source=chunk.source or doc_file.name,
                    metadata=chunk.metadata,
                    embedding_status="PENDING",
                    vector_id=None,
                ))
            total_chunks += len(chunks)
            print(f"    -> {len(chunks)} chunks")
        except Exception as exc:
            print(f"    Error: {exc}")

    doc_count = len(list(DOCUMENTS_DIR.glob("*.md")))
    kb.document_count = doc_count
    kb.chunk_count = total_chunks
    db.commit()
    db.close()
    print(f"Seeded {doc_count} documents, {total_chunks} chunks")
    return kb.id


def seed_sql_data():
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
    db.commit()

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


if __name__ == "__main__":
    print("Initializing database...")
    init_db()
    print("Seeding documents...")
    kb_id = seed_documents()
    print("Seeding SQL data...")
    seed_sql_data()
    print(f"\nDone! Knowledge Base ID: {kb_id}")
