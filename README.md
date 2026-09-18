# 企业知识库 RAG 问答系统

面向企业制度、FAQ、合同、会议纪要、产品手册和表格类文档的 RAG 问答项目。系统覆盖文档解析、结构优先自适应分块、父子块检索生成、向量与 BM25 混合召回、RRF 融合、Cross-Encoder 重排、上下文压缩、引用生成和检索效果验证。

## 核心能力

- 端到端 RAG：支持 PDF、DOCX、Markdown、TXT、CSV 和 XLSX 文档解析、向量化和问答。
- 结构优先分块：识别 FAQ、合同条款、制度章节、表格和会议纪要等结构，按 token 预算合并原子语义单元。
- 父子块检索生成：child chunk 用于召回与精排，parent chunk 用于补充完整章节、条款或表格上下文。
- 混合检索：BGE-M3 + Milvus 向量检索与 Elasticsearch BM25 关键词检索，通过 RRF 融合候选结果。
- 精排优化：使用 BGE Reranker Cross-Encoder 对候选结果重新排序，提升数字、编号和专业术语的命中效果。
- 引用生成：回答只保留实际使用的文档来源，支持文档名、页码、章节和原文片段展示。
- 效果验证：支持 Recall@K、Precision@K、MRR、LLM Judge 答案评分、忠实度和相关性指标。

## 技术栈

- 后端：Python 3.11+、FastAPI、Pydantic v2、SQLAlchemy 2
- 检索：BAAI/bge-m3、Milvus、Elasticsearch BM25、RRF、BAAI/bge-reranker-v2-m3
- 存储：PostgreSQL、Redis、本地文件系统
- 前端：React、TypeScript、Vite、Ant Design
- 部署：Docker Compose、Nginx
- 测试：pytest、pytest-asyncio

## RAG 架构

```text
用户问题
  -> Query Rewrite / optional HyDE / optional decomposition
  -> Milvus vector search + Elasticsearch BM25
  -> RRF fusion
  -> child chunk reranking
  -> parent context expansion
  -> context compression
  -> LLM answer generation
  -> citation filtering
```

检索版本：

| 版本 | 向量 | BM25 | RRF | Reranker | Rewrite |
|---|---:|---:|---:|---:|---:|
| `baseline` | 是 | 否 | 否 | 否 | 否 |
| `hybrid` | 是 | 是 | 是 | 否 | 否 |
| `hybrid_rerank` | 是 | 是 | 是 | 是 | 否 |
| `hybrid_rerank_rewrite` | 是 | 是 | 是 | 是 | 是 |

## 文档入库

```text
Upload
  -> validate type / size / hash
  -> parse PDF, DOCX, Markdown, TXT, CSV or XLSX
  -> detect document type
  -> extract atomic semantic units
  -> adaptive token-budget packing
  -> create child chunks and parent chunks
  -> embed and index child chunks
  -> persist parent context and metadata
  -> success and KB counter refresh
```

外部索引写入失败时会回滚，文档不会被错误标记为 `SUCCESS`。重索引采用 copy-on-write，新索引完成后再删除旧版本。

已有文档需要在后端重启并完成数据库兼容迁移后调用：

```text
POST /api/documents/{id}/reindex
```

## 分块策略

固定 `768/96` 字符窗口只作为无法识别文档结构时的 fallback。正常流程优先保留文档结构：

- FAQ：一个问答对作为一个原子单元。
- 合同：按条款、子条款和编号边界切分。
- 制度文件：按标题层级、章节和段落切分。
- 表格：表头与行组作为一个语义单元，避免跨表格截断。
- 会议纪要：按议题、结论和待办事项切分。
- 普通文本：按段落、句子和字符边界递归切分。

可配置 token 预算：

```env
CHUNK_MIN_TOKENS=120
CHUNK_TARGET_TOKENS=400
CHUNK_MAX_TOKENS=800
CHUNK_OVERLAP_TOKENS=80
PARENT_TARGET_TOKENS=1200
PARENT_MAX_TOKENS=1800
```

不同文档类型会在这些基础值上使用不同倍率。只有发生超长单元拆分时才使用重叠，短 FAQ、完整条款和表格不会被无条件重复。

## 父子块检索

```text
Child chunks
  -> vector / BM25 / RRF
  -> reranker
  -> parent_id lookup
  -> deduplicate parents
  -> parent context for LLM
```

- child chunk 保持较短，语义集中，适合向量召回和 Cross-Encoder 精排。
- parent chunk 保存完整章节、条款或表格上下文，适合生成答案。
- 检索接口仍返回 child 级别结果，生成链路自动扩展 parent。
- 旧索引没有 `parent_id` 时，可回退到 `document_id + section` 匹配。

## 评估指标

- Recall@1/3/5/10
- Precision@1/3/5/10
- MRR
- LLM Judge 答案评分
- Faithfulness
- Answer Relevancy
- Average Latency

`LLM Judge 答案评分` 是生成答案与标准答案的 0 到 1 连续评分，不是“答对题数百分比”。数字类问题优先使用规则匹配，开放式问题由 LLM Judge 评分。

当前 154 题检索评测中：

- Hybrid 将 Top-1 召回率从 76.5% 提升至 79.1%。
- Precision@1 从 81.8% 提升至 84.4%。
- MRR 达到 0.899。

## 数据存储

| 存储 | 用途 |
|---|---|
| PostgreSQL | 文档、child/parent chunk、父子关系、会话、评估任务 |
| Milvus | child chunk 向量索引 |
| Elasticsearch | child chunk BM25 索引 |
| Redis | 会话记忆和运行时状态 |

## 快速启动

### 1. 准备环境变量

```bash
cp .env.example .env
```

至少填写一个 LLM API Key：

```env
OPENAI_API_KEY=...
# 或
DASHSCOPE_API_KEY=...
```

### 2. 完整 Docker 模式

```bash
docker compose up -d --build
```

访问地址：

- 前端：http://localhost:3000
- 后端 API：http://localhost:8000
- OpenAPI 文档：http://localhost:8000/docs

### 3. Lite 模式

```bash
docker compose -f docker-compose.lite.yml up -d --build
```

Lite 模式使用 SQLite 和进程内向量、关键词存储，适合开发、测试和单机演示。进程重启后向量索引和关键词索引会丢失，需要重新入库。

### 4. 本地开发

后端：

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

前端：

```bash
cd frontend
npm install
npm run dev
```

前端开发地址：http://localhost:5173

## 测试

```bash
cd backend
python -m pytest tests -q
```

当前后端测试结果：

```text
114 passed, 14 skipped
```

测试使用临时 SQLite 数据库，不需要本机 PostgreSQL、Redis、Milvus 或 Elasticsearch。依赖本地 BGE 模型或外部服务的测试会自动跳过。

## 主要 API

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/auth/register` | 注册，首个用户为管理员 |
| POST | `/api/auth/login` | 登录 |
| GET | `/api/auth/me` | 当前用户 |
| POST | `/api/chat` | 非流式问答 |
| POST | `/api/chat/stream` | SSE 流式问答 |
| GET/POST | `/api/knowledge-bases` | 知识库列表和创建 |
| POST | `/api/documents/upload` | 上传文档 |
| GET | `/api/documents` | 文档列表 |
| GET | `/api/documents/{id}` | 文档和 child chunk 详情 |
| POST | `/api/documents/{id}/reindex` | 重新分块和索引 |
| POST | `/api/retrieval/search` | 检索调试 |
| POST | `/api/evaluation/run` | 创建评估任务 |
| GET | `/api/evaluation/tasks` | 评估任务列表 |
| GET | `/api/health` | 服务健康状态 |

## 项目结构

```text
backend/app/
  api/routes/   FastAPI routes
  core/         config, database, permissions, logging
  embeddings/   BGE embedding
  evaluation/   retrieval and generation metrics
  rag/          pipeline, rewrite, compression, citation
  reranker/     BGE reranker
  retrieval/    Milvus, Elasticsearch, Lite stores, RRF
  services/     parser, adaptive chunking, ingestion
  models/       SQLAlchemy models

frontend/src/
  api/
  components/
  layouts/
  pages/
  stores/
  types/
```

## 已知边界

- Lite 模式是开发和测试模式，不适合多实例生产部署。
- PDF 解析依赖文本层，当前不包含 OCR。
- 多实例部署时应将进程内限流替换为 Redis 限流。
- 生产环境建议使用 Alembic 管理正式数据库迁移，并将上传文件迁移到对象存储。
- 现有向量索引需要重新索引后才能使用新的父子块结构。
