# 企业级 RAG + Agent 智能知识库平台

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?logo=typescript&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Agent-FF6B6B)
![Milvus](https://img.shields.io/badge/Milvus-Vector%20DB-00A3E0)
![Elasticsearch](https://img.shields.io/badge/Elasticsearch-BM25-005571?logo=elasticsearch&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![Tests](https://img.shields.io/badge/tests-112%20passed-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

> 一个完整的企业级 AI 应用项目：真 RAG + 真 Agent + 真评测 + 完整前后端 + Docker 部署 + 自动化测试。

---

## 目录

1. [项目背景](#1-项目背景)
2. [项目目标](#2-项目目标)
3. [项目特点](#3-项目特点)
4. [技术栈](#4-技术栈)
5. [系统架构](#5-系统架构)
6. [RAG 架构](#6-rag-架构)
7. [Agent 架构](#7-agent-架构)
8. [数据流](#8-数据流)
9. [Embedding 方案](#9-embedding-方案)
10. [Hybrid Search](#10-hybrid-search)
11. [RRF 融合](#11-rrf-融合)
12. [Reranker](#12-reranker)
13. [Query Rewrite](#13-query-rewrite)
14. [Context Compression](#14-context-compression)
15. [Agent Tool Calling](#15-agent-tool-calling)
16. [Memory](#16-memory)
17. [Citation 引用来源](#17-citation-引用来源)
18. [RAG Evaluation](#18-rag-evaluation)
19. [Recall@K](#19-recallk)
20. [Precision@K](#20-precisionk)
21. [MRR](#21-mrr)
22. [Accuracy](#22-accuracy)
23. [Faithfulness](#23-faithfulness)
24. [Answer Relevancy](#24-answer-relevancy)
25. [Baseline 对比](#25-baseline-对比)
26. [API 列表](#26-api-列表)
27. [数据库设计](#27-数据库设计)
28. [Docker 部署](#28-docker-部署)
29. [本地启动](#29-本地启动)
30. [测试](#30-测试)
31. [Demo 使用](#31-demo-使用)
32. [核心技术亮点](#32-核心技术亮点)

---

## 1. 项目背景

在企业数字化转型中，内部知识分散在各种文档、政策、产品手册中，员工查找信息效率低。传统搜索引擎无法理解语义，大模型直接回答又存在幻觉问题。RAG（检索增强生成）结合了检索的精准性和大模型的生成能力，是当前企业知识管理的最佳实践。

本项目从零构建了一个**企业级 RAG + Agent 智能知识库平台**，不仅实现了基础的文档问答，还包含 Agent 工具调用、混合检索、重排序、查询重写、上下文压缩、完整评测体系等生产级特性。

## 2. 项目目标

- 构建一个**可真实运行**的企业级知识库平台
- 实现**真 RAG**（非 Mock）：文档解析 → Chunking → Embedding → 向量存储 → 检索 → 生成
- 实现**真 Agent**（非伪 Agent）：LangGraph 状态机 + ReAct + Tool Calling
- 实现**真评测**：Recall/Precision/MRR/Accuracy/Faithfulness/Relevancy 真实计算
- 前后端分离，完整 UI 交互
- Docker 一键部署

## 3. 项目特点

| 特性 | 说明 |
|------|------|
| 真 RAG | BGE-M3 本地 Embedding + Milvus 向量库 + Elasticsearch BM25 |
| 真 Agent | LangGraph 状态机 + ReAct 循环 + 5 个 Tool |
| Hybrid Search | 向量检索 + BM25 关键词检索 + RRF 融合 |
| Reranker | BGE-reranker-v2-m3 本地交叉编码器重排序 |
| Query Rewrite | 基于对话历史的查询重写，解决指代消解 |
| Context Compression | 去重 + 相关性过滤 + 长度截断 |
| 多轮对话 | Redis 短期记忆 + 自动摘要长期记忆 |
| 引用来源 | 答案标注 [1][2]，可展开 Chunk 原文 |
| 完整评测 | 55 题评测集，12 项指标，4 版本对比 |
| Agent Trace | 完整执行链可视化，不暴露隐藏 CoT |
| 文档解析 | PDF/DOCX/TXT/MD/CSV/XLSX 六种格式 |
| 智能 Chunking | 递归分块 + Markdown 标题分块 + 结构感知分块 |
| Docker 部署 | 全栈 Docker Compose，含轻量模式 |
| 自动化测试 | 13 个测试文件，pytest + pytest-asyncio |

## 4. 技术栈

### 前端
- **React 18** + **TypeScript**
- **Vite** 构建工具
- **Tailwind CSS** + **Ant Design 5** UI 组件库
- **ECharts** 数据可视化
- **Axios** HTTP 客户端
- **Zustand** 状态管理
- **React Markdown** 渲染 Markdown 答案

### 后端
- **Python 3.11+** + **FastAPI**
- **Pydantic v2** 数据校验
- **SQLAlchemy 2.0** ORM
- **Uvicorn** ASGI 服务器

### AI / LLM
- **LangChain** + **LangGraph** Agent 框架
- **OpenAI Compatible API**（默认 Qwen3，可切换任意兼容模型）
- 模型通过环境变量配置，**API Key 禁止硬编码**

### Embedding / Reranker（免费、开源、本地）
- **BAAI/bge-m3** — Embedding 模型，本地加载，支持 CPU/CUDA 自动切换
- **BAAI/bge-reranker-v2-m3** — 交叉编码器 Reranker，本地运行

### 数据存储
- **Milvus** — 向量数据库
- **Elasticsearch** — BM25 全文检索
- **PostgreSQL** — 业务数据
- **Redis** — 会话记忆 + 缓存

### 部署 & 测试
- **Docker** + **Docker Compose**
- **pytest** + **pytest-asyncio**
- **Nginx** 前端反向代理

## 5. 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    React Frontend                        │
│  (Chat / KB / Docs / Evaluation / Trace / Status)       │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP / SSE
┌────────────────────────▼────────────────────────────────┐
│                    FastAPI API Layer                     │
│  (Pydantic Schema / Validation / Logging / request_id)   │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                   Agent Service (LangGraph)               │
│  Intent → Rewrite → Plan → ToolCall → Observe → Verify   │
└───────┬──────────────┬──────────────┬───────────────────┘
        │              │              │
┌───────▼──────┐ ┌────▼─────┐ ┌──────▼──────┐
│  RAG Tool    │ │ SQL Tool │ │ Calculator  │
│  (Hybrid)    │ │ (Read-   │ │   Tool      │
│              │ │  only)   │ │             │
└───────┬──────┘ └────┬─────┘ └─────────────┘
        │              │
┌───────▼──────────────▼───────────────────────────────────┐
│              RAG Retrieval Pipeline                       │
│  Query Rewrite → Metadata Filter → Vector+BM25 → RRF    │
│  → Top-K → Reranker → Context Compression                │
└───────┬──────────────────────┬───────────────────────────┘
        │                      │
┌───────▼──────┐        ┌──────▼───────┐
│   Milvus     │        │ Elasticsearch│
│  (Vector)    │        │   (BM25)     │
└──────────────┘        └──────────────┘
```

## 6. RAG 架构

```
用户问题
    │
    ▼
Query Rewrite（结合对话历史，消解指代）
    │
    ▼
Metadata Filter（按文档类型、页码等过滤）
    │
    ├─────────────────┐
    ▼                 ▼
Vector Search      BM25 Search
(Milvus, BGE-M3)  (Elasticsearch)
    │                 │
    └────────┬────────┘
             ▼
        RRF Fusion（倒数排名融合）
             │
             ▼
          Top-K
             │
             ▼
    BGE Reranker（交叉编码器重排序）
             │
             ▼
    Context Compression（去重+过滤+截断）
             │
             ▼
         LLM 生成
             │
             ▼
    Answer + Citation [1][2]
             │
             ▼
        Verification（验证答案忠实度）
```

### 四个 RAG 版本

| 版本 | 向量检索 | BM25 | RRF | Reranker | Query Rewrite |
|------|---------|------|-----|----------|---------------|
| Baseline | ✅ | ❌ | ❌ | ❌ | ❌ |
| Hybrid | ✅ | ✅ | ✅ | ❌ | ❌ |
| Hybrid+Rerank | ✅ | ✅ | ✅ | ✅ | ❌ |
| Hybrid+Rerank+Rewrite | ✅ | ✅ | ✅ | ✅ | ✅ |

## 7. Agent 架构

使用 **LangGraph** 构建状态机式 Agent，节点如下：

1. **analyze_intent** — 分析用户意图，判断需要哪些工具
2. **rewrite_query** — 结合对话历史重写查询
3. **plan** — 制定多步执行计划
4. **tool_call** — 执行工具（Act 阶段）
5. **observe** — 观察工具结果（Observe 阶段）
6. **verify** — 验证答案质量，不通过则重试
7. **final_answer** — 生成最终答案

**防无限循环机制：**
- 最大迭代次数：5（`AGENT_MAX_ITERATIONS`）
- 超时：120 秒（`AGENT_TIMEOUT_SECONDS`）
- 验证失败最多重试 2 次
- 每个节点有独立的错误处理和 fallback

## 8. 数据流

### 文档 Ingestion 流程
```
Upload → Validate(type/size/hash) → Parser(PDF/DOCX/TXT/MD/CSV/XLSX)
→ Cleaning → Structure Detection → Chunking(3种策略)
→ Metadata Extraction → BGE-M3 Embedding → Milvus Insert
→ Elasticsearch Index → Status: SUCCESS
```

### 问答流程
```
User Query → Redis Memory(获取历史) → Agent Intent Analysis
→ Query Rewrite → Hybrid Retrieval(Vector + BM25 + RRF)
→ BGE Reranker → Context Compression → LLM Generation
→ Citation Attachment → Verification → Save to Redis + PostgreSQL
→ SSE Streaming to Frontend
```

## 9. Embedding 方案

**必须使用免费、开源、本地运行的 Embedding 模型。**

默认模型：**BAAI/bge-m3**

特性：
- 通过 `sentence-transformers` 本地加载，**不依赖任何付费 API**
- 自动检测 CUDA，无 GPU 自动使用 CPU
- 支持模型缓存（`EMBEDDING_CACHE_DIR`）
- 支持 Batch Embedding（`EMBEDDING_BATCH_SIZE=32`）
- **Embedding dimension 自动获取**，不写死
- Milvus Collection 创建时根据模型 dimension 动态生成

配置：
```env
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DEVICE=auto      # auto | cpu | cuda
EMBEDDING_BATCH_SIZE=32
```

## 10. Hybrid Search

纯向量检索在处理**专业术语、数字、编号、政策名称、产品型号**时效果不佳。Hybrid Search 结合了两种检索的优势：

- **Vector Search（Milvus）**：语义相似度，理解同义词和上下文
- **BM25 Search（Elasticsearch）**：关键词精确匹配，擅长术语和数字

两种检索结果通过 **RRF 融合**后输出，兼顾语义理解和精确匹配。

## 11. RRF 融合

**Reciprocal Rank Fusion（倒数排名融合）** 是一种无参数的结果融合方法：

```
RRF(d) = Σ 1 / (k + rank(d))
```

其中 `k=60`（默认），`rank(d)` 是文档在某个排序列表中的位置（1-based）。

**优势：**
- 不需要归一化不同检索器的分数
- 对排名靠前的文档给予更高权重
- 被多个检索器同时排到前面的文档得分最高
- 实现简单，效果稳定

## 12. Reranker

使用 **BAAI/bge-reranker-v2-m3** 交叉编码器对候选文档进行精排。

- 免费、开源、本地运行
- 支持 CPU / CUDA 自动切换
- 输入：Query + 候选文档对
- 输出：相关性分数，按分数降序排列
- 默认取 Top-5 进入 LLM

**为什么需要 Reranker？**
向量检索和 BM25 都是"双编码器"架构，查询和文档分别编码，无法捕捉细粒度交互。Reranker 将查询和文档拼接后一起编码，能更精准地判断相关性。

## 13. Query Rewrite

根据当前问题 + 对话历史，生成完整自包含的查询。

**解决的问题：**
- 指代消解："它什么时候生效？" → "公司2026年差旅报销制度什么时候生效？"
- 上下文缺失：多轮对话中省略的信息
- 模糊查询优化

**示例：**
```
上一轮：公司差旅报销制度有什么变化？
当前：什么时候生效？
Rewrite：公司2026年差旅报销制度什么时候生效？
```

## 14. Context Compression

对 Top-K 检索结果进一步压缩，只保留与 Query 高度相关的信息：

1. **分数阈值过滤** — 过滤低相关度 Chunk
2. **冗余去除** — 基于字符 n-gram Jaccard 相似度去重
3. **句子级相关性过滤** — 保留包含查询关键词的句子
4. **长度截断** — 总上下文不超过 8000 字符

**避免的问题：**
- 无关 Chunk 干扰 LLM 判断
- 大量冗余上下文浪费 Token
- 降低幻觉风险

## 15. Agent Tool Calling

Agent 可调用以下 5 个工具：

| 工具 | 功能 | 输入 |
|------|------|------|
| **KnowledgeSearchTool** | 混合检索知识库 | query, top_k, filters |
| **SQLQueryTool** | 只读 SQL 查询 | sql（仅 SELECT） |
| **CalculatorTool** | 数学/统计计算 | expression |
| **DocumentAnalysisTool** | 文档摘要/关键词/结构 | document_id, analysis_type |
| **WebSearchTool** | 网络搜索 | query, num_results |

**SQL 安全机制：**
- 仅允许 SELECT / WITH 语句
- 禁止 DROP/DELETE/UPDATE/INSERT/ALTER/TRUNCATE
- 禁止多语句（分号）
- 必须包含 LIMIT
- 正则匹配 + 关键字白名单双重校验

### 复杂 Agent 场景 Demo

```
用户：帮我查询2026年销售额最高的3个产品，并结合知识库里的营销政策分析原因。

Agent 执行链：
1. analyze_intent → 需要 SQL + RAG
2. plan → [SQL查询Top3产品, RAG检索营销政策, 汇总分析]
3. tool_call(sql_query) → SELECT ... GROUP BY ... LIMIT 3
4. observe → 获取产品名称和销售额
5. tool_call(knowledge_search) → 检索各产品的营销政策
6. observe → 获取政策文档
7. final_answer → 产品排名 + 销售额 + 政策 + 原因分析 + 引用来源
8. verify → 验证答案准确性
```

## 16. Memory

使用 **Redis** 实现多轮对话记忆：

- **短期记忆**：最近 N 条消息（默认 20 轮）
- **长期记忆**：对话摘要（历史超过 15 轮时自动生成）
- **自动摘要**：LLM 总结旧消息，保留核心信息
- **按 Conversation ID 隔离**

## 17. Citation 引用来源

最终答案必须指出具体来源：
- 文档名（如《员工差旅管理制度》）
- 页码（Page 12）
- Section（住宿标准）
- Chunk ID（可展开原文）

前端展示：
```
答案正文...[1]

参考来源：
┌─────────────────────────────────────┐
│ 📄 travel_policy.md  [P3] [住宿标准] │
│ 一线城市普通员工住宿标准为500元/晚... │
└─────────────────────────────────────┘
```

**禁止出现**"根据相关资料……"但找不到具体来源的情况。

## 18. RAG Evaluation

完整评测体系，评测集位于 `data/eval/rag_eval.json`，包含 **55 道题**，覆盖：

| 类别 | 数量 | 示例 |
|------|------|------|
| 简单事实 | 15 | 工作时间是几点到几点？ |
| 政策 | 10 | 迟到超过30分钟怎么处理？ |
| 数字 | 15 | 住宿标准是多少？ |
| 时间 | 8 | 报销时限是多少天？ |
| 产品 | 5 | AI客服当前版本？ |
| 多文档 | 5 | S级和A级客户服务区别？ |
| 多跳/复杂 | 4 | 2026年差旅政策有哪些更新？ |
| 指代问题 | 3 | （需结合上下文） |

每题包含：`question`, `ground_truth`, `expected_sources`, `category`, `difficulty`

## 19. Recall@K

```
Recall@K = 相关文档被召回数量 / Ground Truth 相关文档总数
```

实现：`Recall@1`, `Recall@3`, `Recall@5`, `Recall@10`

**示例：**
```
expected_sources: ["travel_policy.pdf"]
retrieved: ["hr.pdf", "sales.pdf", "travel_policy.pdf"]
→ Recall@3 = 1.0（正确文档在前3名中）
→ Recall@1 = 0.0（正确文档不在第1名）
```

## 20. Precision@K

```
Precision@K = Top-K 中相关文档数量 / K
```

实现：`Precision@1`, `Precision@3`, `Precision@5`, `Precision@10`

**示例：**
```
expected: ["a.pdf"]
retrieved: ["a.pdf", "b.pdf", "c.pdf"]
→ Precision@3 = 1/3 ≈ 0.33
```

## 21. MRR

**Mean Reciprocal Rank（平均倒数排名）**

```
RR = 1 / 第一个正确文档的排名（1-based）
MRR = 所有样本 RR 的平均值
```

**示例：**
```
样本1：正确文档排第1 → RR = 1
样本2：正确文档排第2 → RR = 0.5
样本3：正确文档排第5 → RR = 0.2
MRR = (1 + 0.5 + 0.2) / 3 = 0.567
```

## 22. Accuracy

**Answer Accuracy** 区分检索质量和生成质量：

1. **Rule-based**（数字、日期、固定答案优先）：
   - 提取答案中的数字，与 Ground Truth 比对
   - 数字完全匹配得 1.0，部分匹配按比例

2. **LLM-as-a-Judge**（开放式答案）：
   - LLM 比较生成答案和标准答案
   - 输出 0-1 分和评分理由

## 23. Faithfulness

**答案忠实度**：判断 Answer 是否被 Retrieved Context 支撑。

- 输入：Question + Context + Answer
- 输出：0-1 分 + 理由
- 使用 LLM Judge 评估
- 1.0 = 完全由上下文支撑，无编造
- 0.0 = 与上下文无关或完全编造

## 24. Answer Relevancy

**答案相关性**：判断 Answer 是否回答了用户问题。

- 输入：Question + Answer
- 输出：0-1 分 + 理由
- 1.0 = 完全回答了问题
- 0.0 = 完全不相关

## 25. Baseline 对比

评测页面支持四个 RAG 版本的指标对比，形成 **RAG Version Comparison** 图表：

| 指标 | Baseline | Hybrid | Hybrid+Rerank | +Rewrite |
|------|----------|--------|---------------|----------|
| Recall@5 | - | - | - | - |
| Precision@5 | - | - | - | - |
| MRR | - | - | - | - |
| Accuracy | - | - | - | - |
| Faithfulness | - | - | - | - |
| Relevancy | - | - | - | - |
| Latency | - | - | - | - |

从对比中可以直观看到**每次优化到底解决了什么问题**：
- Hybrid → 提升术语/数字的 Recall
- Reranker → 提升 Precision 和 MRR
- Rewrite → 提升多轮对话 Accuracy

## 26. API 列表

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/chat` | 非流式问答 |
| POST | `/api/chat/stream` | SSE 流式问答 |
| POST | `/api/documents/upload` | 上传文档 |
| GET | `/api/documents` | 文档列表 |
| GET | `/api/documents/{id}` | 文档详情（含 Chunk） |
| DELETE | `/api/documents/{id}` | 删除文档 |
| POST | `/api/documents/{id}/reindex` | 重新索引 |
| GET | `/api/knowledge-bases` | 知识库列表 |
| POST | `/api/knowledge-bases` | 创建知识库 |
| DELETE | `/api/knowledge-bases/{id}` | 删除知识库 |
| POST | `/api/retrieval/search` | 检索测试 |
| POST | `/api/evaluation/run` | 启动评测 |
| GET | `/api/evaluation/tasks` | 评测任务列表 |
| GET | `/api/evaluation/tasks/{id}` | 评测任务详情 |
| GET | `/api/evaluation/results` | 评测样本结果 |
| GET | `/api/agent/traces/{conversation_id}` | Agent 执行链 |
| GET | `/api/health` | 健康检查 |

所有 API 均有：Pydantic Schema 校验、异常处理、logging、request_id。

## 27. 数据库设计

### PostgreSQL 表结构

| 表名 | 说明 | 关键字段 |
|------|------|---------|
| `users` | 用户 | id, username, email, role |
| `knowledge_bases` | 知识库 | id, name, document_count, chunk_count |
| `documents` | 文档 | id, kb_id, filename, status, chunk_count |
| `document_chunks` | 文档块 | id, doc_id, content, page, section, vector_id |
| `conversations` | 对话 | id, user_id, title, kb_id |
| `messages` | 消息 | id, conv_id, role, content, citations |
| `tool_calls` | 工具调用 | id, conv_id, tool_name, input, output, latency |
| `evaluation_tasks` | 评测任务 | id, rag_version, status, metrics |
| `evaluation_samples` | 评测样本 | id, task_id, question, metrics, analysis |
| `agent_traces` | Agent 追踪 | id, conv_id, node, input/output, latency |

### SQL Demo 表（供 SQL Tool 查询）

| 表名 | 说明 |
|------|------|
| `products` | 产品（8条） |
| `sales_orders` | 销售订单（500条，覆盖12个月） |
| `customers` | 客户（50条，5个地区） |
| `marketing_campaigns` | 营销活动（4条） |

所有表均有主键、外键、索引、`created_at`/`updated_at`。

## 28. Docker 部署

### 全栈模式（推荐用于演示）

```bash
# 复制环境变量
cp .env.example .env
# 编辑 .env，填入你的 LLM API Key

# 启动全部服务（frontend + backend + postgres + redis + milvus + es）
docker-compose up -d --build

# 查看日志
docker-compose logs -f backend

# 停止
docker-compose down
```

访问：
- 前端：http://localhost:3000
- 后端 API：http://localhost:8000
- API 文档：http://localhost:8000/docs

### 轻量模式（资源有限时）

```bash
docker-compose -f docker-compose.lite.yml up -d --build
```

> 注意：Milvus 和 Elasticsearch 对内存要求较高（建议 8GB+）。轻量模式仅启动前后端，向量检索和 BM25 会优雅降级。

## 29. 本地启动

### 前置要求
- Python 3.11+
- Node.js 18+
- PostgreSQL 14+
- Redis 7+
- Milvus 2.4+（可选，无则降级）
- Elasticsearch 8+（可选，无则降级）

### 后端

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 初始化数据库
python -c "from app.core.database import init_db; init_db()"

# 启动
uvicorn app.main:app --reload --port 8000
```

### 前端

```bash
cd frontend
npm install
npm run dev
```

访问 http://localhost:5173

### 导入 Demo 数据

```bash
# 在项目根目录
python scripts/seed_demo_data.py
```

## 30. 测试

```bash
cd backend

# 运行全部测试
pytest tests/ -v

# 带覆盖率
pytest tests/ -v --cov=app --cov-report=term-missing

# 运行单个测试文件
pytest tests/test_recall.py -v
```

### 测试文件清单

| 文件 | 测试内容 |
|------|---------|
| `test_chunking.py` | 递归分块、Markdown分块、结构感知分块 |
| `test_embedding.py` | BGE-M3 Embedding、维度检测、归一化 |
| `test_retrieval.py` | Milvus/ES 连接、集合名生成 |
| `test_hybrid_search.py` | RRF 融合、加权融合 |
| `test_reranker.py` | BGE Reranker、Top-N 截断 |
| `test_recall.py` | Recall@1/3/5/10 |
| `test_precision.py` | Precision@1/3/5/10 |
| `test_mrr.py` | Reciprocal Rank、MRR |
| `test_accuracy.py` | 规则匹配、LLM Judge |
| `test_agent.py` | Agent State、Trace Recorder |
| `test_tools.py` | Calculator、KnowledgeSearch |
| `test_sql_safety.py` | SQL 安全检查（13种禁止操作） |
| `test_api.py` | FastAPI 端点校验 |

## 31. Demo 使用

### 1. 启动服务
```bash
docker-compose up -d --build
python scripts/seed_demo_data.py
```

### 2. 知识库管理
- 进入「知识库管理」页面
- 系统已自动创建「企业知识库」，包含 9 份 Demo 文档
- 可上传自己的 PDF/DOCX/TXT/MD/CSV/XLSX

### 3. 智能对话
- 进入「智能对话」页面
- 选择知识库和 RAG 版本
- 尝试以下问题：

**简单问答：**
> 2026年差旅报销标准是什么？

**多轮对话：**
> 第一轮：公司差旅报销制度有什么变化？
> 第二轮：什么时候生效？（测试 Query Rewrite）

**复杂 Agent 场景：**
> 帮我查询2026年销售额最高的3个产品，并结合知识库里的营销政策分析原因。

### 4. RAG 评测
- 进入「RAG 评测」页面
- 选择 RAG 版本，点击「开始评测」
- 查看 Recall/Precision/MRR/Accuracy 等指标
- 点击任务详情查看逐题分析（区分 Retrieval Problem vs Generation Problem）

### 5. Agent Trace
- 进入「Agent Trace」页面
- 输入 Conversation ID
- 查看完整执行链：Intent → Plan → Tool Call → Observe → Verify

## 32. 核心技术亮点

1. **全栈 AI 应用架构**：React+TypeScript 前端（6 页面）、FastAPI 后端（15+ API）、LangGraph Agent、RAG 检索引擎、离线评测体系、Docker 全栈部署，完整闭环可直接运行。

2. **五层 RAG 检索管线**：BGE-M3 本地 Embedding（1024 维，CPU/GPU 自适应）→ Milvus 向量检索 → Elasticsearch BM25 关键词检索 → RRF 分数级融合 → BGE Reranker 交叉编码器重排，全部真实实现，零付费 Embedding API。

3. **LangGraph 状态机 Agent**：7 节点执行图（意图识别→查询改写→规划→工具调用→观察→验证→最终回答），条件边路由（直答模式 vs 工具链模式），最大迭代 5 次 + 超时 + Verification 重试三层防无限循环。

4. **意图路由双模式**：首节点判别用户意图，寒暄/通用常识走直答模式（3 秒返回，0 工具调用），企业内部数据查询走完整 RAG+工具链，避免所有问题强制检索的延迟浪费。

5. **5 个 Tool Calling**：知识库检索、SQL 查询（只读安全校验，仅 SELECT/WITH，强制 LIMIT）、计算器、文档分析、Web 搜索，每个 Tool 含 Pydantic Input Schema、错误处理和 fallback。

6. **完整 RAG 离线评测**：自建 55 题中文评测集（事实/政策/数字/多跳/指代 8 类），真实计算 Recall@K、Precision@K、MRR、Accuracy、Faithfulness、Answer Relevancy 共 12 项指标，区分 Retrieval Problem 与 Generation Problem。

7. **四版本 RAG 对比实验**：Baseline（纯向量）→ Hybrid（+BM25+RRF）→ +Reranker → +Query Rewrite，同一评测集横向对比 8 项指标，量化每次优化的收益与延迟 trade-off。

8. **生产级问答体验**：SSE 流式输出、Markdown/表格/代码块渲染、Citation 引用来源（文档名+页码+Section+Chunk 原文可展开）、Redis 多轮对话记忆（长对话自动摘要）、Agent Trace 全链路节点耗时追踪。

9. **文档 Ingestion 管线**：支持 PDF/DOCX/TXT/Markdown/CSV/XLSX 六种格式，三种分块策略（递归字符/Markdown 标题/结构感知），完整状态机 UPLOADING→PARSING→CHUNKING→EMBEDDING→INDEXING→SUCCESS，Milvus Collection 维度自动适配模型。

10. **工程化质量保障**：Python Type Hint + Pydantic 参数校验 + SQLAlchemy ORM + async/await 全异步 + structured logging（request_id/conversation_id）+ 全局异常处理，112 个 pytest 测试全部通过（13 个测试文件）。

---

## 项目结构

```
rag-agent-platform/
├── backend/
│   ├── app/
│   │   ├── api/routes/          # FastAPI 路由（chat/documents/kb/retrieval/evaluation/agent）
│   │   ├── agents/              # LangGraph Agent（graph.py, state.py, trace.py）
│   │   ├── rag/                 # RAG Pipeline（pipeline, query_rewrite, context_compression, citation）
│   │   ├── retrieval/           # 检索（milvus_store, es_store, hybrid_retriever, rrf）
│   │   ├── embeddings/          # BGE-M3 Embedding
│   │   ├── reranker/            # BGE Reranker
│   │   ├── tools/               # 5 个 Agent Tools
│   │   ├── evaluation/          # 评测（metrics, llm_judge, runner）
│   │   ├── memory/              # Redis 记忆
│   │   ├── models/              # SQLAlchemy 模型
│   │   ├── services/            # 业务服务（document, parser, chunking, llm）
│   │   ├── schemas/             # Pydantic Schemas
│   │   ├── core/                # 配置、日志、数据库、异常
│   │   └── main.py              # FastAPI 入口
│   ├── tests/                   # 13 个测试文件
│   ├── requirements.txt
│   ├── Dockerfile
│   └── pytest.ini
├── frontend/
│   ├── src/
│   │   ├── pages/               # 6 个页面
│   │   ├── components/
│   │   ├── layouts/
│   │   ├── api/                 # Axios API 客户端
│   │   ├── stores/              # Zustand 状态
│   │   ├── types/               # TypeScript 类型
│   │   └── utils/
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   ├── Dockerfile
│   └── nginx.conf
├── data/
│   ├── documents/               # 9 份 Demo 文档
│   ├── eval/                    # 55 题评测集
│   ├── uploads/
│   └── model_cache/
├── scripts/
│   ├── seed_demo_data.py        # 种子数据脚本
│   └── run_evaluation.py        # 评测脚本
├── docker-compose.yml           # 全栈部署
├── docker-compose.lite.yml      # 轻量模式
├── .env.example
├── Makefile
└── README.md
```

---

## License

MIT
