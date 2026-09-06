// API Type Definitions

export interface KnowledgeBase {
  id: string
  name: string
  description: string
  document_count: number
  chunk_count: number
  status: string
  created_at: string
}

export interface Document {
  id: string
  knowledge_base_id: string
  filename: string
  file_type: string
  file_size: number
  status: string
  chunk_count: number
  created_at: string
  error_message?: string | null
}

export interface Chunk {
  id: string
  document_id: string
  chunk_index: number
  content: string
  page: number | null
  section: string | null
  title: string | null
  source: string | null
  metadata: Record<string, any>
  embedding_status: string
}

export interface DocumentDetail extends Document {
  chunks: Chunk[]
}

export interface Citation {
  document_id: string
  filename: string
  page: number | null
  section: string | null
  title: string | null
  chunk_id: string
  content: string
  score: number
}

export interface ChatResponse {
  conversation_id: string
  message_id: string
  answer: string
  citations: Citation[]
  retrieved_docs: number
  latency_ms: number
  token_usage: Record<string, number>
  tool_calls: any[]
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  timestamp?: string
  tool_calls?: any[]
}

export interface RetrievedDoc {
  chunk_id: string
  document_id: string
  content: string
  score: number
  source: string | null
  page: number | null
  section: string | null
  title: string | null
  metadata: Record<string, any>
}

export interface EvaluationTask {
  id: string
  name: string
  dataset_name: string
  rag_version: string
  status: string
  total_samples: number
  completed_samples: number
  metrics: Record<string, any>
  created_at: string
}

export interface EvaluationSample {
  id: string
  task_id: string
  question: string
  ground_truth: string | null
  expected_sources: string[]
  category: string | null
  difficulty: string | null
  retrieved_sources: string[]
  generated_answer: string | null
  metrics: Record<string, any>
  latency_ms: number
  analysis: string | null
}

export interface AgentTrace {
  id: string
  conversation_id: string
  agent_run_id: string
  node: string
  input_data: Record<string, any>
  output_data: Record<string, any>
  tool_name: string | null
  latency_ms: number
  status: string
  token_usage: Record<string, any>
  error_message: string | null
  sequence: number
  created_at: string
}

export interface HealthResponse {
  status: string
  version: string
  services: Record<string, string>
}

export type RagVersion = 'baseline' | 'hybrid' | 'hybrid_rerank' | 'hybrid_rerank_rewrite'

// ── Auth ──────────────────────────────────────────────────
export interface User {
  id: string
  username: string
  email: string
  role: string
  created_at?: string
}

export interface AuthResponse {
  access_token: string
  token_type: string
  user: User
}

export interface Conversation {
  id: string
  title: string
  knowledge_base_id?: string | null
  message_count: number
  created_at?: string
  updated_at?: string
}

export interface ConversationMessage {
  id: string
  role: string
  content: string
  citations?: Citation[]
  token_usage?: Record<string, number>
  created_at?: string
}

export interface ConversationDetail extends Conversation {
  messages: ConversationMessage[]
}
