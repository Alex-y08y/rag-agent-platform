import axios from 'axios'
import type {
  KnowledgeBase,
  Document,
  DocumentDetail,
  ChatResponse,
  RetrievedDoc,
  EvaluationTask,
  EvaluationSample,
  AgentTrace,
  HealthResponse,
  RagVersion,
  User,
  AuthResponse,
  Conversation,
  ConversationDetail,
} from '../types'

const api = axios.create({
  baseURL: '/api',
  timeout: 120000,
  headers: { 'Content-Type': 'application/json' },
})

// ── JWT Token Interceptor ─────────────────────────────────
const TOKEN_KEY = 'rag_token'
export const getToken = () => localStorage.getItem(TOKEN_KEY)
export const setToken = (token: string | null) => {
  if (token) localStorage.setItem(TOKEN_KEY, token)
  else localStorage.removeItem(TOKEN_KEY)
}

api.interceptors.request.use((config) => {
  const token = getToken()
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      setToken(null)
      if (!window.location.pathname.startsWith('/login')) {
        window.location.href = '/login'
      }
    }
    return Promise.reject(error)
  },
)

// ── Auth ──────────────────────────────────────────────────
export const authApi = {
  register: (data: { username: string; email: string; password: string }) =>
    api.post<AuthResponse>('/auth/register', data),
  login: (data: { username: string; password: string }) =>
    api.post<AuthResponse>('/auth/login', data),
  me: () => api.get<User>('/auth/me'),
}

// ── Conversations ─────────────────────────────────────────
export const conversationApi = {
  list: () => api.get<Conversation[]>('/conversations'),
  get: (id: string) => api.get<ConversationDetail>(`/conversations/${id}`),
  create: (data: { title?: string; knowledge_base_id?: string }) =>
    api.post<Conversation>('/conversations', data),
  delete: (id: string) => api.delete(`/conversations/${id}`),
}

// ── Chat ──────────────────────────────────────────────
export const chatApi = {
  send: (data: {
    conversation_id?: string
    knowledge_base_id?: string
    query: string
    stream?: boolean
    rag_version?: RagVersion
    top_k?: number
  }) => api.post<ChatResponse>('/chat', data),

  stream: (
    data: {
      conversation_id?: string
      knowledge_base_id?: string
      query: string
      rag_version?: RagVersion
    },
    onEvent: (event: any) => void,
  ) => {
    const eventSource = new EventSource('/api/chat/stream')
    // Note: SSE with POST body requires fetch + ReadableStream
    return fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    }).then(async (response) => {
      const reader = response.body?.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      while (reader) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const event = JSON.parse(line.slice(6))
              onEvent(event)
            } catch (e) {
              // ignore parse errors
            }
          }
        }
      }
    })
  },
}

// ── Knowledge Bases ───────────────────────────────────
export const kbApi = {
  list: () => api.get<KnowledgeBase[]>('/knowledge-bases'),
  create: (data: { name: string; description?: string }) =>
    api.post<KnowledgeBase>('/knowledge-bases', data),
  delete: (id: string) => api.delete(`/knowledge-bases/${id}`),
}

// ── Documents ─────────────────────────────────────────
export const docApi = {
  list: (kbId?: string) =>
    api.get<Document[]>('/documents', { params: { knowledge_base_id: kbId } }),
  get: (id: string) => api.get<DocumentDetail>(`/documents/${id}`),
  upload: (kbId: string, file: File) => {
    const form = new FormData()
    form.append('knowledge_base_id', kbId)
    form.append('file', file)
    return api.post<Document>('/documents/upload', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
  },
  delete: (id: string) => api.delete(`/documents/${id}`),
  reindex: (id: string) => api.post(`/documents/${id}/reindex`),
}

// ── Retrieval ─────────────────────────────────────────
export const retrievalApi = {
  search: (data: {
    query: string
    knowledge_base_id?: string
    top_k?: number
    filters?: Record<string, any>
    rag_version?: RagVersion
  }) => api.post<{ query: string; rewritten_query?: string; results: RetrievedDoc[]; total: number; latency_ms: number }>('/retrieval/search', data),
}

// ── Evaluation ────────────────────────────────────────
export const evalApi = {
  run: (data: {
    dataset_name?: string
    rag_version?: RagVersion
    name?: string
    max_samples?: number
  }) => api.post<EvaluationTask>('/evaluation/run', data),
  tasks: () => api.get<EvaluationTask[]>('/evaluation/tasks'),
  task: (id: string) => api.get<EvaluationTask>(`/evaluation/tasks/${id}`),
  results: (taskId?: string) =>
    api.get<EvaluationSample[]>('/evaluation/results', { params: { task_id: taskId } }),
}

// ── Agent Trace ───────────────────────────────────────
export const traceApi = {
  get: (conversationId: string) =>
    api.get<AgentTrace[]>(`/agent/traces/${conversationId}`),
}

// ── Health ────────────────────────────────────────────
export const healthApi = {
  check: () => api.get<HealthResponse>('/health'),
}

// ── Settings ──────────────────────────────────────────
export interface SettingsData {
  LLM_PROVIDER: string
  LLM_MODEL: string
  LLM_BASE_URL: string
  OPENAI_API_KEY: string
  DASHSCOPE_API_KEY: string
  _OPENAI_API_KEY_configured: boolean
  _DASHSCOPE_API_KEY_configured: boolean
  LLM_TEMPERATURE: number
  LLM_MAX_TOKENS: number
  LLM_TIMEOUT: number
}

export interface TestResult {
  ok: boolean
  error?: string
  model?: string
  base_url?: string
  reply?: string
}

export const settingsApi = {
  get: () => api.get<SettingsData>('/settings'),
  update: (data: Partial<SettingsData>) => api.post<SettingsData>('/settings', data),
  test: () => api.post<TestResult>('/settings/test'),
  reset: () => api.post('/settings/reset'),
}

export default api
