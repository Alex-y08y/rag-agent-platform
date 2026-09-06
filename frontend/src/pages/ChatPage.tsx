import { useState, useRef, useEffect, useCallback } from 'react'
import {
  Input,
  Button,
  Card,
  Avatar,
  Spin,
  Tag,
  Collapse,
  Select,
  Empty,
  Tooltip,
  List,
  Popconfirm,
  message,
} from 'antd'
import {
  SendOutlined,
  UserOutlined,
  RobotOutlined,
  PlusOutlined,
  FileTextOutlined,
  LoadingOutlined,
  CheckCircleOutlined,
  SearchOutlined,
  DatabaseOutlined,
  DeleteOutlined,
  MessageOutlined,
} from '@ant-design/icons'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useNavigate, useParams } from 'react-router-dom'
import { kbApi, conversationApi } from '../api/client'
import { useAppStore } from '../stores/useStore'
import type {
  ChatMessage,
  Citation,
  KnowledgeBase,
  RagVersion,
  Conversation,
  ConversationMessage,
} from '../types'

const { TextArea } = Input

const NODE_ICONS: Record<string, any> = {
  analyze_intent: <SearchOutlined />,
  intent_route: <CheckCircleOutlined />,
  direct_answer: <RobotOutlined />,
  rewrite_query: <SearchOutlined />,
  plan: <FileTextOutlined />,
  tool_call: <DatabaseOutlined />,
  observe: <CheckCircleOutlined />,
  final_answer: <RobotOutlined />,
  verify: <CheckCircleOutlined />,
}

const NODE_LABELS: Record<string, string> = {
  analyze_intent: '分析问题',
  intent_route: '意图路由',
  direct_answer: '直接回答',
  rewrite_query: '优化查询',
  plan: '制定计划',
  tool_call: '执行工具',
  observe: '观察结果',
  final_answer: '生成回答',
  verify: '验证答案',
}

export default function ChatPage() {
  const navigate = useNavigate()
  const { conversationId: urlConvId } = useParams()
  const [input, setInput] = useState('')
  const [kbs, setKbs] = useState<KnowledgeBase[]>([])
  const [selectedKb, setSelectedKb] = useState<string>('')
  const [ragVersion, setRagVersion] = useState<RagVersion>('hybrid_rerank_rewrite')
  const [loading, setLoading] = useState(false)
  const [agentSteps, setAgentSteps] = useState<Array<{ node: string; message: string; status: string }>>([])
  const [streamAnswer, setStreamAnswer] = useState('')
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [currentConvId, setCurrentConvId] = useState<string>('')
  const [convLoading, setConvLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const { messages, addMessage, setMessages, currentKb, setCurrentKb } = useAppStore()

  // Load knowledge bases
  useEffect(() => {
    kbApi.list().then((res) => {
      setKbs(res.data)
      if (res.data.length > 0 && !selectedKb) {
        setSelectedKb(res.data[0].id)
        setCurrentKb(res.data[0])
      }
    })
  }, [])

  // Load conversation list
  const loadConversations = useCallback(async () => {
    try {
      const res = await conversationApi.list()
      setConversations(res.data)
    } catch {
      // ignore
    }
  }, [])

  useEffect(() => {
    loadConversations()
  }, [loadConversations])

  // Load conversation messages when currentConvId changes
  useEffect(() => {
    if (!currentConvId) {
      setMessages([])
      return
    }
    setConvLoading(true)
    conversationApi
      .get(currentConvId)
      .then((res) => {
        const msgs: ChatMessage[] = res.data.messages.map((m: ConversationMessage) => ({
          id: m.id,
          role: m.role as 'user' | 'assistant',
          content: m.content,
          citations: m.citations,
          timestamp: m.created_at,
        }))
        setMessages(msgs)
      })
      .catch(() => setMessages([]))
      .finally(() => setConvLoading(false))
  }, [currentConvId, setMessages])

  // Sync URL param with current conversation
  useEffect(() => {
    if (urlConvId && urlConvId !== currentConvId) {
      setCurrentConvId(urlConvId)
    }
  }, [urlConvId, currentConvId])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streamAnswer])

  const handleSend = async () => {
    if (!input.trim() || loading) return
    const query = input.trim()
    setInput('')
    setLoading(true)
    setStreamAnswer('')
    setAgentSteps([])

    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: query,
    }
    addMessage(userMsg)

    try {
      const response = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${localStorage.getItem('rag_token') || ''}`,
        },
        body: JSON.stringify({
          query,
          conversation_id: currentConvId || undefined,
          knowledge_base_id: selectedKb || undefined,
          rag_version: ragVersion,
          stream: true,
        }),
      })

      const reader = response.body?.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let answer = ''
      let citations: Citation[] = []
      let newConvId = ''

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
              if (event.type === 'status') {
                setAgentSteps((prev) => {
                  const exists = prev.find((s) => s.node === event.data.node)
                  if (exists) return prev
                  return [...prev, { ...event.data, status: 'active' }]
                })
              } else if (event.type === 'tool_call') {
                setAgentSteps((prev) => [
                  ...prev.map((s) => ({ ...s, status: 'done' })),
                  { node: 'tool_call', message: `调用工具: ${event.data.tool}`, status: 'active' },
                ])
              } else if (event.type === 'token') {
                answer += event.data.content
                setStreamAnswer(answer)
              } else if (event.type === 'citation') {
                citations = event.data.citations || []
              } else if (event.type === 'done') {
                answer = event.data.answer || answer
                citations = event.data.citations || []
                newConvId = event.data.conversation_id
              } else if (event.type === 'error') {
                answer = `错误: ${event.data.message}`
              }
            } catch {
              // ignore
            }
          }
        }
      }

      setAgentSteps((prev) => prev.map((s) => ({ ...s, status: 'done' })))

      const assistantMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: answer,
        citations,
      }
      addMessage(assistantMsg)
      setStreamAnswer('')

      // Update conversation state
      if (newConvId && newConvId !== currentConvId) {
        setCurrentConvId(newConvId)
        navigate(`/chat/${newConvId}`, { replace: true })
      }
      // Refresh conversation list (new title etc.)
      loadConversations()
    } catch (err: any) {
      addMessage({
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `请求失败: ${err.message}`,
      })
    } finally {
      setLoading(false)
    }
  }

  const handleNewChat = () => {
    setCurrentConvId('')
    setMessages([])
    setAgentSteps([])
    setStreamAnswer('')
    navigate('/chat')
  }

  const handleSelectConversation = (conv: Conversation) => {
    setCurrentConvId(conv.id)
    navigate(`/chat/${conv.id}`)
  }

  const handleDeleteConversation = async (e: React.MouseEvent, convId: string) => {
    e.stopPropagation()
    try {
      await conversationApi.delete(convId)
      message.success('对话已删除')
      if (convId === currentConvId) {
        handleNewChat()
      }
      loadConversations()
    } catch {
      message.error('删除失败')
    }
  }

  return (
    <div style={{ display: 'flex', height: 'calc(100vh - 160px)' }}>
      {/* Left sidebar: conversations + settings */}
      <div style={{ width: 260, borderRight: '1px solid #f0f0f0', paddingRight: 12, marginRight: 16, display: 'flex', flexDirection: 'column' }}>
        <Button
          type="primary"
          icon={<PlusOutlined />}
          block
          onClick={handleNewChat}
          style={{ marginBottom: 12 }}
        >
          新建对话
        </Button>

        {/* Conversation list */}
        <div style={{ flex: 1, overflowY: 'auto', marginBottom: 12, minHeight: 0 }}>
          <div style={{ fontSize: 12, color: '#999', marginBottom: 6, fontWeight: 600 }}>
            历史对话 ({conversations.length})
          </div>
          {conversations.length === 0 ? (
            <div style={{ fontSize: 12, color: '#ccc', textAlign: 'center', padding: '16px 0' }}>
              暂无对话
            </div>
          ) : (
            <List
              size="small"
              dataSource={conversations}
              renderItem={(conv) => (
                <div
                  onClick={() => handleSelectConversation(conv)}
                  style={{
                    padding: '8px 10px',
                    borderRadius: 6,
                    cursor: 'pointer',
                    background: conv.id === currentConvId ? '#e6f4ff' : 'transparent',
                    border: conv.id === currentConvId ? '1px solid #91caff' : '1px solid transparent',
                    marginBottom: 4,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                  }}
                >
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div
                      style={{
                        fontSize: 13,
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      <MessageOutlined style={{ marginRight: 6, color: '#1677ff', fontSize: 12 }} />
                      {conv.title || '新对话'}
                    </div>
                    <div style={{ fontSize: 11, color: '#999', marginTop: 2 }}>
                      {conv.message_count} 条消息
                    </div>
                  </div>
                  <Popconfirm
                    title="确定删除这个对话？"
                    onConfirm={(e) => handleDeleteConversation(e as any, conv.id)}
                    okText="删除"
                    cancelText="取消"
                  >
                    <DeleteOutlined
                      style={{ color: '#999', fontSize: 12, padding: 4 }}
                      onClick={(e) => e.stopPropagation()}
                    />
                  </Popconfirm>
                </div>
              )}
            />
          )}
        </div>

        {/* KB and RAG version selectors */}
        <div style={{ borderTop: '1px solid #f0f0f0', paddingTop: 12 }}>
          <div style={{ marginBottom: 10 }}>
            <div style={{ fontSize: 12, color: '#999', marginBottom: 4 }}>选择知识库</div>
            <Select
              style={{ width: '100%' }}
              value={selectedKb || undefined}
              placeholder="选择知识库"
              onChange={(val) => {
                setSelectedKb(val)
                const kb = kbs.find((k) => k.id === val)
                if (kb) setCurrentKb(kb)
              }}
              options={kbs.map((kb) => ({ label: kb.name, value: kb.id }))}
            />
          </div>

          <div style={{ marginBottom: 8 }}>
            <div style={{ fontSize: 12, color: '#999', marginBottom: 4 }}>RAG 版本</div>
            <Select
              style={{ width: '100%' }}
              value={ragVersion}
              onChange={setRagVersion}
              options={[
                { label: 'Baseline (Vector)', value: 'baseline' },
                { label: 'Hybrid (Vector+BM25)', value: 'hybrid' },
                { label: 'Hybrid + Reranker', value: 'hybrid_rerank' },
                { label: 'Hybrid + Reranker + Rewrite', value: 'hybrid_rerank_rewrite' },
              ]}
            />
          </div>
        </div>
      </div>

      {/* Chat area */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        {/* Messages */}
        <div style={{ flex: 1, overflowY: 'auto', paddingRight: 8 }}>
          {convLoading ? (
            <div style={{ textAlign: 'center', padding: 40 }}>
              <Spin />
            </div>
          ) : messages.length === 0 && !loading ? (
            <Empty description="开始与企业知识库对话" style={{ marginTop: 80 }} />
          ) : (
            <>
              {messages.map((msg) => (
                <MessageBubble key={msg.id} message={msg} />
              ))}

              {/* Streaming answer */}
              {streamAnswer && (
                <div style={{ display: 'flex', gap: 12, marginBottom: 16 }}>
                  <Avatar icon={<RobotOutlined />} style={{ backgroundColor: '#1677ff' }} />
                  <div style={{ flex: 1 }}>
                    <Card size="small" style={{ background: '#f6f8fa' }}>
                      <ReactMarkdown remarkPlugins={[remarkGfm]} className="markdown-body">
                        {streamAnswer}
                      </ReactMarkdown>
                    </Card>
                  </div>
                </div>
              )}

              {/* Agent steps */}
              {agentSteps.length > 0 && (
                <div style={{ marginBottom: 16, padding: '8px 12px', background: '#fafafa', borderRadius: 6 }}>
                  <div style={{ fontSize: 12, color: '#999', marginBottom: 6 }}>Agent 执行过程</div>
                  {agentSteps.map((step, i) => (
                    <div key={i} className={`agent-step ${step.status}`}>
                      <span className="step-icon">
                        {step.status === 'active' ? (
                          <LoadingOutlined spin />
                        ) : step.status === 'done' ? (
                          <CheckCircleOutlined />
                        ) : (
                          NODE_ICONS[step.node] || <FileTextOutlined />
                        )}
                      </span>
                      <span>{NODE_LABELS[step.node] || step.node}: {step.message}</span>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div style={{ borderTop: '1px solid #f0f0f0', paddingTop: 12, marginTop: 12 }}>
          <div style={{ display: 'flex', gap: 8 }}>
            <TextArea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  handleSend()
                }
              }}
              placeholder="输入您的问题，按 Enter 发送..."
              autoSize={{ minRows: 1, maxRows: 4 }}
              disabled={loading}
            />
            <Button
              type="primary"
              icon={loading ? <Spin size="small" /> : <SendOutlined />}
              onClick={handleSend}
              disabled={loading || !input.trim()}
              style={{ height: 'auto' }}
            >
              发送
            </Button>
          </div>
          {currentConvId && (
            <div style={{ fontSize: 11, color: '#999', marginTop: 6 }}>
              当前对话 ID: {currentConvId} · 多轮上下文已启用
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === 'user'
  return (
    <div style={{ display: 'flex', gap: 12, marginBottom: 16, flexDirection: isUser ? 'row-reverse' : 'row' }}>
      <Avatar
        icon={isUser ? <UserOutlined /> : <RobotOutlined />}
        style={{ backgroundColor: isUser ? '#52c41a' : '#1677ff' }}
      />
      <div style={{ flex: 1, maxWidth: '85%' }}>
        <Card size="small" style={{ background: isUser ? '#e6f7ff' : '#f6f8fa' }}>
          <ReactMarkdown remarkPlugins={[remarkGfm]} className="markdown-body">
            {message.content}
          </ReactMarkdown>
        </Card>

        {/* Citations */}
        {message.citations && message.citations.length > 0 && (
          <div style={{ marginTop: 8 }}>
            <div style={{ fontSize: 12, color: '#999', marginBottom: 4 }}>
              参考来源 ({message.citations.length})
            </div>
            <Collapse
              size="small"
              ghost
              items={message.citations.map((c, i) => ({
                key: i,
                label: (
                  <span>
                    <FileTextOutlined style={{ marginRight: 6, color: '#1677ff' }} />
                    {c.filename}
                    {c.page && <Tag color="blue" style={{ marginLeft: 8 }}>P{c.page}</Tag>}
                    {c.section && <Tag>{c.section}</Tag>}
                  </span>
                ),
                children: (
                  <div style={{ fontSize: 13, color: '#555', lineHeight: 1.6 }}>
                    {c.content}
                  </div>
                ),
              }))}
            />
          </div>
        )}
      </div>
    </div>
  )
}
