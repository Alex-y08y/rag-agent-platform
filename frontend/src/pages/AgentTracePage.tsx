import { useState, useEffect } from 'react'
import {
  Card,
  Input,
  Button,
  Timeline,
  Tag,
  Table,
  Empty,
  Spin,
  Descriptions,
  Row,
  Col,
  Statistic,
} from 'antd'
import {
  SearchOutlined,
  ApiOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
} from '@ant-design/icons'
import { traceApi } from '../api/client'
import type { AgentTrace } from '../types'

const NODE_LABELS: Record<string, string> = {
  analyze_intent: '分析意图',
  rewrite_query: '查询重写',
  plan: '任务规划',
  retrieve: '检索知识库',
  tool_call: '工具调用',
  observe: '观察结果',
  verify: '验证答案',
  final_answer: '生成最终答案',
  error: '错误',
}

const NODE_COLORS: Record<string, string> = {
  analyze_intent: 'blue',
  rewrite_query: 'cyan',
  plan: 'geekblue',
  retrieve: 'green',
  tool_call: 'orange',
  observe: 'purple',
  verify: 'magenta',
  final_answer: 'success',
  error: 'red',
}

export default function AgentTracePage() {
  const [conversationId, setConversationId] = useState('')
  const [traces, setTraces] = useState<AgentTrace[]>([])
  const [loading, setLoading] = useState(false)
  const [searched, setSearched] = useState(false)

  const handleSearch = async () => {
    if (!conversationId.trim()) return
    setLoading(true)
    setSearched(true)
    try {
      const res = await traceApi.get(conversationId.trim())
      setTraces(res.data)
    } catch (err) {
      setTraces([])
    } finally {
      setLoading(false)
    }
  }

  // Group by agent_run_id
  const groupedTraces = traces.reduce((acc, t) => {
    if (!acc[t.agent_run_id]) acc[t.agent_run_id] = []
    acc[t.agent_run_id].push(t)
    return acc
  }, {} as Record<string, AgentTrace[]>)

  const runIds = Object.keys(groupedTraces)
  const totalLatency = traces.reduce((sum, t) => sum + t.latency_ms, 0)
  const totalTokens = traces.reduce((sum, t) => {
    const tu = t.token_usage || {}
    return sum + (tu.total_tokens || tu.prompt_tokens || 0) + (tu.completion_tokens || 0)
  }, 0)

  return (
    <div>
      <Card style={{ marginBottom: 16 }}>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          <Input
            placeholder="输入 Conversation ID 查看 Agent 执行链"
            value={conversationId}
            onChange={(e) => setConversationId(e.target.value)}
            onPressEnter={handleSearch}
            style={{ width: 400 }}
            prefix={<ApiOutlined />}
          />
          <Button type="primary" icon={<SearchOutlined />} onClick={handleSearch} loading={loading}>
            查询 Trace
          </Button>
        </div>
      </Card>

      {loading && <Spin style={{ display: 'block', marginTop: 40 }} />}

      {!loading && searched && traces.length === 0 && (
        <Empty description="未找到该 Conversation 的 Trace 记录" />
      )}

      {traces.length > 0 && (
        <>
          <Row gutter={16} style={{ marginBottom: 16 }}>
            <Col span={6}><Card><Statistic title="执行轮次" value={runIds.length} /></Card></Col>
            <Col span={6}><Card><Statistic title="总节点数" value={traces.length} /></Card></Col>
            <Col span={6}><Card><Statistic title="总延迟" value={totalLatency} suffix="ms" /></Card></Col>
            <Col span={6}><Card><Statistic title="Token 消耗" value={totalTokens} /></Card></Col>
          </Row>

          {runIds.map((runId) => (
            <Card
              key={runId}
              title={
                <span>
                  <ApiOutlined style={{ marginRight: 8 }} />
                  Agent Run: {runId}
                  <Tag color="blue" style={{ marginLeft: 12 }}>
                    {groupedTraces[runId].length} 个节点
                  </Tag>
                </span>
              }
              style={{ marginBottom: 16 }}
            >
              <Timeline
                items={groupedTraces[runId].map((t) => ({
                  color: t.status === 'error' ? 'red' : t.status === 'success' ? 'green' : 'blue',
                  dot: t.status === 'error' ? <CloseCircleOutlined /> : t.status === 'success' ? <CheckCircleOutlined /> : <ClockCircleOutlined />,
                  children: (
                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                        <Tag color={NODE_COLORS[t.node] || 'default'}>
                          {NODE_LABELS[t.node] || t.node}
                        </Tag>
                        {t.tool_name && <Tag color="orange">{t.tool_name}</Tag>}
                        <span style={{ color: '#999', fontSize: 12 }}>{t.latency_ms}ms</span>
                        <Tag color={t.status === 'success' ? 'success' : 'error'}>{t.status}</Tag>
                      </div>
                      <Descriptions column={2} size="small" bordered style={{ marginTop: 8 }}>
                        <Descriptions.Item label="Input" span={2}>
                          <code style={{ fontSize: 12, wordBreak: 'break-all' }}>
                            {JSON.stringify(t.input_data).slice(0, 300)}
                          </code>
                        </Descriptions.Item>
                        <Descriptions.Item label="Output" span={2}>
                          <code style={{ fontSize: 12, wordBreak: 'break-all' }}>
                            {JSON.stringify(t.output_data).slice(0, 300)}
                          </code>
                        </Descriptions.Item>
                        {t.token_usage && Object.keys(t.token_usage).length > 0 && (
                          <Descriptions.Item label="Token Usage">
                            {JSON.stringify(t.token_usage)}
                          </Descriptions.Item>
                        )}
                        {t.error_message && (
                          <Descriptions.Item label="Error" span={2}>
                            <span style={{ color: '#ff4d4f' }}>{t.error_message}</span>
                          </Descriptions.Item>
                        )}
                      </Descriptions>
                    </div>
                  ),
                }))}
              />
            </Card>
          ))}
        </>
      )}
    </div>
  )
}
