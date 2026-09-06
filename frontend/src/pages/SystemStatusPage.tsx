import { useState, useEffect } from 'react'
import { Card, Row, Col, Statistic, Tag, Table, Button, Space } from 'antd'
import {
  DashboardOutlined,
  DatabaseOutlined,
  CloudOutlined,
  ReloadOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
} from '@ant-design/icons'
import { healthApi, kbApi, docApi } from '../api/client'
import type { HealthResponse, KnowledgeBase, Document } from '../types'

export default function SystemStatusPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [kbs, setKbs] = useState<KnowledgeBase[]>([])
  const [docs, setDocs] = useState<Document[]>([])
  const [loading, setLoading] = useState(true)

  const loadAll = () => {
    setLoading(true)
    Promise.all([healthApi.check(), kbApi.list(), docApi.list()])
      .then(([h, k, d]) => {
        setHealth(h.data)
        setKbs(k.data)
        setDocs(d.data)
      })
      .catch(() => {})
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    loadAll()
  }, [])

  const totalChunks = kbs.reduce((sum, kb) => sum + (kb.chunk_count || 0), 0)
  const successDocs = docs.filter((d) => d.status === 'SUCCESS').length
  const failedDocs = docs.filter((d) => d.status === 'FAILED').length

  const serviceStatus = health?.services || {}

  return (
    <div>
      <div style={{ marginBottom: 16, display: 'flex', justifyContent: 'flex-end' }}>
        <Button icon={<ReloadOutlined />} onClick={loadAll} loading={loading}>
          刷新
        </Button>
      </div>

      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="系统状态"
              value={health?.status || 'unknown'}
              valueStyle={{
                color: health?.status === 'healthy' ? '#52c41a' : '#faad14',
                fontSize: 20,
              }}
              prefix={health?.status === 'healthy' ? <CheckCircleOutlined /> : <CloseCircleOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card><Statistic title="知识库数量" value={kbs.length} prefix={<DatabaseOutlined />} /></Card>
        </Col>
        <Col span={6}>
          <Card><Statistic title="文档总数" value={docs.length} prefix={<CloudOutlined />} /></Card>
        </Col>
        <Col span={6}>
          <Card><Statistic title="Chunk 总数" value={totalChunks} /></Card>
        </Col>
      </Row>

      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={12}>
          <Card title="服务状态">
            <Table
              dataSource={Object.entries(serviceStatus).map(([name, status]) => ({
                key: name,
                name,
                status,
                healthy: status === 'healthy',
              }))}
              columns={[
                { title: '服务', dataIndex: 'name', key: 'name' },
                {
                  title: '状态',
                  dataIndex: 'status',
                  key: 'status',
                  render: (status: string, record: any) => (
                    <Tag color={record.healthy ? 'success' : 'error'}>
                      {record.healthy ? <CheckCircleOutlined /> : <CloseCircleOutlined />} {status}
                    </Tag>
                  ),
                },
              ]}
              pagination={false}
              size="small"
            />
          </Card>
        </Col>
        <Col span={12}>
          <Card title="文档处理状态">
            <Row gutter={16}>
              <Col span={8}>
                <Statistic title="成功" value={successDocs} valueStyle={{ color: '#52c41a' }} />
              </Col>
              <Col span={8}>
                <Statistic title="处理中" value={docs.length - successDocs - failedDocs} valueStyle={{ color: '#1677ff' }} />
              </Col>
              <Col span={8}>
                <Statistic title="失败" value={failedDocs} valueStyle={{ color: '#ff4d4f' }} />
              </Col>
            </Row>
            <div style={{ marginTop: 16 }}>
              <Table
                dataSource={docs.slice(0, 5)}
                columns={[
                  { title: '文件', dataIndex: 'filename', key: 'filename', ellipsis: true },
                  {
                    title: '状态',
                    dataIndex: 'status',
                    key: 'status',
                    render: (v: string) => (
                      <Tag color={v === 'SUCCESS' ? 'success' : v === 'FAILED' ? 'error' : 'processing'}>
                        {v}
                      </Tag>
                    ),
                  },
                ]}
                pagination={false}
                size="small"
              />
            </div>
          </Card>
        </Col>
      </Row>

      <Card title="技术栈信息">
        <Row gutter={16}>
          <Col span={8}>
            <div style={{ marginBottom: 8 }}><strong>前端</strong></div>
            <div>React 18 + TypeScript + Vite</div>
            <div>Ant Design + Tailwind CSS</div>
            <div>ECharts 可视化</div>
          </Col>
          <Col span={8}>
            <div style={{ marginBottom: 8 }}><strong>后端</strong></div>
            <div>FastAPI + Python 3.11+</div>
            <div>SQLAlchemy + Pydantic</div>
            <div>LangGraph + LangChain</div>
          </Col>
          <Col span={8}>
            <div style={{ marginBottom: 8 }}><strong>AI 基础设施</strong></div>
            <div>BGE-M3 Embedding (本地)</div>
            <div>BGE Reranker v2-m3 (本地)</div>
            <div>Milvus + Elasticsearch</div>
            <div>PostgreSQL + Redis</div>
          </Col>
        </Row>
      </Card>
    </div>
  )
}
