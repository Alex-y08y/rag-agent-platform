import { useState, useEffect } from 'react'
import {
  Card,
  Button,
  Table,
  Tag,
  Descriptions,
  Progress,
  Empty,
  Spin,
  Row,
  Col,
  Statistic,
} from 'antd'
import { ArrowLeftOutlined, AlertOutlined } from '@ant-design/icons'
import { useParams, useNavigate } from 'react-router-dom'
import { evalApi } from '../api/client'
import type { EvaluationTask, EvaluationSample } from '../types'

export default function EvaluationDetailPage() {
  const { taskId } = useParams<{ taskId: string }>()
  const navigate = useNavigate()
  const [task, setTask] = useState<EvaluationTask | null>(null)
  const [samples, setSamples] = useState<EvaluationSample[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (taskId) {
      setLoading(true)
      Promise.all([evalApi.task(taskId), evalApi.results(taskId)]).then(([t, s]) => {
        setTask(t.data)
        setSamples(s.data)
        setLoading(false)
      }).catch(() => setLoading(false))
    }
  }, [taskId])

  if (loading) return <Spin style={{ display: 'block', marginTop: 80 }} />
  if (!task) return <Empty description="任务不存在" />

  const m = task.metrics || {}

  const columns = [
    { title: '#', key: 'idx', width: 50, render: (_: any, __: any, i: number) => i + 1 },
    {
      title: '问题',
      dataIndex: 'question',
      key: 'question',
      width: 200,
      ellipsis: true,
    },
    {
      title: '期望来源',
      dataIndex: 'expected_sources',
      key: 'expected_sources',
      width: 150,
      render: (v: string[]) => v?.map((s) => <Tag key={s} color="green">{s}</Tag>),
    },
    {
      title: '召回来源',
      dataIndex: 'retrieved_sources',
      key: 'retrieved_sources',
      width: 150,
      render: (v: string[]) => v?.slice(0, 3).map((s) => <Tag key={s}>{s}</Tag>),
    },
    {
      title: 'Recall@5',
      key: 'recall5',
      width: 90,
      render: (_: any, r: EvaluationSample) => {
        const v = r.metrics?.recall_at_5 ?? 0
        return <Tag color={v >= 1 ? 'success' : v > 0 ? 'warning' : 'error'}>{v.toFixed(2)}</Tag>
      },
    },
    {
      title: 'MRR',
      key: 'mrr',
      width: 80,
      render: (_: any, r: EvaluationSample) => (r.metrics?.mrr ?? 0).toFixed(2),
    },
    {
      title: 'Accuracy',
      key: 'acc',
      width: 90,
      render: (_: any, r: EvaluationSample) => {
        const v = r.metrics?.accuracy ?? 0
        return <Tag color={v >= 0.7 ? 'success' : v >= 0.4 ? 'warning' : 'error'}>{v.toFixed(2)}</Tag>
      },
    },
    {
      title: 'Faithfulness',
      key: 'faith',
      width: 100,
      render: (_: any, r: EvaluationSample) => (r.metrics?.faithfulness ?? 0).toFixed(2),
    },
    {
      title: '延迟',
      dataIndex: 'latency_ms',
      key: 'latency',
      width: 80,
      render: (v: number) => `${v}ms`,
    },
  ]

  return (
    <div>
      <Button icon={<ArrowLeftOutlined />} onClick={() => navigate('/evaluation')} style={{ marginBottom: 16 }}>
        返回评测列表
      </Button>

      <Card title="评测任务概览" style={{ marginBottom: 16 }}>
        <Descriptions column={4} size="small">
          <Descriptions.Item label="任务名">{task.name}</Descriptions.Item>
          <Descriptions.Item label="RAG版本"><Tag color="blue">{task.rag_version}</Tag></Descriptions.Item>
          <Descriptions.Item label="状态">
            <Tag color={task.status === 'COMPLETED' ? 'success' : 'processing'}>{task.status}</Tag>
          </Descriptions.Item>
          <Descriptions.Item label="样本数">{task.total_samples}</Descriptions.Item>
        </Descriptions>
        <div style={{ marginTop: 12 }}>
          <Progress
            percent={task.total_samples ? Math.round((task.completed_samples / task.total_samples) * 100) : 0}
          />
        </div>
      </Card>

      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={3}><Card><Statistic title="Recall@5" value={m.recall_at_5} precision={3} /></Card></Col>
        <Col span={3}><Card><Statistic title="Precision@5" value={m.precision_at_5} precision={3} /></Card></Col>
        <Col span={3}><Card><Statistic title="MRR" value={m.mrr} precision={3} /></Card></Col>
        <Col span={3}><Card><Statistic title="Accuracy" value={m.accuracy} precision={3} /></Card></Col>
        <Col span={3}><Card><Statistic title="Faithfulness" value={m.faithfulness} precision={3} /></Card></Col>
        <Col span={3}><Card><Statistic title="Relevancy" value={m.answer_relevancy} precision={3} /></Card></Col>
        <Col span={3}><Card><Statistic title="Avg Latency" value={m.avg_latency_ms} suffix="ms" /></Card></Col>
        <Col span={3}><Card><Statistic title="总样本" value={m.total_samples} /></Card></Col>
      </Row>

      <Card title="逐题评测详情">
        <Table
          dataSource={samples}
          columns={columns}
          rowKey="id"
          pagination={{ pageSize: 10 }}
          size="small"
          expandable={{
            expandedRowRender: (record: EvaluationSample) => (
              <div style={{ padding: '8px 16px' }}>
                <Descriptions column={1} size="small" bordered>
                  <Descriptions.Item label="问题">{record.question}</Descriptions.Item>
                  <Descriptions.Item label="标准答案">{record.ground_truth || '-'}</Descriptions.Item>
                  <Descriptions.Item label="生成答案">{record.generated_answer || '-'}</Descriptions.Item>
                  <Descriptions.Item label="期望来源">
                    {record.expected_sources?.join(', ') || '-'}
                  </Descriptions.Item>
                  <Descriptions.Item label="召回来源">
                    {record.retrieved_sources?.join(', ') || '-'}
                  </Descriptions.Item>
                  <Descriptions.Item label="指标详情">
                    {JSON.stringify(record.metrics, null, 2)}
                  </Descriptions.Item>
                  {record.analysis && (
                    <Descriptions.Item label={
                      <span><AlertOutlined style={{ color: '#faad14' }} /> 问题分析</span>
                    }>
                      <span style={{ color: record.analysis.includes('Retrieval Problem') ? '#ff4d4f' : record.analysis.includes('Generation Problem') ? '#faad14' : '#52c41a' }}>
                        {record.analysis}
                      </span>
                    </Descriptions.Item>
                  )}
                </Descriptions>
              </div>
            ),
          }}
        />
      </Card>
    </div>
  )
}
