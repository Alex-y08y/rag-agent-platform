import { useState, useEffect } from 'react'
import {
  Card,
  Button,
  Select,
  Row,
  Col,
  Statistic,
  Table,
  Tag,
  Progress,
  Space,
  message,
  Spin,
} from 'antd'
import {
  PlayCircleOutlined,
  BarChartOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons'
import ReactECharts from 'echarts-for-react'
import { useNavigate } from 'react-router-dom'
import { evalApi } from '../api/client'
import type { EvaluationTask, RagVersion } from '../types'

const VERSION_LABELS: Record<string, string> = {
  baseline: 'Baseline (Vector)',
  hybrid: 'Hybrid (Vector+BM25)',
  hybrid_rerank: 'Hybrid + Reranker',
  hybrid_rerank_rewrite: 'Hybrid + Reranker + Rewrite',
}

export default function EvaluationPage() {
  const [tasks, setTasks] = useState<EvaluationTask[]>([])
  const [loading, setLoading] = useState(true)
  const [running, setRunning] = useState(false)
  const [selectedVersion, setSelectedVersion] = useState<RagVersion>('hybrid_rerank_rewrite')
  const [dataset, setDataset] = useState('rag_eval')
  const navigate = useNavigate()

  const loadTasks = () => {
    evalApi.tasks().then((res) => {
      setTasks(res.data)
      setLoading(false)
    })
  }

  useEffect(() => {
    loadTasks()
    const interval = setInterval(() => {
      if (tasks.some((t) => t.status === 'RUNNING' || t.status === 'PENDING')) {
        loadTasks()
      }
    }, 3000)
    return () => clearInterval(interval)
  }, [])

  const handleRun = async () => {
    setRunning(true)
    try {
      await evalApi.run({
        dataset_name: dataset,
        rag_version: selectedVersion,
        name: `${VERSION_LABELS[selectedVersion]} 评测`,
      })
      message.success('评测任务已启动')
      setTimeout(loadTasks, 1000)
    } catch (err: any) {
      message.error('启动失败: ' + err.message)
    } finally {
      setRunning(false)
    }
  }

  const latestTask = tasks.find((t) => t.status === 'COMPLETED')
  const metrics = latestTask?.metrics || {}

  // Comparison chart data
  const completedTasks = tasks.filter((t) => t.status === 'COMPLETED')
  const comparisonOption = {
    title: { text: 'RAG Version Comparison', left: 'center', textStyle: { fontSize: 14 } },
    tooltip: { trigger: 'axis' },
    legend: { bottom: 0 },
    grid: { left: '3%', right: '4%', bottom: '15%', containLabel: true },
    xAxis: {
      type: 'category',
      data: ['Recall@5', 'Precision@5', 'MRR', 'Accuracy', 'Faithfulness', 'Relevancy'],
      axisLabel: { fontSize: 11 },
    },
    yAxis: { type: 'value', max: 1 },
    series: completedTasks.slice(0, 4).map((t) => ({
      name: VERSION_LABELS[t.rag_version] || t.rag_version,
      type: 'bar',
      data: [
        t.metrics?.recall_at_5 || 0,
        t.metrics?.precision_at_5 || 0,
        t.metrics?.mrr || 0,
        t.metrics?.accuracy || 0,
        t.metrics?.faithfulness || 0,
        t.metrics?.answer_relevancy || 0,
      ],
    })),
  }

  // Recall@K chart
  const recallOption = {
    title: { text: 'Recall@K', left: 'center', textStyle: { fontSize: 14 } },
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: ['@1', '@3', '@5', '@10'] },
    yAxis: { type: 'value', max: 1 },
    series: [
      {
        type: 'line',
        data: [
          metrics.recall_at_1 || 0,
          metrics.recall_at_3 || 0,
          metrics.recall_at_5 || 0,
          metrics.recall_at_10 || 0,
        ],
        smooth: true,
        areaStyle: {},
        itemStyle: { color: '#1677ff' },
      },
    ],
  }

  // Latency chart
  const latencyOption = {
    title: { text: '各版本平均延迟 (ms)', left: 'center', textStyle: { fontSize: 14 } },
    tooltip: { trigger: 'axis' },
    xAxis: {
      type: 'category',
      data: completedTasks.map((t) => VERSION_LABELS[t.rag_version]?.split(' ')[0] || t.rag_version),
      axisLabel: { fontSize: 10, rotate: 30 },
    },
    yAxis: { type: 'value' },
    series: [
      {
        type: 'bar',
        data: completedTasks.map((t) => t.metrics?.avg_latency_ms || 0),
        itemStyle: { color: '#52c41a' },
      },
    ],
  }

  const columns = [
    { title: '任务名', dataIndex: 'name', key: 'name' },
    {
      title: 'RAG版本',
      dataIndex: 'rag_version',
      key: 'rag_version',
      render: (v: string) => <Tag color="blue">{VERSION_LABELS[v] || v}</Tag>,
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (v: string) => (
        <Tag color={v === 'COMPLETED' ? 'success' : v === 'FAILED' ? 'error' : 'processing'}>
          {v}
        </Tag>
      ),
    },
    {
      title: '进度',
      key: 'progress',
      render: (_: any, record: EvaluationTask) => (
        <Progress
          percent={record.total_samples ? Math.round((record.completed_samples / record.total_samples) * 100) : 0}
          size="small"
        />
      ),
    },
    { title: '样本数', dataIndex: 'total_samples', key: 'total_samples', width: 80 },
    {
      title: 'Recall@5',
      key: 'recall5',
      render: (_: any, r: EvaluationTask) => r.metrics?.recall_at_5?.toFixed(3) || '-',
    },
    {
      title: 'MRR',
      key: 'mrr',
      render: (_: any, r: EvaluationTask) => r.metrics?.mrr?.toFixed(3) || '-',
    },
    {
      title: '操作',
      key: 'action',
      render: (_: any, record: EvaluationTask) => (
        <Button size="small" onClick={() => navigate(`/evaluation/${record.id}`)}>
          详情
        </Button>
      ),
    },
  ]

  return (
    <div>
      {/* Control bar */}
      <Card style={{ marginBottom: 16 }}>
        <Space wrap>
          <span>数据集:</span>
          <Select value={dataset} onChange={setDataset} style={{ width: 160 }}
            options={[{ label: 'rag_eval (50题)', value: 'rag_eval' }]} />
          <span>RAG版本:</span>
          <Select value={selectedVersion} onChange={setSelectedVersion} style={{ width: 260 }}
            options={Object.entries(VERSION_LABELS).map(([k, v]) => ({ label: v, value: k }))} />
          <Button type="primary" icon={<PlayCircleOutlined />} onClick={handleRun} loading={running}>
            开始评测
          </Button>
        </Space>
      </Card>

      {/* Metrics overview */}
      {latestTask && (
        <Row gutter={16} style={{ marginBottom: 16 }}>
          <Col span={3}><Card><Statistic title="Recall@1" value={metrics.recall_at_1} precision={3} /></Card></Col>
          <Col span={3}><Card><Statistic title="Recall@5" value={metrics.recall_at_5} precision={3} /></Card></Col>
          <Col span={3}><Card><Statistic title="Precision@5" value={metrics.precision_at_5} precision={3} /></Card></Col>
          <Col span={3}><Card><Statistic title="MRR" value={metrics.mrr} precision={3} /></Card></Col>
          <Col span={3}><Card><Statistic title="Accuracy" value={metrics.accuracy} precision={3} /></Card></Col>
          <Col span={3}><Card><Statistic title="Faithfulness" value={metrics.faithfulness} precision={3} /></Card></Col>
          <Col span={3}><Card><Statistic title="Relevancy" value={metrics.answer_relevancy} precision={3} /></Card></Col>
          <Col span={3}><Card><Statistic title="Avg Latency" value={metrics.avg_latency_ms} suffix="ms" /></Card></Col>
        </Row>
      )}

      {/* Charts */}
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={12}>
          <Card title={<span><BarChartOutlined /> 指标对比</span>}>
            {completedTasks.length > 0 ? (
              <ReactECharts option={comparisonOption} style={{ height: 320 }} />
            ) : (
              <div style={{ textAlign: 'center', padding: 60, color: '#999' }}>暂无评测数据</div>
            )}
          </Card>
        </Col>
        <Col span={12}>
          <Card title={<span><ThunderboltOutlined /> Recall@K 趋势</span>}>
            {latestTask ? (
              <ReactECharts option={recallOption} style={{ height: 320 }} />
            ) : (
              <div style={{ textAlign: 'center', padding: 60, color: '#999' }}>暂无评测数据</div>
            )}
          </Card>
        </Col>
      </Row>

      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={24}>
          <Card title="延迟对比">
            {completedTasks.length > 0 ? (
              <ReactECharts option={latencyOption} style={{ height: 240 }} />
            ) : (
              <div style={{ textAlign: 'center', padding: 40, color: '#999' }}>暂无数据</div>
            )}
          </Card>
        </Col>
      </Row>

      {/* Task list */}
      <Card title="评测任务列表">
        <Table
          dataSource={tasks}
          columns={columns}
          rowKey="id"
          loading={loading}
          pagination={{ pageSize: 10 }}
          size="small"
        />
      </Card>
    </div>
  )
}
