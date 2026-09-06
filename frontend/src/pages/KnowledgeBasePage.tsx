import { useState, useEffect } from 'react'
import {
  Card,
  Button,
  Table,
  Tag,
  Modal,
  Input,
  Upload,
  Space,
  Popconfirm,
  message,
  Statistic,
  Row,
  Col,
} from 'antd'
import {
  PlusOutlined,
  UploadOutlined,
  DeleteOutlined,
  ReloadOutlined,
  BookOutlined,
  FileTextOutlined,
  EyeOutlined,
} from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'
import { kbApi, docApi } from '../api/client'
import { useAppStore } from '../stores/useStore'
import type { KnowledgeBase, Document } from '../types'

const STATUS_COLORS: Record<string, string> = {
  UPLOADING: 'default',
  PARSING: 'processing',
  CHUNKING: 'processing',
  EMBEDDING: 'processing',
  INDEXING: 'processing',
  SUCCESS: 'success',
  FAILED: 'error',
}

export default function KnowledgeBasePage() {
  const [kbs, setKbs] = useState<KnowledgeBase[]>([])
  const [docs, setDocs] = useState<Document[]>([])
  const [selectedKb, setSelectedKb] = useState<string>('')
  const [createModal, setCreateModal] = useState(false)
  const [newKbName, setNewKbName] = useState('')
  const [newKbDesc, setNewKbDesc] = useState('')
  const [uploading, setUploading] = useState(false)
  const navigate = useNavigate()
  const { setCurrentKb } = useAppStore()

  const loadKbs = () => {
    kbApi.list().then((res) => {
      setKbs(res.data)
      if (res.data.length > 0 && !selectedKb) {
        setSelectedKb(res.data[0].id)
        setCurrentKb(res.data[0])
      }
    })
  }

  const loadDocs = (kbId: string) => {
    if (!kbId) return
    docApi.list(kbId).then((res) => setDocs(res.data))
  }

  useEffect(() => {
    loadKbs()
  }, [])

  useEffect(() => {
    if (selectedKb) loadDocs(selectedKb)
  }, [selectedKb])

  const handleCreateKb = async () => {
    if (!newKbName.trim()) return
    try {
      await kbApi.create({ name: newKbName, description: newKbDesc })
      message.success('知识库创建成功')
      setCreateModal(false)
      setNewKbName('')
      setNewKbDesc('')
      loadKbs()
    } catch (err: any) {
      message.error('创建失败: ' + err.message)
    }
  }

  const handleDeleteKb = async (id: string) => {
    try {
      await kbApi.delete(id)
      message.success('知识库已删除')
      loadKbs()
      if (selectedKb === id) {
        setSelectedKb('')
        setDocs([])
      }
    } catch (err: any) {
      message.error('删除失败: ' + err.message)
    }
  }

  const handleUpload = async (file: File) => {
    if (!selectedKb) {
      message.warning('请先选择知识库')
      return false
    }
    setUploading(true)
    try {
      await docApi.upload(selectedKb, file)
      message.success('文档上传成功，正在处理...')
      setTimeout(() => loadDocs(selectedKb), 2000)
    } catch (err: any) {
      message.error('上传失败: ' + err.message)
    } finally {
      setUploading(false)
    }
    return false
  }

  const handleDeleteDoc = async (id: string) => {
    try {
      await docApi.delete(id)
      message.success('文档已删除')
      loadDocs(selectedKb)
    } catch (err: any) {
      message.error('删除失败: ' + err.message)
    }
  }

  const handleReindex = async (id: string) => {
    try {
      await docApi.reindex(id)
      message.success('重新索引已启动')
      setTimeout(() => loadDocs(selectedKb), 3000)
    } catch (err: any) {
      message.error('重新索引失败: ' + err.message)
    }
  }

  const currentKb = kbs.find((k) => k.id === selectedKb)

  const columns = [
    {
      title: '文件名',
      dataIndex: 'filename',
      key: 'filename',
      render: (text: string, record: Document) => (
        <a onClick={() => navigate(`/documents/${record.id}`)}>{text}</a>
      ),
    },
    { title: '类型', dataIndex: 'file_type', key: 'file_type', width: 80 },
    {
      title: '大小',
      dataIndex: 'file_size',
      key: 'file_size',
      width: 100,
      render: (v: number) => (v / 1024).toFixed(1) + ' KB',
    },
    { title: 'Chunk数', dataIndex: 'chunk_count', key: 'chunk_count', width: 80 },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 120,
      render: (v: string) => <Tag color={STATUS_COLORS[v] || 'default'}>{v}</Tag>,
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 180,
      render: (v: string) => new Date(v).toLocaleString(),
    },
    {
      title: '操作',
      key: 'actions',
      width: 180,
      render: (_: any, record: Document) => (
        <Space>
          <Button size="small" icon={<EyeOutlined />} onClick={() => navigate(`/documents/${record.id}`)}>
            详情
          </Button>
          <Button size="small" icon={<ReloadOutlined />} onClick={() => handleReindex(record.id)}>
            重索引
          </Button>
          <Popconfirm title="确定删除?" onConfirm={() => handleDeleteDoc(record.id)}>
            <Button size="small" danger icon={<DeleteOutlined />} />
          </Popconfirm>
        </Space>
      ),
    },
  ]

  return (
    <div>
      {/* KB Cards */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        {kbs.map((kb) => (
          <Col span={6} key={kb.id}>
            <Card
              hoverable
              style={{
                borderColor: selectedKb === kb.id ? '#1677ff' : undefined,
                borderWidth: selectedKb === kb.id ? 2 : 1,
              }}
              onClick={() => {
                setSelectedKb(kb.id)
                setCurrentKb(kb)
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start' }}>
                <div>
                  <div style={{ fontWeight: 600, fontSize: 15 }}>{kb.name}</div>
                  <div style={{ color: '#999', fontSize: 12, marginTop: 4 }}>{kb.description}</div>
                </div>
                <Popconfirm title="删除知识库?" onConfirm={() => handleDeleteKb(kb.id)}>
                  <DeleteOutlined style={{ color: '#ff4d4f', cursor: 'pointer' }} />
                </Popconfirm>
              </div>
              <div style={{ marginTop: 12, display: 'flex', gap: 16 }}>
                <Statistic title="文档" value={kb.document_count} valueStyle={{ fontSize: 16 }} />
                <Statistic title="Chunk" value={kb.chunk_count} valueStyle={{ fontSize: 16 }} />
              </div>
            </Card>
          </Col>
        ))}
        <Col span={6}>
          <Card
            hoverable
            style={{ borderStyle: 'dashed', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
            onClick={() => setCreateModal(true)}
          >
            <div style={{ textAlign: 'center', color: '#999' }}>
              <PlusOutlined style={{ fontSize: 24 }} />
              <div style={{ marginTop: 8 }}>创建知识库</div>
            </div>
          </Card>
        </Col>
      </Row>

      {/* Document list */}
      <Card
        title={
          <Space>
            <FileTextOutlined />
            文档列表
            {currentKb && <Tag color="blue">{currentKb.name}</Tag>}
          </Space>
        }
        extra={
          <Upload
            showUploadList={false}
            beforeUpload={handleUpload}
            multiple
            accept=".pdf,.docx,.txt,.md,.csv,.xlsx"
          >
            <Button type="primary" icon={<UploadOutlined />} loading={uploading}>
              上传文档
            </Button>
          </Upload>
        }
      >
        <Table
          dataSource={docs}
          columns={columns}
          rowKey="id"
          pagination={{ pageSize: 10 }}
          size="small"
        />
      </Card>

      {/* Create KB Modal */}
      <Modal
        title="创建知识库"
        open={createModal}
        onOk={handleCreateKb}
        onCancel={() => setCreateModal(false)}
        okText="创建"
        cancelText="取消"
      >
        <div style={{ marginBottom: 16 }}>
          <div style={{ marginBottom: 4 }}>名称</div>
          <Input value={newKbName} onChange={(e) => setNewKbName(e.target.value)} placeholder="知识库名称" />
        </div>
        <div>
          <div style={{ marginBottom: 4 }}>描述</div>
          <Input.TextArea
            value={newKbDesc}
            onChange={(e) => setNewKbDesc(e.target.value)}
            placeholder="知识库描述"
            rows={3}
          />
        </div>
      </Modal>
    </div>
  )
}
