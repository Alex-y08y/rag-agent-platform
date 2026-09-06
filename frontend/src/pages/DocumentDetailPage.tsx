import { useState, useEffect } from 'react'
import { Card, Descriptions, Tag, Table, Input, Button, Space, Empty, Spin } from 'antd'
import { ArrowLeftOutlined, SearchOutlined } from '@ant-design/icons'
import { useParams, useNavigate } from 'react-router-dom'
import { docApi } from '../api/client'
import type { DocumentDetail, Chunk } from '../types'

export default function DocumentDetailPage() {
  const { docId } = useParams<{ docId: string }>()
  const navigate = useNavigate()
  const [doc, setDoc] = useState<DocumentDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [searchText, setSearchText] = useState('')

  useEffect(() => {
    if (docId) {
      setLoading(true)
      docApi.get(docId).then((res) => {
        setDoc(res.data)
        setLoading(false)
      }).catch(() => setLoading(false))
    }
  }, [docId])

  if (loading) return <Spin style={{ display: 'block', marginTop: 80 }} />
  if (!doc) return <Empty description="文档不存在" />

  const filteredChunks = doc.chunks.filter((c) =>
    searchText ? c.content.toLowerCase().includes(searchText.toLowerCase()) : true,
  )

  const columns = [
    { title: '#', dataIndex: 'chunk_index', key: 'chunk_index', width: 60 },
    {
      title: '内容',
      dataIndex: 'content',
      key: 'content',
      ellipsis: true,
      render: (text: string) => (
        <div style={{ maxWidth: 500, whiteSpace: 'pre-wrap', fontSize: 13 }}>{text}</div>
      ),
    },
    { title: '页码', dataIndex: 'page', key: 'page', width: 70 },
    { title: '章节', dataIndex: 'section', key: 'section', width: 150, ellipsis: true },
    {
      title: 'Embedding',
      dataIndex: 'embedding_status',
      key: 'embedding_status',
      width: 100,
      render: (v: string) => (
        <Tag color={v === 'DONE' ? 'success' : v === 'FAILED' ? 'error' : 'processing'}>
          {v}
        </Tag>
      ),
    },
  ]

  return (
    <div>
      <Button
        icon={<ArrowLeftOutlined />}
        onClick={() => navigate('/knowledge-bases')}
        style={{ marginBottom: 16 }}
      >
        返回知识库
      </Button>

      <Card title="文档信息" style={{ marginBottom: 16 }}>
        <Descriptions column={3} size="small">
          <Descriptions.Item label="文件名">{doc.filename}</Descriptions.Item>
          <Descriptions.Item label="类型">{doc.file_type}</Descriptions.Item>
          <Descriptions.Item label="大小">{(doc.file_size / 1024).toFixed(1)} KB</Descriptions.Item>
          <Descriptions.Item label="状态">
            <Tag color={doc.status === 'SUCCESS' ? 'success' : 'processing'}>{doc.status}</Tag>
          </Descriptions.Item>
          <Descriptions.Item label="Chunk 数量">{doc.chunk_count}</Descriptions.Item>
          <Descriptions.Item label="创建时间">{new Date(doc.created_at).toLocaleString()}</Descriptions.Item>
        </Descriptions>
      </Card>

      <Card
        title={`Chunk 列表 (${filteredChunks.length}/${doc.chunks.length})`}
        extra={
          <Space>
            <Input
              placeholder="搜索内容..."
              prefix={<SearchOutlined />}
              value={searchText}
              onChange={(e) => setSearchText(e.target.value)}
              allowClear
              style={{ width: 240 }}
            />
          </Space>
        }
      >
        <Table
          dataSource={filteredChunks}
          columns={columns}
          rowKey="id"
          pagination={{ pageSize: 10 }}
          size="small"
          expandable={{
            expandedRowRender: (record: Chunk) => (
              <div style={{ padding: '8px 16px', background: '#fafafa', borderRadius: 4 }}>
                <div style={{ whiteSpace: 'pre-wrap', fontSize: 13, lineHeight: 1.8 }}>
                  {record.content}
                </div>
                {record.metadata && Object.keys(record.metadata).length > 0 && (
                  <div style={{ marginTop: 8, fontSize: 12, color: '#999' }}>
                    Metadata: {JSON.stringify(record.metadata)}
                  </div>
                )}
              </div>
            ),
          }}
        />
      </Card>
    </div>
  )
}
