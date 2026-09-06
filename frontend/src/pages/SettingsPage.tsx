import { useState, useEffect } from 'react'
import {
  Card,
  Form,
  Input,
  Select,
  Button,
  Space,
  message,
  Divider,
  Typography,
  Tag,
  Alert,
  Tooltip,
} from 'antd'
import {
  KeyOutlined,
  ApiOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ReloadOutlined,
  ExperimentOutlined,
} from '@ant-design/icons'
import { settingsApi, type SettingsData, type TestResult } from '../api/client'

const { Title, Text, Paragraph } = Typography

const PRESETS = [
  {
    label: '通义千问 (百炼 Model Studio)',
    value: 'dashscope',
    base_url: 'https://ws-yd9ijsp7q2ieflqf.cn-beijing.maas.aliyuncs.com/compatible-mode/v1',
    model: 'qwen3.8-flash',
    keyField: 'DASHSCOPE_API_KEY' as const,
    docs: 'https://bailian.console.aliyun.com/?apiKey=1#/api-key',
  },
  {
    label: 'OpenAI',
    value: 'openai',
    base_url: 'https://api.openai.com/v1',
    model: 'gpt-4o-mini',
    keyField: 'OPENAI_API_KEY' as const,
    docs: 'https://platform.openai.com/api-keys',
  },
  {
    label: 'DeepSeek',
    value: 'deepseek',
    base_url: 'https://api.deepseek.com/v1',
    model: 'deepseek-chat',
    keyField: 'OPENAI_API_KEY' as const,
    docs: 'https://platform.deepseek.com/api_keys',
  },
  {
    label: '智谱 (Zhipu)',
    value: 'zhipu',
    base_url: 'https://open.bigmodel.cn/api/paas/v4',
    model: 'glm-4-flash',
    keyField: 'OPENAI_API_KEY' as const,
    docs: 'https://open.bigmodel.cn/usercenter/apikeys',
  },
  {
    label: 'Ollama (本地)',
    value: 'ollama',
    base_url: 'http://host.docker.internal:11434/v1',
    model: 'qwen2.5:7b',
    keyField: 'OPENAI_API_KEY' as const,
    docs: 'https://ollama.com',
  },
]

export default function SettingsPage() {
  const [form] = Form.useForm()
  const [loading, setLoading] = useState(false)
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState<TestResult | null>(null)
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    loadSettings()
  }, [])

  const loadSettings = async () => {
    setLoading(true)
    try {
      const { data } = await settingsApi.get()
      form.setFieldsValue({
        LLM_PROVIDER: data.LLM_PROVIDER || 'dashscope',
        LLM_MODEL: data.LLM_MODEL,
        LLM_BASE_URL: data.LLM_BASE_URL,
        OPENAI_API_KEY: data.OPENAI_API_KEY,
        DASHSCOPE_API_KEY: data.DASHSCOPE_API_KEY,
        LLM_TEMPERATURE: data.LLM_TEMPERATURE,
        LLM_MAX_TOKENS: data.LLM_MAX_TOKENS,
        LLM_TIMEOUT: data.LLM_TIMEOUT,
      })
      setSaved(false)
    } catch (err) {
      message.error('加载配置失败')
    } finally {
      setLoading(false)
    }
  }

  const handlePreset = (value: string) => {
    const preset = PRESETS.find((p) => p.value === value)
    if (preset) {
      form.setFieldsValue({
        LLM_PROVIDER: preset.value,
        LLM_BASE_URL: preset.base_url,
        LLM_MODEL: preset.model,
      })
    }
  }

  const handleSave = async () => {
    try {
      const values = await form.validateFields()
      setLoading(true)
      await settingsApi.update(values)
      setSaved(true)
      setTestResult(null)
      message.success('配置已保存，LLM 客户端已重新加载')
    } catch (err: any) {
      if (err.errorFields) return
      message.error('保存失败: ' + (err.message || ''))
    } finally {
      setLoading(false)
    }
  }

  const handleTest = async () => {
    setTesting(true)
    setTestResult(null)
    try {
      // Save first if there are unsaved changes
      const values = form.getFieldsValue()
      await settingsApi.update(values)
      const { data } = await settingsApi.test()
      setTestResult(data)
      if (data.ok) {
        message.success('连接成功！')
      } else {
        message.error('连接失败: ' + data.error)
      }
    } catch (err: any) {
      message.error('测试请求失败: ' + (err.message || ''))
    } finally {
      setTesting(false)
    }
  }

  const handleReset = async () => {
    try {
      await settingsApi.reset()
      message.success('已恢复 .env 默认配置')
      loadSettings()
    } catch {
      message.error('重置失败')
    }
  }

  return (
    <div style={{ maxWidth: 720, margin: '0 auto' }}>
      <Title level={4} style={{ marginTop: 0 }}>
        <ApiOutlined /> API 配置
      </Title>
      <Paragraph type="secondary">
        配置大模型 API 以启用 RAG 问答、Query Rewrite、Agent 推理等功能。
        配置保存在服务端 <Text code>data/config/settings.json</Text>，无需重启。
      </Paragraph>

      {saved && (
        <Alert
          message="配置已保存"
          description="新的 API 配置已生效，可以直接去对话页面测试。"
          type="success"
          showIcon
          closable
          style={{ marginBottom: 16 }}
        />
      )}

      <Card>
        <Form form={form} layout="vertical">
          <Form.Item label="模型服务商" name="LLM_PROVIDER">
            <Select
              placeholder="选择预设服务商"
              onChange={handlePreset}
              options={PRESETS.map((p) => ({ label: p.label, value: p.value }))}
            />
          </Form.Item>

          <Form.Item
            label="API Base URL"
            name="LLM_BASE_URL"
            rules={[{ required: true, message: '请输入 Base URL' }]}
          >
            <Input placeholder="https://dashscope.aliyuncs.com/compatible-mode/v1" />
          </Form.Item>

          <Form.Item
            label="模型名称"
            name="LLM_MODEL"
            rules={[{ required: true, message: '请输入模型名称' }]}
          >
            <Input placeholder="qwen3-plus / gpt-4o-mini / deepseek-chat ..." />
          </Form.Item>

          <Divider orientation="left">API Key（二选一）</Divider>

          <Form.Item label="DashScope API Key" name="DASHSCOPE_API_KEY">
            <Input.Password
              placeholder="sk-xxxxxxxx"
              prefix={<KeyOutlined />}
              autoComplete="new-password"
            />
          </Form.Item>

          <Form.Item label="OpenAI 兼容 API Key" name="OPENAI_API_KEY">
            <Input.Password
              placeholder="sk-xxxxxxxx"
              prefix={<KeyOutlined />}
              autoComplete="new-password"
            />
          </Form.Item>

          <Alert
            message="API Key 安全说明"
            description="Key 仅保存在服务端本地文件中，不会上传到任何第三方。显示时自动脱敏（前4后4）。"
            type="info"
            showIcon
            style={{ marginBottom: 16 }}
          />

          <Divider orientation="left">高级参数</Divider>

          <Space wrap size="large">
            <Form.Item label="Temperature" name="LLM_TEMPERATURE" style={{ marginBottom: 0 }}>
              <Input type="number" min={0} max={2} step={0.1} style={{ width: 120 }} />
            </Form.Item>
            <Form.Item label="Max Tokens" name="LLM_MAX_TOKENS" style={{ marginBottom: 0 }}>
              <Input type="number" min={1} max={8192} style={{ width: 120 }} />
            </Form.Item>
            <Form.Item label="Timeout (秒)" name="LLM_TIMEOUT" style={{ marginBottom: 0 }}>
              <Input type="number" min={5} max={300} style={{ width: 120 }} />
            </Form.Item>
          </Space>

          <Divider />

          <Space>
            <Button type="primary" onClick={handleSave} loading={loading} icon={<CheckCircleOutlined />}>
              保存配置
            </Button>
            <Button
              onClick={handleTest}
              loading={testing}
              icon={<ExperimentOutlined />}
            >
              测试连接
            </Button>
            <Tooltip title="清除运行时配置，恢复 .env 文件中的默认值">
              <Button onClick={handleReset} icon={<ReloadOutlined />} danger>
                恢复默认
              </Button>
            </Tooltip>
          </Space>
        </Form>
      </Card>

      {testResult && (
        <Card
          style={{ marginTop: 16 }}
          title={
            <Space>
              {testResult.ok ? (
                <Tag color="success" icon={<CheckCircleOutlined />}>连接成功</Tag>
              ) : (
                <Tag color="error" icon={<CloseCircleOutlined />}>连接失败</Tag>
              )}
              <Text type="secondary">{testResult.model}</Text>
            </Space>
          }
        >
          {testResult.ok ? (
            <div>
              <Paragraph>
                <Text strong>Base URL: </Text>{testResult.base_url}
              </Paragraph>
              <Paragraph>
                <Text strong>模型回复: </Text>"{testResult.reply}"
              </Paragraph>
            </div>
          ) : (
            <Alert
              message="连接失败"
              description={testResult.error}
              type="error"
              showIcon
            />
          )}
        </Card>
      )}

      <Card style={{ marginTop: 16 }} title="获取 API Key">
        <ul style={{ paddingLeft: 20, lineHeight: 2 }}>
          {PRESETS.filter((p) => p.value !== 'ollama').map((p) => (
            <li key={p.value}>
              <Text strong>{p.label}: </Text>
              <a href={p.docs} target="_blank" rel="noreferrer">{p.docs}</a>
            </li>
          ))}
          <li>
            <Text strong>Ollama (本地免费): </Text>
            安装 Ollama 后运行 <Text code>ollama pull qwen2.5:7b</Text>，无需 API Key
          </li>
        </ul>
      </Card>
    </div>
  )
}
