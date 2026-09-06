import { useState } from 'react'
import { Form, Input, Button, Card, Typography, Tabs, Alert, Space } from 'antd'
import { UserOutlined, LockOutlined, MailOutlined, RobotOutlined } from '@ant-design/icons'
import { useNavigate, useLocation } from 'react-router-dom'
import { useAuthStore } from '../stores/authStore'

const { Title, Text } = Typography

export default function LoginPage() {
  const [mode, setMode] = useState<'login' | 'register'>('login')
  const [error, setError] = useState('')
  const navigate = useNavigate()
  const location = useLocation()
  const { login, register, loading } = useAuthStore()

  const from = (location.state as any)?.from?.pathname || '/chat'

  const handleLogin = async (values: { username: string; password: string }) => {
    setError('')
    try {
      await login(values.username, values.password)
      navigate(from, { replace: true })
    } catch (err: any) {
      setError(err.message)
    }
  }

  const handleRegister = async (values: {
    username: string
    email: string
    password: string
    confirm: string
  }) => {
    setError('')
    if (values.password !== values.confirm) {
      setError('两次输入的密码不一致')
      return
    }
    try {
      await register(values.username, values.email, values.password)
      navigate(from, { replace: true })
    } catch (err: any) {
      setError(err.message)
    }
  }

  return (
    <div
      style={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      }}
    >
      <Card
        style={{ width: 420, boxShadow: '0 8px 32px rgba(0,0,0,0.15)' }}
        styles={{ body: { padding: '32px 24px' } }}
      >
        <div style={{ textAlign: 'center', marginBottom: 24 }}>
          <RobotOutlined style={{ fontSize: 48, color: '#1677ff' }} />
          <Title level={3} style={{ margin: '12px 0 4px' }}>
            RAG Agent Platform
          </Title>
          <Text type="secondary">企业级智能知识库平台</Text>
        </div>

        <Tabs
          activeKey={mode}
          onChange={(k) => {
            setMode(k as 'login' | 'register')
            setError('')
          }}
          centered
          items={[
            { key: 'login', label: '登录' },
            { key: 'register', label: '注册' },
          ]}
        />

        {error && (
          <Alert message={error} type="error" showIcon style={{ marginBottom: 16 }} />
        )}

        {mode === 'login' ? (
          <Form onFinish={handleLogin} size="large">
            <Form.Item
              name="username"
              rules={[{ required: true, message: '请输入用户名' }]}
            >
              <Input prefix={<UserOutlined />} placeholder="用户名" />
            </Form.Item>
            <Form.Item
              name="password"
              rules={[{ required: true, message: '请输入密码' }]}
            >
              <Input.Password prefix={<LockOutlined />} placeholder="密码" />
            </Form.Item>
            <Form.Item>
              <Button
                type="primary"
                htmlType="submit"
                block
                loading={loading}
              >
                登录
              </Button>
            </Form.Item>
          </Form>
        ) : (
          <Form onFinish={handleRegister} size="large">
            <Form.Item
              name="username"
              rules={[
                { required: true, message: '请输入用户名' },
                { min: 3, message: '用户名至少3个字符' },
              ]}
            >
              <Input prefix={<UserOutlined />} placeholder="用户名（3-64字符）" />
            </Form.Item>
            <Form.Item
              name="email"
              rules={[
                { required: true, message: '请输入邮箱' },
                { type: 'email', message: '邮箱格式不正确' },
              ]}
            >
              <Input prefix={<MailOutlined />} placeholder="邮箱" />
            </Form.Item>
            <Form.Item
              name="password"
              rules={[
                { required: true, message: '请输入密码' },
                { min: 6, message: '密码至少6个字符' },
              ]}
            >
              <Input.Password prefix={<LockOutlined />} placeholder="密码（至少6位）" />
            </Form.Item>
            <Form.Item
              name="confirm"
              rules={[{ required: true, message: '请确认密码' }]}
            >
              <Input.Password prefix={<LockOutlined />} placeholder="确认密码" />
            </Form.Item>
            <Form.Item>
              <Button
                type="primary"
                htmlType="submit"
                block
                loading={loading}
              >
                注册并登录
              </Button>
            </Form.Item>
          </Form>
        )}

        <div style={{ textAlign: 'center', marginTop: 8 }}>
          <Text type="secondary" style={{ fontSize: 12 }}>
            首次使用请先注册，数据保存在本地 PostgreSQL
          </Text>
        </div>
      </Card>
    </div>
  )
}
