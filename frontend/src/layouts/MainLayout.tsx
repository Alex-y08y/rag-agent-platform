import { Layout, Menu, Avatar, Typography, Dropdown, Space } from 'antd'
import {
  MessageOutlined,
  BookOutlined,
  BarChartOutlined,
  ApiOutlined,
  DashboardOutlined,
  SettingOutlined,
  RobotOutlined,
  UserOutlined,
  LogoutOutlined,
} from '@ant-design/icons'
import { Outlet, useNavigate, useLocation } from 'react-router-dom'
import { useAppStore } from '../stores/useStore'
import { useAuthStore } from '../stores/authStore'

const { Sider, Content, Header } = Layout
const { Title } = Typography

const menuItems = [
  { key: '/chat', icon: <MessageOutlined />, label: '智能对话' },
  { key: '/knowledge-bases', icon: <BookOutlined />, label: '知识库管理' },
  { key: '/evaluation', icon: <BarChartOutlined />, label: 'RAG 评测' },
  { key: '/agent-trace', icon: <ApiOutlined />, label: 'Agent Trace' },
  { key: '/system-status', icon: <DashboardOutlined />, label: '系统状态' },
  { key: '/settings', icon: <SettingOutlined />, label: 'API 配置' },
]

export default function MainLayout() {
  const navigate = useNavigate()
  const location = useLocation()
  const { currentKb } = useAppStore()
  const { user, logout } = useAuthStore()

  const selectedKey = menuItems.find((item) =>
    location.pathname.startsWith(item.key),
  )?.key || '/chat'

  const userMenuItems = [
    {
      key: 'logout',
      icon: <LogoutOutlined />,
      label: '退出登录',
      onClick: () => {
        logout()
        navigate('/login')
      },
    },
  ]

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider width={220} theme="dark" style={{ position: 'fixed', height: '100vh', left: 0 }}>
        <div style={{ padding: '20px 16px', textAlign: 'center' }}>
          <RobotOutlined style={{ fontSize: 32, color: '#1677ff' }} />
          <Title level={5} style={{ color: '#fff', margin: '8px 0 0' }}>
            RAG Agent Platform
          </Title>
          <div style={{ color: '#999', fontSize: 11 }}>企业级智能知识库</div>
        </div>
        <Menu
          theme="dark"
          mode="inline"
          selectedKeys={[selectedKey]}
          items={menuItems}
          onClick={({ key }) => navigate(key)}
        />
        {currentKb && (
          <div style={{ position: 'absolute', bottom: 60, left: 16, right: 16, color: '#999', fontSize: 12 }}>
            当前知识库: {currentKb.name}
          </div>
        )}
      </Sider>
      <Layout style={{ marginLeft: 220 }}>
        <Header
          style={{
            background: '#fff',
            padding: '0 24px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            boxShadow: '0 1px 4px rgba(0,0,0,0.08)',
          }}
        >
          <Typography.Text strong style={{ fontSize: 16 }}>
            {menuItems.find((m) => m.key === selectedKey)?.label || 'RAG Agent Platform'}
          </Typography.Text>
          <Dropdown menu={{ items: userMenuItems }} placement="bottomRight">
            <Space style={{ cursor: 'pointer' }}>
              <Avatar style={{ backgroundColor: '#1677ff' }} icon={<UserOutlined />} />
              <span>{user?.username || '用户'}</span>
            </Space>
          </Dropdown>
        </Header>
        <Content style={{ margin: 16, padding: 24, background: '#fff', borderRadius: 8, minHeight: 'calc(100vh - 112px)' }}>
          <Outlet />
        </Content>
      </Layout>
    </Layout>
  )
}
