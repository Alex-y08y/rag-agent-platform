import { useEffect } from 'react'
import { Routes, Route, Navigate, useLocation } from 'react-router-dom'
import { Spin } from 'antd'
import MainLayout from './layouts/MainLayout'
import ChatPage from './pages/ChatPage'
import KnowledgeBasePage from './pages/KnowledgeBasePage'
import DocumentDetailPage from './pages/DocumentDetailPage'
import EvaluationPage from './pages/EvaluationPage'
import EvaluationDetailPage from './pages/EvaluationDetailPage'
import AgentTracePage from './pages/AgentTracePage'
import SystemStatusPage from './pages/SystemStatusPage'
import SettingsPage from './pages/SettingsPage'
import LoginPage from './pages/LoginPage'
import { useAuthStore } from './stores/authStore'

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, loading } = useAuthStore()
  const location = useLocation()

  if (loading && !isAuthenticated) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <Spin size="large" />
      </div>
    )
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" state={{ from: location }} replace />
  }

  return <>{children}</>
}

export default function App() {
  const { initAuth } = useAuthStore()

  useEffect(() => {
    initAuth()
  }, [initAuth])

  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route
        path="/"
        element={
          <ProtectedRoute>
            <MainLayout />
          </ProtectedRoute>
        }
      >
        <Route index element={<Navigate to="/chat" replace />} />
        <Route path="chat" element={<ChatPage />} />
        <Route path="chat/:conversationId" element={<ChatPage />} />
        <Route path="knowledge-bases" element={<KnowledgeBasePage />} />
        <Route path="documents/:docId" element={<DocumentDetailPage />} />
        <Route path="evaluation" element={<EvaluationPage />} />
        <Route path="evaluation/:taskId" element={<EvaluationDetailPage />} />
        <Route path="agent-trace" element={<AgentTracePage />} />
        <Route path="system-status" element={<SystemStatusPage />} />
        <Route path="settings" element={<SettingsPage />} />
      </Route>
    </Routes>
  )
}
