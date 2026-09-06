import { create } from 'zustand'
import type { User } from '../types'
import { authApi, setToken, getToken } from '../api/client'

interface AuthState {
  user: User | null
  token: string | null
  isAuthenticated: boolean
  loading: boolean

  login: (username: string, password: string) => Promise<void>
  register: (username: string, email: string, password: string) => Promise<void>
  logout: () => void
  fetchMe: () => Promise<void>
  initAuth: () => void
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  token: getToken(),
  isAuthenticated: !!getToken(),
  loading: false,

  login: async (username, password) => {
    set({ loading: true })
    try {
      const res = await authApi.login({ username, password })
      const { access_token, user } = res.data
      setToken(access_token)
      set({ user, token: access_token, isAuthenticated: true, loading: false })
    } catch (err: any) {
      set({ loading: false })
      throw new Error(err.response?.data?.detail || '登录失败')
    }
  },

  register: async (username, email, password) => {
    set({ loading: true })
    try {
      const res = await authApi.register({ username, email, password })
      const { access_token, user } = res.data
      setToken(access_token)
      set({ user, token: access_token, isAuthenticated: true, loading: false })
    } catch (err: any) {
      set({ loading: false })
      throw new Error(err.response?.data?.detail || '注册失败')
    }
  },

  logout: () => {
    setToken(null)
    set({ user: null, token: null, isAuthenticated: false })
  },

  fetchMe: async () => {
    try {
      const res = await authApi.me()
      set({ user: res.data, isAuthenticated: true })
    } catch {
      setToken(null)
      set({ user: null, token: null, isAuthenticated: false })
    }
  },

  initAuth: () => {
    const token = getToken()
    if (token) {
      set({ token, isAuthenticated: true })
      // Fetch user info in background
      authApi.me().then((res) => {
        set({ user: res.data })
      }).catch(() => {
        setToken(null)
        set({ user: null, token: null, isAuthenticated: false })
      })
    }
  },
}))
