import { create } from 'zustand'
import type { ChatMessage, KnowledgeBase } from '../types'

interface AppState {
  // Current knowledge base
  currentKb: KnowledgeBase | null
  setCurrentKb: (kb: KnowledgeBase | null) => void

  // Chat
  conversations: Array<{ id: string; title: string }>
  currentConversationId: string | null
  messages: ChatMessage[]
  isStreaming: boolean
  agentSteps: Array<{ node: string; message: string; status: 'pending' | 'active' | 'done' }>

  setCurrentConversation: (id: string | null) => void
  addMessage: (msg: ChatMessage) => void
  setMessages: (msgs: ChatMessage[]) => void
  setStreaming: (v: boolean) => void
  setAgentSteps: (steps: AppState['agentSteps']) => void
  addConversation: (id: string, title: string) => void
  clearMessages: () => void
}

export const useAppStore = create<AppState>((set) => ({
  currentKb: null,
  setCurrentKb: (kb) => set({ currentKb: kb }),

  conversations: [],
  currentConversationId: null,
  messages: [],
  isStreaming: false,
  agentSteps: [],

  setCurrentConversation: (id) => set({ currentConversationId: id }),
  addMessage: (msg) => set((state) => ({ messages: [...state.messages, msg] })),
  setMessages: (msgs) => set({ messages: msgs }),
  setStreaming: (v) => set({ isStreaming: v }),
  setAgentSteps: (steps) => set({ agentSteps: steps }),
  addConversation: (id, title) =>
    set((state) => ({
      conversations: [{ id, title }, ...state.conversations],
    })),
  clearMessages: () => set({ messages: [], agentSteps: [] }),
}))
