'use client';

import { useState, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';

export type VisualDataset = { label?: string; data: number[]; backgroundColor?: string | string[]; borderColor?: string | string[]; fill?: boolean };
export type Visual = {
  type: 'bar' | 'line' | 'pie' | 'doughnut' | 'table' | string;
  title?: string;
  description?: string;
  labels?: string[];
  datasets?: VisualDataset[];
  columns?: string[];
  rows?: any[];
};

export type Message = { id: string; role: 'user' | 'assistant'; content: string; visuals?: Visual[] };

export type Conversation = {
  id: string;
  title: string;
  created_at: string;
  messages: Message[];
};

const STORAGE_KEY = 'gloser_conversations';
const CURRENT_KEY = 'gloser_current_id';

function loadConversations(): Conversation[] {
  if (typeof window === 'undefined') return [];
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return [];
  try {
    return JSON.parse(raw);
  } catch {
    return [];
  }
}

function saveConversations(conversations: Conversation[]) {
    if (typeof window === 'undefined') return;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
}

function generateTitle(firstUserMessage: string): string {
  return firstUserMessage.slice(0, 60) + (firstUserMessage.length > 60 ? '...' : '');
}

export function useChat({ api = '/api/chat' } = { api: '/api/chat' }) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currentId, setCurrentId] = useState<string | null>(null);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  useEffect(() => {
    if (typeof window !== 'undefined') {
      setConversations(loadConversations());
      setCurrentId(localStorage.getItem(CURRENT_KEY));
    }
  }, []);


  const currentConversation = conversations.find((c) => c.id === currentId) || null;
  const messages = currentConversation?.messages || [];

  function handleInputChange(e: React.ChangeEvent<HTMLTextAreaElement>) {
    setInput(e.target.value);
  }

  async function fetchWithRetries(inputUrl: string, opts: RequestInit, attempts = 2) {
    let lastErr: any;
    for (let i = 0; i < attempts; i++) {
      try {
        const resp = await fetch(inputUrl, opts);
        if (!resp.ok) throw new Error(`status:${resp.status}`);
        return resp;
      } catch (e) {
        lastErr = e;
        // exponential-ish backoff
        await new Promise((res) => setTimeout(res, 300 * (i + 1)));
      }
    }
    throw lastErr;
  }

  async function sendMessage(content: string) {
    const trimmed = content?.trim();
    if (!trimmed) return;

    const userMsg: Message = { id: String(Date.now()), role: 'user', content: trimmed };

    let conv = currentConversation;
    if (!conv) {
      conv = {
        id: uuidv4(),
        title: generateTitle(trimmed),
        created_at: new Date().toISOString(),
        messages: [],
      };
      setCurrentId(conv.id);
      localStorage.setItem(CURRENT_KEY, conv.id);
    }

    conv.messages.push(userMsg);
    const updatedConvs = conversations.filter((c) => c.id !== conv!.id).concat([conv]);
    setConversations(updatedConvs);
    saveConversations(updatedConvs);

    setIsLoading(true);
    try {
      const resp = await fetchWithRetries(api, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          input: trimmed,
          session_id: conv.id  // Pass conversation ID as session ID for memory
        }),
      }, 2);

      const data = await resp.json().catch(() => ({ error: 'Invalid response from server' }));

      const serverMessages: Message[] = [];
      if (Array.isArray(data?.messages)) {
        for (const msg of data.messages) {
          if (typeof msg.content === 'string')
            serverMessages.push({ id: msg.id ?? String(Date.now()), role: msg.role || 'assistant', content: msg.content, visuals: msg.visuals || [] });
        }
      } else if (data?.result?.message || data?.result?.final_answer || data?.plan) {
        const content = data?.result?.message ? data.result.message : data?.result?.final_answer ? data.result.final_answer : JSON.stringify(data.plan || data.result || data);
        const visuals = Array.isArray(data?.result?.visuals) ? data.result.visuals : [];
        serverMessages.push({ id: String(Date.now()), role: 'assistant', content, visuals });
      } else if (data && Object.keys(data).length) {
        // avoid showing raw backend objects or error dumps to users
        if (data.error) {
          serverMessages.push({ id: String(Date.now()), role: 'assistant', content: 'Something didn\'t load correctly. Would you like me to try again?' });
        } else {
          serverMessages.push({ id: String(Date.now()), role: 'assistant', content: JSON.stringify(data) });
        }
      } else {
        serverMessages.push({ id: String(Date.now()), role: 'assistant', content: 'No response from server.' });
      }

      conv!.messages.push(...serverMessages);
      const finalConvs = conversations.filter((c) => c.id !== conv!.id).concat([conv!]);
      setConversations(finalConvs);
      saveConversations(finalConvs);
    } catch (err: any) {
      // do not expose raw errors to the user; offer retry
      const friendly: Message = { id: String(Date.now()), role: 'assistant', content: "Something didn't load correctly. I tried a couple of times — would you like me to retry?" };
      conv!.messages.push(friendly);
      const finalConvs = conversations.filter((c) => c.id !== conv!.id).concat([conv!]);
      setConversations(finalConvs);
      saveConversations(finalConvs);
    } finally {
      setIsLoading(false);
    }
  }

  async function handleSubmit(e?: React.FormEvent<HTMLFormElement>) {
    if (e) e.preventDefault();
    const trimmed = input?.trim();
    if (!trimmed) return;
    setInput('');
    await sendMessage(trimmed);
  }

  function newConversation() {
    setCurrentId(null);
      if (typeof window !== 'undefined') {
    localStorage.removeItem(CURRENT_KEY);
    }
  }

  function loadConversation(id: string) {
    setCurrentId(id);
      if (typeof window !== 'undefined') {
    localStorage.setItem(CURRENT_KEY, id);
    }
    if (typeof window !== 'undefined') {
    }
  }

  function deleteConversation(id: string) {
    const updated = conversations.filter((c) => c.id !== id);
    setConversations(updated);
    saveConversations(updated);
    if (currentId === id) {
        if (typeof window !== 'undefined') {
        }
      setCurrentId(null);
      localStorage.removeItem(CURRENT_KEY);
    }
  }

  function exportConversation() {
    if (!currentConversation) return;
    const payload = JSON.stringify(currentConversation, null, 2);
    const blob = new Blob([payload], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `conversation-${currentConversation.title.replace(/[^a-z0-9]/gi, '_')}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return {
    conversations,
    currentConversation,
    messages,
    input,
    setInput,
    handleInputChange,
    handleSubmit,
    isLoading,
    sendMessage,
    newConversation,
    loadConversation,
    deleteConversation,
    exportConversation,
  };
}
