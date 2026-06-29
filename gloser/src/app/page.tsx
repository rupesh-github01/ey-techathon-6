'use client';

import { useChat } from '../hooks/useChat';
import { useRef, useEffect } from 'react';
import MessageBubble from '../components/MessageBubble';
import Sidebar from '../components/Sidebar';

export default function Chat() {
  const {
    conversations,
    currentConversation,
    messages,
    input,
    handleInputChange,
    handleSubmit,
    isLoading,
    newConversation,
    loadConversation,
    deleteConversation,
    exportConversation,
  } = useChat({ api: '/api/chat' });

  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="chat-root">
      <Sidebar
        conversations={conversations}
        currentId={currentConversation?.id || null}
        onNewConversation={newConversation}
        onLoadConversation={loadConversation}
        onDeleteConversation={deleteConversation}
        onExportConversation={exportConversation}
      />
      <div className="chat-main">
        <div className="chat-container">
          <div className="messages-container">
            {messages.length === 0 && (
              <div className="empty-state">
                <h2 className="empty-title">How can I help you today?</h2>
              </div>
            )}
            {messages.map((message) => (
              <MessageBubble key={message.id} message={message} />
            ))}
            {isLoading && (
              <div className="message-wrapper assistant">
                <div className="message-content">
                  <div className="typing-indicator">
                    <span className="typing-dot"></span>
                    <span className="typing-dot"></span>
                    <span className="typing-dot"></span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="input-container">
            <form onSubmit={handleSubmit} className="input-form">
              <textarea
                className="input-textarea"
                value={input}
                onChange={handleInputChange}
                placeholder="Message Gloser AI..."
                rows={1}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit(e as any);
                  }
                }}
              />
              <button
                type="submit"
                disabled={isLoading || !input.trim()}
                className="send-button"
              >
                ↑
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}
