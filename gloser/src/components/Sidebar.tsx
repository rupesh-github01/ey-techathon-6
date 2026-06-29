import React from 'react';
import { Conversation } from '../hooks/useChat';

export default function Sidebar({
  conversations,
  currentId,
  onNewConversation,
  onLoadConversation,
  onDeleteConversation,
  onExportConversation,
}: {
  conversations: Conversation[];
  currentId: string | null;
  onNewConversation: () => void;
  onLoadConversation: (id: string) => void;
  onDeleteConversation: (id: string) => void;
  onExportConversation: () => void;
}) {
  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <h1 className="sidebar-title">Gloser AI</h1>
        <button className="sidebar-button new-chat" onClick={onNewConversation}>
          + New Chat
        </button>
      </div>

      <div className="conversation-list">
        {conversations.length === 0 && (
          <div className="empty-message">No conversations yet</div>
        )}
        {conversations.map((conv) => (
          <div
            key={conv.id}
            className={`conversation-item ${conv.id === currentId ? 'active' : ''}`}
            title={conv.title}
          >
            <button
              className="conversation-title"
              onClick={() => onLoadConversation(conv.id)}
            >
              {conv.title}
            </button>
            <button
              className="delete-conversation"
              onClick={(e) => {
                e.stopPropagation();
                onDeleteConversation(conv.id);
              }}
              aria-label="Delete conversation"
            >
              ×
            </button>
          </div>
        ))}
      </div>

      <div className="sidebar-footer">
        <button className="sidebar-button" onClick={onExportConversation}>
          Export Conversation
        </button>
      </div>
    </div>
  );
}
