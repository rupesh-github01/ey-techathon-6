"""
Memory Manager for Gloser AI
Provides optimized context management to avoid sending full chat history to LLM.

Strategy:
1. Short-term memory: Last N messages (full detail) for immediate context
2. Long-term summary: Compressed summary of older conversation
3. Key entities: Important entities/topics mentioned in conversation
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ChatMessage:
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    has_visuals: bool = False


@dataclass
class ConversationMemory:
    """Holds memory for a single conversation session"""
    session_id: str
    messages: List[ChatMessage] = field(default_factory=list)
    summary: str = ""  # Compressed summary of older messages
    key_entities: List[str] = field(default_factory=list)  # Important topics/drugs/companies mentioned
    last_query_context: str = ""  # Context from last query for continuity
    total_exchanges: int = 0


class MemoryManager:
    """
    Manages conversation memory with optimization for LLM context.
    
    - Keeps last N messages in full detail
    - Summarizes older messages into a compact summary
    - Tracks key entities mentioned in conversation
    """
    
    # Configuration
    SHORT_TERM_LIMIT = 4  # Number of recent message pairs to keep in full
    SUMMARY_TRIGGER = 6   # Number of messages before triggering summarization
    MAX_SUMMARY_LENGTH = 500  # Max chars for summary
    
    def __init__(self):
        self.sessions: Dict[str, ConversationMemory] = {}
    
    def get_or_create_session(self, session_id: str) -> ConversationMemory:
        """Get existing session or create new one"""
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationMemory(session_id=session_id)
        return self.sessions[session_id]
    
    def add_message(self, session_id: str, role: str, content: str, has_visuals: bool = False):
        """Add a message to the conversation history"""
        session = self.get_or_create_session(session_id)
        msg = ChatMessage(role=role, content=content, has_visuals=has_visuals)
        session.messages.append(msg)
        session.total_exchanges = len([m for m in session.messages if m.role == 'user'])
        
        # Extract key entities from user messages
        if role == 'user':
            entities = self._extract_entities(content)
            for entity in entities:
                if entity not in session.key_entities:
                    session.key_entities.append(entity)
            # Keep only last 10 entities
            session.key_entities = session.key_entities[-10:]
        
        # Check if we need to summarize older messages
        if len(session.messages) > self.SUMMARY_TRIGGER:
            self._compress_old_messages(session)
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract important entities from text (simple keyword extraction)"""
        # Common pharmaceutical terms to look for
        pharma_indicators = [
            'drug', 'medicine', 'pharmaceutical', 'clinical', 'trial', 'patent',
            'market', 'import', 'export', 'api', 'formulation'
        ]
        
        entities = []
        words = text.lower().split()
        
        # Look for capitalized words (potential drug names, company names)
        for word in text.split():
            if len(word) > 2 and word[0].isupper() and word not in ['I', 'The', 'What', 'How', 'Why', 'When', 'Where', 'Can', 'Could', 'Would', 'Should', 'Is', 'Are', 'Do', 'Does']:
                clean_word = ''.join(c for c in word if c.isalnum())
                if clean_word and clean_word not in entities:
                    entities.append(clean_word)
        
        return entities[:5]  # Limit to 5 entities per message
    
    def _compress_old_messages(self, session: ConversationMemory):
        """Compress older messages into summary, keeping recent ones in full"""
        if len(session.messages) <= self.SHORT_TERM_LIMIT * 2:
            return
        
        # Messages to summarize (older ones)
        messages_to_summarize = session.messages[:-self.SHORT_TERM_LIMIT * 2]
        
        # Build summary from older messages
        summary_parts = []
        if session.summary:
            summary_parts.append(session.summary)
        
        for msg in messages_to_summarize:
            if msg.role == 'user':
                # Compress user message to key query
                summary_parts.append(f"User asked about: {msg.content[:100]}")
            else:
                # Compress assistant response to key points
                summary_parts.append(f"Assistant provided: {msg.content[:150]}...")
        
        # Combine and truncate summary
        full_summary = " | ".join(summary_parts)
        if len(full_summary) > self.MAX_SUMMARY_LENGTH:
            full_summary = full_summary[:self.MAX_SUMMARY_LENGTH] + "..."
        
        session.summary = full_summary
        
        # Keep only recent messages
        session.messages = session.messages[-self.SHORT_TERM_LIMIT * 2:]
    
    def get_context_for_llm(self, session_id: str, current_query: str) -> Dict[str, Any]:
        """
        Get optimized context for LLM, combining:
        - Conversation summary (compressed history)
        - Recent messages (full detail)
        - Key entities
        - Current query context
        """
        session = self.get_or_create_session(session_id)
        
        context = {
            "has_history": len(session.messages) > 0,
            "conversation_summary": session.summary if session.summary else None,
            "key_topics": session.key_entities if session.key_entities else None,
            "recent_messages": [],
            "total_exchanges": session.total_exchanges,
            "is_follow_up": self._is_follow_up_query(current_query, session)
        }
        
        # Add recent messages (last N pairs)
        for msg in session.messages[-self.SHORT_TERM_LIMIT * 2:]:
            context["recent_messages"].append({
                "role": msg.role,
                "content": msg.content[:500] if len(msg.content) > 500 else msg.content,
                "had_visuals": msg.has_visuals
            })
        
        return context
    
    def _is_follow_up_query(self, query: str, session: ConversationMemory) -> bool:
        """Detect if query is a follow-up to previous conversation"""
        follow_up_indicators = [
            'more', 'also', 'what about', 'how about', 'and', 'additionally',
            'tell me more', 'expand', 'detail', 'specifically', 'same',
            'that', 'this', 'it', 'they', 'those', 'these', 'previous',
            'earlier', 'mentioned', 'you said', 'compare', 'versus', 'vs'
        ]
        
        query_lower = query.lower()
        
        # Check for follow-up indicators
        for indicator in follow_up_indicators:
            if indicator in query_lower:
                return True
        
        # Check if query references entities from conversation
        for entity in session.key_entities:
            if entity.lower() in query_lower:
                return True
        
        return False
    
    def build_memory_prompt(self, session_id: str, current_query: str) -> str:
        """
        Build a compact memory context string to inject into LLM prompt.
        Returns empty string if no relevant history.
        """
        context = self.get_context_for_llm(session_id, current_query)
        
        if not context["has_history"]:
            return ""
        
        parts = []
        
        # Add conversation summary if exists
        if context["conversation_summary"]:
            parts.append(f"[Previous conversation summary: {context['conversation_summary']}]")
        
        # Add key topics if exists
        if context["key_topics"]:
            parts.append(f"[Key topics discussed: {', '.join(context['key_topics'])}]")
        
        # Add recent exchange context
        if context["recent_messages"]:
            parts.append("[Recent conversation:]")
            for msg in context["recent_messages"][-4:]:  # Last 2 exchanges
                role = "User" if msg["role"] == "user" else "Assistant"
                content = msg["content"]
                if len(content) > 200:
                    content = content[:200] + "..."
                parts.append(f"{role}: {content}")
        
        if context["is_follow_up"]:
            parts.append("[Note: This appears to be a follow-up question to the previous conversation]")
        
        return "\n".join(parts)
    
    def get_last_query_result(self, session_id: str) -> Optional[str]:
        """Get the last assistant response for reference"""
        session = self.get_or_create_session(session_id)
        for msg in reversed(session.messages):
            if msg.role == 'assistant':
                return msg.content
        return None
    
    def clear_session(self, session_id: str):
        """Clear a specific session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics about a session"""
        session = self.get_or_create_session(session_id)
        return {
            "session_id": session_id,
            "total_messages": len(session.messages),
            "total_exchanges": session.total_exchanges,
            "has_summary": bool(session.summary),
            "key_entities_count": len(session.key_entities),
            "key_entities": session.key_entities
        }


# Global memory manager instance
memory_manager = MemoryManager()


# Convenience functions for easy import
def add_to_memory(session_id: str, role: str, content: str, has_visuals: bool = False):
    """Add a message to session memory"""
    memory_manager.add_message(session_id, role, content, has_visuals)


def get_memory_context(session_id: str, current_query: str) -> str:
    """Get memory context string for LLM prompt"""
    return memory_manager.build_memory_prompt(session_id, current_query)


def get_memory_for_llm(session_id: str, current_query: str) -> Dict[str, Any]:
    """Get full memory context dict for LLM"""
    return memory_manager.get_context_for_llm(session_id, current_query)


def clear_memory(session_id: str):
    """Clear session memory"""
    memory_manager.clear_session(session_id)
