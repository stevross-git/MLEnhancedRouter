"""
Shared Memory System for AI Collaboration
Provides real-time shared memory and scratchpad for multiple AI agents
"""

import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import logging

logger = logging.getLogger(__name__)

class MessageType(Enum):
    QUERY = "query"
    RESPONSE = "response"
    THOUGHT = "thought"
    QUESTION = "question"
    FACT = "fact"
    CONCLUSION = "conclusion"
    COLLABORATION = "collaboration"

class MessageStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class SharedMessage:
    """A message in the shared memory system"""
    id: str
    agent_id: str
    agent_name: str
    message_type: MessageType
    content: str
    timestamp: datetime
    status: MessageStatus = MessageStatus.PENDING
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    relevance_score: float = 1.0
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['message_type'] = self.message_type.value
        data['status'] = self.status.value
        return data

@dataclass
class CollaborationSession:
    """A collaborative session between AI agents"""
    session_id: str
    original_query: str
    created_at: datetime
    participants: List[str]
    status: str = "active"
    final_response: Optional[str] = None
    confidence_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data

class SharedMemoryManager:
    """Manages shared memory and scratchpad for AI collaboration"""
    
    def __init__(self, max_memory_size: int = 10000, cleanup_interval: int = 3600):
        self.max_memory_size = max_memory_size
        self.cleanup_interval = cleanup_interval
        
        # Shared memory storage
        self.messages: Dict[str, SharedMessage] = {}
        self.sessions: Dict[str, CollaborationSession] = {}
        self.agent_contexts: Dict[str, Dict[str, Any]] = {}
        
        # Real-time updates
        self.subscribers: Dict[str, List[Callable]] = {}
        self.message_index: Dict[str, List[str]] = {}  # session_id -> message_ids
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def create_session(self, query: str, agent_ids: List[str]) -> str:
        """Create a new collaboration session"""
        session_id = str(uuid.uuid4())
        
        with self.lock:
            session = CollaborationSession(
                session_id=session_id,
                original_query=query,
                created_at=datetime.now(),
                participants=agent_ids.copy()
            )
            
            self.sessions[session_id] = session
            self.message_index[session_id] = []
            
            # Initialize agent contexts for this session
            for agent_id in agent_ids:
                if agent_id not in self.agent_contexts:
                    self.agent_contexts[agent_id] = {}
                
                self.agent_contexts[agent_id][session_id] = {
                    'working_memory': {},
                    'scratchpad': [],
                    'last_activity': datetime.now()
                }
            
            logger.info(f"Created collaboration session {session_id} with {len(agent_ids)} agents")
            return session_id
    
    def add_message(self, session_id: str, agent_id: str, agent_name: str, 
                   message_type: MessageType, content: str, 
                   parent_id: Optional[str] = None,
                   metadata: Dict[str, Any] = None) -> str:
        """Add a message to shared memory"""
        message_id = str(uuid.uuid4())
        
        with self.lock:
            if session_id not in self.sessions:
                raise ValueError(f"Session {session_id} not found")
            
            message = SharedMessage(
                id=message_id,
                agent_id=agent_id,
                agent_name=agent_name,
                message_type=message_type,
                content=content,
                timestamp=datetime.now(),
                parent_id=parent_id,
                metadata=metadata or {}
            )
            
            self.messages[message_id] = message
            self.message_index[session_id].append(message_id)
            
            # Update agent context
            if agent_id in self.agent_contexts and session_id in self.agent_contexts[agent_id]:
                self.agent_contexts[agent_id][session_id]['last_activity'] = datetime.now()
            
            # Notify subscribers
            self._notify_subscribers(session_id, message)
            
            logger.debug(f"Added message {message_id} to session {session_id}")
            return message_id
    
    def get_session_messages(self, session_id: str, 
                           message_types: List[MessageType] = None,
                           since: datetime = None) -> List[SharedMessage]:
        """Get messages from a session"""
        with self.lock:
            if session_id not in self.message_index:
                return []
            
            messages = []
            for message_id in self.message_index[session_id]:
                if message_id in self.messages:
                    message = self.messages[message_id]
                    
                    # Filter by message types
                    if message_types and message.message_type not in message_types:
                        continue
                    
                    # Filter by timestamp
                    if since and message.timestamp < since:
                        continue
                    
                    messages.append(message)
            
            return sorted(messages, key=lambda m: m.timestamp)
    
    def update_agent_scratchpad(self, session_id: str, agent_id: str, 
                              scratchpad_entry: Dict[str, Any]) -> None:
        """Update agent's scratchpad for a session"""
        with self.lock:
            if (agent_id in self.agent_contexts and 
                session_id in self.agent_contexts[agent_id]):
                
                scratchpad_entry['timestamp'] = datetime.now().isoformat()
                self.agent_contexts[agent_id][session_id]['scratchpad'].append(scratchpad_entry)
                
                # Keep only last 100 entries
                if len(self.agent_contexts[agent_id][session_id]['scratchpad']) > 100:
                    self.agent_contexts[agent_id][session_id]['scratchpad'].pop(0)
    
    def get_agent_scratchpad(self, session_id: str, agent_id: str) -> List[Dict[str, Any]]:
        """Get agent's scratchpad for a session"""
        with self.lock:
            if (agent_id in self.agent_contexts and 
                session_id in self.agent_contexts[agent_id]):
                return self.agent_contexts[agent_id][session_id]['scratchpad'].copy()
            return []
    
    def update_working_memory(self, session_id: str, agent_id: str, 
                            key: str, value: Any) -> None:
        """Update agent's working memory"""
        with self.lock:
            if (agent_id in self.agent_contexts and 
                session_id in self.agent_contexts[agent_id]):
                self.agent_contexts[agent_id][session_id]['working_memory'][key] = value
    
    def get_working_memory(self, session_id: str, agent_id: str) -> Dict[str, Any]:
        """Get agent's working memory"""
        with self.lock:
            if (agent_id in self.agent_contexts and 
                session_id in self.agent_contexts[agent_id]):
                return self.agent_contexts[agent_id][session_id]['working_memory'].copy()
            return {}
    
    def get_shared_context(self, session_id: str) -> Dict[str, Any]:
        """Get shared context for all agents in a session"""
        with self.lock:
            if session_id not in self.sessions:
                return {}
            
            session = self.sessions[session_id]
            messages = self.get_session_messages(session_id)
            
            # Build shared context
            shared_context = {
                'session_id': session_id,
                'original_query': session.original_query,
                'participants': session.participants,
                'message_count': len(messages),
                'latest_thoughts': [],
                'facts_discovered': [],
                'questions_raised': [],
                'conclusions': []
            }
            
            # Extract different types of information
            for message in messages[-50:]:  # Last 50 messages
                if message.message_type == MessageType.THOUGHT:
                    shared_context['latest_thoughts'].append({
                        'agent': message.agent_name,
                        'content': message.content,
                        'timestamp': message.timestamp.isoformat()
                    })
                elif message.message_type == MessageType.FACT:
                    shared_context['facts_discovered'].append({
                        'agent': message.agent_name,
                        'content': message.content,
                        'timestamp': message.timestamp.isoformat()
                    })
                elif message.message_type == MessageType.QUESTION:
                    shared_context['questions_raised'].append({
                        'agent': message.agent_name,
                        'content': message.content,
                        'timestamp': message.timestamp.isoformat()
                    })
                elif message.message_type == MessageType.CONCLUSION:
                    shared_context['conclusions'].append({
                        'agent': message.agent_name,
                        'content': message.content,
                        'timestamp': message.timestamp.isoformat()
                    })
            
            return shared_context
    
    def subscribe_to_session(self, session_id: str, callback: Callable) -> None:
        """Subscribe to real-time updates for a session"""
        with self.lock:
            if session_id not in self.subscribers:
                self.subscribers[session_id] = []
            self.subscribers[session_id].append(callback)
    
    def unsubscribe_from_session(self, session_id: str, callback: Callable) -> None:
        """Unsubscribe from session updates"""
        with self.lock:
            if session_id in self.subscribers:
                try:
                    self.subscribers[session_id].remove(callback)
                except ValueError:
                    pass
    
    def _notify_subscribers(self, session_id: str, message: SharedMessage) -> None:
        """Notify subscribers of new messages"""
        if session_id in self.subscribers:
            for callback in self.subscribers[session_id]:
                try:
                    callback(message)
                except Exception as e:
                    logger.error(f"Error notifying subscriber: {e}")
    
    def finalize_session(self, session_id: str, final_response: str, 
                        confidence_score: float) -> None:
        """Finalize a collaboration session"""
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id].final_response = final_response
                self.sessions[session_id].confidence_score = confidence_score
                self.sessions[session_id].status = "completed"
                
                logger.info(f"Session {session_id} finalized with confidence {confidence_score}")
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a session"""
        with self.lock:
            if session_id not in self.sessions:
                return {}
            
            session = self.sessions[session_id]
            messages = self.get_session_messages(session_id)
            
            stats = {
                'session_id': session_id,
                'status': session.status,
                'participants': len(session.participants),
                'total_messages': len(messages),
                'message_types': {},
                'agent_activity': {},
                'duration_minutes': (datetime.now() - session.created_at).total_seconds() / 60,
                'final_response': session.final_response,
                'confidence_score': session.confidence_score
            }
            
            # Count message types
            for message in messages:
                msg_type = message.message_type.value
                stats['message_types'][msg_type] = stats['message_types'].get(msg_type, 0) + 1
                
                # Count agent activity
                agent = message.agent_name
                stats['agent_activity'][agent] = stats['agent_activity'].get(agent, 0) + 1
            
            return stats
    
    def _cleanup_loop(self) -> None:
        """Background cleanup of old sessions and messages"""
        while True:
            try:
                time.sleep(self.cleanup_interval)
                self._cleanup_old_data()
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    def _cleanup_old_data(self) -> None:
        """Clean up old sessions and messages"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        with self.lock:
            # Clean up old sessions
            expired_sessions = []
            for session_id, session in self.sessions.items():
                if session.created_at < cutoff_time and session.status == "completed":
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                self._cleanup_session(session_id)
            
            # Clean up orphaned messages
            if len(self.messages) > self.max_memory_size:
                self._cleanup_excess_messages()
    
    def _cleanup_session(self, session_id: str) -> None:
        """Clean up a specific session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
        
        if session_id in self.message_index:
            # Clean up messages from this session
            for message_id in self.message_index[session_id]:
                if message_id in self.messages:
                    del self.messages[message_id]
            del self.message_index[session_id]
        
        if session_id in self.subscribers:
            del self.subscribers[session_id]
        
        # Clean up agent contexts
        for agent_id in self.agent_contexts:
            if session_id in self.agent_contexts[agent_id]:
                del self.agent_contexts[agent_id][session_id]
        
        logger.info(f"Cleaned up session {session_id}")
    
    def _cleanup_excess_messages(self) -> None:
        """Clean up excess messages when memory is full"""
        # Sort messages by timestamp and keep only the newest
        sorted_messages = sorted(self.messages.values(), key=lambda m: m.timestamp, reverse=True)
        
        # Keep only the newest messages
        to_keep = sorted_messages[:self.max_memory_size // 2]
        new_messages = {msg.id: msg for msg in to_keep}
        
        # Update message index
        for session_id in self.message_index:
            self.message_index[session_id] = [
                msg_id for msg_id in self.message_index[session_id] 
                if msg_id in new_messages
            ]
        
        self.messages = new_messages
        logger.info(f"Cleaned up excess messages, keeping {len(new_messages)}")

# Global shared memory manager
_shared_memory_manager = None

def get_shared_memory_manager() -> SharedMemoryManager:
    """Get or create the global shared memory manager"""
    global _shared_memory_manager
    if _shared_memory_manager is None:
        _shared_memory_manager = SharedMemoryManager()
    return _shared_memory_manager