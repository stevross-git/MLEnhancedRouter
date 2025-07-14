"""
Collaborative AI Router
Orchestrates real-time collaboration between multiple AI agents
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import threading
import time

from shared_memory import (
    get_shared_memory_manager, 
    MessageType, 
    MessageStatus,
    SharedMemoryManager
)
from ai_models import AIModelManager, AIProvider
from rag_chat import get_rag_chat
from external_llm_integration import get_external_llm_manager, QueryComplexity

logger = logging.getLogger(__name__)

@dataclass
class CollaborativeAgent:
    """Represents an AI agent in collaborative mode"""
    agent_id: str
    name: str
    model_id: str
    specialization: str
    confidence_threshold: float = 0.7
    max_concurrent_sessions: int = 3
    is_active: bool = True
    current_sessions: List[str] = None
    
    def __post_init__(self):
        if self.current_sessions is None:
            self.current_sessions = []

class CollaborativeRouter:
    """Manages collaborative AI processing with shared memory"""
    
    def __init__(self, ai_model_manager: AIModelManager):
        self.ai_model_manager = ai_model_manager
        self.shared_memory = get_shared_memory_manager()
        self.external_llm_manager = get_external_llm_manager()
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.collaborative_agents: Dict[str, CollaborativeAgent] = {}
        self.session_lock = threading.RLock()
        
        # Initialize default collaborative agents
        self._initialize_collaborative_agents()
        
        # Start session monitoring
        self.monitor_thread = threading.Thread(target=self._monitor_sessions, daemon=True)
        self.monitor_thread.start()
    
    def _initialize_collaborative_agents(self):
        """Initialize collaborative agents with different specializations"""
        specializations = [
            ("analyst", "Analysis and data interpretation", 0.8),
            ("creative", "Creative thinking and ideation", 0.7),
            ("technical", "Technical problem solving", 0.8),
            ("researcher", "Research and fact-finding", 0.7),
            ("synthesizer", "Information synthesis and conclusions", 0.8)
        ]
        
        available_models = self.ai_model_manager.get_all_models()
        
        for i, (spec_name, description, threshold) in enumerate(specializations):
            # Use different models for different agents if available
            model = available_models[i % len(available_models)] if available_models else None
            
            if model:
                agent = CollaborativeAgent(
                    agent_id=f"collab_{spec_name}",
                    name=f"Collaborative {spec_name.title()}",
                    model_id=model.id,
                    specialization=description,
                    confidence_threshold=threshold
                )
                self.collaborative_agents[agent.agent_id] = agent
                logger.info(f"Initialized collaborative agent: {agent.name}")
    
    def update_agent_model(self, agent_id: str, model_id: str) -> bool:
        """Update the AI model for a specific collaborative agent"""
        if agent_id not in self.collaborative_agents:
            return False
        
        # Verify the model exists
        model = self.ai_model_manager.get_model(model_id)
        if not model:
            return False
        
        self.collaborative_agents[agent_id].model_id = model_id
        logger.info(f"Updated agent {agent_id} to use model {model_id}")
        return True
    
    def get_agent_configurations(self) -> Dict[str, Any]:
        """Get current agent configurations"""
        configurations = {}
        available_models = self.ai_model_manager.get_all_models()
        
        for agent_id, agent in self.collaborative_agents.items():
            current_model = self.ai_model_manager.get_model(agent.model_id)
            configurations[agent_id] = {
                'name': agent.name,
                'specialization': agent.specialization,
                'current_model': {
                    'id': current_model.id if current_model else None,
                    'name': current_model.name if current_model else 'Unknown',
                    'provider': current_model.provider.value if current_model else 'Unknown'
                },
                'confidence_threshold': agent.confidence_threshold,
                'is_active': agent.is_active,
                'current_sessions': len(agent.current_sessions),
                'max_concurrent_sessions': agent.max_concurrent_sessions
            }
        
        return {
            'agents': configurations,
            'available_models': [
                {
                    'id': model.id,
                    'name': model.name,
                    'provider': model.provider.value
                }
                for model in available_models
            ]
        }
    
    async def process_collaborative_query(self, query: str, 
                                        enable_rag: bool = False,
                                        max_agents: int = 3,
                                        collaboration_timeout: int = 300,
                                        selected_agents: List[str] = None) -> Dict[str, Any]:
        """Process a query using collaborative AI agents"""
        
        # Select agents - either user-specified or automatically selected
        if selected_agents:
            # User specified agents
            agent_objects = []
            for agent_id in selected_agents:
                if agent_id in self.collaborative_agents:
                    agent = self.collaborative_agents[agent_id]
                    if agent.is_active and len(agent.current_sessions) < agent.max_concurrent_sessions:
                        agent_objects.append(agent)
            
            if not agent_objects:
                return {
                    'error': 'No suitable agents available from selection',
                    'query': query,
                    'timestamp': datetime.now().isoformat()
                }
            
            selected_agents = agent_objects
        else:
            # Auto-select best agents for this query
            selected_agents = self._select_agents_for_query(query, max_agents)
            
            if not selected_agents:
                return {
                    'error': 'No suitable agents available for collaboration',
                    'query': query,
                    'timestamp': datetime.now().isoformat()
                }
        
        # Create collaboration session
        session_id = self.shared_memory.create_session(
            query=query,
            agent_ids=[agent.agent_id for agent in selected_agents]
        )
        
        logger.info(f"Starting collaborative session {session_id} with {len(selected_agents)} agents")
        
        # Initialize session tracking
        with self.session_lock:
            self.active_sessions[session_id] = {
                'query': query,
                'agents': selected_agents,
                'start_time': datetime.now(),
                'status': 'active',
                'enable_rag': enable_rag,
                'responses': {},
                'final_response': None
            }
        
        # Start collaborative processing
        try:
            result = await self._run_collaborative_session(
                session_id, 
                query, 
                selected_agents, 
                enable_rag, 
                collaboration_timeout
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in collaborative session {session_id}: {e}")
            return {
                'error': f'Collaboration failed: {str(e)}',
                'session_id': session_id,
                'query': query,
                'timestamp': datetime.now().isoformat()
            }
        
        finally:
            # Clean up session
            with self.session_lock:
                if session_id in self.active_sessions:
                    del self.active_sessions[session_id]
    
    def _select_agents_for_query(self, query: str, max_agents: int) -> List[CollaborativeAgent]:
        """Select the best agents for a given query"""
        available_agents = [
            agent for agent in self.collaborative_agents.values()
            if agent.is_active and len(agent.current_sessions) < agent.max_concurrent_sessions
        ]
        
        if not available_agents:
            return []
        
        # Simple selection based on query keywords
        # In production, this would use more sophisticated matching
        query_lower = query.lower()
        scored_agents = []
        
        for agent in available_agents:
            score = 0.5  # Base score
            
            # Score based on specialization keywords
            if agent.specialization:
                spec_words = agent.specialization.lower().split()
                for word in spec_words:
                    if word in query_lower:
                        score += 0.2
            
            # Score based on agent type
            if 'analysis' in query_lower or 'analyze' in query_lower:
                if 'analyst' in agent.agent_id:
                    score += 0.3
            elif 'creative' in query_lower or 'brainstorm' in query_lower:
                if 'creative' in agent.agent_id:
                    score += 0.3
            elif 'technical' in query_lower or 'code' in query_lower:
                if 'technical' in agent.agent_id:
                    score += 0.3
            elif 'research' in query_lower or 'find' in query_lower:
                if 'researcher' in agent.agent_id:
                    score += 0.3
            
            scored_agents.append((agent, score))
        
        # Sort by score and select top agents
        scored_agents.sort(key=lambda x: x[1], reverse=True)
        selected = [agent for agent, score in scored_agents[:max_agents]]
        
        # Always include synthesizer if available and not already selected
        synthesizer = next((a for a in available_agents if 'synthesizer' in a.agent_id), None)
        if synthesizer and synthesizer not in selected and len(selected) < max_agents:
            selected.append(synthesizer)
        
        return selected[:max_agents]
    
    async def _run_collaborative_session(self, session_id: str, query: str, 
                                       agents: List[CollaborativeAgent],
                                       enable_rag: bool, 
                                       timeout: int) -> Dict[str, Any]:
        """Run a collaborative session with multiple agents and external LLM integration"""
        
        # Check if query requires external LLM processing
        requires_external_llm = self.external_llm_manager.is_complex_query(query)
        external_llm_response = None
        
        if requires_external_llm:
            logger.info(f"Complex query detected, using external LLM for initial analysis")
            
            # Get RAG context for external LLM
            rag_context = ""
            if enable_rag:
                rag_context = await self._get_rag_context(query)
            
            try:
                # Process with external LLM
                external_llm_response = await self.external_llm_manager.process_complex_query(
                    query, context=rag_context
                )
                
                # Store external LLM response in shared memory
                self.shared_memory.add_message(
                    agent_id="external_llm",
                    agent_name=f"External LLM ({external_llm_response['provider']})",
                    message_type=MessageType.EXTERNAL_LLM_RESPONSE,
                    content=external_llm_response["response"],
                    metadata={
                        "provider": external_llm_response["provider"],
                        "complexity": external_llm_response["complexity"],
                        "processing_time": external_llm_response["processing_time"],
                        "session_id": session_id
                    }
                )
                
                logger.info(f"External LLM processing completed with {external_llm_response['provider']}")
                
            except Exception as e:
                logger.error(f"External LLM processing failed: {str(e)}")
                # Continue with regular collaborative processing
                external_llm_response = None
        
        # Enhance query with RAG if enabled
        enhanced_query = query
        rag_context = None
        
        if enable_rag:
            try:
                rag_system = get_rag_chat()
                rag_context = await asyncio.get_event_loop().run_in_executor(
                    None, rag_system.enhance_query_with_context, query
                )
                if rag_context and rag_context.get('enhanced_query'):
                    enhanced_query = rag_context['enhanced_query']
            except Exception as e:
                logger.error(f"RAG enhancement failed: {e}")
        
        # Add initial query to shared memory
        self.shared_memory.add_message(
            session_id=session_id,
            agent_id="system",
            agent_name="System",
            message_type=MessageType.QUERY,
            content=enhanced_query,
            metadata={'original_query': query, 'rag_context': rag_context}
        )
        
        # Start agents in parallel
        tasks = []
        for agent in agents:
            task = asyncio.create_task(
                self._run_agent_collaboration(session_id, agent, enhanced_query, timeout)
            )
            tasks.append(task)
        
        # Wait for all agents to complete or timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True), 
                timeout=timeout
            )
            
            # Process results
            agent_responses = {}
            for i, result in enumerate(results):
                agent = agents[i]
                if isinstance(result, Exception):
                    logger.error(f"Agent {agent.name} failed: {result}")
                    agent_responses[agent.agent_id] = {
                        'error': str(result),
                        'agent_name': agent.name
                    }
                else:
                    agent_responses[agent.agent_id] = result
            
            # Synthesize final response
            final_response = await self._synthesize_collaborative_response(
                session_id, agent_responses, query
            )
            
            # Finalize session
            confidence_score = self._calculate_session_confidence(agent_responses)
            self.shared_memory.finalize_session(session_id, final_response, confidence_score)
            
            return {
                'session_id': session_id,
                'query': query,
                'enhanced_query': enhanced_query,
                'agents_used': [agent.name for agent in agents],
                'agent_responses': agent_responses,
                'final_response': final_response,
                'confidence_score': confidence_score,
                'rag_used': enable_rag,
                'rag_context': rag_context,
                'timestamp': datetime.now().isoformat()
            }
            
        except asyncio.TimeoutError:
            logger.warning(f"Collaborative session {session_id} timed out")
            return {
                'error': 'Collaboration session timed out',
                'session_id': session_id,
                'query': query,
                'partial_responses': self._get_partial_responses(session_id),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _run_agent_collaboration(self, session_id: str, agent: CollaborativeAgent, 
                                     query: str, timeout: int) -> Dict[str, Any]:
        """Run a single agent's collaboration process"""
        
        # Add agent to session
        agent.current_sessions.append(session_id)
        
        try:
            # Get shared context
            shared_context = self.shared_memory.get_shared_context(session_id)
            
            # Update agent's working memory
            self.shared_memory.update_working_memory(
                session_id, agent.agent_id, 'specialization', agent.specialization
            )
            
            # Add thinking message
            self.shared_memory.add_message(
                session_id=session_id,
                agent_id=agent.agent_id,
                agent_name=agent.name,
                message_type=MessageType.THOUGHT,
                content=f"Starting analysis of query: {query}",
                metadata={'specialization': agent.specialization}
            )
            
            # Process with AI model
            system_message = f"""You are {agent.name}, specialized in {agent.specialization}.
            
You are collaborating with other AI agents to answer this query: {query}

Your role is to:
1. Provide insights from your specialization perspective
2. Share relevant thoughts and findings
3. Ask clarifying questions if needed
4. Build upon other agents' contributions

Shared context from other agents:
{json.dumps(shared_context, indent=2)}

Provide a focused response based on your expertise."""
            
            # Generate response
            response = await self.ai_model_manager.generate_response(
                query=query,
                system_message=system_message,
                model_id=agent.model_id
            )
            
            # Add response to shared memory
            response_id = self.shared_memory.add_message(
                session_id=session_id,
                agent_id=agent.agent_id,
                agent_name=agent.name,
                message_type=MessageType.RESPONSE,
                content=response.get('response', ''),
                metadata={
                    'model_used': response.get('model_used'),
                    'response_time': response.get('response_time'),
                    'cached': response.get('cached', False)
                }
            )
            
            # Update scratchpad
            self.shared_memory.update_agent_scratchpad(
                session_id, agent.agent_id, {
                    'action': 'generated_response',
                    'response_id': response_id,
                    'query': query,
                    'response_length': len(response.get('response', '')),
                    'specialization': agent.specialization
                }
            )
            
            return {
                'agent_id': agent.agent_id,
                'agent_name': agent.name,
                'response': response.get('response', ''),
                'model_used': response.get('model_used'),
                'response_time': response.get('response_time'),
                'cached': response.get('cached', False),
                'specialization': agent.specialization
            }
            
        except Exception as e:
            logger.error(f"Agent {agent.name} collaboration failed: {e}")
            
            # Add error message
            self.shared_memory.add_message(
                session_id=session_id,
                agent_id=agent.agent_id,
                agent_name=agent.name,
                message_type=MessageType.RESPONSE,
                content=f"Error: {str(e)}",
                metadata={'error': True}
            )
            
            raise e
            
        finally:
            # Remove agent from session
            if session_id in agent.current_sessions:
                agent.current_sessions.remove(session_id)
    
    async def _synthesize_collaborative_response(self, session_id: str, 
                                               agent_responses: Dict[str, Any],
                                               original_query: str) -> str:
        """Synthesize a final response from all agent contributions"""
        
        # Get synthesizer agent
        synthesizer = next((a for a in self.collaborative_agents.values() 
                          if 'synthesizer' in a.agent_id), None)
        
        if not synthesizer:
            # Fallback synthesis
            responses = []
            for agent_id, response in agent_responses.items():
                if not response.get('error'):
                    responses.append(f"**{response.get('agent_name', agent_id)}**: {response.get('response', '')}")
            
            return "\n\n".join(responses)
        
        # Use synthesizer agent
        synthesis_prompt = f"""Synthesize a comprehensive response to this query: {original_query}

Based on contributions from multiple AI agents:

{json.dumps(agent_responses, indent=2)}

Create a cohesive, well-structured response that:
1. Addresses the original query directly
2. Incorporates the best insights from each agent
3. Resolves any contradictions
4. Provides a clear, actionable conclusion

Focus on accuracy, completeness, and clarity."""
        
        try:
            synthesis = await self.ai_model_manager.generate_response(
                query=synthesis_prompt,
                system_message="You are a synthesis specialist. Create comprehensive, accurate responses from multiple AI perspectives.",
                model_id=synthesizer.model_id
            )
            
            # Add synthesis to shared memory
            self.shared_memory.add_message(
                session_id=session_id,
                agent_id=synthesizer.agent_id,
                agent_name=synthesizer.name,
                message_type=MessageType.CONCLUSION,
                content=synthesis.get('response', ''),
                metadata={'synthesis': True}
            )
            
            return synthesis.get('response', '')
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return f"Synthesis error: {str(e)}"
    
    def _calculate_session_confidence(self, agent_responses: Dict[str, Any]) -> float:
        """Calculate confidence score for the session"""
        if not agent_responses:
            return 0.0
        
        successful_responses = [r for r in agent_responses.values() if not r.get('error')]
        if not successful_responses:
            return 0.0
        
        # Base confidence on response quality and agent agreement
        confidence = len(successful_responses) / len(agent_responses)
        
        # Adjust based on response lengths and agent types
        avg_response_length = sum(len(r.get('response', '')) for r in successful_responses) / len(successful_responses)
        if avg_response_length > 100:  # Substantial responses
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _get_partial_responses(self, session_id: str) -> Dict[str, Any]:
        """Get partial responses from a session"""
        messages = self.shared_memory.get_session_messages(
            session_id, 
            message_types=[MessageType.RESPONSE, MessageType.THOUGHT]
        )
        
        partial = {}
        for message in messages:
            if message.agent_id not in partial:
                partial[message.agent_id] = {
                    'agent_name': message.agent_name,
                    'messages': []
                }
            
            partial[message.agent_id]['messages'].append({
                'type': message.message_type.value,
                'content': message.content,
                'timestamp': message.timestamp.isoformat()
            })
        
        return partial
    
    def _monitor_sessions(self):
        """Monitor active sessions for cleanup"""
        while True:
            try:
                time.sleep(60)  # Check every minute
                
                with self.session_lock:
                    current_time = datetime.now()
                    expired_sessions = []
                    
                    for session_id, session_data in self.active_sessions.items():
                        # Check if session is too old
                        if current_time - session_data['start_time'] > timedelta(minutes=30):
                            expired_sessions.append(session_id)
                    
                    for session_id in expired_sessions:
                        logger.info(f"Cleaning up expired session {session_id}")
                        del self.active_sessions[session_id]
                
            except Exception as e:
                logger.error(f"Error in session monitoring: {e}")
    
    def get_active_sessions(self) -> Dict[str, Any]:
        """Get information about active sessions"""
        with self.session_lock:
            return {
                session_id: {
                    'query': data['query'],
                    'agents': [agent.name for agent in data['agents']],
                    'start_time': data['start_time'].isoformat(),
                    'status': data['status'],
                    'duration_minutes': (datetime.now() - data['start_time']).total_seconds() / 60
                }
                for session_id, data in self.active_sessions.items()
            }
    
    def get_session_details(self, session_id: str) -> Dict[str, Any]:
        """Get detailed information about a session"""
        stats = self.shared_memory.get_session_stats(session_id)
        messages = self.shared_memory.get_session_messages(session_id)
        
        return {
            'stats': stats,
            'messages': [msg.to_dict() for msg in messages[-20:]],  # Last 20 messages
            'shared_context': self.shared_memory.get_shared_context(session_id)
        }

# Global collaborative router instance
_collaborative_router = None

def get_collaborative_router(ai_model_manager: AIModelManager = None) -> CollaborativeRouter:
    """Get or create the global collaborative router"""
    global _collaborative_router
    if _collaborative_router is None:
        if ai_model_manager is None:
            from ai_models import AIModelManager
            ai_model_manager = AIModelManager()
        _collaborative_router = CollaborativeRouter(ai_model_manager)
    return _collaborative_router