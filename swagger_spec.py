"""
OpenAPI/Swagger specification for ML Query Router API
"""

swagger_spec = {
    "openapi": "3.0.3",
    "info": {
        "title": "ML Query Router API",
        "description": "A Flask-based ML-Enhanced Query Router with intelligent agent routing, AI model management, and RAG capabilities",
        "version": "1.0.0",
        "contact": {
            "name": "API Support",
            "email": "support@example.com"
        }
    },
    "servers": [
        {
            "url": "http://localhost:5000",
            "description": "Development server"
        }
    ],
    "tags": [
        {
            "name": "Query Routing",
            "description": "Query submission and routing operations"
        },
        {
            "name": "Agent Management",
            "description": "Agent registration and management"
        },
        {
            "name": "AI Models",
            "description": "AI model management and configuration"
        },
        {
            "name": "Authentication",
            "description": "User authentication and authorization"
        },
        {
            "name": "Settings",
            "description": "Application configuration and settings"
        },
        {
            "name": "Configuration",
            "description": "Advanced system configuration"
        },
        {
            "name": "Cache Management",
            "description": "AI response caching operations"
        },
        {
            "name": "Chat Interface",
            "description": "Chat messaging and session management"
        },
        {
            "name": "RAG System",
            "description": "Document upload and retrieval-augmented generation"
        }
    ],
    "paths": {
        "/api/submit": {
            "post": {
                "tags": ["Query Routing"],
                "summary": "Submit a query for routing",
                "description": "Submit a user query to be classified and routed to appropriate AI agent",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "The user query to be processed"
                                    },
                                    "user_id": {
                                        "type": "string",
                                        "description": "Optional user identifier"
                                    }
                                },
                                "required": ["query"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Query processed successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "query_id": {"type": "string"},
                                        "category": {"type": "string"},
                                        "confidence": {"type": "number"},
                                        "agent_id": {"type": "string"},
                                        "agent_name": {"type": "string"},
                                        "response": {"type": "string"},
                                        "response_time": {"type": "number"}
                                    }
                                }
                            }
                        }
                    },
                    "400": {
                        "description": "Invalid request",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"}
                            }
                        }
                    }
                }
            }
        },
        "/api/agents": {
            "get": {
                "tags": ["Agent Management"],
                "summary": "Get list of available agents",
                "description": "Retrieve all registered agents with their capabilities",
                "responses": {
                    "200": {
                        "description": "List of agents",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "array",
                                    "items": {"$ref": "#/components/schemas/Agent"}
                                }
                            }
                        }
                    }
                }
            },
            "post": {
                "tags": ["Agent Management"],
                "summary": "Register a new agent",
                "description": "Register a new AI agent with the system",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/AgentRegistration"}
                        }
                    }
                },
                "responses": {
                    "201": {
                        "description": "Agent registered successfully",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Agent"}
                            }
                        }
                    },
                    "400": {
                        "description": "Invalid agent data",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"}
                            }
                        }
                    }
                }
            }
        },
        "/api/agents/{agent_id}": {
            "delete": {
                "tags": ["Agent Management"],
                "summary": "Unregister an agent",
                "description": "Remove an agent from the system",
                "parameters": [
                    {
                        "name": "agent_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "The agent ID to unregister"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Agent unregistered successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "message": {"type": "string"}
                                    }
                                }
                            }
                        }
                    },
                    "404": {
                        "description": "Agent not found",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"}
                            }
                        }
                    }
                }
            }
        },
        "/api/stats": {
            "get": {
                "tags": ["Query Routing"],
                "summary": "Get routing statistics",
                "description": "Retrieve system performance and routing statistics",
                "responses": {
                    "200": {
                        "description": "System statistics",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "total_queries": {"type": "integer"},
                                        "successful_routes": {"type": "integer"},
                                        "failed_routes": {"type": "integer"},
                                        "average_response_time": {"type": "number"},
                                        "active_agents": {"type": "integer"},
                                        "categories": {
                                            "type": "object",
                                            "additionalProperties": {"type": "integer"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/ai-models": {
            "get": {
                "tags": ["AI Models"],
                "summary": "Get all AI models",
                "description": "Retrieve all configured AI models",
                "responses": {
                    "200": {
                        "description": "List of AI models",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "array",
                                    "items": {"$ref": "#/components/schemas/AIModel"}
                                }
                            }
                        }
                    }
                }
            },
            "post": {
                "tags": ["AI Models"],
                "summary": "Create a new AI model",
                "description": "Add a new AI model configuration",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/AIModelCreate"}
                        }
                    }
                },
                "responses": {
                    "201": {
                        "description": "AI model created successfully",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/AIModel"}
                            }
                        }
                    }
                }
            }
        },
        "/api/ai-models/{model_id}": {
            "get": {
                "tags": ["AI Models"],
                "summary": "Get specific AI model",
                "description": "Retrieve details of a specific AI model",
                "parameters": [
                    {
                        "name": "model_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "The model ID"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "AI model details",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/AIModel"}
                            }
                        }
                    },
                    "404": {
                        "description": "Model not found",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"}
                            }
                        }
                    }
                }
            },
            "put": {
                "tags": ["AI Models"],
                "summary": "Update AI model",
                "description": "Update an existing AI model configuration",
                "parameters": [
                    {
                        "name": "model_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "The model ID"
                    }
                ],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/AIModelUpdate"}
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "AI model updated successfully",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/AIModel"}
                            }
                        }
                    }
                }
            },
            "delete": {
                "tags": ["AI Models"],
                "summary": "Delete AI model",
                "description": "Remove an AI model from the system",
                "parameters": [
                    {
                        "name": "model_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "The model ID"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "AI model deleted successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "message": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/ai-models/{model_id}/activate": {
            "post": {
                "tags": ["AI Models"],
                "summary": "Activate AI model",
                "description": "Set an AI model as active for use",
                "parameters": [
                    {
                        "name": "model_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "The model ID"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "AI model activated successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "message": {"type": "string"},
                                        "model": {"$ref": "#/components/schemas/AIModel"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/ai-models/{model_id}/test": {
            "post": {
                "tags": ["AI Models"],
                "summary": "Test AI model",
                "description": "Test an AI model with a sample query",
                "parameters": [
                    {
                        "name": "model_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "The model ID"
                    }
                ],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "system_message": {"type": "string"}
                                },
                                "required": ["query"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Test completed successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "response": {"type": "string"},
                                        "response_time": {"type": "number"},
                                        "status": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/ai-models/active": {
            "get": {
                "tags": ["AI Models"],
                "summary": "Get active AI model",
                "description": "Retrieve the currently active AI model",
                "responses": {
                    "200": {
                        "description": "Active AI model",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/AIModel"}
                            }
                        }
                    },
                    "404": {
                        "description": "No active model found",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"}
                            }
                        }
                    }
                }
            }
        },
        "/api/cache/stats": {
            "get": {
                "tags": ["Cache Management"],
                "summary": "Get cache statistics",
                "description": "Retrieve AI response cache statistics",
                "responses": {
                    "200": {
                        "description": "Cache statistics",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "total_entries": {"type": "integer"},
                                        "hit_rate": {"type": "number"},
                                        "miss_rate": {"type": "number"},
                                        "cache_size_mb": {"type": "number"},
                                        "expired_entries": {"type": "integer"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/cache/entries": {
            "get": {
                "tags": ["Cache Management"],
                "summary": "Get cache entries",
                "description": "Retrieve cached AI responses",
                "parameters": [
                    {
                        "name": "limit",
                        "in": "query",
                        "schema": {"type": "integer", "default": 50},
                        "description": "Maximum number of entries to return"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Cache entries",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "array",
                                    "items": {"$ref": "#/components/schemas/CacheEntry"}
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/cache/clear": {
            "post": {
                "tags": ["Cache Management"],
                "summary": "Clear cache",
                "description": "Clear all or specific cache entries",
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "model_id": {"type": "string"},
                                    "expired_only": {"type": "boolean"}
                                }
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Cache cleared successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "message": {"type": "string"},
                                        "cleared_entries": {"type": "integer"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/chat/message": {
            "post": {
                "tags": ["Chat Interface"],
                "summary": "Send chat message",
                "description": "Send a message to the AI chat interface",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "message": {"type": "string"},
                                    "model": {"type": "string"},
                                    "system_message": {"type": "string"},
                                    "enable_rag": {"type": "boolean", "default": False}
                                },
                                "required": ["message"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Message processed successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "response": {"type": "string"},
                                        "model_used": {"type": "string"},
                                        "cached": {"type": "boolean"},
                                        "rag_used": {"type": "boolean"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/chat/stream": {
            "post": {
                "tags": ["Chat Interface"],
                "summary": "Stream chat response",
                "description": "Stream AI response using Server-Sent Events",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "message": {"type": "string"},
                                    "model": {"type": "string"},
                                    "system_message": {"type": "string"},
                                    "enable_rag": {"type": "boolean", "default": False}
                                },
                                "required": ["message"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Streaming response",
                        "content": {
                            "text/event-stream": {
                                "schema": {
                                    "type": "string",
                                    "description": "Server-Sent Events stream"
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/rag/upload": {
            "post": {
                "tags": ["RAG System"],
                "summary": "Upload document for RAG",
                "description": "Upload a document for retrieval-augmented generation",
                "requestBody": {
                    "required": True,
                    "content": {
                        "multipart/form-data": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "file": {
                                        "type": "string",
                                        "format": "binary",
                                        "description": "Document file (PDF, DOCX, TXT, MD, HTML, JSON, CSV)"
                                    }
                                },
                                "required": ["file"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Document uploaded successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "message": {"type": "string"},
                                        "document_id": {"type": "string"},
                                        "filename": {"type": "string"}
                                    }
                                }
                            }
                        }
                    },
                    "400": {
                        "description": "Invalid file or upload error",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"}
                            }
                        }
                    }
                }
            }
        },
        "/api/rag/documents": {
            "get": {
                "tags": ["RAG System"],
                "summary": "List uploaded documents",
                "description": "Get list of all uploaded documents for RAG",
                "responses": {
                    "200": {
                        "description": "List of documents",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "documents": {
                                            "type": "array",
                                            "items": {"$ref": "#/components/schemas/RAGDocument"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/rag/documents/{doc_id}": {
            "delete": {
                "tags": ["RAG System"],
                "summary": "Delete RAG document",
                "description": "Delete a document from the RAG system",
                "parameters": [
                    {
                        "name": "doc_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "The document ID"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Document deleted successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "message": {"type": "string"}
                                    }
                                }
                            }
                        }
                    },
                    "404": {
                        "description": "Document not found",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"}
                            }
                        }
                    }
                }
            }
        },
        "/api/rag/search": {
            "post": {
                "tags": ["RAG System"],
                "summary": "Search documents",
                "description": "Search uploaded documents using vector similarity",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "max_results": {"type": "integer", "default": 3}
                                },
                                "required": ["query"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Search results",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "results": {
                                            "type": "array",
                                            "items": {"$ref": "#/components/schemas/RAGSearchResult"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/rag/stats": {
            "get": {
                "tags": ["RAG System"],
                "summary": "Get RAG statistics",
                "description": "Get statistics about the RAG system",
                "responses": {
                    "200": {
                        "description": "RAG statistics",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "total_documents": {"type": "integer"},
                                        "total_chunks": {"type": "integer"},
                                        "file_types": {
                                            "type": "object",
                                            "additionalProperties": {"type": "integer"}
                                        },
                                        "collection_initialized": {"type": "boolean"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/collaborate": {
            "post": {
                "tags": ["Collaborative AI"],
                "summary": "Submit collaborative query",
                "description": "Submit a query for collaborative AI processing with multiple specialized agents",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "The query to process collaboratively",
                                        "example": "What are the pros and cons of artificial intelligence?"
                                    },
                                    "enable_rag": {
                                        "type": "boolean",
                                        "description": "Whether to enable RAG (Retrieval-Augmented Generation) for document context",
                                        "default": False
                                    },
                                    "max_agents": {
                                        "type": "integer",
                                        "description": "Maximum number of agents to use (only applies to automatic selection)",
                                        "default": 3,
                                        "minimum": 1,
                                        "maximum": 5
                                    },
                                    "timeout": {
                                        "type": "integer",
                                        "description": "Collaboration timeout in seconds",
                                        "default": 300,
                                        "minimum": 60,
                                        "maximum": 600
                                    },
                                    "selected_agents": {
                                        "type": "array",
                                        "items": {
                                            "type": "string"
                                        },
                                        "description": "Specific agents to use for collaboration (overrides automatic selection)",
                                        "example": ["collab_analyst", "collab_technical"],
                                        "enum": ["collab_analyst", "collab_creative", "collab_technical", "collab_researcher", "collab_synthesizer"]
                                    }
                                },
                                "required": ["query"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Collaborative processing completed",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/CollaborativeResult"}
                            }
                        }
                    }
                }
            }
        },
        "/api/collaborate/sessions": {
            "get": {
                "tags": ["Collaborative AI"],
                "summary": "Get active collaboration sessions",
                "description": "Get information about active collaboration sessions",
                "responses": {
                    "200": {
                        "description": "Active sessions",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "sessions": {
                                            "type": "object",
                                            "additionalProperties": {"$ref": "#/components/schemas/CollaborativeSession"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/collaborate/sessions/{session_id}": {
            "get": {
                "tags": ["Collaborative AI"],
                "summary": "Get session details",
                "description": "Get detailed information about a specific collaboration session",
                "parameters": [
                    {
                        "name": "session_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "Session ID"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Session details",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/CollaborativeSession"}
                            }
                        }
                    }
                }
            }
        },
        "/api/collaborate/agents": {
            "get": {
                "tags": ["Collaborative AI"],
                "summary": "Get collaborative agent configurations",
                "description": "Get current configurations for all collaborative agents",
                "responses": {
                    "200": {
                        "description": "Agent configurations",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/CollaborativeAgentConfig"}
                            }
                        }
                    }
                }
            }
        },
        "/api/collaborate/agents/{agent_id}/model": {
            "put": {
                "tags": ["Collaborative AI"],
                "summary": "Update agent model",
                "description": "Update the AI model for a specific collaborative agent",
                "parameters": [
                    {
                        "name": "agent_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "Agent ID"
                    }
                ],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "model_id": {
                                        "type": "string",
                                        "description": "ID of the AI model to use"
                                    }
                                },
                                "required": ["model_id"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Agent model updated successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "message": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/shared-memory/stats": {
            "get": {
                "tags": ["Shared Memory"],
                "summary": "Get shared memory statistics",
                "description": "Get statistics about the shared memory system",
                "responses": {
                    "200": {
                        "description": "Shared memory statistics",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "total_messages": {"type": "integer"},
                                        "active_sessions": {"type": "integer"},
                                        "agent_contexts": {"type": "integer"},
                                        "message_index_size": {"type": "integer"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/shared-memory/sessions/{session_id}/messages": {
            "get": {
                "tags": ["Shared Memory"],
                "summary": "Get session messages",
                "description": "Get messages from a specific collaboration session",
                "parameters": [
                    {
                        "name": "session_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "Session ID"
                    },
                    {
                        "name": "limit",
                        "in": "query",
                        "required": False,
                        "schema": {"type": "integer", "default": 50},
                        "description": "Maximum number of messages to return"
                    },
                    {
                        "name": "types",
                        "in": "query",
                        "required": False,
                        "schema": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "description": "Message types to filter by"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Session messages",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "session_id": {"type": "string"},
                                        "messages": {
                                            "type": "array",
                                            "items": {"$ref": "#/components/schemas/SharedMemoryMessage"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/shared-memory/sessions/{session_id}/context": {
            "get": {
                "tags": ["Shared Memory"],
                "summary": "Get session context",
                "description": "Get shared context for a collaboration session",
                "parameters": [
                    {
                        "name": "session_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "Session ID"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Session context",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "additionalProperties": True
                                }
                            }
                        }
                    }
                }
            }
        },
        "/health": {
            "get": {
                "tags": ["System"],
                "summary": "Health check",
                "description": "Check system health and status",
                "responses": {
                    "200": {
                        "description": "System is healthy",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "status": {"type": "string"},
                                        "timestamp": {"type": "string"},
                                        "version": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    },
    "components": {
        "schemas": {
            "Error": {
                "type": "object",
                "properties": {
                    "error": {"type": "string"},
                    "message": {"type": "string"},
                    "code": {"type": "integer"}
                }
            },
            "Agent": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "endpoint": {"type": "string"},
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "capabilities": {
                        "type": "object",
                        "additionalProperties": True
                    },
                    "is_active": {"type": "boolean"},
                    "created_at": {"type": "string", "format": "date-time"},
                    "last_seen": {"type": "string", "format": "date-time"}
                }
            },
            "AgentRegistration": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "endpoint": {"type": "string"},
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "capabilities": {
                        "type": "object",
                        "additionalProperties": True
                    }
                },
                "required": ["name", "endpoint", "categories"]
            },
            "AIModel": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "provider": {"type": "string"},
                    "model_id": {"type": "string"},
                    "description": {"type": "string"},
                    "config": {
                        "type": "object",
                        "additionalProperties": True
                    },
                    "is_active": {"type": "boolean"},
                    "created_at": {"type": "string", "format": "date-time"},
                    "updated_at": {"type": "string", "format": "date-time"}
                }
            },
            "AIModelCreate": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "provider": {"type": "string"},
                    "model_id": {"type": "string"},
                    "description": {"type": "string"},
                    "config": {
                        "type": "object",
                        "additionalProperties": True
                    }
                },
                "required": ["name", "provider", "model_id"]
            },
            "AIModelUpdate": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "config": {
                        "type": "object",
                        "additionalProperties": True
                    }
                }
            },
            "CacheEntry": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "cache_key": {"type": "string"},
                    "query": {"type": "string"},
                    "response": {"type": "string"},
                    "model_id": {"type": "string"},
                    "created_at": {"type": "string", "format": "date-time"},
                    "expires_at": {"type": "string", "format": "date-time"},
                    "hit_count": {"type": "integer"},
                    "last_accessed": {"type": "string", "format": "date-time"}
                }
            },
            "RAGDocument": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "filename": {"type": "string"},
                    "file_type": {"type": "string"},
                    "chunk_count": {"type": "integer"},
                    "added_at": {"type": "string", "format": "date-time"}
                }
            },
            "RAGSearchResult": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "document_id": {"type": "string"},
                            "document_name": {"type": "string"},
                            "file_type": {"type": "string"},
                            "chunk_index": {"type": "integer"}
                        }
                    },
                    "similarity_score": {"type": "number"},
                    "rank": {"type": "integer"}
                }
            },
            "CollaborativeResult": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "query": {"type": "string"},
                    "enhanced_query": {"type": "string"},
                    "final_response": {"type": "string"},
                    "agents_used": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "agent_responses": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "object",
                            "properties": {
                                "agent_id": {"type": "string"},
                                "agent_name": {"type": "string"},
                                "specialization": {"type": "string"},
                                "response": {"type": "string"},
                                "model_used": {"type": "string"},
                                "response_time": {"type": "number"},
                                "cached": {"type": "boolean"}
                            }
                        }
                    },
                    "confidence_score": {"type": "number"},
                    "rag_used": {"type": "boolean"},
                    "rag_context": {"type": "string"},
                    "timestamp": {"type": "string", "format": "date-time"}
                }
            },
            "CollaborativeSession": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "query": {"type": "string"},
                    "agents": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "status": {"type": "string"},
                    "duration_minutes": {"type": "number"},
                    "created_at": {"type": "string", "format": "date-time"}
                }
            },
            "CollaborativeAgentConfig": {
                "type": "object",
                "properties": {
                    "agents": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "specialization": {"type": "string"},
                                "current_model": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "name": {"type": "string"},
                                        "provider": {"type": "string"}
                                    }
                                },
                                "confidence_threshold": {"type": "number"},
                                "is_active": {"type": "boolean"},
                                "current_sessions": {"type": "integer"},
                                "max_concurrent_sessions": {"type": "integer"}
                            }
                        }
                    },
                    "available_models": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "name": {"type": "string"},
                                "provider": {"type": "string"}
                            }
                        }
                    }
                }
            },
            "SharedMemoryMessage": {
                "type": "object",
                "properties": {
                    "message_id": {"type": "string"},
                    "session_id": {"type": "string"},
                    "message_type": {"type": "string"},
                    "content": {"type": "string"},
                    "agent_id": {"type": "string"},
                    "timestamp": {"type": "string", "format": "date-time"},
                    "metadata": {
                        "type": "object",
                        "additionalProperties": True
                    }
                }
            }
        }
    }
}