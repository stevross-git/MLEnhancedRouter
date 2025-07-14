# ML Query Router

## Overview

This is a Flask-based ML-enhanced query routing system that intelligently routes user queries to appropriate AI agents based on query classification and analysis. The system uses machine learning models (DistilBERT) for query categorization and maintains a registry of specialized agents to handle different types of queries.

## User Preferences

Preferred communication style: Simple, everyday language.

## Recent Changes (July 14, 2025)

✓ Added comprehensive AI model integration supporting 10+ providers
✓ Implemented AI model management interface with selection and configuration
✓ Created authentication system with API key management and JWT tokens
✓ Added support for OpenAI, Anthropic, Google, xAI, Perplexity, local Ollama, and custom endpoints
✓ Built AI model testing and validation functionality
✓ Integrated API key status monitoring across all providers
✓ Added model activation, configuration, and cost tracking features
✓ Created dedicated Settings page for enterprise API key management
✓ Built comprehensive Configuration page for advanced model and routing settings
✓ Added export/import functionality for configuration backup and deployment
✓ Implemented AI response caching system with database backend (SQLite/PostgreSQL)
✓ Added intelligent cache management with TTL, hit counting, and automatic cleanup
✓ Integrated cache functionality into AI model manager for improved performance
✓ Created cache management interface in Settings page with statistics and controls
✓ Replaced file-based cache with database models (AICacheEntry, AICacheStats)
✓ Added comprehensive cache endpoints for statistics, entries, and management
✓ Implemented database-backed cache with hit tracking and expiration management
✓ Integrated RAG (Retrieval-Augmented Generation) system with ChromaDB
✓ Added document upload support (PDF, DOCX, TXT, MD, HTML, JSON, CSV)
✓ Implemented document chunking and vector embedding for semantic search
✓ Created RAG management interface in chat sidebar with upload and statistics
✓ Added RAG panel modal for comprehensive document management
✓ Integrated RAG context enhancement for AI responses with document search
✓ Built RAG API endpoints for document upload, management, and search
✓ Fixed database model metadata conflicts with SQLAlchemy reserved attributes
✓ Created comprehensive Swagger/OpenAPI 3.0 documentation for all API endpoints
✓ Implemented interactive API documentation interface with Swagger UI
✓ Added API documentation navigation link and dark theme compatibility
✓ Fixed missing error templates (404.html, 500.html) for proper error handling
✓ Implemented shared memory system for real-time AI collaboration
✓ Created collaborative router with specialized AI agents (Analyst, Creative, Technical, Researcher, Synthesizer)
✓ Added collaborative AI interface with real-time session monitoring and shared context
✓ Integrated shared memory with agent scratchpads and working memory for context persistence
✓ Built collaborative API endpoints for multi-agent query processing and session management

## System Architecture

### Frontend Architecture
- **Framework**: Flask with Jinja2 templates
- **UI Components**: Bootstrap 5 with dark theme, Font Awesome icons
- **JavaScript**: Vanilla JavaScript for dynamic interactions
- **Pages**: Home (query submission), Dashboard (metrics), Agents (management)

### Backend Architecture
- **Framework**: Flask with SQLAlchemy ORM
- **Database**: SQLite (default) with PostgreSQL support via DATABASE_URL
- **ML Integration**: DistilBERT for query classification, SentenceTransformers for semantic analysis
- **Caching**: Redis support for distributed caching
- **Rate Limiting**: Flask-Limiter for API protection

### Data Storage
- **Primary Database**: SQLite/PostgreSQL via SQLAlchemy
- **Models**: QueryLog, AgentRegistration, RouterMetrics
- **Caching Layer**: Redis (optional) for performance optimization

## Key Components

### 1. Query Classification System
- **ML Model**: DistilBERT-based classification
- **Categories**: Analysis, Creative, Technical, Mathematical, Coding, Research, Philosophical, Practical, Educational, Conversational
- **Confidence Thresholds**: Configurable routing confidence levels

### 2. Agent Management
- **Registration**: Dynamic agent registration with capabilities
- **Load Balancing**: Concurrent query limits and load penalties
- **Health Monitoring**: Agent availability and performance tracking

### 3. Routing Engine
- **Smart Routing**: ML-enhanced routing with fallback mechanisms
- **Consensus Algorithm**: Multi-agent consensus for complex queries
- **Retry Logic**: Exponential backoff for failed requests

### 4. Monitoring and Metrics
- **Query Logging**: Complete audit trail of all queries
- **Performance Metrics**: Response times, success rates, agent utilization
- **Real-time Dashboard**: Live statistics and system health

## Data Flow

1. **Query Submission**: User submits query through web interface
2. **Classification**: ML model analyzes query and determines category
3. **Agent Selection**: Router selects appropriate agent(s) based on classification
4. **Load Balancing**: System considers agent load and availability
5. **Query Routing**: Request forwarded to selected agent
6. **Response Processing**: Agent response validated and returned
7. **Logging**: Complete interaction logged for analytics

## External Dependencies

### Required Libraries
- Flask ecosystem (Flask, SQLAlchemy, Limiter)
- ML libraries (transformers, sentence-transformers, torch)
- Database drivers (sqlite3, psycopg2 for PostgreSQL)
- Redis client (optional, for caching)

### Optional Integrations
- **Redis**: Distributed caching and session storage
- **Prometheus**: Metrics collection and monitoring
- **JWT**: Authentication and authorization

### Model Dependencies
- DistilBERT models for text classification
- SentenceTransformers for semantic similarity
- Local model storage in ./models/ directory

## Deployment Strategy

### Development Environment
- SQLite database for local development
- Debug mode enabled
- Hot reloading for development

### Production Considerations
- PostgreSQL database recommended
- Redis for distributed caching
- Rate limiting and security headers
- Environment-based configuration
- Containerization ready (Docker/Kubernetes)

### Configuration Management
- Environment variables for sensitive data
- Configurable thresholds and limits
- Feature flags for optional components
- Separate configs for dev/staging/prod

### Scalability Features
- Async operation support
- Connection pooling
- Distributed caching
- Load balancing across agents
- Horizontal scaling capability

### Security Measures
- Rate limiting on API endpoints
- Input validation and sanitization
- Secure session management
- CSRF protection
- Proxy-aware deployment (ProxyFix)