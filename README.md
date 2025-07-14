# ML-Enhanced Collaborative AI Router

A Flask-based Collaborative AI Routing Platform that enables intelligent, real-time multi-agent problem-solving and knowledge sharing with enterprise-grade AI integration.

## 🚀 Features

### Collaborative AI System
- **Multi-Agent Collaboration**: 5 specialized AI agents (Analyst, Creative, Technical, Researcher, Synthesizer) working together
- **Real-time Shared Memory**: Agents share context, thoughts, and working memory across sessions
- **Intelligent Agent Selection**: Automatic agent selection based on query type or manual selection
- **User-Configurable Models**: Select different AI models for each specialized agent
- **Session Management**: Track and monitor active collaboration sessions

### AI Model Integration
- **16+ AI Models**: Support for OpenAI, Anthropic, Google, xAI, Perplexity, Cohere, Mistral, and local models
- **Model Switching**: Change AI models for agents on-the-fly without restarting
- **Caching System**: Intelligent response caching with database backend for improved performance
- **Authentication**: Enterprise-grade API key management and JWT authentication

### RAG (Retrieval-Augmented Generation)
- **Document Upload**: Support for PDF, DOCX, TXT, MD, HTML, JSON, CSV files
- **Vector Search**: ChromaDB integration for semantic document search
- **Context Enhancement**: Automatically enhance AI responses with relevant document context
- **Document Management**: Upload, search, and manage document collections

### Advanced Features
- **Interactive API Documentation**: Comprehensive Swagger/OpenAPI 3.0 documentation
- **Real-time Chat**: Advanced chat interface with streaming responses
- **Query Classification**: ML-enhanced query routing with DistilBERT
- **Performance Monitoring**: Real-time metrics, statistics, and health monitoring
- **Configuration Management**: Export/import configurations for deployment

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Flask Web Application                        │
├─────────────────────────────────────────────────────────────────┤
│  Collaborative AI Router                                        │
│  ├── Agent Manager (5 Specialized Agents)                      │
│  ├── Shared Memory System                                      │
│  ├── Session Management                                        │
│  └── Model Selection Engine                                    │
├─────────────────────────────────────────────────────────────────┤
│  AI Model Manager                                              │
│  ├── Multi-Provider Support (OpenAI, Anthropic, Google, etc.)  │
│  ├── Model Configuration                                       │
│  ├── Response Caching                                          │
│  └── Authentication System                                     │
├─────────────────────────────────────────────────────────────────┤
│  RAG System                                                    │
│  ├── Document Processing                                       │
│  ├── Vector Storage (ChromaDB)                                 │
│  ├── Semantic Search                                           │
│  └── Context Enhancement                                       │
├─────────────────────────────────────────────────────────────────┤
│  ML Router                                                     │
│  ├── Query Classification (DistilBERT)                         │
│  ├── Agent Registration                                        │
│  ├── Load Balancing                                            │
│  └── Performance Monitoring                                    │
└─────────────────────────────────────────────────────────────────┘
```

## 🔧 Installation

### Prerequisites
- Python 3.8+
- PostgreSQL (optional, SQLite by default)
- Redis (optional, for distributed caching)

### Setup
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set environment variables:
   ```bash
   export DATABASE_URL="your_database_url"
   export SESSION_SECRET="your_session_secret"
   # Add API keys for AI providers
   export OPENAI_API_KEY="your_openai_key"
   export ANTHROPIC_API_KEY="your_anthropic_key"
   # ... other API keys
   ```
4. Run the application:
   ```bash
   python main.py
   ```

## 🔑 API Keys

The system supports the following AI providers:

### Required for Full Functionality
- **OpenAI**: `OPENAI_API_KEY`
- **Anthropic**: `ANTHROPIC_API_KEY`
- **Google**: `GEMINI_API_KEY`

### Optional Providers
- **xAI**: `XAI_API_KEY`
- **Perplexity**: `PERPLEXITY_API_KEY`
- **Cohere**: `COHERE_API_KEY`
- **Mistral**: `MISTRAL_API_KEY`

## 📚 Usage

### Collaborative AI
1. Navigate to `/collaborate`
2. Enter your query in the collaborative query form
3. Choose between automatic or manual agent selection
4. Configure AI models for each agent (optional)
5. Submit and watch agents collaborate in real-time

### Model Configuration
1. Click "Configure Agents" in the collaborative interface
2. Select different AI models for each specialized agent
3. Save configurations for optimal performance

### RAG Document Management
1. Go to the Chat interface (`/chat`)
2. Upload documents using the RAG panel
3. Enable RAG in collaborative queries for context-aware responses

### API Documentation
- Interactive API docs: `/api/docs`
- OpenAPI specification: `/api/openapi.json`

## 🔗 API Endpoints

### Collaborative AI
- `POST /api/collaborate` - Submit collaborative query
- `GET /api/collaborate/sessions` - Get active sessions
- `GET /api/collaborate/agents` - Get agent configurations
- `PUT /api/collaborate/agents/{id}/model` - Update agent model

### AI Models
- `GET /api/ai-models` - List available models
- `POST /api/ai-models` - Create custom model
- `PUT /api/ai-models/{id}/activate` - Activate model

### RAG System
- `POST /api/rag/upload` - Upload document
- `GET /api/rag/documents` - List documents
- `POST /api/rag/search` - Search documents

### Authentication
- `POST /api/auth/token` - Generate JWT token
- `GET /api/auth/users` - List users
- `POST /api/auth/regenerate` - Regenerate API key

## 🔍 Monitoring

### Dashboard
- Real-time metrics: `/dashboard`
- Agent management: `/agents`
- Model management: `/models`
- System configuration: `/config`

### Health Checks
- Application health: `/health`
- Database status: Built-in monitoring
- Cache statistics: `/api/cache/stats`

## 🛡️ Security

- JWT-based authentication
- API key management
- Rate limiting
- Input validation
- Secure session management
- Environment-based configuration

## 📊 Performance

- **Caching**: Intelligent response caching with configurable TTL
- **Load Balancing**: Distributed query processing across agents
- **Connection Pooling**: Optimized database connections
- **Async Operations**: Non-blocking collaborative processing

## 🔧 Configuration

### Environment Variables
```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost/db

# Session
SESSION_SECRET=your-secret-key

# AI Providers
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
GEMINI_API_KEY=your-google-key

# Optional
REDIS_URL=redis://localhost:6379
DEBUG=true
```

### Model Configuration
Configure AI models through the web interface or API:
- Model selection for each agent
- Provider-specific settings
- Performance thresholds
- Caching preferences

## 🚀 Deployment

### Production Considerations
- Use PostgreSQL for production database
- Configure Redis for distributed caching
- Set up proper environment variables
- Enable SSL/TLS
- Configure rate limiting
- Set up monitoring and logging

### Docker Deployment
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "main:app"]
```

## 📈 Metrics

The system tracks:
- Query processing times
- Agent utilization
- Model performance
- Cache hit rates
- Session statistics
- Error rates

## 🔄 Updates

### Recent Enhancements (July 2025)
- Enhanced collaborative AI with user-selectable models
- Added agent configuration interface
- Implemented manual agent selection
- Created comprehensive agent management
- Built model switching API endpoints
- Added real-time configuration updates

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For support and questions:
- Documentation: `/api/docs`
- Issues: GitHub Issues
- Configuration: `/config` interface

---

Built with ❤️ using Flask, SQLAlchemy, and cutting-edge AI technologies.