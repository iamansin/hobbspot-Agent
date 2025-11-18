# Chat Agent API

A production-ready FastAPI backend service that provides personalized AI chat capabilities with multi-provider LLM support, intelligent context management, and web search integration.

## Features

- **Multi-Provider LLM Support**: Seamlessly switch between OpenAI and Google Gemini
- **Intelligent Caching**: DiskCache-based caching with 10-minute TTL for optimal performance
- **Context Management**: Automatic conversation summarization when history exceeds limits
- **Web Search Integration**: Function calling with Brave Search API for real-time information
- **Persistent Storage**: Appwrite database integration for user data and chat history
- **Production Ready**: Comprehensive logging, error handling, rate limiting, and CORS support
- **Async-First Architecture**: Non-blocking I/O operations for high concurrency
- **Docker Support**: Containerized deployment with health checks

## Architecture

```
Client → FastAPI → [Cache Manager ↔ DiskCache]
                 ↓
                 [Database Service ↔ Appwrite]
                 ↓
                 [AI Agent ↔ OpenAI/Gemini]
                 ↓
                 [Search Service ↔ Brave API]
```

## Requirements

- Python 3.11+
- Docker (optional, for containerized deployment)
- Appwrite account and database setup
- OpenAI API key and/or Google Gemini API key
- Brave Search API key

## Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.template .env
   # Edit .env with your actual credentials
   ```

5. **Run the application**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

6. **Access the API**
   - API: http://localhost:8000
   - Interactive docs: http://localhost:8000/docs
   - Health check: http://localhost:8000/health

### Docker Deployment

1. **Build the Docker image**
   ```bash
   docker build -t chat-agent .
   ```

2. **Run with Docker**
   ```bash
   docker run -p 8000:8000 --env-file .env chat-agent
   ```

3. **Or use Docker Compose**
   ```bash
   docker-compose up
   ```

## Environment Variables

### Context Management
| Variable | Description | Default |
|----------|-------------|---------|
| `PREVIOUS_MESSAGE_CONTEXT_LENGTH` | Number of recent messages to include in context | `10` |
| `OVERLAP_COUNT` | Buffer before triggering summarization | `5` |
| `CACHE_TTL_SECONDS` | Cache time-to-live in seconds | `600` (10 min) |

### Appwrite Configuration
| Variable | Description | Required |
|----------|-------------|----------|
| `APPWRITE_ENDPOINT` | Appwrite API endpoint | Yes |
| `APPWRITE_PROJECT_ID` | Your Appwrite project ID | Yes |
| `APPWRITE_API_KEY` | Appwrite API key with database permissions | Yes |
| `APPWRITE_DATABASE_ID` | Database ID for user data | Yes |
| `APPWRITE_COLLECTION_ID` | Collection ID for chat history | Yes |

### LLM Provider Configuration
| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required if using OpenAI |
| `GEMINI_API_KEY` | Google Gemini API key | Required if using Gemini |
| `DEFAULT_LLM_PROVIDER` | Default provider (`openai` or `gemini`) | `openai` |

### Brave Search Configuration
| Variable | Description | Required |
|----------|-------------|----------|
| `BRAVE_API_KEY` | Brave Search API key | Yes |

### DiskCache Configuration
| Variable | Description | Default |
|----------|-------------|---------|
| `CACHE_DIRECTORY` | Directory for cache storage | `./cache` |

### Logging Configuration
| Variable | Description | Default |
|----------|-------------|---------|
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | `INFO` |
| `LOG_ROTATION` | Log file rotation size | `100 MB` |
| `LOG_RETENTION` | Log file retention period | `30 days` |

## API Documentation

### POST /chat

Send a chat message and receive an AI-generated response.

**Request Body:**
```json
{
  "userId": "user123",
  "userMessage": "Tell me about quantum computing",
  "chatInterest": false,
  "interestTopic": null
}
```

**First-Time User Request:**
```json
{
  "userId": "newuser456",
  "userMessage": "",
  "chatInterest": true,
  "interestTopic": "I'm interested in learning about artificial intelligence"
}
```

**Response:**
```json
{
  "response": "Quantum computing is a revolutionary approach to computation..."
}
```

**Status Codes:**
- `200`: Success
- `422`: Validation error (invalid request payload)
- `429`: Rate limit exceeded (10 requests/minute)
- `500`: Internal server error

### GET /health

Check the health status of the service.

**Response:**
```json
{
  "status": "healthy",
  "service": "chat-agent",
  "version": "1.0.0"
}
```

## Database Schema

### Appwrite Collection Structure

```json
{
  "$id": "string (user ID)",
  "chatHistory": [
    {
      "role": "user|assistant",
      "content": "string"
    }
  ],
  "chatInterest": "string",
  "userSummary": "string",
  "birthdate": "string (ISO date)",
  "topics": ["string"]
}
```

**Required Attributes:**
- `chatHistory` (array)
- `chatInterest` (string, optional)
- `userSummary` (string)
- `birthdate` (string, optional)
- `topics` (array)

## Testing

### Run All Tests
```bash
pytest
```

### Run with Coverage
```bash
pytest --cov=app --cov-report=html
```

### Run Specific Test Categories
```bash
# Unit tests only
pytest tests/test_*.py -k "not integration"

# Integration tests
pytest tests/test_main.py
```

### Run Async Tests
```bash
pytest -v --asyncio-mode=auto
```

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── ai_agent.py       # LLM provider abstraction
│   ├── cache.py          # DiskCache manager
│   ├── config.py         # Configuration management
│   ├── db_service.py     # Appwrite database service
│   ├── models.py         # Pydantic models
│   ├── search.py         # Brave Search integration
│   └── utils.py          # Logging and utilities
├── tests/
│   ├── test_ai_agent.py
│   ├── test_cache.py
│   ├── test_config.py
│   ├── test_db_service.py
│   ├── test_main.py
│   ├── test_models.py
│   ├── test_search.py
│   └── test_utils.py
├── cache/                # DiskCache storage
├── logs/                 # Application logs
├── main.py               # FastAPI application
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose setup
├── .env.template         # Environment variable template
└── README.md             # This file
```

## How It Works

### First-Time User Flow
1. User sends request with `chatInterest: true` and `interestTopic`
2. System creates new user context with the interest topic
3. AI generates personalized response based on the topic
4. Chat history is initialized and stored in cache and database

### Returning User Flow
1. User sends request with `chatInterest: false`
2. System checks cache for user context (10-minute TTL)
3. On cache miss, fetches from Appwrite database
4. AI generates response using recent message history and user summary
5. Chat history is updated in both cache and database

### Automatic Summarization
1. When chat history exceeds `PREVIOUS_MESSAGE_CONTEXT_LENGTH + OVERLAP_COUNT`
2. Oldest messages are sent to LLM for summarization
3. Summary is appended to `userSummary` field
4. Chat history is trimmed to keep only recent messages
5. Updated context is saved to cache and database

### Function Calling (Web Search)
1. LLM determines when web search is needed
2. System calls Brave Search API with the query
3. Search results are formatted and provided to LLM
4. LLM generates response incorporating search results

## Performance Considerations

- **Caching**: 10-minute TTL reduces database queries by ~90%
- **Async Operations**: All I/O is non-blocking for high concurrency
- **Context Management**: Summarization prevents token limit issues
- **Rate Limiting**: 10 requests/minute per IP prevents abuse

## Security Best Practices

- Store all API keys in environment variables (never commit to git)
- Configure CORS for specific origins in production
- Use HTTPS in production deployments
- Implement authentication/authorization for production use
- Regularly rotate API keys
- Monitor rate limits and adjust as needed

## Deployment Platforms

This application can be deployed to:
- **Railway**: One-click deployment with automatic HTTPS
- **Render**: Free tier available, automatic deployments
- **Fly.io**: Global edge deployment
- **AWS ECS/Fargate**: Enterprise-grade scalability
- **Google Cloud Run**: Serverless container deployment
- **Azure Container Instances**: Simple container hosting

## Troubleshooting

### Common Issues

**Cache directory not writable**
```bash
mkdir -p cache logs
chmod 755 cache logs
```

**Appwrite connection errors**
- Verify `APPWRITE_ENDPOINT` is correct
- Check API key has database read/write permissions
- Ensure database and collection IDs are correct

**LLM API errors**
- Verify API keys are valid and have sufficient credits
- Check rate limits on your LLM provider account
- Review logs for detailed error messages

**Rate limit errors**
- Adjust rate limit in `main.py` if needed
- Implement user-based rate limiting for production

### Logs

Application logs are stored in `logs/` directory with automatic rotation:
- Console output: Real-time logs with color coding
- File output: `logs/chat_agent_YYYY-MM-DD.log`
- Rotation: Every 100 MB
- Retention: 30 days

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run tests and ensure they pass
5. Submit a pull request

## License

[Your License Here]

## Support

For issues, questions, or contributions, please open an issue on GitHub.
