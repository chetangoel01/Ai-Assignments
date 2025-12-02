# Erica - AI Course Tutor with GraphRAG

An AI-powered tutor for the Introduction to AI course, built with GraphRAG (Graph-based Retrieval Augmented Generation).

## Project Structure

```
erica-tutor/
├── docker-compose.yml      # Container orchestration
├── Dockerfile              # Main app container
├── requirements.txt        # Python dependencies
├── .env.example           # Environment template
├── .env                   # Your local config (git-ignored)
│
├── src/                   # Source code
│   ├── config/           # Configuration management
│   ├── ingestion/        # M2: Data ingestion pipeline
│   ├── graph/            # M3: Knowledge graph construction
│   ├── retrieval/        # M4: Query & retrieval
│   └── generation/       # M4: Answer generation
│
├── notebooks/             # Jupyter notebooks for development
│   ├── 01_ingestion.ipynb
│   ├── 02_graph_construction.ipynb
│   ├── 03_retrieval_testing.ipynb
│   └── 04_generation_testing.ipynb
│
├── data/                  # Data storage
│   ├── raw/              # Raw ingested content
│   ├── processed/        # Processed chunks
│   └── exports/          # Graph exports, visualizations
│
└── config/               # Configuration files
    └── prompts/          # LLM prompt templates
```

## Quick Start (Mac M1 Pro + OpenRouter)

### 1. Prerequisites

- Docker Desktop for Mac (Apple Silicon)
- OpenRouter API key ([get one here](https://openrouter.ai/keys))
- ~8GB RAM allocated to Docker

### 2. Setup

```bash
# Clone and enter directory
cd erica-tutor

# Copy environment file
cp .env.example .env

# Edit .env - ADD YOUR OPENROUTER API KEY
nano .env
# Set: OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Start all services
docker-compose up -d

# Check everything is running
docker-compose ps
```

### 3. Access Points

| Service       | URL                          | Purpose                    |
|---------------|------------------------------|----------------------------|
| Jupyter       | http://localhost:8888        | Development notebooks      |
| Neo4j Browser | http://localhost:7474        | Knowledge graph visualization |
| Mongo Express | http://localhost:8081        | Document storage browser   |
| ChromaDB      | http://localhost:8000        | Vector DB API             |

### 4. Verify Installation

Open Jupyter at http://localhost:8888 and run `notebooks/00_verify_environment.ipynb`

## Cost Management (Important!)

You're using OpenRouter which charges per token. To avoid surprise bills:

1. **Set a spending limit** in [OpenRouter dashboard](https://openrouter.ai/settings/limits)
2. Start with a smaller model: `qwen/qwen-2.5-7b-instruct` (~$0.0003/1K tokens)
3. Scale up to `qwen/qwen-2.5-72b-instruct` only when needed

Estimated costs for this project (rough):
- Entity extraction for ~100 documents: $1-5
- Query/generation testing: $0.50-2
- Total project: likely under $10 if careful

## Milestones

- [ ] **M1**: Environment and Tooling ← You are here
- [ ] **M2**: Ingestion Pipeline
- [ ] **M3**: Knowledge Graph Construction
- [ ] **M4**: Query and Generation

## Docker Desktop Settings (Mac)

In Docker Desktop > Settings > Resources:
- Memory: 8GB minimum (10GB+ recommended)
- CPU: 4+ cores
- Disk: 20GB+

## Development Workflow

1. Start services: `docker-compose up -d`
2. Open Jupyter: http://localhost:8888
3. Work in notebooks for experimentation
4. Move stable code to `src/` modules
5. Commit often with meaningful messages
