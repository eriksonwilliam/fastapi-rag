# fastapi-rag

A reference RAG service with replaceable retrieval and generation components.
Built with Python and FastAPI; the default hashing embedder and stub generator
run offline. Optional adapters connect FastEmbed and Ollama.

### Reliability and retrieval evaluation

- Reingesting an existing `doc_id` atomically replaces its previous chunks in the
  in-memory store. Other documents are preserved. Embeddings are prepared before
  replacement, so an embedding failure leaves the previous document available.
- Concurrent replacements cannot mix versions within one document. The last
  completed replacement wins; there is no durable storage or version history.
- `RagService.retrieve()` separates retrieval from generation for evaluation.

```bash
pip install -r requirements-dev.txt
python scripts/evaluate_retrieval.py --top-k 3
pytest
```

The committed fixture contains **4 short documents and 6 questions in PT/EN**.
With the default hashing embedder, the measured mean document recall in the top
3 chunks is **0.8333**, and mean reciprocal rank over deduplicated document IDs
is **0.7222**. One cross-language question misses its expected document. Tied
scores retain ingestion order, so a weak result can still appear in the top 3.
These are diagnostic fixture results, not evidence of production accuracy,
semantic understanding, answer faithfulness, or multilingual capability.
Use a representative labeled dataset before comparing embedding models.
`EMBEDDER=fastembed` selects the optional real adapter; no generation is run by
this evaluation script. The top-k limit counts chunks, not unique documents.

### Documentação em português

Serviço de **RAG** (Retrieval-Augmented Generation) em **Python + FastAPI**:
você envia documentos, faz uma pergunta, e o serviço **recupera** os trechos
relevantes e **gera** uma resposta ancorada neles. Template limpo da pipeline de
RAG, com arquitetura em camadas e **cobertura de testes 100%**.

```
  documento ──▶ chunking ──▶ embedding ──▶ [índice vetorial]
  pergunta  ──▶ embedding ──▶ busca (top-k) ──▶ contexto ──▶ LLM ──▶ resposta
```

## Destaques

- **Pipeline de RAG completa** — chunking, embedding, busca por similaridade
  (cosseno) e geração, cada etapa isolada e testável.
- **LLM plugável (port)** — o gerador fica atrás de uma interface. Vem com um
  `StubLLM` determinístico (roda **offline**, sem GPU/modelo) e um `OllamaLLM`
  para plugar um LLM local de verdade.
- **Sem dependências pesadas** — o embedding é um *hashing bag-of-words* puro
  em Python (determinístico e leve). Troque por um modelo semântico via adapter.
- **Arquitetura em camadas** — domínio (chunking/embedding) → aplicação (serviço
  + ports) → infraestrutura (índice em memória, LLMs) → API.
- **Cobertura 100%** — verificada na CI; o bootstrap e o adapter Ollama (rede)
  ficam fora da métrica.
- **OpenAPI/Swagger** — documentação interativa automática em `/docs`.

## Stack

- **Linguagem:** Python 3.12
- **Framework:** FastAPI (Uvicorn)
- **Testes/cobertura:** pytest + pytest-cov (gate de 100%)
- **Qualidade:** ruff (lint + format)

## Arquitetura

```
app/
├── domain/          # chunking, embedding, erros (Python puro)
├── application/     # RagService + ports (VectorStore, LLM)
├── infrastructure/  # índice em memória, StubLLM, OllamaLLM
├── api/             # FastAPI: rotas + DTOs (pydantic)
└── main.py          # composição (bootstrap)
```

## Como rodar

```bash
python -m venv .venv && . .venv/Scripts/activate   # (Linux/mac: source .venv/bin/activate)
pip install -r requirements.txt
uvicorn app.main:app --reload      # http://localhost:8000  (Swagger em /docs)
```

### Docker

```bash
docker compose up --build
```

### Modelos reais (sem tocar no resto do código)

O **embedder** e o **LLM** são *ports* — trocáveis por variável de ambiente:

```bash
pip install -r requirements-real.txt   # adiciona o fastembed (embedder semântico, ONNX)

# sobe a API + um Ollama local (perfil "real"):
docker compose --profile real up --build
# apontando os adapters reais:
#   EMBEDDER=fastembed   -> embeddings semânticos de verdade (BAAI/bge-small)
#   LLM=ollama           -> geração por um LLM local (Ollama)
```

Os padrões (`hashing` + `stub`) rodam **offline**, sem baixar modelo.

## Endpoints

| Método | Rota         | Descrição                              |
| ------ | ------------ | -------------------------------------- |
| POST   | `/documents` | Ingesta um documento (`text`, `doc_id?`) |
| POST   | `/query`     | Pergunta (`question`, `top_k?`) → resposta + fontes |
| GET    | `/docs`      | Swagger UI                             |
| GET    | `/health`    | Health check                           |

Exemplo:

```bash
curl -X POST localhost:8000/documents -H "Content-Type: application/json" \
  -d '{"doc_id":"d1","text":"Python foi criado por Guido van Rossum..."}'

curl -X POST localhost:8000/query -H "Content-Type: application/json" \
  -d '{"question":"quem criou o python?","top_k":2}'
# -> { "answer": "...", "sources": [{ "doc_id": "d1", "text": "...", "score": 0.23 }] }
```

## Testes e cobertura

```bash
pip install -r requirements-dev.txt
pytest            # exige 100% (configurado no pyproject.toml)
ruff check . && ruff format --check .
```

## Licença

MIT.
