# fastapi-rag

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

### Usar um LLM real (Ollama)

Suba um [Ollama](https://ollama.com) local e, em `app/main.py`, troque
`StubLLM()` por `OllamaLLM()` (usa `OLLAMA_HOST`, padrão `http://localhost:11434`).

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
