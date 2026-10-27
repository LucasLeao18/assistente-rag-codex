# 📚 Assistente RAG — Codex + RAG

Assistente de perguntas e respostas sobre PDFs de uma disciplina. Ele **responde citando arquivo e página** e **admite quando não encontra** a informação.

Projeto da disciplina **TOPAES I 2026.2 (UPE)**. A I Unidade definiu a arquitetura; a **II Unidade** é a implementação com **Scrum**, com sprints semanais até a live demo de **08/12/2026**.

> 📋 **Board de gestão:** aba **Projects** deste repositório (GitHub Projects)

## Arquitetura

| Camada | Ferramenta | Arquivo |
|---|---|---|
| Interface | Streamlit (upload + chat) | `app.py` |
| Ingestão | pypdf → splitter (~650/120 tokens) → embeddings | `ingest.py` |
| Armazenamento | Chroma persistente | `chroma_db/` |
| Consulta | Retriever top-4 + limiar de relevância + LLM | `rag.py` |
| Modelos | OpenAI API (`text-embedding-3-small` + chat) | `config.py` |
| Engenharia | Codex + Git + pytest + GitHub Actions | `test_*.py` |

```
PDFs → Parser (texto + página) → Chunking → Embeddings → Chroma
Pergunta → Retriever (top-4) → Contexto (trechos + fontes) → LLM → Resposta + citações
```

## Como rodar

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # e preencha OPENAI_API_KEY
streamlit run app.py
pytest                       # testes
```

## Time Scrum

| Integrante | Papel | Frente técnica |
|---|---|---|
| Lucas Leão | Product Owner | Produto e avaliação (20 perguntas) |
| Vinícios Menezes | Scrum Master | Engenharia, Git, CI e revisão de PRs |
| Marcelo Barbosa | Dev | Ingestão: parser, chunking, Chroma |
| Ruan Galvão | Dev | Consulta: retriever, prompt, LLM |
| João Flavio | Dev | Interface Streamlit e QA |

## Product Backlog (MVP — 45 pts)

| ID | Como estudante, quero… | Prioridade | Pts | Sprint |
|---|---|---|---|---|
| US01 | enviar um ou mais PDFs pela interface | Must | 3 | S1 |
| US02 | extrair o texto mantendo arquivo e página | Must | 5 | S1 |
| US03 | ter os documentos indexados no Chroma | Must | 8 | S2 |
| US04 | recuperar os 4 trechos mais relevantes | Must | 5 | S3 |
| US05 | receber resposta com fonte e página | Must | 8 | S3 |
| US06 | ser avisado quando não houver evidência | Must | 3 | S4 |
| US07 | conversar com histórico e ver os trechos | Should | 5 | S4 |
| US08 | medir a qualidade com 20 perguntas | Should | 5 | S5 |
| US09 | ver tempo e custo de cada resposta | Could | 3 | S5 |

**Fora do MVP:** login/multiusuário, fine-tuning, deploy com escala, OCR, DOCX/web.

## Sprints (II Unidade)

| Sprint | Review (terça) | Meta | Histórias |
|---|---|---|---|
| S0 | 27/10 | Fundação: repositório, board, ambiente, app vazio | — |
| S1 | 03/11 | Upload e extração com arquivo + página | US01, US02 |
| S2 | 10/11 | Índice vetorial (chunking, embeddings, Chroma) | US03 |
| S3 | 17/11 | Primeira resposta RAG ponta a ponta | US04, US05 |
| S4 | 24/11 | Chat com histórico + “não encontrei” · **Seminário Scrum 27/11** | US06, US07 |
| S5 | 01/12 | Avaliação com 20 perguntas, métricas e ajustes | US08, US09 |
| 🎯 | **08/12** | **Live demo da II Unidade** (code freeze 05/12) | — |

**Cerimônias:** Planning (quarta, 30 min) · Daily assíncrona (seg/qua/sex) · Refinamento (segunda) · Review (terça, em aula) · Retro (terça, 15 min).

## Definition of Done

- [ ] Critérios de aceite da issue atendidos
- [ ] Testes passando no CI
- [ ] PR revisado por outro integrante (código do Codex nunca entra sem revisão humana)
- [ ] Nenhum segredo no código (`.env` fora do Git)
- [ ] Fonte e página preservadas nas respostas
- [ ] Demonstrável rodando na terça

## Fluxo de trabalho

1. Pegue uma issue da coluna **Todo** do board e mova para **In Progress**.
2. Crie a branch `usXX-descricao` e peça ao Codex um **plano antes de alterar**.
3. Abra um PR com `Closes #N` → card em revisão; após merge, a issue fecha e vai para **Done**.
