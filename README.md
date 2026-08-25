# Assistente Cia Agro

Chatbot de IA especializado em enfezamentos do milho e na cigarrinha-do-milho
(*Dalbulus maidis*), desenvolvido como projeto de Iniciação Científica (IC).
Combina RAG (Retrieval-Augmented Generation) com uma base curada de artigos para
responder com fonte e página verificáveis.


## Funcionalidades

### Chat com IA
- Interface web interativa via **Streamlit**
- Respostas geradas por modelos de linguagem (LLM) via **Groq**, com **NVIDIA AI Endpoints** como fallback
- Suporte a histórico de conversas persistente entre sessões
- Múltiplas conversas por usuário, com títulos automáticos e exclusão individual
- Indicação visual quando a resposta utilizou documentos indexados (RAG)

### RAG Híbrido (Retrieval-Augmented Generation)
- Upload de arquivos PDF pelo painel web
- Indexação de documentos com embeddings multilíngues (`paraphrase-multilingual-MiniLM-L12-v2`)
- Busca semântica via **pgvector** no Supabase
- **Base global**: documentos disponíveis para todos os usuários
- **Base pessoal**: documentos privados de cada usuário
- Combinação automática das duas bases em cada consulta
- Roteamento de domínio e limiar mínimo de relevância para não apresentar qualquer vizinho vetorial como evidência
- Painel administrativo com perguntas cegas para diagnosticar a recuperação sem gastar tokens da LLM final

### Autenticação
- Cadastro e login via **Supabase Auth**
- Verificação de email obrigatória
- Proteção contra cadastro com email já utilizado
- Controle de acesso por sessão (Streamlit session state)
- Perfil de administrador com painel exclusivo

### Bot do Telegram
- Bot integrado ao mesmo sistema de RAG e LLM do painel web
- Histórico de conversas salvo no banco de dados
- Vinculação da conta Telegram com a conta do painel web (acesso aos documentos pessoais)
- Usuários não vinculados têm acesso à base global

**Comandos disponíveis:**
| Comando | Descrição |
|---|---|
| `/start` | Iniciar o chat |
| `/nova` | Começar uma nova conversa |
| `/status` | Ver se a conta está vinculada |
| `/vincular <codigo>` | Conectar a conta do painel |

---

## Arquitetura

```
ChatbotIC/
├── app.py                  # Ponto de entrada — roteamento de páginas
├── auth/
│   ├── authenticator.py    # Login, registro, logout, is_admin
│   └── login_page.py       # Interface de login/cadastro
├── db/
│   ├── supabase_client.py  # Singleton do cliente Supabase
│   ├── conversations.py    # CRUD de conversas e mensagens
│   └── documents.py        # CRUD de documentos indexados
├── llm/
│   ├── client.py           # Inicialização do LLM (Groq / NVIDIA)
│   └── chat.py             # Montagem de mensagens com RAG e invocação do LLM
├── rag/
│   ├── embeddings.py       # Modelo de embeddings (singleton)
│   ├── retriever.py        # Busca semântica híbrida via Supabase RPC
│   └── pdf_processor.py    # Processamento e indexação de PDFs
├── pages/
│   ├── chat.py             # Página de chat
│   ├── history.py          # Página de histórico de conversas
│   └── admin.py            # Painel administrativo
└── telegram_bot/
    ├── bot.py              # Handlers dos comandos e mensagens do Telegram
    └── runner.py           # Inicialização do bot em thread background
```

---

## Tecnologias

| Camada | Tecnologia |
|---|---|
| Frontend | Streamlit |
| Banco de dados | Supabase (PostgreSQL + pgvector) |
| Autenticação | Supabase Auth |
| LLM | Groq (`openai/gpt-oss-120b`) / NVIDIA AI Endpoints |
| Embeddings | sentence-transformers (HuggingFace) |
| RAG | LangChain + pgvector |
| Bot | python-telegram-bot |
| Deploy | Streamlit Community Cloud |

---

## Configuração

### Pré-requisitos
- Python 3.12 (versão recomendada para o deploy)
- Conta no [Supabase](https://supabase.com)
- Token de bot no [Telegram BotFather](https://t.me/botfather)
- Chave de API no [Groq](https://console.groq.com) ou [NVIDIA](https://build.nvidia.com)

### Instalação local

```bash
pip install -r requirements.txt
```

Crie o arquivo `.streamlit/secrets.toml`:

```toml
SUPABASE_URL             = "https://<project-id>.supabase.co"
SUPABASE_PUBLISHABLE_KEY = "sua-anon-key"
SUPABASE_SECRET_KEY      = "sua-service-role-key"
TELEGRAM_TOKEN           = "seu-token-do-bot"
GROQ_API_KEY             = "sua-groq-api-key"
NVIDIA_API_KEY           = "sua-nvidia-api-key"
# Opcional: evita downloads anônimos e lentos dos modelos de embeddings
HF_TOKEN                 = "seu-token-huggingface"
# O Multi-Query custa uma chamada extra e fica desligado até demonstrar ganho
ENABLE_MULTI_QUERY       = false
# O reranqueador atual é inglês; mantenha desligado para perguntas em português
ENABLE_RERANKER          = false
# Opcionais: permitem trocar modelos sem editar o código
GROQ_ANSWER_MODEL        = "openai/gpt-oss-120b"
GROQ_UTILITY_MODEL       = "openai/gpt-oss-20b"
RAG_MIN_VECTOR_SIMILARITY = 0.30
```

Inicie a aplicação:

```bash
streamlit run app.py
```

### Deploy no Streamlit Cloud

1. Suba o código para um repositório GitHub público
2. Acesse [share.streamlit.io](https://share.streamlit.io) e conecte o repositório
3. Em **Advanced settings**, selecione **Python 3.12**
4. Configure os secrets em **Settings → Secrets** (mesmo conteúdo do `secrets.toml` acima)
5. Deploy automático a cada `git push`


## Observações

- O projeto é um **protótipo de pesquisa** — o servidor entra em modo de economia de energia quando inativo no plano gratuito do Streamlit Cloud
- O bot do Telegram é iniciado automaticamente em uma thread background quando o painel web é acessado
- Usuários sem conta podem interagir com o bot do Telegram usando apenas a base global de documentos
- O arquivo `evals/questions_blind.json` contém 24 perguntas definidas antes da inspeção do corpus; veja `evals/README.md` para a rubrica de correção, recuperação, fidelidade, citação e segurança
