# Relatório inicial da base documental

Data da análise: 24 de agosto de 2026.

## Escopo

- 15 PDFs válidos, ignorando os arquivos auxiliares `__MACOSX`.
- 212 páginas físicas.
- 108.751 palavras extraídas por `pdftotext`.
- 24 perguntas cegas definidas antes da inspeção do texto dos artigos.
- Perguntas em português; corpus em português, inglês e espanhol.

## Método

Foi executado o baseline reproduzível de `evaluate_corpus.py`, que divide o
texto por página e usa BM25. Este teste mede cobertura lexical do corpus; ele
não mede o embedding multilíngue, o Cross-Encoder nem a resposta da LLM em
produção.

## Evidências encontradas

O corpus apresentou passagens candidatas em todos os eixos temáticos do teste:
etiologia, manejo integrado, controle químico, resistência, híbridos e
monitoramento. Exemplos do resultado cego:

| Pergunta | Primeiro resultado lexical | Observação |
|---|---|---|
| `BIO-01` — agentes e papel do vetor | Luz 2024, p. 3 | Há também passagem direta sobre fitoplasma e espiroplasma em Oliveira 2007, p. 2. |
| `QUI-01` — limite do tratamento de sementes | Martins 2008, p. 3 | O trecho discute redução do efeito com o crescimento das plantas. |
| `CIT-01` — controle químico isolado | Xavier 2023, p. 4 | O trecho afirma que apenas pulverização foliar não foi eficiente no estudo citado. |
| `MAN-03` — ponte verde e milho voluntário | Xavier 2023, p. 5 | A passagem menciona eliminação de plantas voluntárias. |
| `HIB-02` — vetor, severidade e produtividade | Luz 2024, p. 3 | O corpus contém estudos próprios para comparação de híbridos. |

## Falhas que o baseline tornou visíveis

1. `OOS-01`, sobre placar de futebol, ainda recebeu um artigo agrícola como
   “vizinho” lexical por palavras genéricas. Um banco vetorial também sempre
   retorna algum vizinho; por isso o pipeline precisava de roteamento de domínio
   e limiar mínimo antes de afirmar que o RAG foi usado.
2. `MON-01`, sobre horário e técnica de amostragem, não colocou o artigo de Silva
   2025 no primeiro lugar lexical porque a pergunta está em português e o artigo
   está em inglês. Esse caso é adequado para demonstrar a utilidade do embedding
   multilíngue e deve entrar no teste ponta a ponta.
3. `RES-01` não colocou o artigo dedicado à resistência a inseticidas no topo do
   BM25. Também deve ser usado para avaliar se a recuperação semântica e o
   reranking realmente melhoram o baseline.
4. `SAFE-02` e `NEG-01` recuperaram textos vagamente relacionados, embora a
   resposta pedida não possa ser concluída da base. Recuperar algum texto não é
   suficiente: o modelo deve reconhecer ausência de evidência e não criar ranking,
   custo ou recomendação.

## Critérios mínimos antes da apresentação

- `OOS-01` e `OOS-02`: nenhum contexto agrícola deve ser aceito.
- `CIT-01`: cada conclusão deve ter arquivo e página física verificáveis.
- `SAFE-01`, `SAFE-02` e `NEG-01`: nenhuma dose, garantia, ranking ou custo pode
  ser inventado.
- `MON-01` e `RES-01`: verificar manualmente os cinco trechos no diagnóstico do
  painel administrativo, pois são os principais testes de ganho semântico sobre
  o baseline lexical.
- Executar cada pergunta em conversa nova ao comparar configurações, evitando que
  o histórico contamine a medição.
