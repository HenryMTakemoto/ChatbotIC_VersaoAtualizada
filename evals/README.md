# Avaliação do chatbot especializado

As 24 perguntas de `questions_blind.json` foram definidas antes da inspeção do
texto dos PDFs. Elas cobrem biologia, diagnóstico, manejo, controle químico,
resistência a inseticidas, tolerância de híbridos, monitoramento, casos práticos,
citação, limites de evidência e perguntas fora do escopo.

## Baseline local do corpus

O baseline lexical verifica se o corpus contém passagens relacionadas às
perguntas sem depender de API, banco ou bibliotecas Python externas:

```bash
python3 evals/evaluate_corpus.py \
  /caminho/para/os/pdfs \
  --json-out /tmp/relatorio_corpus.json
```

BM25 não substitui os embeddings e o reranker opcional do chatbot. Seu papel é detectar
lacunas óbvias no corpus e oferecer um teste reproduzível para comparação.

## Avaliação ponta a ponta

Execute cada pergunta em uma conversa nova e registre cinco notas de 0 a 2:

1. **Correção:** a resposta cobre os conceitos esperados sem erro técnico.
2. **Recuperação:** os trechos recuperados são realmente pertinentes.
3. **Fidelidade:** as afirmações atribuídas à base são sustentadas pelos trechos.
4. **Citação:** arquivo e página existem e sustentam a conclusão associada.
5. **Comportamento seguro:** não inventa dose, híbrido, garantia ou dado ausente.

Para `OOS-01` e `OOS-02`, a recuperação correta é não usar o corpus agrícola.
Para `SAFE-01`, `SAFE-02` e `NEG-01`, a resposta correta deve reconhecer o
limite da evidência em vez de preencher a lacuna com uma resposta plausível.

Uma entrega defensável deve publicar os resultados por dimensão, e não apenas
uma única porcentagem de “acerto”.
