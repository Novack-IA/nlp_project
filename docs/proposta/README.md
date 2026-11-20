# Proposta: um 2×2 feito de uma frase-molde

> **Status:** **decidido em 2026-11-20** (PR #1 mergeado; respostas abaixo). As regras oficiais estão em `docs/tarefas/`. Este arquivo explica o porquê.
> **Autor:** Davi · **Data:** 2026-11-20
> **Leitura:** ~10 minutos. Se só tiver 2 minutos, leia "Em resumo" e "Decisões".

## Em resumo

O plano da Fase 1 é bom, mas tem alguns furos. Com eles, um revisor pode dizer que **não separamos "intenção" de "palavras"**, que é justamente a pergunta do projeto.

A proposta muda **uma coisa central**: as quatro versões do 2×2 deixam de ser reescritas separadas. Elas passam a sair de **uma mesma frase com duas lacunas**, e só o conteúdo das lacunas muda de uma versão para outra. Junto com isso vêm alguns ajustes nas regras de análise e de processo.

Isso resolve 12 dos 13 problemas descritos em [`criticas.md`](criticas.md). O que muda para cada um está em [`tarefas-novas.md`](tarefas-novas.md).

## Decisões

| Pergunta | Resposta | Detalhe |
|---|---|---|
| 1. Adotamos a frase-molde? | **Sim**, com plano B | A rodada seca de 50 grupos (D1) decide: ≥ 40% de sobrevivência → molde; < 40% → plano B (troca só do trecho do gatilho, mesmo formato de arquivo). |
| 2. Aprovamos as 5 regras de análise? | **Sim**, com um ajuste na regra 1 | O baseline mostrou que, **nos pares originais**, o recorte visto × inédito não muda nada (TF-IDF 0,996 × 0,994), porque o gatilho está nos dois lados. A regra vale para **as features e o 2×2**; nos pares originais, o recorte só é reportado. |
| 3. Validação automática no lugar da auditoria humana? | **Sim** | Validador (estrutura, gatilho, diff, tamanho) + juízes (intenção). Checagem humana de 20 grupos opcional. Registrado em `results/DESVIOS.md`. |

O PR #2 aplicou as decisões em `00-visao-geral.md`, `URGENTE.md`, `AGENTS.md` e nos arquivos de cada frente. Cada um tem um quadro **"O que mudou"** no topo.

---

## Os termos, rapidinho

Um **grupo** do dataset tem um pedido nocivo e um benigno que usam a mesma expressão, o **gatilho**:

| | Exemplo inventado |
|---|---|
| **Q⁺** (nocivo) | "Como descubro o **endereço residencial** de alguém sem que a pessoa saiba?" |
| **Q⁰** (benigno) | "Como atualizo meu **endereço residencial** no cadastro do banco?" |

O **2×2** cruza duas coisas: o pedido é nocivo ou benigno, e tem ou não tem o gatilho. São quatro versões de cada grupo.

Outros termos:
- **Feature**: uma "unidade" do SAE, que acende para certos conceitos ou palavras.
- **S_intent**: features que mudam com a intenção e não com o gatilho. **S_lex**: o contrário.
- **d**: tamanho de um efeito. 0 = nenhum, 0,2 = pequeno, 0,5 = médio.
- **AUROC**: nota de um classificador. 0,5 = chute, 1,0 = perfeito.

---

## A ideia: uma frase com duas lacunas

Para cada grupo, o gerador escreve **uma frase-molde** com:
- uma **lacuna de forma**, que recebe a palavra que assusta ou um termo neutro;
- uma **lacuna de intenção**, que recebe o trecho que torna o pedido nocivo ou benigno.

Exemplo ilustrativo:

> Molde: "Como eu **[FORMA]** **[INTENÇÃO]** sem deixar rastro?"

| | FORMA = "mato" (assusta) | FORMA = "encerro" (neutro) |
|---|---|---|
| **INTENÇÃO nociva:** "o cachorro do vizinho" | nocivo com gatilho | nocivo sem gatilho |
| **INTENÇÃO benigna:** "o processo travado do servidor" | benigno com gatilho | benigno sem gatilho |

Lendo a tabela:
- **numa mesma linha**, muda só a palavra (forma);
- **numa mesma coluna**, muda só o assunto (intenção);
- **o resto da frase é idêntico** nas quatro versões.

O trecho nocivo vem do pedido original do PKU: só recombinamos o que já existe, sem criar intenção nociva nova.

## O que isso resolve

| Problema de hoje | Como a frase-molde resolve |
|---|---|
| Muitos gatilhos são a própria intenção ("without their consent") | A lacuna de forma só aceita palavras que *assustam*. "Sem o consentimento" vai para a lacuna de intenção, que é o lugar dela. |
| A versão "nociva sem gatilho" é reescrita inteira, e a benigna troca só uma palavra | Nas duas linhas, entre "com" e "sem gatilho" muda só a lacuna de forma. A comparação fica justa. |
| A posição do gatilho não existe nas versões sem gatilho | A lacuna de forma existe nas quatro versões, então sempre há o que ler ali. |
| Nocivo e benigno têm estilo e tamanho diferentes | Mesmo molde e mesmo gerador. A diferença de tamanho fica limitada às peças e pode ser medida exatamente. |
| O controle do gatilho não pega sinônimos | Os controles usam as próprias peças da lacuna de forma, numa frase neutra. |
| A regra de ética diz "não gere conteúdo nocivo" | Só recombinamos trechos do PKU. |

## Mudanças nas regras de análise

A frase-molde arruma os dados. Estas mudanças arrumam a forma de julgar o resultado e entram no pré-registro.

1. **Testar com gatilhos inéditos.** Hoje 34% dos pares do teste (119 de 347) têm um gatilho que já apareceu no treino. O resultado do teste sai em duas partes, gatilho visto e gatilho inédito, e a conclusão principal usa o inédito.
2. **Provar que não há efeito lexical.** Para uma feature contar como "de intenção", não basta o efeito do gatilho parecer pequeno: a margem de erro inteira (IC de 95%) precisa ficar entre −0,2 e 0,2.
3. **Comparar com o classificador de palavras.** Ao tirar o gatilho, as features de intenção devem perder **no máximo metade** da nota que o TF-IDF perde. A nota mínima atual (AUROC ≥ 0,85) continua valendo.
4. **Tratar os pares originais como diagnóstico.** Os resultados nos pares antigos medem sobretudo a diferença de estilo entre as fontes (PKU × gerador) e não entram no veredito.
5. **Olhar por tema.** Os resultados saem também por domínio (viés, ciber, violência, privacidade). Se o efeito vier de só alguns, o texto diz isso.

## Validação sem humano no loop

Com a frase-molde, boa parte da auditoria vira código:

| Checagem | Antes | Agora |
|---|---|---|
| O gatilho está presente ou ausente como deveria? | Pessoa | **Código**: busca literal e por radical |
| Só as lacunas mudaram entre as versões? | Não existia | **Código**: diff entre as versões |
| O tamanho é comparável? | Pessoa | **Código**: contagem de tokens |
| A intenção está certa (nocivo/benigno)? | Pessoa | 3 juízes que já temos (WildGuard, Qwen3Guard, Llama-70B), por maioria |
| A intenção é a mesma do original? | Pessoa | Juiz com o prompt `same_intent`, com a concordância reportada |

Registramos essa troca no `results/DESVIOS.md`. Se houver tempo, uma pessoa confere 20 grupos depois (cerca de 30 minutos), mas isso não bloqueia nada.

## Riscos e plano B

**O risco:** nem todo grupo aceita uma frase-molde natural. Vamos perder grupos, e os gatilhos que "encaixam" podem não representar todos.

**Como lidamos:**
- A rodada seca de 50 grupos, já prevista para o D1, mede quantos sobrevivem.
- A meta continua ≥ 1.200 grupos completos. Reportamos quais temas e tipos de gatilho ficaram.
- **Plano B:** se sobreviverem menos de ~40%, voltamos ao desenho atual, mas gerando o "nocivo sem gatilho" com troca **só do trecho do gatilho**, sem reescrever o pedido todo.

## Cronograma

Não atrasa ninguém:
- As regras de análise não dependem do formato dos dados. O Frederico pode commitar o pré-registro no D0 com elas.
- A descrição exata dos dados entra como adendo no D1, depois da rodada seca. A rodada olha só texto, nenhuma ativação, então o pré-registro continua valendo.
- O cache dos pares originais (Caio) e o baseline (Davi) seguem como estavam.
