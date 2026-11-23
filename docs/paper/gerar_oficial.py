"""Gera a versão oficial do paper, sem apêndices, a partir da versão completa.

    python docs/paper/gerar_oficial.py          # lê docs/paper/paper.md e escreve docs/paper/paper_oficial.md

Só o paper.md é editado à mão. Este script corta tudo a partir do "## Apêndice A" e reescreve as frases do corpo
que citam um apêndice ou uma figura do apêndice, para que o texto se sustente sozinho. Se uma dessas frases mudar
no paper.md, o script para com um erro em vez de gerar uma versão quebrada; nesse caso, atualize a lista abaixo.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
ENTRADA, SAIDA = AQUI / "paper.md", AQUI / "paper_oficial.md"
CORTE = "\n## Apêndice A"

# (trecho na versão completa, trecho na versão oficial); cada trecho precisa aparecer uma única vez no corpo
TROCAS: list[tuple[str, str]] = [
    # seção 4, abertura
    ("As regras de análise e os limiares do que contaria como resposta foram fixados antes de ver os dados. "
     "O Apêndice A os lista, junto com os desvios, e o Apêndice B traz os detalhes de implementação.",
     "As regras de análise e os limiares do que contaria como resposta foram fixados antes de ver os dados e estão "
     "registrados no repositório do projeto, junto com os desvios e os detalhes de implementação."),
    # 4.1
    ("servem só de checagem (Apêndice C).", "servem só de checagem."),
    # 4.2
    ("ficam de fora porque o efeito da palavra delas passa de 0,2 (Apêndice C).",
     "ficam de fora porque o efeito da palavra delas passa de 0,2."),
    # 4.4
    (" As intervenções exploratórias estão no Apêndice C.", ""),
    # 5.1
    ("Em nenhuma das oito combinações há feature da palavra ou feature mista (Figura 5).",
     "Em nenhuma das oito combinações há feature da palavra ou feature mista."),
    ("O que não existe é uma feature lexical comum a muitos gatilhos (Apêndice C).",
     "O que não existe é uma feature lexical comum a muitos gatilhos."),
    ("e ficam fora da seleção (Apêndice C).", "e ficam fora da seleção."),
    # 5.2
    ("No SAE de 262k os valores são praticamente os mesmos (Apêndice C).",
     "No SAE de 262k os valores são praticamente os mesmos."),
    ("Isso é coerente com não haver features da palavra (Apêndice C).",
     "Isso é coerente com não haver features da palavra."),
    ("foi acrescentado depois de ver o resultado (Apêndice A).",
     "foi acrescentado depois de ver o resultado e está registrado como desvio do pré-registro."),
    (" As intervenções exploratórias estão na Figura 7.", ""),
    # 5.3
    ("sobrevivem a um controle de estilo (Apêndice C).", "sobrevivem a um controle de estilo."),
    # 5.4
    ("Pela regra fixada antes da análise (Apêndice A), o veredito é **parcial**.",
     "Pela regra fixada antes da análise, o veredito é **parcial**. Para um sim completo, a regra pedia que o sinal "
     "existisse, fosse além das palavras, passasse nos dois testes de que não é a palavra e se mantivesse em outro "
     "dicionário e nos recortes dos dados."),
    # 6
    ("O controle de estilo (Apêndice C) é um ajuste estatístico.",
     "O controle de estilo, uma regressão do efeito de intenção nos 14 atributos de estilo, é um ajuste estatístico."),
    ("dois controles que dependiam delas ficam indefinidos (Apêndice A).",
     "dois controles que dependiam delas ficam indefinidos."),
    # 7
    ("A mesma ablação reduz a recusa a pedidos benignos que o modelo recusa indevidamente, e o empurrão médio do "
     "gatilho na direção de \"nocivo\" é pequeno e não muda o comportamento quando removido (Apêndice C).",
     "A mesma ablação reduz a recusa a pedidos benignos que o modelo recusa indevidamente, e nenhuma feature "
     "responde à palavra de forma consistente."),
]

CABECALHO = ("<!-- Versão oficial, sem apêndices. Gerada por docs/paper/gerar_oficial.py a partir de docs/paper/paper.md; "
             "não edite este arquivo. A versão completa, com os apêndices, é o paper.md. -->\n\n")


def gerar(texto: str) -> str:
    if CORTE not in texto:
        raise SystemExit(f"não achei o início dos apêndices ({CORTE.strip()!r}) no paper.md")
    corpo = texto.split(CORTE)[0].rstrip() + "\n"
    for velho, novo in TROCAS:
        n = corpo.count(velho)
        if n != 1:
            raise SystemExit(f"o trecho aparece {n} vezes no corpo (esperado 1); atualize TROCAS:\n  {velho[:120]}")
        corpo = corpo.replace(velho, novo)
    sem_comentarios = re.sub(r"<!--.*?-->", "", corpo, flags=re.S)
    sobra = re.findall(r"[^.\n]*(?:Apêndice|Figura [5-9]\b|Tabela [A-C]\d)[^.\n]*", sem_comentarios)
    if sobra:
        raise SystemExit("ainda há menções a apêndices na versão oficial:\n  " + "\n  ".join(s.strip() for s in sobra))
    return CABECALHO + corpo


def main() -> int:
    SAIDA.write_text(gerar(ENTRADA.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"escrito {SAIDA.relative_to(AQUI.parents[1])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
