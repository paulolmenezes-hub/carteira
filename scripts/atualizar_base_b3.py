"""Acrescenta à base de cotações da B3 (dados/) os meses fechados que faltam.

Roda sozinho no GitHub do dia 2 ao dia 10 de cada mês
(.github/workflows/atualizar_base_b3.yml) e
pode ser rodado à mão: ``python scripts/atualizar_base_b3.py``. Sai com
código 0 quando não há nada a fazer, quando acrescentou algum mês ou quando o
mês ainda não pôde ser baixado antes do dia 10 (a B3 publica o arquivo nos
primeiros dias úteis; a tarefa tenta de novo no dia seguinte, sem alarme).
Sai com código 1 — e o GitHub avisa por e-mail — só se no dia 10 o mês
ainda não tiver sido baixado."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from carteira_analise.fontes.b3_cotahist import (acrescentar_meses, caminho_base_existente,  # noqa: E402
                                                 meses_faltando, salvar_base)


def main() -> int:
    caminho = caminho_base_existente()
    if not os.path.exists(caminho):
        print(f"Base não encontrada em {caminho}. Gere a base inicial pela Seção 15 do notebook.")
        return 1
    base = pd.read_csv(caminho, parse_dates=["data"])
    faltam = meses_faltando(base)
    print(f"Base: {base['ticker'].nunique()} papéis, de {base['mes'].min()} a {base['mes'].max()}.")
    if not faltam:
        print("Nenhum mês fechado faltando.")
        return 0
    print("Meses a acrescentar: " + ", ".join(m.strftime('%m/%Y') for m in faltam))
    nova, ok, falhas = acrescentar_meses(base, faltam)
    if ok:
        salvar_base(nova, caminho)
        print(f"Base atualizada: {nova['ticker'].nunique()} papéis, de {nova['mes'].min()} a {nova['mes'].max()}.")
    if falhas:
        print("Não baixados: " + "; ".join(falhas))
    if ok:
        return 0
    if pd.Timestamp.today().day < 10:
        print("Ainda não publicado ou B3 indisponível — nova tentativa amanhã.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
