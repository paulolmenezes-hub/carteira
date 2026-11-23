"""Metas de alocação: compara a carteira de hoje com os percentuais que o
usuário escolheu por classe. Não sugere ativos nem compra ou venda: mostra
quanto falta ou sobra em cada classe e, se o usuário informar um aporte,
como dividi-lo entre as classes abaixo da meta."""
from __future__ import annotations

CLASSES = ["Ações B3", "FIIs", "ETFs B3", "EUA (ações e ETFs)"]
METAS_PADRAO = {"Ações B3": 45, "FIIs": 30, "ETFs B3": 10, "EUA (ações e ETFs)": 15}
TOLERANCIA_PP = 2.0  # diferença de até 2 p.p. conta como "na meta"


def classe_do_ativo(mercado: str, tipo: str) -> str:
    if mercado == "EUA":
        return "EUA (ações e ETFs)"
    return {"Ações": "Ações B3", "FIIs": "FIIs", "ETFs": "ETFs B3"}.get(tipo, "Ações B3")


def comparar(valores: dict[str, float], metas: dict[str, float]) -> list[dict]:
    """Uma linha por classe: meta (%), hoje (%), valor da meta, valor atual,
    diferença em reais (positivo = falta para chegar à meta) e % atingido."""
    total = sum(valores.get(c, 0.0) for c in CLASSES)
    linhas = []
    for c in CLASSES:
        tem = float(valores.get(c, 0.0))
        meta = float(metas.get(c, 0.0))
        alvo = total * meta / 100
        hoje = tem / total * 100 if total else 0.0
        linhas.append({"classe": c, "meta": meta, "hoje": hoje, "alvo": alvo, "tem": tem, "falta": alvo - tem,
                       "atingido": (tem / alvo * 100) if alvo else None,
                       "na_meta": abs(hoje - meta) <= TOLERANCIA_PP})
    return linhas


def dividir_aporte(linhas: list[dict], aporte: float) -> dict[str, float]:
    """Divide um aporte entre as classes que estão abaixo da meta, na
    proporção do que falta a cada uma, sem passar da meta (o que sobrar vai
    na proporção das metas). Não vende nada."""
    if aporte <= 0:
        return {l["classe"]: 0.0 for l in linhas}
    total_novo = sum(l["tem"] for l in linhas) + aporte
    falta = {l["classe"]: max(total_novo * l["meta"] / 100 - l["tem"], 0.0) for l in linhas}
    soma_falta = sum(falta.values())
    if soma_falta <= 0:
        soma_metas = sum(l["meta"] for l in linhas) or 1
        return {l["classe"]: aporte * l["meta"] / soma_metas for l in linhas}
    if soma_falta >= aporte:
        return {c: aporte * f / soma_falta for c, f in falta.items()}
    sobra = aporte - soma_falta
    soma_metas = sum(l["meta"] for l in linhas) or 1
    return {l["classe"]: falta[l["classe"]] + sobra * l["meta"] / soma_metas for l in linhas}


def metas_de_texto(texto: str | None) -> dict[str, float] | None:
    """Lê as metas guardadas no endereço da página ("45-30-10-15"). None se
    o texto não tiver 4 números somando 100."""
    try:
        numeros = [float(x) for x in str(texto).split("-")]
    except (TypeError, ValueError):
        return None
    if len(numeros) != len(CLASSES) or abs(sum(numeros) - 100) > 0.01 or any(n < 0 for n in numeros):
        return None
    return dict(zip(CLASSES, numeros))


def metas_para_texto(metas: dict[str, float]) -> str:
    return "-".join(f"{metas.get(c, 0):g}" for c in CLASSES)
