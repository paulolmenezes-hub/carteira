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


# Limite por ativo: peso máximo (e mínimo opcional) de cada ativo na
# carteira. Só aponta quem passou; não sugere compra nem venda.
LIMITES_PADRAO = {"maximo": 10.0, "minimo": 0.0}


def limites_de_texto(texto: str | None) -> dict[str, float] | None:
    """Lê "10-0" (máximo-mínimo, em %) do endereço da página."""
    try:
        maximo, minimo = (float(x) for x in str(texto).split("-"))
    except (TypeError, ValueError):
        return None
    if not (0 < maximo <= 100) or not (0 <= minimo < maximo):
        return None
    return {"maximo": maximo, "minimo": minimo}


def limites_para_texto(limites: dict[str, float]) -> str:
    return f"{limites['maximo']:g}-{limites['minimo']:g}"


def situacao_peso(valor: float, total: float, limites: dict[str, float]) -> dict:
    """Peso do ativo (%) e situação: "acima" (com o valor além do máximo),
    "abaixo" (com o que falta para o mínimo) ou "dentro"."""
    peso = valor / total * 100 if total else 0.0
    maximo, minimo = limites["maximo"], limites.get("minimo") or 0.0
    if peso > maximo:
        return {"peso": peso, "situacao": "acima", "diferenca": valor - total * maximo / 100}
    if minimo and peso < minimo:
        return {"peso": peso, "situacao": "abaixo", "diferenca": total * minimo / 100 - valor}
    return {"peso": peso, "situacao": "dentro", "diferenca": 0.0}


def pesos_dos_ativos(valores: dict[str, float], limites: dict[str, float]) -> list[dict]:
    """Uma linha por ativo (maior peso primeiro): ticker, valor e situação."""
    total = sum(v for v in valores.values() if v > 0)
    linhas = [{"ticker": t, "valor": v, **situacao_peso(v, total, limites)} for t, v in valores.items() if v > 0]
    return sorted(linhas, key=lambda l: -l["valor"])


def metas_e_limites(linhas: list[dict], valores_por_classe: dict[str, list[float]],
                    limites: dict[str, float]) -> tuple[dict[str, dict], list[str]]:
    """Liga a meta de cada classe ao limite por ativo.

    ``linhas``: saída de ``comparar``; ``valores_por_classe``: classe -> valores
    (R$) dos ativos dela. Por classe: n ativos, quantos acima do limite,
    espaço até o limite (quanto ainda cabe nos ativos atuais sem nenhum passar
    do máximo) e a leitura: "na_meta" (dentro da tolerância), "acima_da_meta", "sem_ativos", "cabe" ou
    "nao_cabe" (com quanto falta de espaço). Também devolve os avisos de
    metas e limites que não fecham (texto)."""
    total = sum(v for vs in valores_por_classe.values() for v in vs)
    maximo, minimo = limites["maximo"], limites.get("minimo") or 0.0
    teto_r = total * maximo / 100
    saida, avisos = {}, []
    for l in linhas:
        c = l["classe"]
        vs = [v for v in valores_por_classe.get(c, []) if v > 0]
        n = len(vs)
        espaco = sum(max(teto_r - v, 0.0) for v in vs)
        falta = l["falta"]
        if l.get("na_meta"):
            leitura = "na_meta"
        elif falta < 1:
            leitura = "acima_da_meta"
        elif n == 0:
            leitura = "sem_ativos"
        elif espaco >= falta:
            leitura = "cabe"
        else:
            leitura = "nao_cabe"
        saida[c] = {"n": n, "acima": sum(1 for v in vs if total and v / total * 100 > maximo), "espaco": espaco,
                    "leitura": leitura, "falta_espaco": max(falta - espaco, 0.0)}
        if n and n * maximo < l["meta"]:
            avisos.append(f"{c}: com {n} ativo{'s' if n > 1 else ''} e limite de {maximo:g}%, a classe chega no "
                          f"máximo a {n * maximo:g}%, abaixo da meta de {l['meta']:g}%.")
        if n and minimo and n * minimo > l["meta"]:
            avisos.append(f"{c}: {n} ativos com mínimo de {minimo:g}% somam {n * minimo:g}%, acima da meta de "
                          f"{l['meta']:g}%.")
    return saida, avisos
