"""Rendimentos de FIIs em mais de uma fonte, com conferência cruzada.

Os rendimentos por cota de um FII podem vir de três fontes gratuitas:

- **Fundamentus**: proventos pagos, pela data-com (página pública do FII);
- **CVM**: informes mensais oficiais enviados pelo próprio fundo
  (dividend yield do mês x valor patrimonial da cota);
- **Yahoo Finance**: dividendos pela data-ex (tem lacunas de anos em
  muitos FIIs).

Em vez de confiar numa fonte só, ``conferir_fontes`` registra quais fontes
responderam, escolhe a usada pela ordem de preferência e compara mês a mês
os rendimentos dela com os das outras. O resultado é um selo por FII:

- ``confirmado``: outra fonte concorda com a usada na maioria dos meses;
- ``divergente``: há outra fonte com meses em comum, mas nenhuma concorda;
- ``fonte_unica``: só uma fonte tem dados suficientes (sem conferência);
- ``sem_dados``: nenhuma fonte respondeu.

As fontes datam o mesmo rendimento de formas diferentes (data-com, data-ex,
mês de referência do informe), o que pode deslocar a série inteira em um
mês. Por isso a comparação testa defasagens de -1, 0 e +1 mês e fica com a
que mais concorda. Meses em que só uma das fontes tem valor são lacunas,
não discordância.
"""
from __future__ import annotations

import pandas as pd

ORDEM_PADRAO = ("fundamentus", "cvm", "yahoo")
NOMES_FONTES = {"fundamentus": "Fundamentus", "cvm": "CVM", "yahoo": "Yahoo"}
TOLERANCIA_MES = 0.05  # diferença relativa aceita num mês (arredondamentos)
TAXA_MINIMA = 0.80  # fração dos meses em comum que precisa concordar
MESES_MINIMOS = 12  # meses em comum para a conferência valer

SELOS = {
    "confirmado": "✅ confirmado",
    "divergente": "⚠️ divergente",
    "fonte_unica": "◻️ fonte única",
    "sem_dados": "— sem dados",
}


def rendimentos_por_mes(serie: pd.Series | None) -> pd.Series:
    """Soma dos rendimentos por cota em cada mês (índice mensal Period)."""
    if serie is None or len(serie) == 0:
        return pd.Series(dtype=float)
    s = pd.Series(serie).dropna()
    s = s[s > 0]
    if s.empty:
        return pd.Series(dtype=float)
    indice = pd.DatetimeIndex(s.index)
    if indice.tz is not None:
        indice = indice.tz_localize(None)
    s.index = indice
    return s.groupby(s.index.to_period("M")).sum()


def comparar_mes_a_mes(a: pd.Series | None, b: pd.Series | None,
                       tolerancia: float = TOLERANCIA_MES) -> dict:
    """Compara os rendimentos mensais de duas fontes.

    Devolve meses em comum, meses que concordam (diferença relativa até
    ``tolerancia``), a taxa de concordância, a defasagem usada (em meses,
    aplicada em ``b``) e os meses que divergiram."""
    ma, mb = rendimentos_por_mes(a), rendimentos_por_mes(b)
    melhor = {"meses_comuns": 0, "meses_concordam": 0, "taxa": None, "defasagem": 0,
              "meses_divergentes": []}
    if ma.empty or mb.empty:
        return melhor
    for defasagem in (0, -1, 1):
        deslocada = mb.copy()
        deslocada.index = deslocada.index + defasagem
        comuns = ma.index.intersection(deslocada.index)
        if len(comuns) == 0:
            continue
        va, vb = ma.reindex(comuns), deslocada.reindex(comuns)
        erro = (va - vb).abs() / vb.where(vb > 0)
        concordam = erro <= tolerancia
        r = {"meses_comuns": int(len(comuns)), "meses_concordam": int(concordam.sum()),
             "taxa": float(concordam.mean()), "defasagem": defasagem,
             "meses_divergentes": [str(m) for m in comuns[~concordam.values]]}
        # maior taxa de concordância vence; empate fica com a defasagem 0 (testada primeiro)
        if melhor["taxa"] is None or r["taxa"] > melhor["taxa"]:
            melhor = r
    return melhor


def conferir_fontes(series: dict[str, pd.Series | None], ordem=ORDEM_PADRAO,
                    tolerancia: float = TOLERANCIA_MES, taxa_minima: float = TAXA_MINIMA,
                    meses_minimos: int = MESES_MINIMOS) -> dict:
    """Escolhe a fonte usada e confere com as demais.

    ``series``: {nome da fonte: série de rendimentos por cota ou None}.
    Devolve um dict com ``disponiveis`` (fontes com dados), ``usada``,
    ``serie`` (a da fonte usada), ``comparacoes`` (por fonte comparada),
    ``confirmada_por``, ``selo`` e ``texto`` (frase para o usuário)."""
    mensais = {nome: rendimentos_por_mes(series.get(nome)) for nome in series}
    disponiveis = [n for n in ordem if n in mensais and not mensais[n].empty]
    disponiveis += [n for n in mensais if n not in ordem and not mensais[n].empty]
    resultado = {"disponiveis": disponiveis, "usada": None, "serie": pd.Series(dtype=float),
                 "comparacoes": {}, "confirmada_por": [], "selo": "sem_dados", "texto": ""}
    if not disponiveis:
        resultado["texto"] = "Nenhuma fonte de rendimentos respondeu."
        return resultado
    usada = disponiveis[0]
    resultado["usada"], resultado["serie"] = usada, series[usada]
    comparaveis = False
    for outra in disponiveis[1:]:
        c = comparar_mes_a_mes(series[usada], series[outra], tolerancia)
        resultado["comparacoes"][outra] = c
        if c["meses_comuns"] >= meses_minimos:
            comparaveis = True
            if c["taxa"] is not None and c["taxa"] >= taxa_minima:
                resultado["confirmada_por"].append(outra)
    nome = NOMES_FONTES.get(usada, usada)
    if resultado["confirmada_por"]:
        resultado["selo"] = "confirmado"
        outras = " e ".join(NOMES_FONTES.get(o, o) for o in resultado["confirmada_por"])
        resultado["texto"] = f"Rendimentos: {nome}, conferido com {outras} ✅"
    elif comparaveis:
        resultado["selo"] = "divergente"
        outras = " e ".join(NOMES_FONTES.get(o, o) for o, c in resultado["comparacoes"].items()
                            if c["meses_comuns"] >= meses_minimos)
        resultado["texto"] = f"Rendimentos: {nome} — ⚠️ diverge de {outras}; confira antes de usar"
    else:
        resultado["selo"] = "fonte_unica"
        resultado["texto"] = f"Rendimentos: {nome} — ◻️ só esta fonte tinha dados suficientes, sem conferência"
    return resultado


def resumo_conferencia(conferencia: dict) -> dict:
    """Linha de tabela com o resultado de ``conferir_fontes`` (para o
    quadro de cobertura do notebook)."""
    def taxa(fonte):
        c = conferencia["comparacoes"].get(fonte)
        if not c or not c["meses_comuns"]:
            return "—"
        defas = f", defasagem {c['defasagem']:+d} mês" if c["defasagem"] else ""
        return f"{c['taxa'] * 100:.0f}% de {c['meses_comuns']} meses{defas}"
    usada = conferencia["usada"]
    linha = {
        "Fontes com dados": ", ".join(NOMES_FONTES.get(f, f) for f in conferencia["disponiveis"]) or "—",
        "Fonte usada": NOMES_FONTES.get(usada, usada) if usada else "—",
        "Conferência": SELOS[conferencia["selo"]],
    }
    for fonte in ORDEM_PADRAO:
        linha[f"Concorda c/ {NOMES_FONTES[fonte]}"] = "(usada)" if fonte == usada else taxa(fonte)
    return linha
