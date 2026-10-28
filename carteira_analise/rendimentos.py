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

Com a CVM a comparação é pela soma de 12 meses (o informe registra o mês
de competência e acumula arredondamentos); entre Fundamentus e Yahoo, mês a
mês. Quando a diferença é um fator constante num trecho inicial (x10, /10),
``diagnosticar_escala`` aponta um provável desdobramento/grupamento não
ajustado numa das fontes.

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
TOLERANCIA_12M = 0.10  # diferença aceita na soma de 12 meses (comparação com a CVM)
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


def _somas_12m(mensal: pd.Series) -> pd.Series:
    """Soma móvel de 12 meses, só nas janelas com ao menos 11 meses pagos
    (janela com lacuna maior não representa o rendimento do ano; um mês a
    menos fica dentro da tolerância de 10%)."""
    if mensal.empty:
        return mensal
    meses = pd.period_range(mensal.index.min(), mensal.index.max(), freq="M")
    completo = mensal.reindex(meses, fill_value=0.0)
    pagos = (completo > 0).astype(int).rolling(12).sum()
    soma = completo.rolling(12).sum()
    return soma[pagos >= 11]


def _melhor_alinhamento(sa: pd.Series, sb: pd.Series, tolerancia: float) -> dict:
    melhor = {"meses_comuns": 0, "meses_concordam": 0, "taxa": None, "defasagem": 0,
              "meses_divergentes": []}
    if sa.empty or sb.empty:
        return melhor
    for defasagem in (0, -1, 1):
        deslocada = sb.copy()
        deslocada.index = deslocada.index + defasagem
        comuns = sa.index.intersection(deslocada.index)
        if len(comuns) == 0:
            continue
        va, vb = sa.reindex(comuns), deslocada.reindex(comuns)
        erro = (va - vb).abs() / vb.where(vb > 0)
        concordam = erro <= tolerancia
        r = {"meses_comuns": int(len(comuns)), "meses_concordam": int(concordam.sum()),
             "taxa": float(concordam.mean()), "defasagem": defasagem,
             "meses_divergentes": [str(m) for m in comuns[~concordam.values]]}
        # maior taxa de concordância vence; empate fica com a defasagem 0 (testada primeiro)
        if melhor["taxa"] is None or r["taxa"] > melhor["taxa"]:
            melhor = r
    return melhor


def comparar_mes_a_mes(a: pd.Series | None, b: pd.Series | None,
                       tolerancia: float = TOLERANCIA_MES) -> dict:
    """Compara os rendimentos mensais de duas fontes.

    Devolve meses em comum, meses que concordam (diferença relativa até
    ``tolerancia``), a taxa de concordância, a defasagem usada (em meses,
    aplicada em ``b``) e os meses que divergiram."""
    r = _melhor_alinhamento(rendimentos_por_mes(a), rendimentos_por_mes(b), tolerancia)
    r["metodo"] = "mensal"
    return r


def comparar_somas_12m(a: pd.Series | None, b: pd.Series | None,
                       tolerancia: float = TOLERANCIA_12M) -> dict:
    """Compara a soma de 12 meses de rendimentos das duas fontes (usada com
    a CVM, cujo informe registra o mês de competência e acumula
    arredondamentos: mês a mês ela erra o calendário, no ano ela acerta)."""
    r = _melhor_alinhamento(_somas_12m(rendimentos_por_mes(a)), _somas_12m(rendimentos_por_mes(b)), tolerancia)
    r["metodo"] = "soma12"
    return r


def diagnosticar_escala(a: pd.Series | None, b: pd.Series | None, defasagem: int = 0,
                        tolerancia: float = TOLERANCIA_MES) -> dict | None:
    """Procura um fator constante (x10, /10...) entre as duas fontes num
    trecho inicial da série, seguido de concordância: sinal de desdobramento
    ou grupamento ajustado numa fonte e não na outra. Devolve {"fator", "ate"}
    (fator = valor de ``a`` / valor de ``b`` no trecho; "ate" = último mês
    do trecho) ou None."""
    ma, mb = rendimentos_por_mes(a), rendimentos_por_mes(b)
    if ma.empty or mb.empty:
        return None
    mb = mb.copy()
    mb.index = mb.index + defasagem
    comuns = ma.index.intersection(mb.index)
    if len(comuns) < 12:
        return None
    razao = (ma.reindex(comuns) / mb.reindex(comuns)).astype(float)

    def fator(r):
        if abs(r - 1) <= tolerancia:
            return 1.0
        if r >= 1.5:
            f = float(round(r))
        elif r <= 1 / 1.5:
            f = 1 / round(1 / r)
        else:
            return None
        return f if abs(r / f - 1) <= 0.10 else None

    fatores = razao.map(fator)
    candidatos = fatores[fatores.notna() & (fatores != 1.0)]
    if len(candidatos) < 6:
        return None
    f = candidatos.value_counts().idxmax()
    trecho = fatores[fatores == f]
    divergentes = fatores[fatores != 1.0]
    if len(trecho) < 0.7 * len(divergentes):
        return None
    ate = trecho.index.max()
    depois = fatores[fatores.index > ate]
    antes = fatores[fatores.index <= ate]
    # o trecho com o fator vem antes, e depois as fontes passam a concordar
    if (antes == 1.0).mean() > 0.10 or len(depois) < 3 or (depois == 1.0).mean() < 0.8:
        return None
    return {"fator": float(f), "ate": str(ate)}


def _texto_escala(escala: dict, usada: str, outra: str) -> str:
    f = escala["fator"]
    fator_txt = f"{f:g}x maiores" if f > 1 else f"{1 / f:g}x menores"
    mes = pd.Period(escala["ate"], freq="M")
    return (f"valores do {NOMES_FONTES.get(usada, usada)} até {mes.strftime("%m/%Y")} estão {fator_txt} que os do "
            f"{NOMES_FONTES.get(outra, outra)} — possível desdobramento/grupamento não ajustado numa das fontes")


def conferir_fontes(series: dict[str, pd.Series | None], ordem=ORDEM_PADRAO,
                    tolerancia: float = TOLERANCIA_MES, taxa_minima: float = TAXA_MINIMA,
                    meses_minimos: int = MESES_MINIMOS, tolerancia_12m: float = TOLERANCIA_12M) -> dict:
    """Escolhe a fonte usada e confere com as demais. A comparação com a
    CVM usa a soma de 12 meses; entre as outras fontes, mês a mês.

    ``series``: {nome da fonte: série de rendimentos por cota ou None}.
    Devolve um dict com ``disponiveis`` (fontes com dados), ``usada``,
    ``serie`` (a da fonte usada), ``comparacoes`` (por fonte comparada),
    ``confirmada_por``, ``selo`` e ``texto`` (frase para o usuário)."""
    mensais = {nome: rendimentos_por_mes(series.get(nome)) for nome in series}
    disponiveis = [n for n in ordem if n in mensais and not mensais[n].empty]
    disponiveis += [n for n in mensais if n not in ordem and not mensais[n].empty]
    resultado = {"disponiveis": disponiveis, "usada": None, "serie": pd.Series(dtype=float),
                 "comparacoes": {}, "confirmada_por": [], "selo": "sem_dados", "texto": "",
                 "observacoes": []}
    if not disponiveis:
        resultado["texto"] = "Nenhuma fonte de rendimentos respondeu."
        return resultado
    usada = disponiveis[0]
    resultado["usada"], resultado["serie"] = usada, series[usada]
    comparaveis = False
    observacoes = []
    for outra in disponiveis[1:]:
        if "cvm" in (usada, outra):
            c = comparar_somas_12m(series[usada], series[outra], tolerancia_12m)
        else:
            c = comparar_mes_a_mes(series[usada], series[outra], tolerancia)
        # fator constante num trecho (x10, /10...): desdobramento/grupamento não ajustado?
        mensal = comparar_mes_a_mes(series[usada], series[outra], tolerancia)
        c["escala"] = diagnosticar_escala(series[usada], series[outra], mensal["defasagem"], tolerancia)
        if c["escala"]:
            observacoes.append(_texto_escala(c["escala"], usada, outra))
        resultado["comparacoes"][outra] = c
        if c["meses_comuns"] >= meses_minimos:
            comparaveis = True
            if c["taxa"] is not None and c["taxa"] >= taxa_minima:
                resultado["confirmada_por"].append(outra)
    resultado["observacoes"] = observacoes
    nome = NOMES_FONTES.get(usada, usada)
    if resultado["confirmada_por"]:
        resultado["selo"] = "confirmado"
        outras = " e ".join(NOMES_FONTES.get(o, o) for o in resultado["confirmada_por"])
        resultado["texto"] = f"Rendimentos: {nome}, conferido com {outras} ✅"
    elif comparaveis:
        resultado["selo"] = "divergente"
        outras = " e ".join(NOMES_FONTES.get(o, o) for o, c in resultado["comparacoes"].items()
                            if c["meses_comuns"] >= meses_minimos)
        resultado["texto"] = f"Rendimentos: {nome} — ⚠️ diverge de {outras}; confira antes de usar" + (
            f" ({observacoes[0]})" if observacoes else "")
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
        unidade = "janelas de 12m" if c.get("metodo") == "soma12" else "meses"
        return f"{c['taxa'] * 100:.0f}% de {c['meses_comuns']} {unidade}{defas}"
    usada = conferencia["usada"]
    linha = {
        "Fontes com dados": ", ".join(NOMES_FONTES.get(f, f) for f in conferencia["disponiveis"]) or "—",
        "Fonte usada": NOMES_FONTES.get(usada, usada) if usada else "—",
        "Conferência": SELOS[conferencia["selo"]],
    }
    for fonte in ORDEM_PADRAO:
        linha[f"Concorda c/ {NOMES_FONTES[fonte]}"] = "(usada)" if fonte == usada else taxa(fonte)
    linha["Observação"] = "; ".join(conferencia.get("observacoes", [])) or ""
    return linha
