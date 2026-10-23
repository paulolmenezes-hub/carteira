"""Contexto de cada ativo para os cards do painel.

Três peças, todas sem I/O (recebem séries e dicionários já baixados):

- ``regua_faixas``: onde o preço de hoje está em relação às faixas da regra
  (aumentar / fora das faixas / realizar), com os preços em que cada faixa
  começa e o próximo degrau;
- ``faixa_12m``: mínima e máxima dos últimos 12 meses e a posição do preço;
- ``indicadores_contexto``: os poucos indicadores que o card mostra, por tipo
  de ativo, com a mesma leitura dos limiares do motor de pontuação;
- ``preco_atrativo_hoje`` (FIIs): o critério de "preço atrativo" do teste da
  Seção 13 do notebook (yield 15% acima da média de 24 meses do próprio FII,
  com rendimentos estáveis), aplicado ao mês de hoje.

O contexto descreve o ativo; não muda os cenários nem a regra de faixas.
"""
from __future__ import annotations

import math

import pandas as pd

from .oportunidades import DESCONTO_MINIMO_YIELD, avaliar_fii_mes_a_mes
from .regra_posicao import CONFIG_REGRA_PADRAO

# Escala da régua: trecho de queda | fora das faixas | realização, nas
# proporções 1:2:1. Os extremos da escala só posicionam o marcador.
_PISO_ESCALA = -0.40
_TETO_ESCALA = 1.00
_PROPORCOES = (0.25, 0.50, 0.25)


def _cfg(config):
    return CONFIG_REGRA_PADRAO if config is None else {**CONFIG_REGRA_PADRAO, **config}


def _sem_fuso(serie):
    if serie is None:
        return pd.Series(dtype=float)
    if getattr(serie.index, "tz", None) is not None:
        serie = serie.copy()
        serie.index = serie.index.tz_localize(None)
    return serie.sort_index()


def _interpolar(v, a, b):
    return 0.0 if b == a else min(max((v - a) / (b - a), 0.0), 1.0)


def regua_faixas(preco_medio: float, preco: float, config: dict | None = None) -> dict | None:
    """Situação do preço nas faixas da regra, para desenhar a régua.

    Devolve ``situacao`` ('queda' | 'fora' | 'realizacao'), ``variacao``,
    os preços de início das faixas (``preco_queda`` e ``preco_realizacao``),
    ``posicao`` do marcador (0 a 1, na escala 1:2:1), o ``degrau`` atingido
    (ex.: 0,60 = faixa de +60%) e o ``proximo`` degrau na mesma direção
    (``{"faixa", "preco", "venda_total"}``), quando houver."""
    if not preco_medio or preco_medio <= 0 or preco is None or preco <= 0:
        return None
    cfg = _cfg(config)
    vendas = sorted(f[0] for f in cfg["faixas_venda"])
    compras = sorted((f[0] for f in cfg["faixas_compra"]), reverse=True)  # -0,15, -0,25
    inicio_real, inicio_queda = vendas[0], compras[0]
    var = preco / preco_medio - 1
    q, f, r = _PROPORCOES
    if var <= inicio_queda + 1e-9:
        situacao = "queda"
        posicao = q * _interpolar(var, _PISO_ESCALA, inicio_queda)
        atingidos = [c for c in compras if var <= c + 1e-9]
        degrau = atingidos[-1]
        mais_fundos = [c for c in compras if c < degrau - 1e-9]
        proximo = ({"faixa": mais_fundos[0], "preco": preco_medio * (1 + mais_fundos[0]), "venda_total": False}
                   if mais_fundos else None)
    elif var >= inicio_real - 1e-9:
        situacao = "realizacao"
        posicao = q + f + r * _interpolar(var, inicio_real, _TETO_ESCALA)
        degrau = max(v for v in vendas if var >= v - 1e-9)
        acima = [v for v in vendas if v > degrau + 1e-9]
        fracao_de = {fx: fr for fx, fr in cfg["faixas_venda"]}
        proximo = ({"faixa": acima[0], "preco": preco_medio * (1 + acima[0]),
                    "venda_total": fracao_de.get(acima[0], 0) >= 1.0} if acima else None)
    else:
        situacao = "fora"
        posicao = q + f * _interpolar(var, inicio_queda, inicio_real)
        degrau, proximo = None, None
    return {
        "situacao": situacao, "variacao": var, "preco": preco, "preco_medio": preco_medio,
        "preco_queda": preco_medio * (1 + inicio_queda), "preco_realizacao": preco_medio * (1 + inicio_real),
        "posicao": min(max(posicao, 0.02), 0.98), "degrau": degrau, "proximo": proximo,
    }


def faixa_12m(precos: pd.Series) -> dict | None:
    """Mínima e máxima dos últimos 12 meses, o preço de hoje e a posição dele
    entre as duas (0 = na mínima, 1 = na máxima)."""
    p = _sem_fuso(precos).dropna()
    if p.empty:
        return None
    janela = p[p.index > p.index[-1] - pd.Timedelta(days=365)]
    minima, maxima, preco = float(janela.min()), float(janela.max()), float(p.iloc[-1])
    if minima <= 0 or maxima <= 0:
        return None
    return {
        "minima": minima, "maxima": maxima, "preco": preco,
        "abaixo_da_maxima": max(0.0, 1 - preco / maxima),
        "acima_da_minima": max(0.0, preco / minima - 1),
        "posicao": _interpolar(preco, minima, maxima) if maxima > minima else 0.5,
    }


def preco_atrativo_hoje(precos: pd.Series, dividendos: pd.Series) -> dict | None:
    """Critério de preço atrativo de FIIs (caminho 2 do teste da Seção 13)
    no mês de hoje. ``None`` quando o histórico não permite calcular (menos
    de 3 anos de rendimentos completos ou lacunas recentes)."""
    df = avaliar_fii_mes_a_mes(precos, dividendos, None)
    if df.empty:
        return None
    ultima = df.iloc[-1]
    y, media = ultima.get("yield"), ultima.get("yield_medio_24m")
    if y is None or media is None or pd.isna(y) or pd.isna(media) or media <= 0:
        return None
    limiar = (1 + DESCONTO_MINIMO_YIELD) * media
    estavel = bool(ultima.get("estavel")) if pd.notna(ultima.get("estavel")) else None
    return {
        "yield": float(y), "yield_medio_24m": float(media), "limiar": float(limiar),
        "estavel": estavel, "atende": bool(y >= limiar and estavel),
    }


# ---------------------------------------------------------------------------
# Indicadores do card (mesmos limiares do motor de pontuação)
# ---------------------------------------------------------------------------

BOM, NEUTRO, ATENCAO = "bom", "neutro", "atencao"


def _ind(nome, valor, leitura, nivel):
    return {"nome": nome, "valor": valor, "leitura": leitura, "nivel": nivel}


def _num(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _pct(v, casas=1):
    return f"{v * 100:.{casas}f}%".replace(".", ",")


def _dec(v, casas=2):
    return f"{v:.{casas}f}".replace(".", ",")


def _tendencia(tec):
    sma200 = getattr(tec, "sma200", None)
    preco = getattr(tec, "preco_atual", None)
    if sma200 is None or preco is None:
        return _ind("Tendência (MM200)", "—", "sem histórico", NEUTRO)
    if preco > sma200:
        return _ind("Tendência (MM200)", "acima da média", "alta", BOM)
    return _ind("Tendência (MM200)", "abaixo da média", "baixa", ATENCAO)


def _rsi(tec):
    rsi = _num(getattr(tec, "rsi14", None))
    if rsi is None:
        return _ind("RSI 14", "—", "sem histórico", NEUTRO)
    if rsi < 40:
        return _ind("RSI 14", _dec(rsi, 0), "não sobrecomprado", BOM)
    if rsi > 70:
        return _ind("RSI 14", _dec(rsi, 0), "sobrecomprado", ATENCAO)
    return _ind("RSI 14", _dec(rsi, 0), "neutro", NEUTRO)


def _dy(div, limite, nome="Dividend yield 12m"):
    dy = _num(getattr(div, "dy_12m_real", None)) if div else None
    if dy is None:
        return _ind(nome, "—", "sem pagamentos", NEUTRO)
    return _ind(nome, _pct(dy), "acima de " + _pct(limite, 0) if dy > limite else "abaixo de " + _pct(limite, 0),
                BOM if dy > limite else NEUTRO)


def _pl(fund, bom_abaixo, atencao_acima):
    pl = _num((fund or {}).get("pl"))
    if pl is None or pl <= 0:
        return _ind("P/L", "—", "não aplicável" if pl is not None else "sem dado", NEUTRO)
    if pl < bom_abaixo:
        return _ind("P/L", _dec(pl, 1), "baixo", BOM)
    if pl > atencao_acima:
        return _ind("P/L", _dec(pl, 1), "alto", ATENCAO)
    return _ind("P/L", _dec(pl, 1), "intermediário", NEUTRO)


def eh_fii_de_tijolo(fund: dict | None) -> bool:
    """Mesmo critério do motor (pontuar_fii): FII com imóveis cadastrados."""
    return (fund or {}).get("qtd_imoveis") not in (None, 0)


def indicadores_contexto(tipo: str | None, tec, fund: dict | None, div,
                         preco_atrativo: dict | None = None) -> list[dict]:
    """Os indicadores do bloco "Contexto do ativo", por tipo.

    FIIs: yield 12 meses (contra a própria média, quando houver), P/VP,
    rendimentos estáveis e vacância (só FII de tijolo) — sem tendência nem
    RSI, que num FII refletem mais o ciclo de juros e o curtíssimo prazo.
    Ações e ETFs: tendência (MM200), RSI, P/L (ações) e dividend yield."""
    fund = fund or {}
    if tipo == "fii":
        itens = []
        dy = _num(getattr(div, "dy_12m_real", None)) if div else None
        if preco_atrativo:
            y, m = preco_atrativo["yield"], preco_atrativo["yield_medio_24m"]
            itens.append(_ind("Yield 12 meses", _pct(y), f"média de 2 anos: {_pct(m)}", BOM if y >= m else NEUTRO))
        elif dy is not None:
            itens.append(_ind("Yield 12 meses", _pct(dy), "sem média de 2 anos", NEUTRO))
        else:
            itens.append(_ind("Yield 12 meses", "—", "sem pagamentos", NEUTRO))
        pvp = _num(fund.get("pvp"))
        if pvp is None:
            itens.append(_ind("P/VP", "—", "sem dado", NEUTRO))
        elif pvp < 0.95:
            itens.append(_ind("P/VP", _dec(pvp), "com desconto", BOM))
        elif pvp > 1.10:
            itens.append(_ind("P/VP", _dec(pvp), "com ágio", ATENCAO))
        else:
            itens.append(_ind("P/VP", _dec(pvp), "próximo do patrimônio", NEUTRO))
        estavel = preco_atrativo.get("estavel") if preco_atrativo else None
        if estavel is None and div is not None and _num(getattr(div, "crescimento_yoy", None)) is not None:
            estavel = div.crescimento_yoy >= -0.10
        if estavel is None:
            itens.append(_ind("Rendimentos", "—", "sem histórico", NEUTRO))
        else:
            itens.append(_ind("Rendimentos", "estáveis" if estavel else "em queda",
                              "queda de até 10%" if estavel else "caíram mais de 10%", BOM if estavel else ATENCAO))
        if eh_fii_de_tijolo(fund):
            vac = _num(fund.get("vacancia_media"))
            if vac is None:
                itens.append(_ind("Vacância", "—", "sem dado", NEUTRO))
            elif vac < 0.05:
                itens.append(_ind("Vacância", _pct(vac), "baixa", BOM))
            elif vac > 0.15:
                itens.append(_ind("Vacância", _pct(vac), "alta", ATENCAO))
            else:
                itens.append(_ind("Vacância", _pct(vac), "moderada", NEUTRO))
        return itens
    itens = [_tendencia(tec), _rsi(tec)]
    if tipo == "acao":
        itens += [_pl(fund, 15, 25), _dy(div, 0.06)]
    elif tipo == "acao_us":
        itens += [_pl(fund, 25, 40), _dy(div, 0.02)]
    elif tipo in ("etf_br", "etf_us"):
        itens.append(_dy(div, 0.04 if tipo == "etf_br" else 0.015))
    else:
        itens.append(_dy(div, 0.06))
    return itens


def contexto_do_ativo(resultado, precos: pd.Series | None = None,
                      dividendos_fii: pd.Series | None = None) -> dict | None:
    """Junta, a partir do resultado de ``analisar_ativo``, o que o card
    mostra: leitura combinada, indicadores, detalhes da pontuação e, para
    FIIs, o preço atrativo de hoje (com os rendimentos de ``dividendos_fii``)."""
    if resultado is None:
        return None
    tipo = getattr(resultado, "tipo", None)
    tec = getattr(resultado, "tec", None)
    pa = None
    if tipo == "fii" and dividendos_fii is not None and len(dividendos_fii):
        serie_precos = precos if precos is not None else getattr(tec, "historico", None)
        if serie_precos is not None:
            try:
                pa = preco_atrativo_hoje(serie_precos, dividendos_fii)
            except Exception:
                pa = None
    return {
        "leitura": getattr(resultado, "leitura_combinada_texto", None),
        "indicadores": indicadores_contexto(tipo, tec, getattr(resultado, "fund", None),
                                            getattr(resultado, "div", None), pa),
        "detalhes_timing": list(getattr(resultado, "detalhes_timing", []) or []),
        "detalhes_renda": list(getattr(resultado, "detalhes_renda", []) or []),
        "preco_atrativo": pa,
    }
