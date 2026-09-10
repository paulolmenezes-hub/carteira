"""Orquestração: junta indicadores técnicos, fundamentalistas, dividendos
e o motor de pontuação para produzir a análise completa de um ativo, ou de
uma carteira inteira.

É a única camada, além de ``fontes/``, que conhece a existência de uma
fonte de dados concreta — por isso recebe o módulo de fonte como parâmetro
(injeção de dependência simples), o que facilita tanto os testes (usando
uma fonte falsa/mock) quanto uma futura troca ou combinação de fontes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .fundamentalistas import (
    HistoricoDividendos,
    calcular_historico_dividendos,
    extrair_fundamentalistas_acao,
    extrair_fundamentalistas_fii,
)
from .pontuacao import (
    avaliar_renda,
    classificar,
    classificar_renda,
    pontuar_acao,
    pontuar_acao_us,
    pontuar_etf,
    pontuar_fii,
    leitura_combinada,
)
from .tecnicos import IndicadoresTecnicos, calcular_indicadores_tecnicos

# `fonte` é qualquer objeto (tipicamente um módulo, ex.: carteira_analise.fontes.yahoo)
# que exponha as três funções abaixo com essa assinatura. Não é um Protocol formal
# de classe porque, na prática, o valor passado é o próprio módulo — módulos não têm
# `self` — e forçar um Protocol baseado em classe aqui atrapalharia mais do que ajudaria.
FonteDados = Any


def parse_tickers(texto: str) -> list[str]:
    """Quebra a lista de tickers por vírgula. Corrige automaticamente um
    erro de digitação comum: vírgula no lugar do ponto antes do sufixo de
    bolsa (ex.: 'VALE3, SA' vira, sem querer, dois tokens 'VALE3' e 'SA' —
    aqui a gente detecta esse padrão e reconstrói como 'VALE3.SA')."""
    brutos = [t.strip().upper() for t in texto.split(",") if t.strip()]

    tickers: list[str] = []
    i = 0
    while i < len(brutos):
        atual = brutos[i]
        proximo = brutos[i + 1] if i + 1 < len(brutos) else None
        if proximo in ("SA", "US") and not atual.endswith("." + proximo):
            tickers.append(f"{atual}.{proximo}")
            i += 2
        else:
            tickers.append(atual)
            i += 1
    return tickers


@dataclass
class AnaliseAtivo:
    ticker: str
    tipo: str  # "acao" | "fii"
    tec: IndicadoresTecnicos
    fund: dict
    div: HistoricoDividendos | None
    pontos_timing: int
    detalhes_timing: list[str]
    pontos_renda: int | str | None
    detalhes_renda: list[str]
    sinal_timing: str = field(init=False)
    qualidade_renda: str = field(init=False)
    leitura_combinada_texto: str = field(init=False)

    def __post_init__(self) -> None:
        self.sinal_timing = classificar(self.pontos_timing)
        self.qualidade_renda = classificar_renda(self.pontos_renda)
        self.leitura_combinada_texto = leitura_combinada(self.pontos_timing, self.pontos_renda)


def analisar_ativo(ticker: str, tipo: str, periodo: str, fonte: FonteDados) -> AnaliseAtivo | None:
    """Executa a análise completa de um único ativo. Retorna ``None`` se
    não houver dados de preço suficientes."""
    fechamento_bruto = fonte.baixar_precos(ticker, periodo)
    tec = calcular_indicadores_tecnicos(fechamento_bruto)
    if tec is None:
        return None

    info = fonte.baixar_info(ticker)
    dividendos = fonte.baixar_dividendos(ticker)
    div = calcular_historico_dividendos(dividendos, tec.preco_atual)

    if tipo == "fii":
        fund = extrair_fundamentalistas_fii(info)
        fund_fundamentus = None
        try:
            from .fontes.fundamentus import buscar_fundamentos_fii
            fund_fundamentus = buscar_fundamentos_fii(ticker)
        except Exception:
            fund_fundamentus = None  # Fundamentus indisponível — segue só com Yahoo Finance
        if fund_fundamentus is not None:
            # Fundamentus é preferido para P/VP e Dividend Yield de FIIs (o
            # campo do Yahoo Finance é pouco confiável/ausente para FIIs —
            # ver Seção 3.2.2 do Relatório). Dados extras (sem equivalente
            # no Yahoo) também entram no dict, para uso futuro no motor.
            fund["pvp"] = fund_fundamentus.pvp if fund_fundamentus.pvp is not None else fund.get("pvp")
            fund["dividend_yield_fundamentus"] = fund_fundamentus.dividend_yield
            fund["segmento"] = fund_fundamentus.segmento
            fund["ffo_yield"] = fund_fundamentus.ffo_yield
            fund["vacancia_media"] = fund_fundamentus.vacancia_media
            fund["cap_rate"] = fund_fundamentus.cap_rate
            fund["qtd_imoveis"] = fund_fundamentus.qtd_imoveis
        resultado_timing = pontuar_fii(tec, fund, div)
    elif tipo == "acao_us":
        fund = extrair_fundamentalistas_acao(info)  # mesmos campos do Yahoo servem p/ EUA
        resultado_timing = pontuar_acao_us(tec, fund, div)
    elif tipo in ("etf_br", "etf_us"):
        fund = extrair_fundamentalistas_acao(info)  # geralmente vazio p/ ETF, não quebra nada
        resultado_timing = pontuar_etf(tec, div, tipo)
    else:
        fund = extrair_fundamentalistas_acao(info)
        fund_fundamentus = None
        try:
            from .fontes.fundamentus import buscar_fundamentos_acao
            fund_fundamentus = buscar_fundamentos_acao(ticker)
        except Exception:
            fund_fundamentus = None  # Fundamentus indisponível — segue só com Yahoo Finance
        if fund_fundamentus is not None:
            # Fundamentus é preferido para ações B3 (sua base original e
            # mais consolidada — ver Seção 3.2.2 do Relatório), substituindo
            # os campos do Yahoo Finance quando disponíveis. Não se aplica
            # a ações dos EUA (tratadas no ramo "acao_us" acima).
            if fund_fundamentus.pl is not None:
                fund["pl"] = fund_fundamentus.pl
            if fund_fundamentus.pvp
