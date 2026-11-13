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
    classificar_fii,
    precisa_conferir,
    regularidade_dividendos,
    segmento_generico,
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
    leitura_fundamentos,
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
    regularidade: dict | None = None  # ações B3: "Os dividendos são regulares?" (5 anos fechados)
    pontos_tecnicos: int = 0  # parte de pontos_timing vinda de MM200, RSI e piso/topo (fora da leitura)
    sinal_timing: str = field(init=False)
    qualidade_renda: str = field(init=False)
    leitura_combinada_texto: str = field(init=False)

    def __post_init__(self) -> None:
        self.sinal_timing = classificar(self.pontos_timing)
        self.qualidade_renda = classificar_renda(self.pontos_renda)
        # leitura combinada: fundamentos e renda, sem os sinais técnicos (revisão de 06/11/2026)
        self.leitura_combinada_texto = leitura_fundamentos(self.pontos_timing - self.pontos_tecnicos,
                                                           self.pontos_renda)


def _medias_b3(ticker: str, fonte) -> pd.Series | None:
    """Médias mensais da base B3 (dados/cotacoes_b3_mensais.csv.gz), na escala
    do Yahoo; None sem a base ou sem o papel."""
    try:
        from .fontes.b3_cotahist import carregar_base, medias_mensais
        base = carregar_base()
        if base is None:
            return None
        try:
            eventos = fonte.baixar_eventos(ticker)
        except Exception:
            eventos = None
        medias = medias_mensais(base, ticker, eventos)
        return medias if len(medias) else None
    except Exception:
        return None


def analisar_ativo(ticker: str, tipo: str, periodo: str, fonte: FonteDados,
                   cdi_liquido_aa: float | None = None) -> AnaliseAtivo | None:
    """Executa a análise completa de um único ativo. Retorna ``None`` se
    não houver dados de preço suficientes. ``cdi_liquido_aa`` (CDI de 12
    meses, líquido de IR) liga a régua do dividend yield aos juros; sem ele,
    valem os limites fixos (ver ``limite_dy``)."""
    fechamento_bruto = fonte.baixar_precos(ticker, periodo)
    tec = calcular_indicadores_tecnicos(fechamento_bruto)
    if tec is None:
        return None

    info = fonte.baixar_info(ticker)
    dividendos = fonte.baixar_dividendos(ticker)
    div = calcular_historico_dividendos(dividendos, tec.preco_atual, data_ref=tec.historico.index[-1])

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
            if fund_fundamentus.qtd_imoveis or classificar_fii(fund) == "tijolo":
                # a participação dos imóveis no patrimônio separa papel de tijolo quando o
                # segmento é genérico (caso RECR11) e confere o segmento de tijolo (caso MXRF11)
                try:
                    from .fontes.fundamentus import buscar_imoveis_pl_fii
                    fund["imoveis_pl"] = buscar_imoveis_pl_fii(ticker)
                except Exception:
                    fund["imoveis_pl"] = None
        resultado_timing = pontuar_fii(tec, fund, div, cdi_liquido_aa)
    elif tipo == "acao_us":
        fund = extrair_fundamentalistas_acao(info)  # mesmos campos do Yahoo servem p/ EUA
        resultado_timing = pontuar_acao_us(tec, fund, div)
    elif tipo in ("etf_br", "etf_us"):
        fund = extrair_fundamentalistas_acao(info)  # geralmente vazio p/ ETF, não quebra nada
        resultado_timing = pontuar_etf(tec, div, tipo, cdi_liquido_aa)
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
            if fund_fundamentus.pvp is not None:
                fund["pvp"] = fund_fundamentus.pvp
            if fund_fundamentus.roe is not None:
                fund["roe"] = fund_fundamentus.roe
            fund["roic"] = fund_fundamentus.roic
            fund["margem_liquida"] = fund_fundamentus.margem_liquida
            fund["divida_bruta_patrimonio"] = fund_fundamentus.divida_bruta_patrimonio
            fund["liquidez_corrente"] = fund_fundamentus.liquidez_corrente
        resultado_timing = pontuar_acao(tec, fund, div, cdi_liquido_aa)

    regularidade = None
    if tipo == "acao" and dividendos is not None and len(dividendos):
        # 5 anos civis fechados pedem ~6 anos de preços: busca 10 anos só para isso
        try:
            precos_10a = fonte.baixar_precos(ticker, "10y")
            medias_b3 = _medias_b3(ticker, fonte)  # anos que o Yahoo não cobre: médias mensais da B3
            regularidade = regularidade_dividendos(dividendos, precos_10a, data_ref=tec.historico.index[-1],
                                                   medias_mensais=medias_b3)
            if precisa_conferir(regularidade):
                # ano sem nenhum provento no Yahoo: confere no Fundamentus se é lacuna da fonte
                from .fontes.fundamentus import buscar_proventos_acao
                conferencia = buscar_proventos_acao(ticker)
                if conferencia is not None and len(conferencia):
                    regularidade = regularidade_dividendos(dividendos, precos_10a, data_ref=tec.historico.index[-1],
                                                           conferencia=conferencia, medias_mensais=medias_b3)
                    if regularidade is not None:
                        regularidade["conferencia"] = "ok"
                        # proventos completos (Fundamentus): o preço-teto do card usa esta série
                        regularidade["proventos_conferidos"] = conferencia
                elif regularidade is not None:
                    # o card avisa: sem a conferência, um ano zerado pode ser lacuna do Yahoo
                    from .fontes.fundamentus import ERRO_PROVENTOS_ACAO
                    regularidade["conferencia"] = "falhou"
                    regularidade["conferencia_motivo"] = ERRO_PROVENTOS_ACAO.get(
                        ticker.upper().replace(".SA", ""), "a página não trouxe proventos")
        except Exception:
            regularidade = None
    resultado_renda = avaliar_renda(div, tipo, cdi_liquido_aa, regularidade)

    return AnaliseAtivo(
        ticker=ticker,
        tipo=tipo,
        tec=tec,
        fund=fund,
        div=div,
        pontos_timing=resultado_timing.pontos,
        detalhes_timing=resultado_timing.detalhes,
        pontos_renda=resultado_renda.pontos,
        detalhes_renda=resultado_renda.detalhes,
        regularidade=regularidade,
        pontos_tecnicos=getattr(resultado_timing, "pontos_tecnicos", 0),
    )


def analisar_carteira(
    tickers: list[str], tipo: str, periodo: str, fonte: FonteDados, cdi_liquido_aa: float | None = None
) -> list[AnaliseAtivo]:
    """Executa a análise de uma lista de tickers do mesmo tipo (todos ação
    ou todos FII), pulando silenciosamente os que não tiverem dados
    suficientes (quem chama pode comparar `len(resultado)` com
    `len(tickers)` para saber se algum foi pulado)."""
    resultados = []
    for ticker in tickers:
        analise = analisar_ativo(ticker, tipo, periodo, fonte, cdi_liquido_aa)
        if analise is not None:
            resultados.append(analise)
    return resultados
