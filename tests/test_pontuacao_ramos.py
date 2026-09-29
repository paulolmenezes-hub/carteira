"""Ramos da pontuação que não tinham teste dedicado: endividamento e
liquidez de ações B3 (dados do Fundamentus), limiares do mercado
americano, ETFs, FIIs de tijolo x papel e renda de ETFs."""
from __future__ import annotations

import pandas as pd
import pytest

from carteira_analise.fundamentalistas import HistoricoDividendos
from carteira_analise.pontuacao import (
    avaliar_renda,
    leitura_combinada,
    pontuar_acao,
    pontuar_acao_us,
    pontuar_etf,
    pontuar_fii,
)
from carteira_analise.tecnicos import IndicadoresTecnicos


def _tec(**over):
    base = dict(preco_atual=10.0, sma50=9.0, sma200=9.0, rsi14=50.0, dist_maxima_pct=-20.0,
                dist_minima_pct=20.0, n_outliers_corrigidos=0, n_dias_truncados=0, dias_usados=500,
                historico=pd.Series([10.0]))
    base.update(over)
    return IndicadoresTecnicos(**base)


def _div(**over):
    base = dict(soma_12m=1.0, soma_anterior_12m=1.0, crescimento_yoy=0.0, dy_12m_real=0.05, n_pagamentos_12m=4)
    base.update(over)
    return HistoricoDividendos(**base)


def _tem(resultado, trecho):
    return any(trecho in d for d in resultado.detalhes)


# ------------------------------------------------------------ ações B3
class TestAcaoB3EndividamentoLiquidez:
    def test_divida_alta_penaliza(self):
        r = pontuar_acao(_tec(), {"divida_bruta_patrimonio": 2.0}, None)
        assert _tem(r, "-1 alto endividamento")

    def test_divida_baixa_bonifica(self):
        r = pontuar_acao(_tec(), {"divida_bruta_patrimonio": 0.3}, None)
        assert _tem(r, "+1 baixo endividamento")

    def test_divida_intermediaria_neutra(self):
        r = pontuar_acao(_tec(), {"divida_bruta_patrimonio": 1.0}, None)
        assert not _tem(r, "endividamento")

    def test_liquidez_apertada_penaliza_e_saudavel_bonifica(self):
        assert _tem(pontuar_acao(_tec(), {"liquidez_corrente": 0.8}, None), "-1 liquidez de curto prazo apertada")
        assert _tem(pontuar_acao(_tec(), {"liquidez_corrente": 2.0}, None), "+1 liquidez de curto prazo saudável")
        assert not _tem(pontuar_acao(_tec(), {"liquidez_corrente": 1.2}, None), "liquidez")

    @pytest.mark.parametrize("campo", ["divida_bruta_patrimonio", "liquidez_corrente"])
    def test_banco_zero_ou_ausente_nao_pontua(self, campo):
        # Regressão da Seção 4.2.2 do Relatório (BBAS3): zero = não aplicável
        assert pontuar_acao(_tec(), {campo: 0.0}, None).pontos == pontuar_acao(_tec(), {}, None).pontos
        assert pontuar_acao(_tec(), {campo: None}, None).pontos == pontuar_acao(_tec(), {}, None).pontos

    def test_roic_e_margem(self):
        r = pontuar_acao(_tec(), {"roic": 0.20, "margem_liquida": 0.15}, None)
        assert _tem(r, "+1 ROIC saudável") and _tem(r, "+1 margem líquida saudável")
        r = pontuar_acao(_tec(), {"roic": 0.05, "margem_liquida": 0.05}, None)
        assert not _tem(r, "ROIC") and not _tem(r, "margem")

    def test_valor_100x_seria_penalizado(self):
        # Documenta o efeito do bug de leitura corrigido na Sprint 3: dívida
        # de 0,45 lida como 45 viraria penalização indevida.
        assert _tem(pontuar_acao(_tec(), {"divida_bruta_patrimonio": 45.0}, None), "-1 alto endividamento")
        assert _tem(pontuar_acao(_tec(), {"divida_bruta_patrimonio": 0.45}, None), "+1 baixo endividamento")


# ------------------------------------------------------------ ações EUA
class TestAcaoEUA:
    def test_limiares_americanos(self):
        r = pontuar_acao_us(_tec(), {"pl": 45.0, "pvp": 12.0}, None)
        assert _tem(r, "-1 P/L alto mesmo p/ padrão americano") and _tem(r, "-1 P/VP alto mesmo p/ padrão americano")
        r = pontuar_acao_us(_tec(), {"pl": 30.0, "pvp": 6.0}, None)
        assert not _tem(r, "P/L") and not _tem(r, "P/VP")

    def test_tecnico_negativo(self):
        r = pontuar_acao_us(_tec(preco_atual=8.0, sma200=9.0, rsi14=75.0), {}, None)
        assert _tem(r, "-1 preço abaixo da MM200") and _tem(r, "-1 RSI > 70")

    def test_rsi_baixo(self):
        assert _tem(pontuar_acao_us(_tec(rsi14=30.0), {}, None), "+1 RSI < 40")


# ------------------------------------------------------------ ETFs
class TestETF:
    def test_tecnico_positivo_e_negativo(self):
        assert _tem(pontuar_etf(_tec(rsi14=30.0), None, "etf_br"), "+1 RSI < 40")
        r = pontuar_etf(_tec(preco_atual=8.0, sma200=9.0, rsi14=80.0), None, "etf_us")
        assert _tem(r, "-1 preço abaixo da MM200") and _tem(r, "-1 RSI > 70")

    def test_limiar_de_yield_por_mercado(self):
        div = _div(dy_12m_real=0.03)
        assert pontuar_etf(_tec(), div, "etf_us").pontos > pontuar_etf(_tec(), div, "etf_br").pontos


# ------------------------------------------------------------ FIIs
class TestFII:
    def test_tijolo_vacancia_e_cap_rate(self):
        bom = pontuar_fii(_tec(), {"qtd_imoveis": 10, "vacancia_media": 0.02, "cap_rate": 0.09}, None)
        assert _tem(bom, "+1 vacância baixa") and _tem(bom, "+1 cap rate atrativo")
        ruim = pontuar_fii(_tec(), {"qtd_imoveis": 10, "vacancia_media": 0.20, "cap_rate": 0.04}, None)
        assert _tem(ruim, "-1 vacância alta") and _tem(ruim, "-1 cap rate baixo")

    def test_papel_ignora_vacancia_e_cap_rate(self):
        r = pontuar_fii(_tec(), {"qtd_imoveis": 0, "vacancia_media": 1.0, "cap_rate": 0.0}, None)
        assert not _tem(r, "vacância") and not _tem(r, "cap rate")

    def test_agio_e_tecnico_negativo(self):
        r = pontuar_fii(_tec(preco_atual=8.0, sma200=9.0, rsi14=75.0), {"pvp": 1.2}, None)
        assert _tem(r, "-1 cota com ágio") and _tem(r, "-1 cota abaixo da MM200") and _tem(r, "-1 RSI > 70")


# ------------------------------------------------------------ renda
class TestRenda:
    @pytest.mark.parametrize("tipo", ["etf_br", "etf_us", "acao_us"])
    def test_sem_historico_nao_aplicavel(self, tipo):
        assert avaliar_renda(None, tipo).pontos == "NA"

    def test_sem_historico_acao_b3(self):
        assert avaliar_renda(None, "acao").pontos is None

    @pytest.mark.parametrize("tipo,dy,atrativo", [
        ("etf_br", 0.045, True), ("etf_br", 0.035, False),
        ("etf_us", 0.02, True), ("etf_us", 0.01, False),
        ("acao_us", 0.025, True), ("acao_us", 0.015, False),
    ])
    def test_limiar_de_yield_por_tipo(self, tipo, dy, atrativo):
        r = avaliar_renda(_div(dy_12m_real=dy), tipo)
        assert _tem(r, "atrativo") == atrativo

    def test_leitura_combinada_na_e_neutra(self):
        assert leitura_combinada(0, "NA") == leitura_combinada(0, None)
