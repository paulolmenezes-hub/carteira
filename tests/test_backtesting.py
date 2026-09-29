from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from carteira_analise.backtesting import (
    PontoBacktest,
    _classe_a_partir_de_pontos,
    calcular_f1_macro,
    calcular_mae_r2_renda,
    executar_backtest_ticker,
    rotular_retorno_futuro,
)


def _serie_tendencia_alta(n=600, seed=1):
    """Série com tendência de alta consistente — um backtest sobre essa
    série deveria prever majoritariamente 'compra', e o gabarito também
    deveria ser majoritariamente 'compra' (alta real observada depois)."""
    rng = np.random.default_rng(seed)
    datas = pd.bdate_range("2022-01-01", periods=n)
    # tendência de alta de ~0.15%/dia + ruído pequeno
    precos = 50 * np.cumprod(1 + rng.normal(0.0015, 0.008, n))
    return pd.Series(precos, index=datas)


def _serie_lateral_ruidosa(n=600, seed=2):
    """Série sem tendência (passeio aleatório de média zero) — o backtest
    não deveria conseguir prever bem o retorno futuro (F1 baixo esperado)."""
    rng = np.random.default_rng(seed)
    datas = pd.bdate_range("2022-01-01", periods=n)
    precos = 50 + np.cumsum(rng.normal(0, 0.3, n))
    precos = np.clip(precos, 20, None)
    return pd.Series(precos, index=datas)


class TestRotularRetornoFuturo:
    def test_alta_forte_rotula_compra(self):
        datas = pd.bdate_range("2024-01-01", periods=200)
        precos = np.concatenate([np.full(60, 10.0), np.full(140, 12.0)])  # +20% a partir do índice 60
        serie = pd.Series(precos, index=datas)
        rotulo = rotular_retorno_futuro(serie, datas[10], horizonte_dias=90, limiar=0.05)
        assert rotulo == "compra"

    def test_queda_forte_rotula_venda(self):
        datas = pd.bdate_range("2024-01-01", periods=200)
        precos = np.concatenate([np.full(60, 10.0), np.full(140, 8.0)])  # -20% a partir do índice 60
        serie = pd.Series(precos, index=datas)
        rotulo = rotular_retorno_futuro(serie, datas[10], horizonte_dias=90, limiar=0.05)
        assert rotulo == "venda"

    def test_preco_estavel_rotula_manter(self):
        datas = pd.bdate_range("2024-01-01", periods=200)
        precos = np.full(200, 10.0)
        serie = pd.Series(precos, index=datas)
        rotulo = rotular_retorno_futuro(serie, datas[50], horizonte_dias=60, limiar=0.05)
        assert rotulo == "manter"

    def test_sem_dado_futuro_suficiente_retorna_none(self):
        datas = pd.bdate_range("2024-01-01", periods=100)
        serie = pd.Series(np.full(100, 10.0), index=datas)
        # data de corte perto do fim da série, sem 90 dias de folga
        rotulo = rotular_retorno_futuro(serie, datas[-5], horizonte_dias=90)
        assert rotulo is None


class TestClasseAPartirDePontos:
    def test_limiares_padrao(self):
        assert _classe_a_partir_de_pontos(3) == "compra"
        assert _classe_a_partir_de_pontos(2) == "manter"  # não atinge o limiar padrão (3)
        assert _classe_a_partir_de_pontos(-2) == "venda"
        assert _classe_a_partir_de_pontos(-1) == "manter"

    def test_limiar_de_compra_customizado(self):
        # com limiar_compra=2, uma pontuação de 2 agora classifica como compra
        assert _classe_a_partir_de_pontos(2, limiar_compra=2) == "compra"
        assert _classe_a_partir_de_pontos(1, limiar_compra=2) == "manter"

    def test_limiares_simetricos_customizados(self):
        assert _classe_a_partir_de_pontos(2, limiar_compra=2, limiar_venda=-2) == "compra"
        assert _classe_a_partir_de_pontos(-2, limiar_compra=2, limiar_venda=-2) == "venda"
        assert _classe_a_partir_de_pontos(0, limiar_compra=2, limiar_venda=-2) == "manter"


class TestExecutarBacktestTicker:
    def test_gera_registros_para_serie_longa_o_suficiente(self):
        serie = _serie_tendencia_alta()
        registros = executar_backtest_ticker(serie, ticker="TESTE3.SA")
        assert len(registros) > 0
        assert all(isinstance(r, PontoBacktest) for r in registros)
        assert all(r.classe_prevista in ("compra", "manter", "venda") for r in registros)
        assert all(r.classe_real in ("compra", "manter", "venda") for r in registros)

    def test_serie_curta_nao_gera_registros(self):
        serie = _serie_tendencia_alta(n=100)  # menor que JANELA_MINIMA_DIAS (200)
        registros = executar_backtest_ticker(serie, ticker="CURTA3.SA")
        assert registros == []

    def test_limiar_de_compra_mais_baixo_gera_mais_previsoes_de_compra(self):
        # Com um limiar de compra mais permissivo (2 em vez de 3), o motor deve
        # classificar "compra" com mais frequência sobre a MESMA série — valida
        # que o parâmetro realmente se propaga até o resultado final.
        serie = _serie_tendencia_alta()
        registros_padrao = executar_backtest_ticker(serie, ticker="TESTE3.SA")
        registros_limiar2 = executar_backtest_ticker(serie, ticker="TESTE3.SA", limiar_compra=2)

        n_compra_padrao = sum(1 for r in registros_padrao if r.classe_prevista == "compra")
        n_compra_limiar2 = sum(1 for r in registros_limiar2 if r.classe_prevista == "compra")
        assert n_compra_limiar2 >= n_compra_padrao

    def test_tendencia_de_alta_forte_tende_a_acertar_mais_bull(self):
        # Não é uma garantia estatística (é uma série com ruído), mas o
        # gabarito real deve ser predominantemente "compra" numa série com
        # tendência de alta consistente — validação de sanidade do rótulo.
        serie = _serie_tendencia_alta()
        registros = executar_backtest_ticker(serie, ticker="ALTA3.SA")
        reais = [r.classe_real for r in registros]
        prop_compra_real = reais.count("compra") / len(reais)
        assert prop_compra_real > 0.5


class TestCalcularF1Macro:
    def test_previsoes_perfeitas_geram_f1_1(self):
        registros = [
            PontoBacktest("T", pd.Timestamp("2024-01-01"), "compra", "compra", None, None),
            PontoBacktest("T", pd.Timestamp("2024-02-01"), "venda", "venda", None, None),
            PontoBacktest("T", pd.Timestamp("2024-03-01"), "manter", "manter", None, None),
        ]
        r = calcular_f1_macro(registros)
        assert r.f1_macro == pytest.approx(1.0)
        assert r.n_total == 3

    def test_previsoes_sempre_erradas_geram_f1_baixo(self):
        registros = [
            PontoBacktest("T", pd.Timestamp("2024-01-01"), "compra", "venda", None, None),
            PontoBacktest("T", pd.Timestamp("2024-02-01"), "venda", "compra", None, None),
        ]
        r = calcular_f1_macro(registros)
        assert r.f1_macro == pytest.approx(0.0)

    def test_lista_vazia_nao_quebra(self):
        r = calcular_f1_macro([])
        assert r.f1_macro == 0.0
        assert r.n_total == 0

    def test_modelo_que_so_preve_uma_classe_e_penalizado(self):
        # Prevê sempre "compra", mas o real varia — F1 macro deve punir isso
        # (é exatamente o vício que motivou usar F1 em vez de acurácia simples)
        registros = (
            [PontoBacktest("T", pd.Timestamp("2024-01-01"), "compra", "compra", None, None)] * 8
            + [PontoBacktest("T", pd.Timestamp("2024-01-01"), "compra", "venda", None, None)] * 1
            + [PontoBacktest("T", pd.Timestamp("2024-01-01"), "compra", "manter", None, None)] * 1
        )
        r = calcular_f1_macro(registros)
        # recall de "venda" e "manter" é 0 (nunca previstos) -> f1 delas é 0
        assert r.por_classe["venda"].f1 == 0.0
        assert r.por_classe["manter"].f1 == 0.0
        assert r.f1_macro < 0.5  # média macro puxada pra baixo pelas classes nunca previstas


class TestCalcularMaeR2Renda:
    def test_previsao_perfeita_gera_mae_zero_e_r2_um(self):
        registros = [
            PontoBacktest("T", pd.Timestamp("2024-01-01"), "manter", "manter", 0.05, 0.05),
            PontoBacktest("T", pd.Timestamp("2024-02-01"), "manter", "manter", 0.08, 0.08),
            PontoBacktest("T", pd.Timestamp("2024-03-01"), "manter", "manter", 0.03, 0.03),
        ]
        r = calcular_mae_r2_renda(registros)
        assert r.mae == pytest.approx(0.0)
        assert r.r2 == pytest.approx(1.0)
        assert r.n == 3

    def test_sem_pares_validos_retorna_none(self):
        registros = [
            PontoBacktest("T", pd.Timestamp("2024-01-01"), "manter", "manter", None, None),
        ]
        r = calcular_mae_r2_renda(registros)
        assert r.mae is None
        assert r.r2 is None
        assert r.n == 0

    def test_mae_reflete_erro_absoluto_medio(self):
        registros = [
            PontoBacktest("T", pd.Timestamp("2024-01-01"), "manter", "manter", 0.10, 0.08),  # erro 0.02
            PontoBacktest("T", pd.Timestamp("2024-02-01"), "manter", "manter", 0.05, 0.09),  # erro 0.04
        ]
        r = calcular_mae_r2_renda(registros)
        assert r.mae == pytest.approx(0.03)  # média de 0.02 e 0.04
