from __future__ import annotations

import pandas as pd

from carteira_analise import pontuacao
from carteira_analise.fundamentalistas import HistoricoDividendos
from carteira_analise.pontuacao import (
    avaliar_renda,
    classificar,
    classificar_renda,
    pontuar_acao,
    pontuar_acao_us,
    pontuar_fii,
    leitura_combinada,
)
from carteira_analise.tecnicos import IndicadoresTecnicos


def _tec(**overrides) -> IndicadoresTecnicos:
    base = dict(
        preco_atual=10.0,
        sma50=9.0,
        sma200=9.0,
        rsi14=50.0,
        dist_maxima_pct=-20.0,
        dist_minima_pct=20.0,
        n_outliers_corrigidos=0,
        n_dias_truncados=0,
        dias_usados=500,
        historico=pd.Series([10.0]),
    )
    base.update(overrides)
    return IndicadoresTecnicos(**base)


def _div(**overrides) -> HistoricoDividendos:
    base = dict(
        soma_12m=1.0,
        soma_anterior_12m=1.0,
        crescimento_yoy=0.0,
        dy_12m_real=0.05,
        n_pagamentos_12m=4,
    )
    base.update(overrides)
    return HistoricoDividendos(**base)


class TestPontuarAcao:
    def test_acao_barata_e_saudavel_pontua_positivo(self):
        tec = _tec(preco_atual=10.0, sma200=9.0, rsi14=35.0, dist_minima_pct=10.0, dist_maxima_pct=-20.0)
        fund = {"pl": 6.0, "pvp": 0.8, "roe": 0.20, "divida_patrimonio": 0.3}
        div = _div(dy_12m_real=0.08)
        resultado = pontuar_acao(tec, fund, div)
        assert resultado.pontos >= 5
        assert any("P/L baixo" in d for d in resultado.detalhes)

    def test_acao_cara_e_sobrecomprada_pontua_negativo(self):
        tec = _tec(preco_atual=10.0, sma200=12.0, rsi14=80.0, dist_minima_pct=90.0, dist_maxima_pct=-1.0)
        fund = {"pl": 40.0, "pvp": 6.0, "roe": 0.02, "divida_patrimonio": 2.0}
        div = _div(dy_12m_real=0.01)
        resultado = pontuar_acao(tec, fund, div)
        assert resultado.pontos <= -2

    def test_yield_usa_dy_12m_real_nao_campo_bruto(self):
        # Regressão do bug do dividendYield inconsistente: a pontuação NUNCA
        # deve usar um campo de yield fora do HistoricoDividendos calculado.
        tec = _tec()
        fund = {"pl": None, "pvp": None, "roe": None, "divida_patrimonio": None}
        div = _div(dy_12m_real=0.10)
        resultado = pontuar_acao(tec, fund, div)
        assert any("yield efetivo" in d and "10.0%" in d for d in resultado.detalhes)


class TestPontuarFII:
    def test_fii_com_desconto_e_bom_yield_pontua_positivo(self):
        tec = _tec(preco_atual=10.0, sma200=9.5, rsi14=35.0, dist_minima_pct=10.0, dist_maxima_pct=-20.0)
        fund = {"pvp": 0.85}
        div = _div(dy_12m_real=0.10)
        resultado = pontuar_fii(tec, fund, div)
        assert resultado.pontos >= 3

    def test_fii_com_agio_pontua_negativo(self):
        tec = _tec(preco_atual=10.0, sma200=9.5, rsi14=50.0, dist_minima_pct=90.0, dist_maxima_pct=-1.0)
        fund = {"pvp": 1.3}
        div = _div(dy_12m_real=0.03)
        resultado = pontuar_fii(tec, fund, div)
        assert resultado.pontos <= 0


class TestClassificar:
    def test_limiares_de_classificacao(self):
        assert classificar(3) == "🟢 Indicadores técnicos majoritariamente favoráveis"
        assert classificar(1) == "🟡 Indicadores técnicos mistos, leve viés favorável"
        assert classificar(-1) == "🟡 Indicadores técnicos mistos, sem direção clara"
        assert classificar(-2) == "🔴 Indicadores técnicos majoritariamente desfavoráveis"


class TestAvaliarRenda:
    def test_sem_dividendos_retorna_none(self):
        resultado = avaliar_renda(None, "acao")
        assert resultado.pontos is None

    def test_renda_crescente_e_consistente_pontua_positivo(self):
        div = _div(dy_12m_real=0.09, crescimento_yoy=0.20, n_pagamentos_12m=12)
        resultado = avaliar_renda(div, "fii")
        assert resultado.pontos >= 2

    def test_renda_caindo_pontua_negativo(self):
        div = _div(dy_12m_real=0.02, crescimento_yoy=-0.30, n_pagamentos_12m=0)
        resultado = avaliar_renda(div, "acao")
        assert resultado.pontos < 0

    def test_limiar_de_yield_dependente_do_tipo(self):
        # 7% é bom pra ação (limite 6%) mas neutro pra FII (limite 8%)
        div = _div(dy_12m_real=0.07, crescimento_yoy=0.0, n_pagamentos_12m=1)
        resultado_acao = avaliar_renda(div, "acao")
        resultado_fii = avaliar_renda(div, "fii")
        assert any("atrativo" in d for d in resultado_acao.detalhes)
        assert not any("atrativo" in d for d in resultado_fii.detalhes)


class TestPontuarAcaoUS:
    def test_acao_de_crescimento_sem_dividendo_nao_e_penalizada(self):
        tec = _tec(preco_atual=180.0, sma200=150.0, rsi14=35.0, dist_minima_pct=10.0, dist_maxima_pct=-10.0)
        fund = {"pl": 30.0, "pvp": 8.0, "roe": 0.25, "divida_patrimonio": None}
        resultado = pontuacao.pontuar_acao_us(tec, fund, None)
        # P/L=30 e P/VP=8 seriam "caros" pelos limiares de ação BR, mas não pelos limiares US
        assert not any("P/L alto" in d for d in resultado.detalhes)
        assert not any("P/VP alto" in d for d in resultado.detalhes)
        assert resultado.pontos > 0

    def test_limiares_us_sao_mais_permissivos_que_acao_br(self):
        tec = _tec(preco_atual=100.0, sma200=100.0, rsi14=50.0, dist_minima_pct=50.0, dist_maxima_pct=-50.0)
        fund = {"pl": 20.0, "pvp": 3.0, "roe": 0.0, "divida_patrimonio": None}
        resultado_us = pontuacao.pontuar_acao_us(tec, fund, None)
        resultado_br = pontuacao.pontuar_acao(tec, fund, None)
        # Mesmos fundamentos: pontuação US deve ser >= BR (P/L=20 e P/VP=3 pontuam bem só nos limiares US)
        assert resultado_us.pontos > resultado_br.pontos

    def test_yield_baixo_ainda_pontua_pelo_limiar_us(self):
        tec = _tec()
        fund = {"pl": None, "pvp": None, "roe": None, "divida_patrimonio": None}
        div = _div(dy_12m_real=0.025)  # 2.5%: baixo p/ padrão BR (6%), ok p/ padrão US (2%)
        resultado = pontuacao.pontuar_acao_us(tec, fund, div)
        assert any("yield efetivo" in d for d in resultado.detalhes)


class TestAvaliarRendaNaoAplicavel:
    def test_acao_us_sem_dividendo_retorna_na(self):
        resultado = avaliar_renda(None, "acao_us")
        assert resultado.pontos == "NA"

    def test_acao_br_sem_dividendo_retorna_none_nao_na(self):
        resultado = avaliar_renda(None, "acao")
        assert resultado.pontos is None
        assert resultado.pontos != "NA"

    def test_classificar_renda_na(self):
        assert classificar_renda("NA") == "⚪ Não aplicável — foco em crescimento, não renda"

    def test_leitura_combinada_trata_na_como_neutro(self):
        # não deve lançar exceção nem tratar 'NA' como sinal negativo
        assert leitura_combinada(pontos_timing=3, pontos_renda="NA").startswith("🟢")
        assert not leitura_combinada(pontos_timing=0, pontos_renda="NA").startswith("🔴")

    def test_limiares(self):
        assert classificar_renda(None) == "⚪ dados de dividendos insuficientes"
        assert classificar_renda(2) == "🟢 renda saudável e crescente"
        assert classificar_renda(0) == "🟡 renda estável"
        assert classificar_renda(-1) == "🔴 atenção: renda em deterioração"


class TestPontuarETF:
    def test_etf_nao_usa_fundamentalistas(self):
        tec = _tec(preco_atual=100.0, sma200=90.0, rsi14=35.0, dist_minima_pct=10.0, dist_maxima_pct=-10.0)
        resultado = pontuacao.pontuar_etf(tec, None, "etf_br")
        assert not any("P/L" in d or "P/VP" in d or "ROE" in d for d in resultado.detalhes)

    def test_etf_br_e_etf_us_tem_limiares_de_yield_diferentes(self):
        tec = _tec()
        div = _div(dy_12m_real=0.02)  # 2%: bom p/ ETF EUA (limiar 1.5%), insuficiente p/ ETF BR (limiar 4%)
        resultado_br = pontuacao.pontuar_etf(tec, div, "etf_br")
        resultado_us = pontuacao.pontuar_etf(tec, div, "etf_us")
        assert not any("yield" in d for d in resultado_br.detalhes)
        assert any("yield" in d for d in resultado_us.detalhes)


class TestAvaliarRendaETF:
    def test_etf_sem_dividendo_retorna_na(self):
        assert avaliar_renda(None, "etf_br").pontos == "NA"
        assert avaliar_renda(None, "etf_us").pontos == "NA"

    def test_mensagem_na_de_etf_menciona_acumulacao(self):
        resultado = avaliar_renda(None, "etf_br")
        assert any("acumulação" in d for d in resultado.detalhes)


class TestLeituraCombinada:
    def test_renda_piorando_prevalece_sobre_timing_bom(self):
        assert leitura_combinada(pontos_timing=4, pontos_renda=-1).startswith("🔴")

    def test_preco_esticado_com_renda_ok_sugere_realizar(self):
        resultado = leitura_combinada(pontos_timing=-3, pontos_renda=1)
        assert resultado.startswith("🔵")

    def test_timing_muito_bom_sugere_comprar(self):
        resultado = leitura_combinada(pontos_timing=3, pontos_renda=1)
        assert resultado.startswith("🟢")

    def test_renda_boa_sem_sinal_forte_sugere_manter(self):
        resultado = leitura_combinada(pontos_timing=0, pontos_renda=2)
        assert "Renda saudável" in resultado
