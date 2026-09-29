from __future__ import annotations

import numpy as np
import pandas as pd

from carteira_analise.leitura import alerta_minima_com_rsi_baixo, leitura_automatica
from carteira_analise.tecnicos import calcular_indicadores_tecnicos


def _serie_com_dois_ciclos_de_minima():
    """Reproduz o caso real do POMO4: mínima em abril/2025, recuperação até
    setembro/2025, e nova mínima em agosto/2026."""
    datas = pd.bdate_range("2024-08-01", periods=500)
    rng = np.random.default_rng(7)
    precos = 6.0 + np.cumsum(rng.normal(0, 0.02, 500))
    serie = pd.Series(precos, index=datas)
    # força um vale reconhecível perto do índice 170 (~abr/2025) e outro no fim
    serie.iloc[165:172] = np.linspace(5.4, 5.2, 7)
    serie.iloc[-5:] = np.linspace(5.15, 5.1, 5)
    return serie


def _serie_preco_no_meio_do_range():
    # Construída deliberadamente (não um passeio aleatório puro) para GARANTIR
    # que o preço final fique no meio do range: sobe, desce até um fundo bem
    # definido, e recupera parcialmente até um valor claramente intermediário
    # entre o mínimo e o máximo do período.
    subida = np.linspace(40, 60, 170)
    descida = np.linspace(60, 20, 170)
    recuperacao_parcial = np.linspace(20, 40, 160)
    precos = np.concatenate([subida, descida, recuperacao_parcial])
    rng = np.random.default_rng(11)
    precos = precos + rng.normal(0, 0.1, len(precos))  # ruído pequeno, não muda a forma
    datas = pd.bdate_range("2024-08-01", periods=len(precos))
    return pd.Series(precos, index=datas)


class TestLeituraAutomatica:
    def test_preco_no_meio_do_range_da_leitura_tranquila(self):
        serie = _serie_preco_no_meio_do_range()
        tec = calcular_indicadores_tecnicos(serie)
        assert tec is not None
        texto = leitura_automatica(serie, tec)
        assert "faixa intermediária" in texto
        assert "Nada de especial" in texto

    def test_leitura_nao_quebra_com_serie_minima_valida(self):
        serie = _serie_com_dois_ciclos_de_minima()
        tec = calcular_indicadores_tecnicos(serie)
        assert tec is not None
        texto = leitura_automatica(serie, tec)
        assert isinstance(texto, str)
        assert len(texto) > 0


class TestAlertaMinimaComRSIBaixo:
    def test_sem_alerta_quando_preco_nao_esta_perto_da_minima(self):
        serie = _serie_preco_no_meio_do_range()
        tec = calcular_indicadores_tecnicos(serie)
        assert tec is not None
        assert alerta_minima_com_rsi_baixo(tec) is None

    def test_alerta_dispara_com_minima_e_rsi_baixo_simultaneos(self):
        from carteira_analise.tecnicos import IndicadoresTecnicos

        tec = IndicadoresTecnicos(
            preco_atual=4.5,
            sma50=5.0,
            sma200=6.0,
            rsi14=28.0,
            dist_maxima_pct=-38.0,
            dist_minima_pct=-11.0,
            n_outliers_corrigidos=0,
            n_dias_truncados=0,
            dias_usados=500,
            historico=pd.Series([4.5]),
        )
        alerta = alerta_minima_com_rsi_baixo(tec)
        assert alerta is not None
        assert "RSI baixo" in alerta
