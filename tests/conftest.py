"""Fixtures compartilhadas: séries de preço e dividendos sintéticas, no
mesmo espírito dos testes exploratórios feitos durante o desenvolvimento
do notebook original (sem depender de rede ou de dados reais)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _serie_base(n: int = 500, inicio: str = "2024-08-01", seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    datas = pd.bdate_range(inicio, periods=n)
    precos = 100 + np.cumsum(rng.normal(0, 0.3, n))
    return pd.Series(precos, index=datas, name="TESTE3.SA")


@pytest.fixture
def serie_precos_normal() -> pd.Series:
    """Série de preços "limpa", sem outliers nem mudanças de nível."""
    return _serie_base()


@pytest.fixture
def serie_com_tick_isolado() -> pd.Series:
    """Série com um erro pontual de poucos dias (tick ruim), cercada de
    dados normais — deve ser corrigida por interpolação local."""
    serie = _serie_base(seed=1)
    serie.iloc[150:153] = [0.137, 0.138, 0.137]
    return serie


@pytest.fixture
def serie_com_bloco_fora_de_escala() -> pd.Series:
    """Série com um bloco antigo extenso (370 de 500 dias) numa escala
    ~10x menor que o preço atual — simula um agrupamento de cotas não
    ajustado retroativamente pelo provedor de dados."""
    rng = np.random.default_rng(2)
    antigos = pd.Series(10.5 + np.cumsum(rng.normal(0, 0.03, 370)))
    recentes = pd.Series(100 + np.cumsum(rng.normal(0, 0.3, 130)))
    serie = pd.concat([antigos, recentes], ignore_index=True)
    serie.index = pd.bdate_range("2024-08-01", periods=len(serie))
    serie.name = "TESTE3.SA"
    return serie


@pytest.fixture
def serie_curta() -> pd.Series:
    """Série com menos de 30 dias — deve ser tratada como dado
    insuficiente pela camada de indicadores técnicos."""
    return _serie_base(n=10, seed=3)


@pytest.fixture
def dividendos_crescentes() -> pd.Series:
    """Histórico de dividendos mensais crescentes ano contra ano."""
    datas_antigas = pd.date_range("2023-08-01", periods=12, freq="MS")
    datas_recentes = pd.date_range("2024-08-01", periods=12, freq="MS")
    valores_antigos = np.full(12, 0.5)
    valores_recentes = np.full(12, 0.8)  # +60% a/a
    serie = pd.concat(
        [
            pd.Series(valores_antigos, index=datas_antigas),
            pd.Series(valores_recentes, index=datas_recentes),
        ]
    )
    return serie
