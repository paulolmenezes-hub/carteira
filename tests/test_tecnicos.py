from __future__ import annotations

import numpy as np
import pandas as pd

from carteira_analise.tecnicos import (
    agrupar_datas_em_ciclos,
    calcular_indicadores_tecnicos,
    calcular_rsi,
    limpar_outliers_precos,
)


class TestCalcularRSI:
    def test_rsi_fica_entre_0_e_100(self, serie_precos_normal):
        rsi = calcular_rsi(serie_precos_normal)
        valores_validos = rsi.dropna()
        assert (valores_validos >= 0).all()
        assert (valores_validos <= 100).all()

    def test_serie_predominantemente_de_altas_gera_rsi_alto(self):
        # Uma série 100% monotônica (perda = 0) gera RSI = NaN pela própria
        # fórmula (divisão por zero) — por isso criamos uma única queda real
        # PERTO DO FIM (dentro da janela de 14 períodos do último valor),
        # que ainda deixa o RSI bem alto sem cair no caso degenerado.
        precos = pd.Series(np.arange(1, 51, dtype=float))
        precos.iloc[-5] -= 2  # queda de fato (não só um incremento menor)
        rsi = calcular_rsi(precos, janela=14)
        assert rsi.iloc[-1] > 90


class TestAgruparDatasEmCiclos:
    def test_datas_proximas_ficam_no_mesmo_ciclo(self):
        datas = pd.to_datetime(["2025-01-01", "2025-01-05", "2025-01-10"])
        ciclos = agrupar_datas_em_ciclos(list(datas))
        assert len(ciclos) == 1
        assert len(ciclos[0]) == 3

    def test_datas_distantes_formam_ciclos_separados(self):
        # Reproduz o padrão real do caso POMO4 discutido durante o desenvolvimento:
        # um bloco de mínimas em abril/2025 e outro isolado em agosto/2026.
        datas = pd.to_datetime(
            [
                "2025-04-04", "2025-04-07", "2025-04-08", "2025-04-09",
                "2025-04-10", "2025-04-11", "2025-04-14",
                "2026-08-05", "2026-08-06", "2026-08-07",
            ]
        )
        ciclos = agrupar_datas_em_ciclos(list(datas), limite_dias=45)
        assert len(ciclos) == 2
        assert len(ciclos[0]) == 7
        assert len(ciclos[1]) == 3

    def test_lista_vazia_nao_quebra(self):
        assert agrupar_datas_em_ciclos([]) == []


class TestLimparOutliersPrecos:
    def test_serie_normal_nao_e_alterada_significativamente(self, serie_precos_normal):
        limpa, n_corrigidos, n_truncados = limpar_outliers_precos(serie_precos_normal)
        assert n_corrigidos == 0
        assert n_truncados == 0
        assert len(limpa) == len(serie_precos_normal)

    def test_tick_isolado_e_corrigido_por_interpolacao(self, serie_com_tick_isolado):
        assert serie_com_tick_isolado.min() < 1  # confirma que o outlier está lá
        limpa, n_corrigidos, n_truncados = limpar_outliers_precos(serie_com_tick_isolado)
        assert n_truncados == 0  # não é um bloco, é ponto isolado
        assert n_corrigidos >= 3
        assert limpa.min() > 50  # outlier removido, série volta à faixa normal

    def test_bloco_antigo_fora_de_escala_e_truncado_nao_interpolado(
        self, serie_com_bloco_fora_de_escala
    ):
        limpa, n_corrigidos, n_truncados = limpar_outliers_precos(serie_com_bloco_fora_de_escala)
        # O bloco de 370 dias antigos deve ser descartado (truncado), não "consertado"
        assert n_truncados >= 300
        assert len(limpa) < len(serie_com_bloco_fora_de_escala)
        # o que sobrou deve estar todo na escala atual (~100), não na antiga (~10)
        assert limpa.min() > 50


class TestCalcularIndicadoresTecnicos:
    def test_serie_curta_retorna_none(self, serie_curta):
        assert calcular_indicadores_tecnicos(serie_curta) is None

    def test_serie_normal_retorna_indicadores_coerentes(self, serie_precos_normal):
        tec = calcular_indicadores_tecnicos(serie_precos_normal)
        assert tec is not None
        assert tec.preco_atual == serie_precos_normal.iloc[-1]
        assert tec.dias_usados == len(serie_precos_normal)
        assert tec.n_outliers_corrigidos == 0
        assert tec.n_dias_truncados == 0
        # sma200 deve existir pois a série tem 500 dias (>= 200)
        assert tec.sma200 is not None

    def test_distancia_min_max_usa_percentil_nao_extremos_absolutos(
        self, serie_com_tick_isolado
    ):
        # Regressão do bug original: um único tick isolado a 0.137 não pode
        # dominar o cálculo de distância da mínima do período.
        tec = calcular_indicadores_tecnicos(serie_com_tick_isolado)
        assert tec is not None
        assert abs(tec.dist_minima_pct) < 50  # não deve disparar pra milhares de %

    def test_bloco_fora_de_escala_nao_distorce_distancia(
        self, serie_com_bloco_fora_de_escala
    ):
        tec = calcular_indicadores_tecnicos(serie_com_bloco_fora_de_escala)
        assert tec is not None
        assert abs(tec.dist_minima_pct) < 50
        assert tec.n_dias_truncados > 0


class TestLimpezaNaoApagaQuedasReais:
    """Regressão (Sprint 4): a limpeza descartava qualquer trecho antigo 60%
    acima/abaixo do preço atual, apagando a história de ativos que caíram de
    verdade (ex.: FIIs de recebíveis com inadimplência) — viés de sobrevivência."""

    def _serie(self, valores):
        s = pd.Series(valores, dtype=float)
        s.index = pd.bdate_range("2021-01-04", periods=len(s))
        return s

    def test_queda_gradual_de_80_por_cento_e_mantida(self):
        rng = np.random.default_rng(5)
        tendencia = np.linspace(100, 20, 1000)
        serie = self._serie(tendencia * (1 + rng.normal(0, 0.01, 1000)))
        limpa, _, n_truncados = limpar_outliers_precos(serie)
        assert n_truncados == 0 and len(limpa) == 1000

    def test_alta_gradual_de_300_por_cento_e_mantida(self):
        serie = self._serie(np.linspace(10, 40, 800))
        assert limpar_outliers_precos(serie)[2] == 0

    def test_grupamento_10_para_1_abrupto_e_descartado(self):
        rng = np.random.default_rng(6)
        antes = 9 + rng.normal(0, 0.05, 600)
        depois = 90 + rng.normal(0, 0.5, 400)
        limpa, _, n_truncados = limpar_outliers_precos(self._serie(np.concatenate([antes, depois])))
        assert 600 <= n_truncados <= 615
        assert limpa.min() > 80

    def test_queda_real_seguida_de_grupamento(self):
        # cota cai de 50 para 10 ao longo de 2 anos (real) e depois agrupa 10:1
        queda = np.linspace(50, 10, 500)
        depois = np.full(300, 100.0)
        limpa, _, n_truncados = limpar_outliers_precos(self._serie(np.concatenate([queda, depois])))
        assert n_truncados >= 500 and limpa.min() > 80

    def test_serie_curta_sem_janela_suficiente(self):
        assert limpar_outliers_precos(self._serie([10.0] * 15))[2] == 0
