from __future__ import annotations

import datetime

import pandas as pd
import pytest

from carteira_analise.ganhos import (
    GanhoPosicaoAberta,
    Operacao,
    ResultadoOperacoes,
    avaliar_operacoes,
    calcular_ganho_posicao_aberta,
)


class TestCalcularGanhoPosicaoAberta:
    def test_ganho_de_preco_simples_sem_dividendos(self):
        r = calcular_ganho_posicao_aberta(
            quantidade=100, preco_medio=10.0, data_inicio="2024-01-01", preco_atual=12.0
        )
        assert r.valor_investido == 1000.0
        assert r.valor_atual == 1200.0
        assert r.ganho_preco == 200.0
        assert r.ganho_preco_pct == pytest.approx(0.20)
        assert r.renda_recebida == 0.0
        assert r.ganho_total == 200.0

    def test_perda_de_preco(self):
        r = calcular_ganho_posicao_aberta(
            quantidade=50, preco_medio=20.0, data_inicio="2024-01-01", preco_atual=18.0
        )
        assert r.ganho_preco == -100.0
        assert r.ganho_preco_pct == pytest.approx(-0.10)

    def test_dividendos_apos_data_inicio_sao_somados(self):
        datas = pd.to_datetime(["2024-02-01", "2024-06-01", "2024-11-01"])
        dividendos = pd.Series([0.5, 0.5, 0.5], index=datas)
        r = calcular_ganho_posicao_aberta(
            quantidade=100, preco_medio=10.0, data_inicio="2024-01-01",
            preco_atual=10.0, dividendos=dividendos,
        )
        # preço não mudou (ganho_preco=0), renda = 3 pagamentos de 0.5 * 100 cotas
        assert r.ganho_preco == 0.0
        assert r.renda_recebida == pytest.approx(150.0)
        assert r.ganho_total == pytest.approx(150.0)

    def test_dividendos_antes_da_data_inicio_nao_sao_contados(self):
        datas = pd.to_datetime(["2023-06-01", "2024-06-01"])  # um antes, um depois
        dividendos = pd.Series([0.5, 0.5], index=datas)
        r = calcular_ganho_posicao_aberta(
            quantidade=100, preco_medio=10.0, data_inicio="2024-01-01",
            preco_atual=10.0, dividendos=dividendos,
        )
        assert r.renda_recebida == pytest.approx(50.0)  # só o de 2024-06-01

    def test_sem_dividendos_informados_nao_quebra(self):
        r = calcular_ganho_posicao_aberta(
            quantidade=10, preco_medio=5.0, data_inicio="2024-01-01", preco_atual=6.0
        )
        assert r.renda_recebida == 0.0

    def test_quantidade_invalida_levanta_erro(self):
        with pytest.raises(ValueError):
            calcular_ganho_posicao_aberta(quantidade=0, preco_medio=10, data_inicio="2024-01-01", preco_atual=10)

    def test_preco_medio_invalido_levanta_erro(self):
        with pytest.raises(ValueError):
            calcular_ganho_posicao_aberta(quantidade=10, preco_medio=-5, data_inicio="2024-01-01", preco_atual=10)


class TestAvaliarOperacoes:
    def test_uma_compra_sem_venda(self):
        ops = [Operacao(data=datetime.date(2024, 1, 1), tipo="compra", quantidade=100, preco=10.0)]
        r = avaliar_operacoes(ops, preco_atual=12.0)
        assert r.quantidade_aberta == 100
        assert r.preco_medio_aberto == 10.0
        assert r.ganho_realizado == 0.0
        assert r.ganho_nao_realizado == pytest.approx(200.0)

    def test_compra_e_venda_total_com_lucro(self):
        ops = [
            Operacao(data=datetime.date(2024, 1, 1), tipo="compra", quantidade=100, preco=10.0),
            Operacao(data=datetime.date(2024, 6, 1), tipo="venda", quantidade=100, preco=15.0),
        ]
        r = avaliar_operacoes(ops)
        assert r.quantidade_aberta == 0
        assert r.preco_medio_aberto is None
        assert r.ganho_realizado == pytest.approx(500.0)
        assert r.ganho_nao_realizado is None

    def test_compra_e_venda_parcial(self):
        ops = [
            Operacao(data=datetime.date(2024, 1, 1), tipo="compra", quantidade=100, preco=10.0),
            Operacao(data=datetime.date(2024, 6, 1), tipo="venda", quantidade=40, preco=15.0),
        ]
        r = avaliar_operacoes(ops, preco_atual=14.0)
        assert r.quantidade_aberta == 60
        assert r.preco_medio_aberto == pytest.approx(10.0)
        assert r.ganho_realizado == pytest.approx((15.0 - 10.0) * 40)
        assert r.ganho_nao_realizado == pytest.approx((14.0 - 10.0) * 60)

    def test_custo_medio_ponderado_em_compras_multiplas(self):
        ops = [
            Operacao(data=datetime.date(2024, 1, 1), tipo="compra", quantidade=100, preco=10.0),
            Operacao(data=datetime.date(2024, 3, 1), tipo="compra", quantidade=100, preco=20.0),
        ]
        r = avaliar_operacoes(ops, preco_atual=20.0)
        # custo médio ponderado: (100*10 + 100*20) / 200 = 15
        assert r.preco_medio_aberto == pytest.approx(15.0)
        assert r.quantidade_aberta == 200

    def test_venda_maior_que_posicao_levanta_erro(self):
        ops = [
            Operacao(data=datetime.date(2024, 1, 1), tipo="compra", quantidade=50, preco=10.0),
            Operacao(data=datetime.date(2024, 6, 1), tipo="venda", quantidade=100, preco=15.0),
        ]
        with pytest.raises(ValueError):
            avaliar_operacoes(ops)

    def test_operacoes_fora_de_ordem_sao_processadas_cronologicamente(self):
        ops = [
            Operacao(data=datetime.date(2024, 6, 1), tipo="venda", quantidade=50, preco=15.0),
            Operacao(data=datetime.date(2024, 1, 1), tipo="compra", quantidade=100, preco=10.0),
        ]
        r = avaliar_operacoes(ops)  # não deve levantar erro, mesmo fora de ordem na lista
        assert r.quantidade_aberta == 50
        assert r.ganho_realizado == pytest.approx(250.0)

    def test_lista_vazia_levanta_erro(self):
        with pytest.raises(ValueError):
            avaliar_operacoes([])

    def test_operacao_com_tipo_invalido_levanta_erro(self):
        with pytest.raises(ValueError):
            Operacao(data=datetime.date(2024, 1, 1), tipo="transferencia", quantidade=10, preco=10.0)

    def test_ganho_total_soma_realizado_nao_realizado_e_renda(self):
        datas = pd.to_datetime(["2024-02-01"])
        dividendos = pd.Series([1.0], index=datas)
        ops = [
            Operacao(data=datetime.date(2024, 1, 1), tipo="compra", quantidade=100, preco=10.0),
            Operacao(data=datetime.date(2024, 6, 1), tipo="venda", quantidade=40, preco=15.0),
        ]
        r = avaliar_operacoes(ops, preco_atual=14.0, dividendos=dividendos)
        esperado = r.ganho_realizado + r.ganho_nao_realizado + r.renda_recebida
        assert r.ganho_total == pytest.approx(esperado)
