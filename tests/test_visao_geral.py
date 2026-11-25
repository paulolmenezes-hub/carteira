import pandas as pd
import pytest

from carteira_analise.visao_geral import (cdi_acumulado, fator_escala, linha_do_tempo_ativo, meses_entre,
                                          rentabilidade_cotas, resumo_periodo, somar)


def _op(data, tipo, q, p=0.0):
    return {"data": pd.Timestamp(data), "tipo": tipo, "quantidade": q, "preco": p}


MESES = meses_entre("2025-01", "2025-04")


def test_posicao_patrimonio_aportes_e_vendas():
    ops = [_op("2025-01-10", "compra", 100, 10), _op("2025-03-05", "venda", 40, 12)]
    precos = pd.Series([10.0, 11.0, 12.0, 13.0], index=pd.to_datetime(["2025-01-31", "2025-02-28",
                                                                       "2025-03-31", "2025-04-30"]))
    df = linha_do_tempo_ativo(ops, MESES, precos)
    assert list(df["quantidade"]) == [100, 100, 60, 60]
    assert list(df["patrimonio"]) == [1000, 1100, 720, 780]
    assert df["aportes"].sum() == 1000 and df["vendas"].sum() == 480
    assert df["custo"].iloc[-1] == pytest.approx(600)  # 60 ao preço médio de 10


def test_encerrada_conta_so_enquanto_aberta():
    ops = [_op("2025-01-10", "compra", 10, 10), _op("2025-02-10", "venda", 10, 15)]
    precos = pd.Series(20.0, index=pd.to_datetime(["2025-01-31", "2025-02-28", "2025-03-31"]))
    df = linha_do_tempo_ativo(ops, MESES, precos)
    assert list(df["patrimonio"]) == [200, 0, 0, 0]


def test_proventos_pela_quantidade_antes_da_data_ex():
    ops = [_op("2025-01-10", "compra", 100, 10), _op("2025-02-15", "compra", 100, 10)]
    divs = pd.Series([0.10, 0.10], index=pd.to_datetime(["2025-02-15", "2025-03-15"]))
    df = linha_do_tempo_ativo(ops, MESES, pd.Series(dtype=float), divs)
    assert list(df["proventos"]) == pytest.approx([0, 10, 20, 0])  # compra do dia da data-ex não recebe


def test_desdobramento_com_precos_ajustados_do_yahoo():
    # 100 cotas a R$ 100; desdobramento 1:10 em 15/03; Yahoo mostra todo o histórico em R$ 10
    ops = [_op("2025-01-10", "compra", 100, 100), _op("2025-03-15", "desdobramento", 900)]
    eventos = pd.Series([10.0], index=pd.to_datetime(["2025-03-15"]))
    precos = pd.Series(10.0, index=pd.to_datetime(["2025-01-31", "2025-02-28", "2025-03-31", "2025-04-30"]))
    divs = pd.Series([0.1, 0.1], index=pd.to_datetime(["2025-02-10", "2025-04-10"]))  # escala de hoje
    df = linha_do_tempo_ativo(ops, MESES, precos, divs, eventos)
    assert list(df["patrimonio"]) == pytest.approx([10000] * 4)  # sem salto no desdobramento
    assert list(df["proventos"]) == pytest.approx([0, 100, 0, 100])
    assert fator_escala(eventos, "2025-01-31") == 10 and fator_escala(eventos, "2025-03-31") == 1


def test_base_b3_ja_na_escala_da_epoca():
    ops = [_op("2025-01-10", "compra", 100, 100), _op("2025-03-15", "desdobramento", 900)]
    eventos = pd.Series([10.0], index=pd.to_datetime(["2025-03-15"]))
    brutos = pd.Series([100.0, 100.0, 10.0, 10.0], index=pd.to_datetime(["2025-01-31", "2025-02-28",
                                                                         "2025-03-31", "2025-04-30"]))
    df = linha_do_tempo_ativo(ops, MESES, brutos, eventos=eventos, precos_ajustados=False)
    assert list(df["patrimonio"]) == pytest.approx([10000] * 4)


def test_bonificacao_nao_e_aporte():
    ops = [_op("2025-01-10", "compra", 100, 10), _op("2025-02-10", "bonificacao", 10, 5)]
    df = linha_do_tempo_ativo(ops, MESES, pd.Series(dtype=float))
    assert df["aportes"].sum() == 1000 and df["custo"].iloc[-1] == 1050


def test_rentabilidade_ignora_aportes():
    # dobra o dinheiro com aporte e o preço não muda: rentabilidade zero
    df = pd.DataFrame({"patrimonio": [1000.0, 2000.0, 2200.0], "aportes": [1000.0, 1000.0, 0.0],
                       "vendas": [0.0, 0.0, 0.0], "proventos": [0.0, 0.0, 0.0]})
    r = rentabilidade_cotas(df)
    assert r.iloc[1] == pytest.approx(0.0) and r.iloc[2] == pytest.approx(0.10)


def test_rentabilidade_soma_proventos_e_encadeia():
    df = pd.DataFrame({"patrimonio": [1000.0, 1000.0], "aportes": [0.0, 0.0], "vendas": [0.0, 0.0],
                       "proventos": [10.0, 10.0]})
    r = rentabilidade_cotas(df, patrimonio_inicial=1000.0)
    assert r.iloc[-1] == pytest.approx(1.01 ** 2 - 1)


def test_somar_converte_dolar_e_filtra():
    idx = meses_entre("2025-01", "2025-02")
    base = pd.DataFrame({"quantidade": 1.0, "patrimonio": [100.0, 110.0], "custo": 100.0, "aportes": [100.0, 0],
                         "vendas": 0.0, "proventos": 0.0}, index=idx)
    total = somar({"A.SA": base, "IVV": base}, cambio={"IVV": pd.Series([5.0, 6.0], index=idx)})
    assert list(total["patrimonio"]) == [600, 770]
    assert list(somar({"A.SA": base, "IVV": base}, tickers=["A.SA"])["patrimonio"]) == [100, 110]


def test_cdi_acumulado_por_mes():
    dias = pd.bdate_range("2025-01-01", "2025-02-28")
    cdi = pd.Series(0.05, index=dias)  # 0,05% ao dia
    acc = cdi_acumulado(cdi, meses_entre("2025-01", "2025-02"))
    n_jan = sum(d.month == 1 for d in dias)
    assert acc.iloc[0] == pytest.approx(1.0005 ** n_jan - 1)
    assert acc.iloc[1] == pytest.approx(1.0005 ** len(dias) - 1)


def test_resumo_do_periodo():
    df = pd.DataFrame({"patrimonio": [1100.0, 1210.0], "aportes": [0.0, 0.0], "vendas": [0.0, 0.0],
                       "proventos": [5.0, 5.0]}, index=meses_entre("2025-01", "2025-02"))
    r = resumo_periodo(df, patrimonio_inicial=1000.0)
    assert r["proventos"] == 10 and r["proventos_mes"] == 5 and r["patrimonio_final"] == 1210
    assert r["rentabilidade"] == pytest.approx(1.105 * (1215 / 1100) - 1)


def test_compra_no_fim_do_mes_rende_sobre_o_preco_de_compra():
    # compra no dia 24 de julho a 100 e fechamento a 105: rentabilidade de 5% (o "meio do mês" dava 10%)
    ops = [_op("2019-07-24", "compra", 100, 100)]
    precos = pd.Series([105.0], index=pd.to_datetime(["2019-07-31"]))
    df = linha_do_tempo_ativo(ops, meses_entre("2019-07", "2019-07"), precos)
    assert rentabilidade_cotas(df).iloc[0] == pytest.approx(0.05)


def test_venda_rende_ate_o_preco_de_venda():
    # 100 cotas valendo 10 (V0 = 1000); vende 50 a 12 e as outras fecham a 12: +20%
    df = pd.DataFrame({"patrimonio": [600.0], "aportes": [0.0], "vendas": [600.0], "proventos": [0.0]})
    assert rentabilidade_cotas(df, patrimonio_inicial=1000.0).iloc[0] == pytest.approx(0.20)


def test_tir_de_aporte_unico_igual_a_rentabilidade():
    # aplica 1000 no início e tem 1100 depois de 12 meses: TIR 10% ao ano
    idx = meses_entre("2025-01", "2025-12")
    df = pd.DataFrame({"patrimonio": [1000.0] * 11 + [1100.0], "aportes": [0.0] * 12, "vendas": [0.0] * 12,
                       "proventos": [0.0] * 12}, index=idx)
    from carteira_analise.visao_geral import tir_anual
    assert tir_anual(df, patrimonio_inicial=1000.0) == pytest.approx(0.10, abs=1e-6)


def test_tir_pesa_o_dinheiro_e_cotas_nao():
    # 100 rendem 100% no 1º mês; depois aporta 10.000 que ficam parados: cotas +100%, TIR baixa
    from carteira_analise.visao_geral import resultado_em_reais, tir_anual
    df = pd.DataFrame({"patrimonio": [200.0, 10200.0], "aportes": [0.0, 10000.0], "vendas": [0.0, 0.0],
                       "proventos": [0.0, 0.0]}, index=meses_entre("2025-01", "2025-02"))
    assert rentabilidade_cotas(df, 100.0).iloc[-1] == pytest.approx(1.0)
    assert resultado_em_reais(df, 100.0) == pytest.approx(100.0)
    assert tir_anual(df, 100.0) > 0  # positiva, mas o ganho em R$ é só 100


def test_por_ano_encadeia_patrimonio_e_soma_proventos():
    from carteira_analise.visao_geral import por_ano
    idx = meses_entre("2024-12", "2025-01")
    df = pd.DataFrame({"patrimonio": [1100.0, 1210.0], "aportes": [0.0, 0.0], "vendas": [0.0, 0.0],
                       "proventos": [10.0, 0.0]}, index=idx)
    t = por_ano(df, 1000.0)
    assert list(t["ano"]) == [2024, 2025]
    assert t["rentabilidade"].iloc[0] == pytest.approx(0.11) and t["rentabilidade"].iloc[1] == pytest.approx(0.10)
    assert t["resultado"].iloc[0] == pytest.approx(110.0) and t["proventos"].sum() == 10


def test_indice_acumulado_e_por_ano():
    import pandas as pd
    from carteira_analise.visao_geral import indice_acumulado, indice_por_ano, meses_entre
    s = pd.Series([100.0, 110.0, 121.0, 99.0], index=pd.to_datetime(["2024-12-30", "2025-01-31", "2025-02-28",
                                                                     "2025-03-31"]))
    acc = indice_acumulado(s, meses_entre("2025-01", "2025-03"))
    assert [round(v, 4) for v in acc] == [0.10, 0.21, -0.01]
    assert indice_acumulado(s, meses_entre("2024-12", "2025-01")) is None  # sem base em nov/2024
    assert round(indice_por_ano(s, [2025], pd.Period("2025-03", "M"))[2025], 4) == -0.01
