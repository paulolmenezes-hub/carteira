import numpy as np
import pandas as pd
import pytest

from carteira_analise.backtest_cenarios import (
    ESTRATEGIAS_FII,
    VARIANTES_ACOES,
    CaixaRendaFixa,
    aliquota_iof,
    aliquota_ir_renda_fixa,
    datas_de_entrada,
    frase_evidencia,
    indice_cdi_acumulado,
    resumir_cenarios,
    rodar_em_varias_datas,
    simular_cenario_regra,
    simular_rendimentos_fii,
)


def serie(valores, inicio="2021-01-04"):
    return pd.Series(valores, index=pd.bdate_range(inicio, periods=len(valores)), dtype=float)


# ---------------------------------------------------------------- tributação
@pytest.mark.parametrize("dias,aliq", [(1, .225), (180, .225), (181, .20), (360, .20), (361, .175),
                                       (720, .175), (721, .15), (3000, .15)])
def test_ir_regressivo(dias, aliq):
    assert aliquota_ir_renda_fixa(dias) == aliq


def test_iof_regressivo():
    assert aliquota_iof(0) == 0 and aliquota_iof(1) == .96 and aliquota_iof(29) == .03
    assert aliquota_iof(30) == 0 and aliquota_iof(400) == 0


# ---------------------------------------------------------------- índice do CDI
def test_indice_cdi_com_serie_do_bc_e_feriados():
    cdi = pd.Series(0.05, index=pd.bdate_range("2024-01-01", periods=10))  # 0,05% a.d.
    datas = pd.DatetimeIndex(["2024-01-01", "2024-01-03", "2024-01-12"])
    ind = indice_cdi_acumulado(datas, cdi)
    assert ind.iloc[1] / ind.iloc[0] == pytest.approx(1.0005 ** 2)
    assert ind.iloc[2] / ind.iloc[0] == pytest.approx(1.0005 ** 9)


def test_indice_cdi_taxa_constante_e_sem_serie():
    datas = pd.bdate_range("2024-01-01", periods=253)
    ind = indice_cdi_acumulado(datas, None, cdi_aa=0.10)
    assert ind.iloc[-1] / ind.iloc[0] == pytest.approx(1.10)
    assert indice_cdi_acumulado(datas, pd.Series(dtype=float)).iloc[-1] == 1.0


def test_indice_cdi_com_fuso():
    cdi = pd.Series(0.05, index=pd.bdate_range("2024-01-01", periods=5, tz="UTC"))
    assert indice_cdi_acumulado(pd.bdate_range("2024-01-01", periods=5), cdi).iloc[-1] > 1


# ---------------------------------------------------------------- caixa lote a lote
def test_caixa_ir_por_lote():
    c = CaixaRendaFixa()
    c.depositar("2024-01-01", 1000, 1.0)
    c.depositar("2024-11-01", 1000, 1.0)  # índice igual: lote novo ainda sem rendimento
    # avaliação em 2025-01-01, índice 1.10 (10% para os dois lotes, por simplicidade)
    v = c.valor_liquido("2025-01-01", 1.10)
    assert v == pytest.approx(1000 + 100 * (1 - .175) + 1000 + 100 * (1 - .225))


def test_caixa_iof_abaixo_de_30_dias():
    c = CaixaRendaFixa(); c.depositar("2024-01-01", 1000, 1.0)
    assert c.valor_liquido("2024-01-11", 1.01) == pytest.approx(1000 + 10 * (1 - .66) * (1 - .225))


def test_caixa_resgate_fifo_e_parcial():
    c = CaixaRendaFixa()
    c.depositar("2023-01-01", 100, 1.0); c.depositar("2024-01-01", 100, 1.0)
    total = c.valor_liquido("2024-06-01", 1.0)
    assert total == pytest.approx(200)
    assert c.resgatar_liquido("2024-06-01", 150, 1.0) == pytest.approx(150)
    assert len(c.lotes) == 1 and c.lotes[0]["principal"] == pytest.approx(50)
    assert c.resgatar_liquido("2024-06-01", 500, 1.0) == pytest.approx(50)  # só o que existe
    assert c.lotes == [] and c.valor_liquido("2024-06-01", 1.0) == 0


def test_caixa_ignora_deposito_zero():
    c = CaixaRendaFixa(); c.depositar("2024-01-01", 0, 1.0)
    assert c.lotes == []


# ---------------------------------------------------------------- datas de entrada
def test_datas_de_entrada():
    d = datas_de_entrada(1252, 13)
    assert len(d) == 13 and d[0] == 252 and d[-1] == 1000
    assert datas_de_entrada(400, 13) == []


def test_datas_de_entrada_historico_curto_tem_menos_datas():
    d = datas_de_entrada(252 + 252 + 100, 13)  # janela de 100 pregões
    assert len(d) == 5 and all(b - a >= 21 for a, b in zip(d, d[1:]))
    assert datas_de_entrada(504, 13) == [252]  # janela de um único dia


# ---------------------------------------------------------------- ações: variantes
def test_variantes_isolam_cenarios_em_alta():
    s = serie([100.0] * 260 + list(np.linspace(100, 220, 120)))
    manter = simular_cenario_regra(s, variante="Manter")
    realizar = simular_cenario_regra(s, variante="Só realizar lucro nas faixas")
    aumentar = simular_cenario_regra(s, variante="Só aumentar nas quedas")
    assert manter["n_vendas"] == 0 and realizar["n_vendas"] >= 4 and realizar["n_compras"] == 0
    assert aumentar["n_compras"] == 0 and aumentar["valor_final"] == pytest.approx(manter["valor_final"])
    assert realizar["valor_de_100"] < manter["valor_de_100"]  # alta contínua: realizar rende menos


def test_variante_so_aumentar_em_queda():
    s = serie([100.0] * 260 + list(np.linspace(100, 60, 120)))
    realizar = simular_cenario_regra(s, variante="Só realizar lucro nas faixas")
    aumentar = simular_cenario_regra(s, variante="Só aumentar nas quedas")
    assert realizar["n_compras"] == 0 and realizar["n_vendas"] == 0
    assert aumentar["n_compras"] == 2 and aumentar["aportado"] > 10000


def test_regra_completa_e_caixa_com_cdi():
    s = serie([100.0] * 260 + list(np.linspace(100, 220, 120)) + [220.0] * 300)
    ind = indice_cdi_acumulado(s.index, None, cdi_aa=0.12)
    sem = simular_cenario_regra(s, variante="Regra completa")
    com = simular_cenario_regra(s, variante="Regra completa", indice_cdi=ind)
    assert com["valor_final"] > sem["valor_final"]


def test_proventos_vao_para_caixa_nas_duas_estrategias():
    s = serie([100.0] * 400)
    d = pd.Series([1.0], index=[s.index[300]])
    r = simular_cenario_regra(s, d, variante="Manter")
    assert r["valor_final"] == pytest.approx(10000 + 100)


def test_serie_curta():
    assert simular_cenario_regra(serie([100.0] * 260)) is None
    assert simular_rendimentos_fii(serie([100.0] * 260)) is None


# ---------------------------------------------------------------- FIIs
def _fii(precos, div_mensal=1.0):
    s = serie(precos)
    datas = pd.date_range(s.index[0], s.index[-1], freq="MS")
    return s, pd.Series(div_mensal, index=datas)


def test_reinvestimento_preco_constante_patrimonio_parecido():
    s, d = _fii([100.0] * 600)
    sem = simular_rendimentos_fii(s, d, "Não reinvestir")
    cem = simular_rendimentos_fii(s, d, "Reinvestir 100%")
    assert cem["cotas_finais"] > sem["cotas_finais"]
    assert cem["renda_mensal_final"] > sem["renda_mensal_final"]
    assert cem["valor_final"] > sem["valor_final"]  # sem CDI, as cotas novas rendem mais


def test_cdi_alto_pode_superar_reinvestir_em_queda():
    s, d = _fii([100.0] * 252 + list(np.linspace(100, 80, 348)))
    ind = indice_cdi_acumulado(s.index, None, cdi_aa=0.14)
    sem = simular_rendimentos_fii(s, d, "Não reinvestir", indice_cdi=ind)
    cem = simular_rendimentos_fii(s, d, "Reinvestir 100%", indice_cdi=ind)
    assert sem["valor_final"] > cem["valor_final"]
    assert cem["renda_mensal_final"] > sem["renda_mensal_final"]


def test_reinvestir_compra_so_cotas_inteiras():
    s, d = _fii([100.0] * 600, div_mensal=0.3)  # 100 cotas × 0,30 = R$ 30/mês
    r = simular_rendimentos_fii(s, d, "Reinvestir 100%")
    assert abs(r["cotas_finais"] - round(r["cotas_finais"])) < 1e-9 or r["cotas_finais"] % 1 == pytest.approx(0)
    assert r["caixa_final"] < 100  # sobra parada abaixo de uma cota


def test_cdi_compra_na_queda_dispara_e_nao_dispara():
    s, d = _fii([100.0] * 252 + list(np.linspace(100, 70, 200)) + [70.0] * 200)
    pm = simular_rendimentos_fii(s, d, "CDI + compra na queda (preço médio)")
    mx = simular_rendimentos_fii(s, d, "CDI + compra na queda (máxima de 12 meses)")
    assert pm["houve_gatilho"] and pm["n_compras"] >= 1
    assert mx["houve_gatilho"]
    s2, d2 = _fii([100.0] * 600)
    nada = simular_rendimentos_fii(s2, d2, "CDI + compra na queda (preço médio)")
    sem = simular_rendimentos_fii(s2, d2, "Não reinvestir")
    assert not nada["houve_gatilho"] and nada["valor_final"] == pytest.approx(sem["valor_final"])


def test_estrategias_cadastradas():
    assert len(ESTRATEGIAS_FII) == 5 and len(VARIANTES_ACOES) == 4


# ---------------------------------------------------------------- várias datas e resumo
def test_rodar_e_resumir():
    s1 = serie([100.0] * 260 + list(np.linspace(100, 220, 400)) + [220.0] * 600)
    s2 = serie([100.0] * 260 + list(np.linspace(100, 60, 400)) + [60.0] * 600)
    nomes = list(VARIANTES_ACOES)
    res = {"A": rodar_em_varias_datas(simular_cenario_regra, s1, None, nomes),
           "B": rodar_em_varias_datas(simular_cenario_regra, s2, None, nomes)}
    assert len(res["A"]["Manter"]) == 13
    df = resumir_cenarios(res, {"A": "Ações B3", "B": "Ações B3"}, "Manter")
    linha = df[df["cenario"] == "Só realizar lucro nas faixas"].iloc[0]
    assert linha["ativos"] == 2 and linha["casos"] == 26 and 0 <= linha["pct_supera_referencia"] <= 100
    assert pd.isna(df[df["cenario"] == "Manter"].iloc[0]["pct_supera_referencia"])
    frase = frase_evidencia(df, "Ações B3", "Só realizar lucro nas faixas", "Manter", "ações B3")
    assert "em 2 ações B3" in frase and "R$ 100" in frase and "% dos casos" in frase
    assert frase_evidencia(df, "FIIs", "x", "Manter", "FIIs") is None
    so_a = resumir_cenarios({"A": res["A"]}, {"A": "Ações EUA"}, "Manter")
    assert "em 1 ação dos EUA," in frase_evidencia(so_a, "Ações EUA", "Regra completa", "Manter",
                                                   "ações dos EUA", "US$", "ação dos EUA")


def test_resumo_fii_com_gatilho():
    s, d = _fii([100.0] * 252 + list(np.linspace(100, 70, 500)) + [70.0] * 500)
    nomes = list(ESTRATEGIAS_FII)
    res = {"F": rodar_em_varias_datas(simular_rendimentos_fii, s, d, nomes)}
    df = resumir_cenarios(res, {"F": "FIIs"}, "Não reinvestir")
    g = df[df["cenario"] == "CDI + compra na queda (preço médio)"].iloc[0]
    assert not pd.isna(g["pct_com_compra_na_queda"])
    assert pd.isna(df[df["cenario"] == "Reinvestir 100%"].iloc[0]["pct_com_compra_na_queda"])
    frase = frase_evidencia(df, "FIIs", "CDI + compra na queda (preço médio)", "Não reinvestir", "FIIs", "R$", "FII")
    assert "\"CDI + compra na queda (preço médio)\"" in frase and "em 1 FII," in frase
