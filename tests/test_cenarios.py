import json

import numpy as np
import pandas as pd
import pytest

from carteira_analise.backtest_cenarios import (
    ESTRATEGIAS_FII,
    VARIANTES_ACOES,
    exportar_evidencias,
    indice_cdi_acumulado,
    resumir_cenarios,
    rodar_em_varias_datas,
    simular_cenario_regra,
    simular_rendimentos_fii,
)
from carteira_analise.cenarios import (
    carregar_evidencias,
    distancia_da_maxima_12m,
    frase_evidencia_card,
    frase_evidencia_fiis,
    gerar_cenarios,
    precos_das_faixas,
    retrospecto_reinvestimento,
    situacao_por_variacao,
    tabela_cenario,
)
from carteira_analise.regra_posicao import decidir_posicao, novo_estado, quantidade_para_acao


# ---------------------------------------------------------------- situação e faixas
@pytest.mark.parametrize("var,sit", [(0.25, "realizacao"), (0.76, "realizacao"), (0.249, "fora"),
                                     (-0.149, "fora"), (-0.15, "queda"), (-0.37, "queda"), (0.0, "fora")])
def test_situacao_por_variacao(var, sit):
    assert situacao_por_variacao(var) == sit


def test_precos_das_faixas():
    f = precos_das_faixas(100.0)
    assert f["realizacao"] == pytest.approx(125.0) and f["queda"] == pytest.approx(85.0)


def test_distancia_da_maxima_12m():
    datas = pd.bdate_range("2025-01-01", periods=300)
    p = pd.Series(np.r_[np.linspace(90, 100, 150), np.linspace(100, 94, 150)], index=datas)
    assert distancia_da_maxima_12m(p) == pytest.approx(0.06)
    assert distancia_da_maxima_12m(pd.Series(dtype=float)) is None


# ---------------------------------------------------------------- tabela padronizada
def _linhas(t):
    return {nome: (v1, v2) for nome, _, v1, v2 in t["linhas"]}


def test_tabela_realizar_40_como_no_prototipo():
    e = novo_estado(500, 8.0)
    d = decidir_posicao(14.07, e)
    t = tabela_cenario(e, 14.07, d, quantidade_para_acao(e, d), renda_12m_por_cota=1.1278)
    assert t["colunas"] == ["Manter tudo", "Realizar 40%"]
    assert t["titulo"] == "Cenários na faixa de +60%: realizar 40% da posição"
    l = _linhas(t)
    assert [nome for nome, *_ in t["linhas"]] == [
        "Valor investido (custo)", "Valor da posição hoje", "Movimento de caixa", "Resultado realizado",
        "Resultado não realizado", "Preço médio", "Variação sobre o preço médio", "Renda estimada/mês"]
    assert l["Valor investido (custo)"] == pytest.approx((4000, 2400))
    assert l["Valor da posição hoje"] == pytest.approx((7035, 4221))
    assert l["Movimento de caixa"][0] is None and l["Movimento de caixa"][1] == ("venda", pytest.approx(2814))
    assert l["Resultado realizado"][1] == pytest.approx(1214)
    assert l["Resultado não realizado"] == pytest.approx((3035, 1821))
    assert l["Preço médio"] == pytest.approx((8, 8))
    assert l["Renda estimada/mês"][1] == pytest.approx(1.1278 / 12 * 300)


def test_tabela_aumentar_25_como_no_prototipo():
    e = novo_estado(120, 6.5)
    d = decidir_posicao(4.11, e)
    t = tabela_cenario(e, 4.11, d, quantidade_para_acao(e, d))
    l = _linhas(t)
    assert t["colunas"] == ["Manter tudo", "Aumentar 25%"]
    assert t["titulo"] == "Cenários na faixa de −25%: aumentar 25% da posição"
    assert l["Valor investido (custo)"] == pytest.approx((780, 903.3))
    assert l["Movimento de caixa"][1] == ("aporte", pytest.approx(-123.3))
    assert l["Resultado não realizado"] == pytest.approx((-286.8, -286.8))
    assert l["Preço médio"][1] == pytest.approx(903.3 / 150)
    assert l["Renda estimada/mês"] == (None, None)


def test_tabela_venda_total():
    e = novo_estado(24.11, 36.67)
    d = decidir_posicao(77.73, e)
    t = tabela_cenario(e, 77.73, d, quantidade_para_acao(e, d, permitir_fracionario=True))
    l = _linhas(t)
    assert t["colunas"][1] == "Venda total" and "venda total" in t["titulo"]
    assert l["Valor da posição hoje"][1] is None and l["Preço médio"][1] is None
    assert l["Resultado realizado"][1] == pytest.approx((77.73 - 36.67) * 24.11)


def test_tabela_sem_cenario():
    e = novo_estado(100, 100.0)
    assert tabela_cenario(e, 100.0, decidir_posicao(100.0, e), 0) is None
    d = decidir_posicao(130.0, e)
    assert tabela_cenario(e, 130.0, d, 0) is None


# ---------------------------------------------------------------- retrospecto dos FIIs
def _fii(precos, div=1.0, inicio="2024-06-03"):
    datas = pd.bdate_range(inicio, periods=len(precos))
    s = pd.Series(precos, index=datas, dtype=float)
    d = pd.Series(div, index=pd.date_range(datas[0], datas[-1], freq="MS"))
    return s, d


def test_retrospecto_preco_constante_reinvestir_aumenta_cotas_e_renda():
    s, d = _fii([100.0] * 400)
    ops = [{"data": s.index[0], "tipo": "compra", "quantidade": 100, "preco": 100.0}]
    r = retrospecto_reinvestimento(ops, s, d)
    lin = {nome: vals for nome, _, *vals in r["linhas"]}
    assert lin["Cotas hoje"][0] == 100 and lin["Cotas hoje"][2] > lin["Cotas hoje"][1] > 100
    assert lin["Renda mensal hoje"][2] > lin["Renda mensal hoje"][0]
    assert r["meses"] == 12 and r["colunas"] == ["Real", "Reinvest. 50%", "Reinvest. 100%"]
    assert not r["compra_na_queda"]["atingida"]
    assert r["compra_na_queda"]["reserva"] == pytest.approx(lin["Em caixa (CDI líquido)"][0])


def test_retrospecto_com_cdi_e_queda():
    s, d = _fii(list(np.linspace(100, 80, 400)))
    ops = [{"data": s.index[0], "tipo": "compra", "quantidade": 100, "preco": 100.0}]
    ind = indice_cdi_acumulado(s.index, None, cdi_aa=0.14)
    r = retrospecto_reinvestimento(ops, s, d, ind)
    lin = {nome: vals for nome, _, *vals in r["linhas"]}
    assert lin["Patrimônio hoje"][0] > lin["Patrimônio hoje"][2]  # cota caiu e o CDI rendeu
    q = r["compra_na_queda"]
    assert q["atingida"] and q["cotas"] > 100 and q["preco_faixa"] == pytest.approx(85.0)


def test_retrospecto_compra_recente_e_operacoes_no_periodo():
    s, d = _fii([100.0] * 400)
    ops = [{"data": s.index[300], "tipo": "compra", "quantidade": 50, "preco": 100.0},
           {"data": s.index[350], "tipo": "compra", "quantidade": 50, "preco": 100.0},
           {"data": s.index[380], "tipo": "venda", "quantidade": 20, "preco": 100.0}]
    r = retrospecto_reinvestimento(ops, s, d)
    assert r["inicio"] == s.index[300] and r["meses"] < 12
    assert r["linhas"][0][2] == 80  # cotas reais hoje


def test_retrospecto_sem_dados():
    s, d = _fii([100.0] * 10)
    assert retrospecto_reinvestimento([], s, d) is None
    assert retrospecto_reinvestimento([{"data": s.index[-1], "tipo": "compra", "quantidade": 1, "preco": 1.0}],
                                      s, d) is None


# ---------------------------------------------------------------- evidências
def test_evidencias_do_repositorio_e_frases():
    e = carregar_evidencias()
    assert e["fiis"]["ativos"] == 28 and "Ações B3" in e["acoes"]
    f = frase_evidencia_card(e, "acao", "realizacao")
    assert "72%" in f and "manter rendeu mais" in f
    assert "65%" in frase_evidencia_card(e, "acao", "queda")
    assert "só 27%" in frase_evidencia_card(e, "etf_us", "realizacao")
    assert "nunca foi atingida" in frase_evidencia_card(e, "acao_us", "queda")
    assert frase_evidencia_card(e, "fii", "realizacao") is None
    assert frase_evidencia_card(e, "acao", "fora") is None and frase_evidencia_card(None, "acao", "queda") is None
    assert "84%" in frase_evidencia_fiis(e) and "empatou na prática" in frase_evidencia_fiis(e)
    assert frase_evidencia_fiis(None) is None


def test_frase_fii_leituras(tmp_path):
    e = carregar_evidencias()
    e["fiis"]["compra_na_queda"]["pct_vence_quando_agiu"] = 70
    assert "foi melhor na maioria" in frase_evidencia_fiis(e)
    e["fiis"]["compra_na_queda"]["pct_vence_quando_agiu"] = 20
    assert "foi pior na maioria" in frase_evidencia_fiis(e)
    assert carregar_evidencias(tmp_path / "nao_existe.json") is None


def test_exportar_evidencias_gera_o_formato_lido_pelo_painel(tmp_path):
    datas = pd.bdate_range("2021-01-04", periods=1250)
    s = pd.Series(100 * np.exp(np.linspace(0, 0.8, 1250) + 0.1 * np.sin(np.arange(1250) / 40)), index=datas)
    d = pd.Series(0.8, index=pd.date_range(datas[0], datas[-1], freq="MS"))
    ra = {"A": rodar_em_varias_datas(simular_cenario_regra, s, None, list(VARIANTES_ACOES))}
    rf = {"F": rodar_em_varias_datas(simular_rendimentos_fii, s, d, list(ESTRATEGIAS_FII))}
    dfa = resumir_cenarios(ra, {"A": "Ações B3"}, "Manter")
    dff = resumir_cenarios(rf, {"F": "FIIs"}, "Não reinvestir")
    ev = exportar_evidencias(dfa, dff, "2022 e 2026", "2026-10-01", "teste")
    arq = tmp_path / "ev.json"
    arq.write_text(json.dumps(ev), encoding="utf-8")
    lido = carregar_evidencias(arq)
    assert lido["acoes"]["Ações B3"]["ativos"] == 1 and "realizar" in lido["acoes"]["Ações B3"]
    assert frase_evidencia_card(lido, "acao", "realizacao") is not None
    assert frase_evidencia_fiis(lido) is not None
    lido["fiis"]["compra_na_queda"]["pct_vence_quando_agiu"] = None
    assert "não foi atingida" in frase_evidencia_fiis(lido)
    assert exportar_evidencias(pd.DataFrame(), pd.DataFrame(), "p", "g", "f")["fiis"] is None


# ---------------------------------------------------------------- cards completos
class FonteFalsa:
    def __init__(self, precos, dividendos):
        self.precos, self.dividendos = precos, dividendos

    def baixar_precos(self, ticker, periodo):
        if ticker == "ERRO3.SA":
            raise RuntimeError("rede")
        return self.precos.get(ticker, pd.Series(dtype=float))

    def baixar_dividendos(self, ticker):
        if ticker == "SEMD3.SA":
            raise RuntimeError("rede")
        return self.dividendos.get(ticker, pd.Series(dtype=float))


def test_gerar_cenarios_carteira_completa():
    datas = pd.bdate_range(end="2026-09-30", periods=700)
    def linha(a, b): return pd.Series(np.linspace(a, b, 700), index=datas)
    precos = {"ITSA4.SA": linha(9, 14.07), "POMO4.SA": linha(7, 4.11), "VALE3.SA": linha(66, 69.45),
              "KNIP11.SA": linha(92, 89.30), "IAU": linha(40, 77.73), "SEMD3.SA": linha(10, 10)}
    mens = pd.date_range(datas[0], datas[-1], freq="MS")
    divs = {"KNIP11.SA": pd.Series(0.84, index=mens), "ITSA4.SA": pd.Series(0.09, index=mens)}
    resumo = pd.DataFrame([[t, q, pm, None, "2023-01-02"] for t, q, pm in
                           [("ITSA4", 500, 8.0), ("POMO4", 120, 6.5), ("VALE3", 200, 68.5), ("KNIP11", 38, 90.96),
                            ("IAU", 24.11, 36.67), ("SEMD3", 10, 10.0), ("ERRO3", 1, 1.0), ("SUMI3", 1, 1.0)]],
                          columns=["ticker", "quantidade", "preco_medio", "valor_investido", "data_inicio"])
    tipos = {"ITSA4.SA": "acao", "POMO4.SA": "acao", "VALE3.SA": "acao", "KNIP11.SA": "fii", "IAU": "etf_us"}
    cards, avisos = gerar_cenarios(resumo, None, FonteFalsa(precos, divs), tipos,
                                   {"POMO4.SA": {"divida_bruta_patrimonio": 2.0}},
                                   indice_cdi_por_data=lambda idx: indice_cdi_acumulado(idx, None, 0.14),
                                   evidencias=carregar_evidencias())
    por = {c["ticker"]: c for c in cards}
    # ordem: B3 (Ações, FIIs, Outros) e depois EUA
    assert [c["ticker"] for c in cards] == ["ITSA4.SA", "POMO4.SA", "VALE3.SA", "KNIP11.SA", "SEMD3.SA", "IAU"]
    assert por["ITSA4.SA"]["situacao"] == "realizacao" and por["ITSA4.SA"]["tabela"] is not None
    assert "72%" in por["ITSA4.SA"]["evidencia"] and por["ITSA4.SA"]["renda_mensal"] > 0
    assert por["POMO4.SA"]["situacao"] == "queda" and por["POMO4.SA"]["tabela"]["colunas"][1] == "Aumentar 25%"
    assert any("dívida" in a for a in por["POMO4.SA"]["alertas"])  # filtro vira alerta, não bloqueio
    assert por["VALE3.SA"]["situacao"] == "fora" and por["VALE3.SA"]["tabela"] is None
    assert por["VALE3.SA"]["evidencia"] is None and por["VALE3.SA"]["abaixo_da_maxima_12m"] is not None
    assert por["KNIP11.SA"]["retrospecto"] is not None and por["KNIP11.SA"]["grupo"] == "FIIs"
    assert por["IAU"]["tabela"]["colunas"][1] == "Venda total" and por["IAU"]["moeda"] == "US$"
    venda_total = {nome: v2 for nome, _, v1, v2 in por["IAU"]["tabela"]["linhas"]}
    assert venda_total["Valor da posição hoje"] is None  # nada sobra: 24,11 cotas vendidas por inteiro
    assert venda_total["Movimento de caixa"][1] == pytest.approx(24.11 * 77.73)
    assert por["SEMD3.SA"]["grupo"] == "Outros" and por["SEMD3.SA"]["renda_mensal"] is None
    assert any("ERRO3.SA" in a for a in avisos) and any("SUMI3.SA" in a for a in avisos)


def test_gerar_cenarios_notas_e_tipo_nao_confirmado():
    datas = pd.bdate_range(end="2026-09-30", periods=300)
    precos = {"X.SA": pd.Series(np.linspace(100, 126, 300), index=datas),
              "Y.SA": pd.Series(np.linspace(100, 80, 300), index=datas),
              "Z.SA": pd.Series(np.linspace(100, 80, 300), index=datas)}
    resumo = pd.DataFrame([["X.SA", 3, 100.0, None, "2025-01-02"], ["Y.SA", 100, 100.0, None, "2025-01-02"],
                           ["Z.SA", 100, 100.0, None, "2025-01-02"]],
                          columns=["ticker", "quantidade", "preco_medio", "valor_investido", "data_inicio"])
    ops = pd.DataFrame([["X.SA", "venda", 1, 126.0, "2026-09-29"]], columns=["ticker", "tipo", "quantidade", "preco", "data"])
    cards, avisos = gerar_cenarios(resumo, ops, FonteFalsa(precos, {}), {"X.SA": "acao", "Z.SA": "acao"})
    por = {c["ticker"]: c for c in cards}
    # X: já vendeu na faixa de +25% -> sem cenário novo, com nota
    assert por["X.SA"]["situacao"] == "realizacao" and por["X.SA"]["tabela"] is None and por["X.SA"]["notas"]
    assert any("Y.SA" in a and "não confirmado" in a for a in avisos)
    assert por["Z.SA"]["tabela"] is not None


def test_posicao_pequena_gera_nota():
    datas = pd.bdate_range(end="2026-09-30", periods=300)
    precos = {"P.SA": pd.Series(np.linspace(100, 130, 300), index=datas)}
    resumo = pd.DataFrame([["P.SA", 3, 100.0, None, "2025-01-02"]],
                          columns=["ticker", "quantidade", "preco_medio", "valor_investido", "data_inicio"])
    cards, _ = gerar_cenarios(resumo, None, FonteFalsa(precos, {}), {"P.SA": "acao"})
    assert cards[0]["tabela"] is None and "pequena demais" in cards[0]["notas"][0]


def test_fii_sem_rendimentos_nao_mostra_retrospecto_zerado():
    datas = pd.bdate_range(end="2026-09-30", periods=300)
    precos = {"HASH11.SA": pd.Series(np.linspace(45, 57.7, 300), index=datas)}
    resumo = pd.DataFrame([["HASH11", 194, 45.83, None, "2025-01-02"]],
                          columns=["ticker", "quantidade", "preco_medio", "valor_investido", "data_inicio"])
    cards, _ = gerar_cenarios(resumo, None, FonteFalsa(precos, {}), {"HASH11.SA": "fii"})
    assert cards[0]["retrospecto"] is None
    assert any("Sem rendimentos" in n and "corrija o tipo" in n for n in cards[0]["notas"])


def test_pico_falso_de_cotacao_nao_distorce_a_maxima():
    datas = pd.bdate_range(end="2026-09-30", periods=300)
    serie = pd.Series(np.linspace(70, 78, 300), index=datas)
    serie.iloc[250] = 150.0  # erro pontual do provedor
    resumo = pd.DataFrame([["IAU", 24.11, 36.67, None, "2025-01-02"]],
                          columns=["ticker", "quantidade", "preco_medio", "valor_investido", "data_inicio"])
    cards, _ = gerar_cenarios(resumo, None, FonteFalsa({"IAU": serie}, {}), {"IAU": "etf_us"})
    assert cards[0]["abaixo_da_maxima_12m"] < 0.01  # sem o pico, o preço atual é a máxima


def test_retrospecto_com_lista_de_rendimentos_vazia_nao_quebra():
    s, _ = _fii([100.0] * 400)
    ops = [{"data": s.index[0], "tipo": "compra", "quantidade": 10, "preco": 100.0}]
    r = retrospecto_reinvestimento(ops, s, pd.Series(dtype=float))
    lin = {nome: vals for nome, _, *vals in r["linhas"]}
    assert lin["Cotas hoje"] == [10, 10, 10] and lin["Em caixa (CDI líquido)"] == [0, 0, 0]
