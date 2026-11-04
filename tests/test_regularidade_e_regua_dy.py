"""Revisão do motor (04/11/2026): régua do dividend yield ligada ao CDI e
bloco "Os dividendos são regulares?" das ações."""
import pandas as pd
import pytest

from carteira_analise.fundamentalistas import (HistoricoDividendos, dy_por_ano, regularidade_dividendos,
                                               texto_regularidade)
from carteira_analise.pontuacao import avaliar_renda, limite_dy, pontuar_acao

DIAS = pd.bdate_range("2020-01-01", "2026-11-03")
PRECOS = pd.Series(100.0, index=DIAS)


def _divs(por_ano: list[float], primeiro_ano=2021, extra=None):
    d = {pd.Timestamp(f"{primeiro_ano + i}-06-15"): v for i, v in enumerate(por_ano) if v > 0}
    d.update(extra or {})
    return pd.Series(d, dtype=float)


# ------------------------------------------------------------------ regularidade

def test_regular_oscilante_irregular():
    assert regularidade_dividendos(_divs([13.5, 13.3, 8.4, 9.2, 9.8]), PRECOS)["selo"] == "regular"
    osc = regularidade_dividendos(_divs([1.4, 5.9, 6.8, 6.6, 7.3]), PRECOS)
    assert osc["selo"] == "oscilante" and osc["ano_pior"] == 2021
    irr = regularidade_dividendos(_divs([3.4, 3.3, 0, 0, 0]), PRECOS)
    assert irr["selo"] == "irregular" and irr["pagou"] == 2
    assert "3 anos sem pagar" in texto_regularidade(irr)


def test_ano_extraordinario_marcado_e_mediana_como_referencia():
    r = regularidade_dividendos(_divs([19.9, 58.8, 29.6, 21.2, 9.0]), PRECOS)
    assert r["extraordinarios"] == [2022]
    assert r["mediana"] == pytest.approx(0.212) and r["media"] == pytest.approx(0.277)
    assert "fora da curva" in texto_regularidade(r)


def test_so_anos_fechados_e_ano_corrente_a_parte():
    r = regularidade_dividendos(_divs([10, 10, 10, 10, 10], extra={pd.Timestamp("2026-03-10"): 2.0}), PRECOS)
    assert [a["ano"] for a in r["anos"]] == [2021, 2022, 2023, 2024, 2025]
    assert r["ano_corrente"]["ano"] == 2026 and r["ano_corrente"]["dy"] == pytest.approx(0.02)


def test_dy_usa_preco_medio_do_ano():
    precos = pd.Series([50.0 if dt.month <= 6 else 150.0 for dt in DIAS], index=DIAS)  # média ~100
    dy = dy_por_ano(pd.Series({pd.Timestamp("2023-06-15"): 10.0}), precos, [2023])[2023]
    assert dy == pytest.approx(0.10, rel=0.03)


def test_sem_dados_suficientes_devolve_none():
    curto = PRECOS[PRECOS.index >= "2023-01-01"]  # só 3 anos de preços
    assert regularidade_dividendos(_divs([5, 5, 5, 5, 5]), curto) is None
    assert regularidade_dividendos(pd.Series(dtype=float), PRECOS) is None  # sem proventos: dado ausente


def test_fuso_horario_nos_proventos_do_yahoo():
    d = _divs([8, 8, 8, 8, 8])
    d.index = d.index.tz_localize("America/Sao_Paulo")
    assert regularidade_dividendos(d, PRECOS)["selo"] == "regular"


# ------------------------------------------------------------------ régua do DY ligada ao CDI

def test_limite_dy_acompanha_o_cdi_com_piso():
    assert limite_dy("acao") == 0.06 and limite_dy("fii") == 0.08  # sem CDI: limites fixos de antes
    assert limite_dy("acao", 0.123) == pytest.approx(0.0615)        # calibrado: ~6% com o CDI de nov/2026
    assert limite_dy("fii", 0.123) == pytest.approx(0.080, abs=0.001)
    assert limite_dy("acao", 0.02) == 0.03                          # piso com juros muito baixos
    assert limite_dy("acao_us", 0.12) == 0.02                       # EUA: fixo


def test_pontuar_acao_texto_da_regua():
    from carteira_analise.tecnicos import calcular_indicadores_tecnicos
    tec = calcular_indicadores_tecnicos(pd.Series(range(100, 400), index=pd.bdate_range("2024-01-01", periods=300),
                                                  dtype=float))
    div = HistoricoDividendos(soma_12m=1.0, soma_anterior_12m=1.0, dy_12m_real=0.07, crescimento_yoy=None, n_pagamentos_12m=2)
    det = pontuar_acao(tec, {}, div, cdi_liquido_aa=0.12).detalhes
    assert any("yield efetivo" in d and "50% do CDI líquido" in d for d in det)
    det_alto = pontuar_acao(tec, {}, div, cdi_liquido_aa=0.16).detalhes  # régua 8%: 7% não pontua
    assert not any("yield efetivo" in d for d in det_alto)


# ------------------------------------------------------------------ regularidade no motor de renda

def test_regularidade_substitui_contagem_de_pagamentos_nas_acoes():
    div = HistoricoDividendos(soma_12m=1.0, soma_anterior_12m=1.0, dy_12m_real=0.05, crescimento_yoy=0.0, n_pagamentos_12m=1)
    reg = regularidade_dividendos(_divs([8, 8, 8, 8, 8]), PRECOS)
    r = avaliar_renda(div, "acao", regularidade=reg)
    assert any("dividendos regulares" in d for d in r.detalhes)
    assert not any("pagamentos regulares" in d for d in r.detalhes)
    irr = avaliar_renda(div, "acao", regularidade=regularidade_dividendos(_divs([3, 3, 0, 0, 0]), PRECOS))
    assert any(d.startswith("-1 dividendos irregulares") for d in irr.detalhes)
    # sem regularidade (ou FII), continua a regra antiga
    assert not any("regulares:" in d for d in avaliar_renda(div, "acao").detalhes)
    assert not any("regulares:" in d for d in avaliar_renda(div, "fii", regularidade=reg).detalhes)


def test_analisar_ativo_calcula_regularidade_com_10_anos(monkeypatch):
    import carteira_analise.fontes.fundamentus as fm
    from carteira_analise.carteira import analisar_ativo
    monkeypatch.setattr(fm, "buscar_fundamentos_acao", lambda t: None)

    class Fonte:
        periodos = []

        def baixar_precos(self, ticker, periodo):
            self.periodos.append(periodo)
            return PRECOS if periodo == "10y" else PRECOS[PRECOS.index >= "2025-11-01"]

        def baixar_info(self, ticker):
            return {}

        def baixar_dividendos(self, ticker):
            return _divs([8, 8, 8, 8, 8], extra={pd.Timestamp("2026-06-15"): 8.0})

    fonte = Fonte()
    r = analisar_ativo("TAEE11.SA", "acao", "1y", fonte, cdi_liquido_aa=0.12)
    assert "10y" in fonte.periodos and r.regularidade["selo"] == "regular"
    assert any("dividendos regulares" in d for d in r.detalhes_renda)


# ------------------------------------------------------------------ lacunas do Yahoo conferidas no Fundamentus

def test_ano_zerado_no_yahoo_mas_pago_no_fundamentus_fica_fora_da_conta():
    yahoo = _divs([13.0, 9.9, 8.6, 9.0, 0])          # 2025 sem registro no Yahoo
    fundamentus = _divs([1.3, 1.0, 0.9, 1.0, 0.8])     # escala diferente: só confere se houve pagamento
    sem = regularidade_dividendos(yahoo, PRECOS)
    assert sem["selo"] == "irregular"
    com = regularidade_dividendos(yahoo, PRECOS, conferencia=fundamentus)
    assert com["selo"] == "regular" and com["lacunas"] == [2025] and com["n_anos"] == 4
    assert com["anos"][-1]["dy"] is None and com["anos"][-1]["sem_registro"]
    assert "2025: o Yahoo não registra" in texto_regularidade(com)
    # sem pagamento também no Fundamentus: continua irregular
    assert regularidade_dividendos(yahoo, PRECOS, conferencia=_divs([1, 1, 1, 1, 0]))["selo"] == "irregular"


def test_lacunas_demais_ficam_sem_selo():
    yahoo = _divs([13.0, 9.9, 8.6, 0, 0])
    r = regularidade_dividendos(yahoo, PRECOS, conferencia=_divs([1, 1, 1, 1, 1]))
    assert r["selo"] == "incompleto" and r["lacunas"] == [2024, 2025]
    assert texto_regularidade(r).startswith("Sem selo")
    div = HistoricoDividendos(soma_12m=1.0, soma_anterior_12m=1.0, dy_12m_real=0.05, crescimento_yoy=0.0,
                              n_pagamentos_12m=2)
    assert not any("regulares:" in d or "irregulares" in d for d in avaliar_renda(div, "acao", regularidade=r).detalhes)


def test_ano_corrente_sem_registro():
    yahoo = _divs([8, 8, 8, 8, 8])
    r = regularidade_dividendos(yahoo, PRECOS, conferencia=_divs([1, 1, 1, 1, 1, 1]))
    assert r["ano_corrente"]["sem_registro"] is True


def test_proventos_acao_de_html_divide_por_quantas_acoes():
    from carteira_analise.fontes.fundamentus import proventos_acao_de_html
    html = ("<table><tr><th>Data</th><th>Valor</th><th>Tipo</th><th>Data de Pagamento</th>"
            "<th>Por quantas Ações</th></tr>"
            "<tr><td>29/04/2025</td><td>2,50</td><td>DIVIDENDO</td><td>30/06/2025</td><td>1</td></tr>"
            "<tr><td>29/04/2025</td><td>0,30</td><td>JRS CAP PROPRIO</td><td>30/06/2025</td><td>1</td></tr>"
            "<tr><td>10/12/2024</td><td>100,00</td><td>DIVIDENDO</td><td>-</td><td>100</td></tr></table>")
    s = proventos_acao_de_html(html)
    assert s.loc["2025-04-29"] == pytest.approx(2.80) and s.loc["2024-12-10"] == pytest.approx(1.0)


def test_analisar_ativo_confere_lacuna_no_fundamentus(monkeypatch):
    import carteira_analise.fontes.fundamentus as fm
    from carteira_analise.carteira import analisar_ativo
    monkeypatch.setattr(fm, "buscar_fundamentos_acao", lambda t: None)
    chamadas = []
    monkeypatch.setattr(fm, "buscar_proventos_acao", lambda t: chamadas.append(t) or _divs([1, 1, 1, 1, 1]))

    class Fonte:
        def baixar_precos(self, ticker, periodo):
            return PRECOS if periodo == "10y" else PRECOS[PRECOS.index >= "2025-11-01"]

        def baixar_info(self, ticker):
            return {}

        def baixar_dividendos(self, ticker):
            return _divs([13.0, 9.9, 8.6, 9.0, 0], extra={pd.Timestamp("2026-05-15"): 8.0})

    r = analisar_ativo("CPFE3.SA", "acao", "1y", Fonte())
    assert chamadas == ["CPFE3.SA"] and r.regularidade["lacunas"] == [2025] and r.regularidade["selo"] == "regular"
