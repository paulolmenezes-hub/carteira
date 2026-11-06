"""Base de cotações mensais da B3 (COTAHIST) — caso AFHI11 (05/11/2026)."""
import io
import zipfile

import pandas as pd
import pytest

from carteira_analise.fontes import b3_cotahist as b3


def _linha(data, ticker, preco, codbdi="12", tpmerc="010", fator=1):
    s = "01" + data + codbdi + ticker.ljust(12) + tpmerc
    s = s.ljust(108) + f"{round(preco * 100):013d}"
    s = s.ljust(210) + f"{fator:07d}"
    return s.ljust(245)


def test_le_layout_oficial_e_filtra_mercado_a_vista():
    linhas = [
        "00COTAHIST.2023BOVESPA 20231229".ljust(245),
        _linha("20230102", "AFHI11", 97.50),
        _linha("20230102", "PETR4", 22.10, codbdi="02"),
        _linha("20230102", "PETR4F", 22.12, codbdi="96"),          # fracionário: fora
        _linha("20230102", "PETRA250", 1.50, codbdi="78", tpmerc="070"),  # opção: fora
        _linha("20230102", "XPTO3", 1500.00, codbdi="02", fator=1000),     # cotado por lote de mil
    ]
    df = b3.ler_linhas_cotahist(linhas)
    assert set(df["ticker"]) == {"AFHI11", "PETR4", "XPTO3"}
    assert df.loc[df["ticker"] == "AFHI11", "fechamento"].iloc[0] == pytest.approx(97.50)
    assert df.loc[df["ticker"] == "XPTO3", "fechamento"].iloc[0] == pytest.approx(1.50)


def test_zip_e_resumo_mensal():
    corpo = "\n".join([_linha("20230102", "AFHI11", 97.0), _linha("20230131", "AFHI11", 99.0),
                       _linha("20230201", "AFHI11", 98.0)]).encode("latin-1")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("COTAHIST_A2023.TXT", corpo)
    resumo = b3.resumo_mensal(b3.ler_zip_cotahist(buf.getvalue()))
    jan = resumo[resumo["mes"] == "2023-01"].iloc[0]
    assert jan["fechamento"] == 99.0 and jan["media"] == 98.0 and jan["pregoes"] == 2
    assert jan["data"] == pd.Timestamp("2023-01-31")


def _base():
    meses = pd.date_range("2021-06-30", "2023-12-31", freq="ME")
    return pd.DataFrame({"ticker": "AFHI11", "mes": meses.to_period("M").astype(str), "data": meses,
                         "fechamento": 95.0, "media": 95.0, "pregoes": 21})


def test_completa_antes_do_primeiro_pregao_do_yahoo():
    yahoo = pd.Series(93.0, index=pd.bdate_range("2024-01-02", "2026-11-03"))
    completa, inicio = b3.completar_precos(yahoo, _base(), "AFHI11.SA")
    assert inicio == pd.Timestamp("2024-01-02")
    assert completa.index[0] == pd.Timestamp("2021-06-30") and completa.index.is_monotonic_increasing
    assert (completa[completa.index < "2024-01-01"] == 95.0).all()
    assert len(completa) == len(yahoo) + 31                         # 31 meses da B3 (06/2021 a 12/2023)
    # Yahoo já cobre o período: nada muda
    longo = pd.Series(93.0, index=pd.bdate_range("2020-01-02", "2026-11-03"))
    mesmo, nada = b3.completar_precos(longo, _base(), "AFHI11.SA")
    assert nada is None and len(mesmo) == len(longo)
    # sem a base ou sem o papel: só o Yahoo
    assert b3.completar_precos(yahoo, None, "AFHI11.SA")[1] is None
    assert b3.completar_precos(yahoo, _base(), "XPTO11.SA")[1] is None


def test_desdobramento_poe_a_b3_na_escala_do_yahoo():
    yahoo = pd.Series(9.5, index=pd.bdate_range("2024-01-02", "2024-12-31"))
    eventos = pd.Series([10.0], index=[pd.Timestamp("2023-06-15")])  # 1 para 10
    completa, _ = b3.completar_precos(yahoo, _base(), "AFHI11.SA", eventos)
    assert completa.loc["2023-05-31"] == pytest.approx(9.5)        # antes do evento: dividido por 10
    assert completa.loc["2023-07-31"] == pytest.approx(95.0)       # depois: preço da B3 já na escala nova


def test_dy_anual_usa_medias_da_b3_quando_o_yahoo_nao_cobre_o_ano():
    from carteira_analise.fundamentalistas import dy_por_ano
    yahoo = pd.Series(100.0, index=pd.bdate_range("2024-01-02", "2025-12-31"))
    divs = pd.Series([5.0, 6.0], index=pd.to_datetime(["2022-06-15", "2024-06-15"]))
    medias = b3.medias_mensais(_base().assign(media=50.0), "AFHI11.SA")
    anos_b3 = []
    dys = dy_por_ano(divs, yahoo, [2022, 2024], medias_mensais=medias, anos_b3=anos_b3)
    assert dys[2022] == pytest.approx(0.10) and dys[2024] == pytest.approx(0.06) and anos_b3 == [2022]


def test_base_ausente_devolve_none(tmp_path):
    assert b3.carregar_base(str(tmp_path / "nao_existe.csv.gz")) is None


def test_filtra_bdrs_e_direitos():
    df = pd.DataFrame({"data": pd.Timestamp("2023-01-02"), "fechamento": 1.0,
                       "ticker": ["PETR4", "TAEE11", "AAPL34", "MGLU1", "AFHI11", "VALE3", "BBDC12"]})
    assert set(b3.filtrar_papeis(df)["ticker"]) == {"PETR4", "TAEE11", "AFHI11", "VALE3"}


def test_carregar_base_aceita_csv_descompactado(tmp_path):
    from carteira_analise.fontes import b3_cotahist
    csv = tmp_path / "cotacoes_b3_mensais.csv"
    csv.write_text("ticker,mes,data,fechamento,media,pregoes\nAFHI11,2021-04,2021-04-30,99.89,100.1,15\n")
    base = b3_cotahist.carregar_base(str(tmp_path / "cotacoes_b3_mensais.csv.gz"))
    assert base is not None and list(base["ticker"]) == ["AFHI11"]
