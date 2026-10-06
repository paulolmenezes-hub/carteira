"""Informes Mensais Estruturados de FIIs (CVM, Portal de Dados Abertos).

Fonte oficial e gratuita, com histórico desde 2016, usada para obter os
rendimentos mensais e o valor patrimonial por cota dos FIIs — o histórico de
rendimentos do Yahoo Finance tem lacunas de vários anos para muitos fundos.

Arquivos: um ZIP por ano (inf_mensal_fii_AAAA.zip) com três CSVs separados
por ';' — "geral" (cadastro, inclusive o código ISIN), "complemento"
(patrimônio, cotas, valor patrimonial, rentabilidade e dividend yield do mês)
e "ativo_passivo" (inclusive os rendimentos a distribuir). Os nomes das
colunas mudam entre os anos (ex.: CNPJ_Fundo -> CNPJ_Fundo_Classe, em 2026),
por isso as colunas são localizadas por palavras-chave.

O rendimento por cota de cada mês pode ser obtido de duas formas, comparadas
com os valores pagos (Yahoo) na função ``comparar_com_referencia``:
  (a) "dy": dividend yield do mês x valor patrimonial da cota;
  (b) "passivo": rendimentos a distribuir / número de cotas emitidas.
"""
from __future__ import annotations

import io
import unicodedata
import urllib.request
import zipfile

import numpy as np
import pandas as pd

URL_CVM_FII = "https://dados.cvm.gov.br/dados/FII/DOC/INF_MENSAL/DADOS/inf_mensal_fii_{ano}.zip"


def _normalizar(texto: str) -> str:
    sem_acento = "".join(c for c in unicodedata.normalize("NFD", str(texto)) if unicodedata.category(c) != "Mn")
    return sem_acento.lower().replace(" ", "_")


def _achar(df: pd.DataFrame, *pistas: str, evitar: tuple = ()) -> str | None:
    """Primeira coluna cujo nome normalizado contém todas as pistas."""
    for col in df.columns:
        nome = _normalizar(col)
        if all(p in nome for p in pistas) and not any(e in nome for e in evitar):
            return col
    return None


def _numero(serie: pd.Series) -> pd.Series:
    """Converte texto numérico da CVM (ponto ou vírgula decimal) em float."""
    s = serie.astype(str).str.strip()
    tem_virgula = s.str.contains(",", regex=False)
    s = s.where(~tem_virgula, s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
    return pd.to_numeric(s, errors="coerce")


def ler_zip_informes(conteudo: bytes) -> dict[str, pd.DataFrame]:
    """Lê os CSVs de um ZIP anual: {'geral': df, 'complemento': df, 'ativo_passivo': df}."""
    saida = {}
    with zipfile.ZipFile(io.BytesIO(conteudo)) as z:
        for nome in z.namelist():
            base = nome.lower()
            chave = ("geral" if "geral" in base else "complemento" if "complemento" in base
                     else "ativo_passivo" if "ativo" in base else None)
            if chave is None or not base.endswith(".csv"):
                continue
            bruto = z.read(nome)
            for codificacao in ("latin-1", "utf-8"):
                try:
                    saida[chave] = pd.read_csv(io.BytesIO(bruto), sep=";", dtype=str, encoding=codificacao)
                    break
                except UnicodeDecodeError:
                    continue
    return saida


def _ultima_versao(df: pd.DataFrame, c_cnpj: str, c_data: str) -> pd.DataFrame:
    """Mantém só a versão mais recente de cada informe (reapresentações)."""
    c_versao = _achar(df, "versao")
    if c_versao:
        df = df.assign(_v=pd.to_numeric(df[c_versao], errors="coerce").fillna(0))
        df = df.sort_values("_v").drop(columns="_v")
    return df.drop_duplicates([c_cnpj, c_data], keep="last")


def ticker_do_isin(isin) -> str | None:
    """ISIN de cota de FII (ex.: BRHGLGCTF004) -> ticker (HGLG11)."""
    isin = str(isin).strip().upper()
    if len(isin) == 12 and isin.startswith("BR") and isin[2:6].isalpha():
        return f"{isin[2:6]}11"
    return None


def montar_base_fii(tabelas_por_ano: list[dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """Une os anos numa base mensal por fundo: ticker, data, valor patrimonial
    por cota, cotas emitidas, dividend yield do mês e rendimentos a
    distribuir, com o rendimento por cota pelas duas formas."""
    partes = []
    for tabs in tabelas_por_ano:
        geral, comp = tabs.get("geral"), tabs.get("complemento")
        if geral is None or comp is None:
            continue
        c_cnpj_g, c_data_g = _achar(geral, "cnpj"), _achar(geral, "data_referencia")
        c_isin = _achar(geral, "isin")
        c_cnpj_c, c_data_c = _achar(comp, "cnpj"), _achar(comp, "data_referencia")
        if not all([c_cnpj_g, c_data_g, c_isin, c_cnpj_c, c_data_c]):
            raise ValueError(f"formato inesperado dos informes da CVM: colunas {list(geral.columns)[:8]} / "
                             f"{list(comp.columns)[:8]}")
        g = _ultima_versao(geral, c_cnpj_g, c_data_g)[[c_cnpj_g, c_data_g, c_isin]]
        g.columns = ["cnpj", "data", "isin"]
        c = _ultima_versao(comp, c_cnpj_c, c_data_c)
        colunas = {
            "cnpj": c_cnpj_c, "data": c_data_c,
            "vp_cota": _achar(c, "valor_patrimonial", "cota"),
            "cotas": _achar(c, "cotas_emitidas"),
            "dy_mes": _achar(c, "dividend_yield"),
            "pl": _achar(c, "patrimonio_liquido"),
        }
        c = pd.DataFrame({k: (c[v] if v else np.nan) for k, v in colunas.items()})
        base = c.merge(g, on=["cnpj", "data"], how="left")
        ap = tabs.get("ativo_passivo")
        if ap is not None:
            c_cnpj_a, c_data_a = _achar(ap, "cnpj"), _achar(ap, "data_referencia")
            c_rend = _achar(ap, "rendimentos", "distribuir")
            if c_cnpj_a and c_data_a and c_rend:
                a = _ultima_versao(ap, c_cnpj_a, c_data_a)[[c_cnpj_a, c_data_a, c_rend]]
                a.columns = ["cnpj", "data", "rendimentos_distribuir"]
                base = base.merge(a, on=["cnpj", "data"], how="left")
        partes.append(base)
    if not partes:
        return pd.DataFrame()
    df = pd.concat(partes, ignore_index=True)
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    for col in ("vp_cota", "cotas", "dy_mes", "pl", "rendimentos_distribuir"):
        if col in df:
            df[col] = _numero(df[col])
    df["ticker"] = df["isin"].map(ticker_do_isin)
    df = df.dropna(subset=["data", "ticker"])
    # dividend yield em percentual (ex.: 0,85) ou em fração (0,0085)? Um FII
    # paga tipicamente 0,5% a 1,2% ao mês: a mediana decide a escala.
    dy = df["dy_mes"].where(df["dy_mes"] > 0)
    if dy.notna().any() and dy.median() > 0.2:
        df["dy_mes"] = df["dy_mes"] / 100
    df["rend_cota_dy"] = df["dy_mes"] * df["vp_cota"]
    if "rendimentos_distribuir" in df:
        df["rend_cota_passivo"] = df["rendimentos_distribuir"] / df["cotas"].where(df["cotas"] > 0)
    else:
        df["rend_cota_passivo"] = np.nan
    df = df.sort_values(["ticker", "data"]).drop_duplicates(["ticker", "data"], keep="last")
    return df.reset_index(drop=True)


def serie_rendimentos(base: pd.DataFrame, ticker: str, forma: str = "dy") -> pd.Series:
    """Rendimento por cota de cada mês, indexado pela data de referência do
    informe — no formato dos dividendos usados pelo pacote."""
    t = ticker.upper().replace(".SA", "")
    col = "rend_cota_dy" if forma == "dy" else "rend_cota_passivo"
    s = base.loc[base["ticker"] == t].set_index("data")[col]
    return s[s > 0].sort_index()


def serie_valor_patrimonial(base: pd.DataFrame, ticker: str) -> pd.Series:
    """Valor patrimonial por cota mês a mês (permite o P/VP histórico)."""
    t = ticker.upper().replace(".SA", "")
    s = base.loc[base["ticker"] == t].set_index("data")["vp_cota"]
    return s[s > 0].sort_index()


def comparar_com_referencia(cvm: pd.Series, referencia: pd.Series) -> dict:
    """Compara, nos meses em que as duas fontes existem, a soma de 12 meses
    de rendimentos por cota da CVM com a da referência (valores pagos).
    Devolve o número de meses comparados e o erro relativo mediano."""
    def mensal(s):
        if s is None or len(s) == 0:
            return pd.Series(dtype=float)
        s = s.copy()
        if getattr(s.index, "tz", None) is not None:
            s.index = s.index.tz_localize(None)
        return s.groupby(s.index.to_period("M")).sum()

    a, b = mensal(cvm), mensal(referencia)
    meses = a.index.intersection(b.index)
    if len(meses) < 12:
        return {"meses": int(len(meses)), "erro_mediano_12m": None}
    a12 = a.reindex(meses).rolling(12, min_periods=12).sum()
    b12 = b.reindex(meses).rolling(12, min_periods=12).sum()
    ok = a12.notna() & b12.notna() & (b12 > 0)
    erro = ((a12[ok] - b12[ok]) / b12[ok]).abs()
    return {"meses": int(len(meses)), "erro_mediano_12m": float(erro.median()) if len(erro) else None}


def baixar_informes(anos, timeout: int = 120, baixar=None) -> list[dict[str, pd.DataFrame]]:
    """Baixa e lê os ZIPs anuais. `baixar(url) -> bytes` pode ser substituído
    (cache, testes). Anos que falharem são ignorados."""
    def padrao(url):
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read()
    baixar = baixar or padrao
    saida = []
    for ano in anos:
        try:
            saida.append(ler_zip_informes(baixar(URL_CVM_FII.format(ano=ano))))
        except Exception:
            continue
    return saida
