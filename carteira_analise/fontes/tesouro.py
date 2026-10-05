"""Taxas históricas do Tesouro IPCA+ (Tesouro Transparente).

O arquivo oficial traz preços e taxas de todos os títulos do Tesouro Direto
desde 2002 (CSV separado por ';', números com vírgula, datas dd/mm/aaaa).
Daqui sai o juro REAL de referência do mercado (Tesouro IPCA+, a antiga
NTN-B), usado para medir o prêmio dos FIIs sobre a renda fixa sem risco
de crédito."""
from __future__ import annotations

import io
import unicodedata
import urllib.request

import pandas as pd

URL_TESOURO = ("https://www.tesourotransparente.gov.br/ckan/dataset/df56aa42-484a-4a59-8184-7676580c81e3/"
               "resource/796d2059-14e9-44e3-80c9-2d9e30b405c1/download/precotaxatesourodireto.csv")


def _sem_acento(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(texto)) if unicodedata.category(c) != "Mn").lower()


def _coluna(df: pd.DataFrame, *pistas: str) -> str | None:
    for col in df.columns:
        nome = _sem_acento(col)
        if all(p in nome for p in pistas):
            return col
    return None


def juro_real_tesouro_ipca(csv_texto: str, prazo_min_anos: float = 5, prazo_max_anos: float = 15) -> pd.Series:
    """Taxa real diária de referência (fração ao ano): mediana das taxas dos
    títulos Tesouro IPCA+ (com e sem juros semestrais) com vencimento entre
    `prazo_min_anos` e `prazo_max_anos` a partir da data. Usa a taxa de
    compra e, quando ela não existe no dia (título fora de oferta), a de venda."""
    df = pd.read_csv(io.StringIO(csv_texto), sep=";", dtype=str)
    c_tipo, c_venc = _coluna(df, "tipo"), _coluna(df, "vencimento")
    c_data = _coluna(df, "data base") or _coluna(df, "data", "base")
    c_compra, c_venda = _coluna(df, "taxa", "compra"), _coluna(df, "taxa", "venda")
    if not all([c_tipo, c_venc, c_data]) or not (c_compra or c_venda):
        raise ValueError("formato inesperado do arquivo do Tesouro")
    tipo = df[c_tipo].map(_sem_acento)
    ipca = tipo.str.startswith("tesouro ipca") & ~tipo.str.contains("renda|educa")
    df = df[ipca].copy()

    def numero(col):
        return pd.to_numeric(df[col].str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
                             errors="coerce") if col else pd.Series(float("nan"), index=df.index)

    taxa = numero(c_compra)
    taxa = taxa.where(taxa > 0, numero(c_venda))
    data = pd.to_datetime(df[c_data], dayfirst=True, errors="coerce")
    venc = pd.to_datetime(df[c_venc], dayfirst=True, errors="coerce")
    prazo = (venc - data).dt.days / 365.25
    ok = taxa.notna() & (taxa > 0) & data.notna() & (prazo >= prazo_min_anos) & (prazo <= prazo_max_anos)
    serie = pd.Series(taxa[ok].values / 100, index=data[ok].values).groupby(level=0).median()
    return serie.sort_index()


def baixar_juro_real_tesouro_ipca(timeout: int = 60) -> pd.Series | None:
    """Baixa o arquivo oficial e devolve a série de juro real, ou None se falhar."""
    try:
        with urllib.request.urlopen(URL_TESOURO, timeout=timeout) as resp:
            texto = resp.read().decode("latin-1")
        return juro_real_tesouro_ipca(texto)
    except Exception:
        return None
