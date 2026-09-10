"""Cache local de dados de mercado (Horizonte 1 do roadmap — Seção 8.1 do manual).

Evita repetir chamadas de rede pro mesmo ticker no mesmo período de tempo,
usando um arquivo SQLite local. Funciona como um "envelope": embrulha
qualquer módulo de fonte que implemente ``baixar_precos``, ``baixar_info`` e
``baixar_dividendos`` (hoje, ``fontes/yahoo.py``) — quem usa
``carteira.analisar_ativo(..., fonte=fonte_com_cache)`` não precisa saber que
o cache existe, porque a interface é idêntica à da fonte original.

Por que essa abordagem (e não cache dentro de ``fontes/yahoo.py`` diretamente):
mantém ``fontes/yahoo.py`` simples e sem estado, e permite ligar/desligar o
cache, trocar o TTL, ou cachear uma fonte totalmente diferente (ex: uma futura
integração com a brapi.dev — Seção 8.2.1 do manual) sem duplicar código.
"""
from __future__ import annotations

import io
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import pandas as pd

CAMINHO_PADRAO = Path.home() / ".carteira_analise_cache.sqlite"
TTL_PADRAO_HORAS = 12.0  # dados de mercado diário não mudam mais de ~2x/dia úteis


class FonteComCache:
    """Embrulha uma fonte de dados (ex: ``fontes.yahoo``) com cache local em
    SQLite. Resultados vazios/falhos NÃO são cacheados de propósito — um erro
    transitório de rede não deve ficar "preso" no cache até o TTL expirar."""

    def __init__(
        self,
        fonte_original: Any,
        caminho_db: Path | str = CAMINHO_PADRAO,
        ttl_horas: float = TTL_PADRAO_HORAS,
    ) -> None:
        self._fonte = fonte_original
        self._ttl_segundos = ttl_horas * 3600
        self._conn = sqlite3.connect(str(caminho_db))
        self._criar_tabelas()

    def _criar_tabelas(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                chave TEXT PRIMARY KEY,
                valor TEXT NOT NULL,
                tipo_valor TEXT NOT NULL,
                salvo_em REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    def _get(self, chave: str) -> Any | None:
        cur = self._conn.execute(
            "SELECT valor, tipo_valor, salvo_em FROM cache WHERE chave = ?", (chave,)
        )
        row = cur.fetchone()
        if row is None:
            return None
        valor, tipo_valor, salvo_em = row
        if time.time() - salvo_em > self._ttl_segundos:
            return None
        return self._desserializar(valor, tipo_valor)

    def _set(self, chave: str, valor: Any) -> None:
        serializado, tipo_valor = self._serializar(valor)
        self._conn.execute(
            "INSERT OR REPLACE INTO cache (chave, valor, tipo_valor, salvo_em) VALUES (?, ?, ?, ?)",
            (chave, serializado, tipo_valor, time.time()),
        )
        self._conn.commit()

    @staticmethod
    def _serializar(valor: Any) -> tuple[str, str]:
        if isinstance(valor, pd.Series):
            return valor.to_json(date_format="iso"), "series"
        return json.dumps(valor), "dict"

    @staticmethod
    def _desserializar(valor: str, tipo_valor: str) -> Any:
        if tipo_valor == "series":
            serie = pd.read_json(io.StringIO(valor), typ="series").astype(float)
            serie.index = pd.to_datetime(serie.index)
            return serie
        return json.loads(valor)

    def baixar_precos(self, ticker: str, periodo: str) -> pd.Series:
        chave = f"precos:{ticker}:{periodo}"
        em_cache = self._get(chave)
        if em_cache is not None:
            return em_cache
        resultado = self._fonte.baixar_precos(ticker, periodo)
        if not resultado.empty:
            self._set(chave, resultado)
        return resultado

    def baixar_info(self, ticker: str) -> dict:
        chave = f"info:{ticker}"
        em_cache = self._get(chave)
        if em_cache is not None:
            return em_cache
        resultado = self._fonte.baixar_info(ticker)
        if resultado:
            self._set(chave, resultado)
        return resultado

    def baixar_dividendos(self, ticker: str) -> pd.Series:
        chave = f"dividendos:{ticker}"
        em_cache = self._get(chave)
        if em_cache is not None:
            return em_cache
        resultado = self._fonte.baixar_dividendos(ticker)
        if not resultado.empty:
            self._set(chave, resultado)
        return resultado

    def limpar(self) -> None:
        """Apaga todo o cache — útil se os dados parecerem desatualizados
        antes do TTL expirar, ou depois de uma correção na fonte original."""
        self._conn.execute("DELETE FROM cache")
        self._conn.commit()

    def fechar(self) -> None:
        self._conn.close()

    def __enter__(self) -> "FonteComCache":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.fechar()
