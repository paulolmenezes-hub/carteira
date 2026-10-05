"""CDI diário do Banco Central (SGS, série 12), em % ao dia útil.

Usado para que o dinheiro guardado nos cenários (rendimentos não
reinvestidos, vendas) renda o CDI real do período. Consulta em janelas de
2 anos, com novas tentativas: a API do Banco Central costuma demorar ou falhar
em consultas longas."""
from __future__ import annotations

import json
import time
import urllib.request

import pandas as pd

JANELA_ANOS = 2
TENTATIVAS = 3

URL_SGS_CDI = ("https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados?formato=json"
               "&dataInicial={ini:%d/%m/%Y}&dataFinal={fim:%d/%m/%Y}")


def _baixar_json(url: str, timeout: int = 45):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _baixar_com_tentativas(url: str, tentativas: int = TENTATIVAS, espera: float = 2.0, dormir=time.sleep):
    """Repete a consulta em caso de falha, esperando 2 s, 4 s, ... entre elas."""
    for k in range(tentativas):
        try:
            return _baixar_json(url)
        except Exception:
            if k == tentativas - 1:
                raise
            dormir(espera * (2 ** k))


def buscar_cdi_diario(data_inicial, data_final, dormir=time.sleep) -> pd.Series | None:
    """Série do CDI (% ao dia útil) entre as datas, ou None se a consulta
    falhar mesmo após as novas tentativas (sem rede ou API fora do ar)."""
    ini, fim = pd.Timestamp(data_inicial), pd.Timestamp(data_final)
    partes = []
    try:
        while ini <= fim:
            fim_janela = min(ini + pd.DateOffset(years=JANELA_ANOS) - pd.Timedelta(days=1), fim)
            dados = _baixar_com_tentativas(URL_SGS_CDI.format(ini=ini, fim=fim_janela), dormir=dormir)
            if dados:
                partes.append(pd.Series([float(d["valor"]) for d in dados],
                                        index=pd.to_datetime([d["data"] for d in dados], dayfirst=True)))
            ini = fim_janela + pd.Timedelta(days=1)
    except Exception:
        return None
    if not partes:
        return None
    serie = pd.concat(partes).sort_index()
    return serie[~serie.index.duplicated()]
