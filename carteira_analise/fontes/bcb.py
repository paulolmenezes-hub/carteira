"""CDI diário do Banco Central (SGS, série 12), em % ao dia útil.

Usado para que o dinheiro guardado nos cenários (rendimentos não
reinvestidos, vendas) renda o CDI real do período. Consulta em janelas de
até 5 anos (a API limita o tamanho das consultas de séries diárias)."""
from __future__ import annotations

import json
import urllib.request

import pandas as pd

URL_SGS_CDI = ("https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados?formato=json"
               "&dataInicial={ini:%d/%m/%Y}&dataFinal={fim:%d/%m/%Y}")


def _baixar_json(url: str, timeout: int = 30):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def buscar_cdi_diario(data_inicial, data_final) -> pd.Series | None:
    """Série do CDI (% ao dia útil) entre as datas, ou None se a consulta
    falhar (sem rede, API fora do ar ou resposta inválida)."""
    ini, fim = pd.Timestamp(data_inicial), pd.Timestamp(data_final)
    partes = []
    try:
        while ini <= fim:
            fim_janela = min(ini + pd.DateOffset(years=5) - pd.Timedelta(days=1), fim)
            dados = _baixar_json(URL_SGS_CDI.format(ini=ini, fim=fim_janela))
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
