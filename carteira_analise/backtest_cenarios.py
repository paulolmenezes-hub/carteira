"""Backtest dos cenários comparativos (Sprint 4).

Gera a evidência histórica mostrada em cada card de cenário ("o que o
histórico mostra"):

- Ações e ETFs: cada cenário comparado com manter, isoladamente —
  "só realizar lucro nas faixas" e "só aumentar a posição nas quedas" —,
  além da regra completa (as duas coisas juntas);
- FIIs: estratégias para os rendimentos — não reinvestir, reinvestir 50%,
  reinvestir 100% e guardar no CDI para comprar cotas na queda (com duas
  definições de queda, fixadas antes de ver os resultados).

O dinheiro guardado é tratado como aplicações em CDB de 100% do CDI: cada
depósito é um lote com o seu próprio prazo, IR pela tabela regressiva
(22,5% a 15%) e IOF regressivo abaixo de 30 dias, calculados como se
resgatado na data de avaliação. Rendimentos de FII e dividendos são
isentos; o IR incide só sobre o ganho da aplicação em renda fixa.

Cada estratégia roda a partir de várias datas de entrada (13, por
padrão), para que o resultado não dependa da sorte de uma única data.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .regra_posicao import (
    aplicar_compra,
    aplicar_resets_por_preco,
    aplicar_venda,
    decidir_posicao,
    novo_estado,
)

# ---------------------------------------------------------------------------
# Tributação da renda fixa (CDB / Tesouro): IR regressivo e IOF regressivo
# ---------------------------------------------------------------------------

# IOF sobre o rendimento, por dia corrido de aplicação (1 a 29); a partir de
# 30 dias, zero.
TABELA_IOF = [96, 93, 90, 86, 83, 80, 76, 73, 70, 66, 63, 60, 56, 53, 50,
              46, 43, 40, 36, 33, 30, 26, 23, 20, 16, 13, 10, 6, 3]


def aliquota_ir_renda_fixa(dias_corridos: int) -> float:
    """Tabela regressiva do IR sobre o rendimento de renda fixa."""
    if dias_corridos <= 180:
        return 0.225
    if dias_corridos <= 360:
        return 0.20
    if dias_corridos <= 720:
        return 0.175
    return 0.15


def aliquota_iof(dias_corridos: int) -> float:
    """IOF regressivo sobre o rendimento (zero a partir de 30 dias)."""
    if 1 <= dias_corridos < 30:
        return TABELA_IOF[dias_corridos - 1] / 100
    return 0.0


def indice_cdi_acumulado(datas: pd.DatetimeIndex, cdi_diario_pct: pd.Series | None = None,
                         cdi_aa: float = 0.0) -> pd.Series:
    """Índice acumulado do CDI alinhado às `datas` dos pregões do ativo.

    `cdi_diario_pct`: série do Banco Central (SGS 12), em % ao dia útil.
    Sem a série, usa uma taxa anual constante `cdi_aa` (252 dias úteis).
    O rendimento de um lote entre duas datas é a razão entre os índices."""
    datas = pd.DatetimeIndex(datas)
    if cdi_diario_pct is not None and len(cdi_diario_pct) > 0:
        serie = cdi_diario_pct.sort_index()
        if getattr(serie.index, "tz", None) is not None:
            serie = serie.copy()
            serie.index = serie.index.tz_localize(None)
        acumulado = (1 + serie.astype(float) / 100).cumprod()
        indice = acumulado.reindex(acumulado.index.union(datas)).ffill().reindex(datas)
        return indice.fillna(1.0)
    # Sem a série do Banco Central (taxa de reserva): compõe pelo tempo corrido,
    # o que dá exatamente `cdi_aa` em um ano — equivalente aos 252 dias úteis
    # do mercado, qualquer que seja o calendário das `datas` (pregões do ativo).
    if len(datas) == 0:
        return pd.Series(dtype=float)
    anos = (datas - datas[0]).days / 365.25
    return pd.Series((1 + cdi_aa) ** np.asarray(anos, dtype=float), index=datas)


class CaixaRendaFixa:
    """Dinheiro guardado em CDB de 100% do CDI, lote a lote.

    Cada depósito guarda o principal, a data e o índice do CDI na data.
    O valor líquido de um lote em uma data é o principal mais o rendimento
    do período, descontados IOF (se < 30 dias) e IR pela faixa do seu
    próprio prazo — como se resgatado naquela data."""

    def __init__(self):
        self.lotes: list[dict] = []

    def depositar(self, data, valor: float, indice: float) -> None:
        if valor > 0:
            self.lotes.append({"data": pd.Timestamp(data), "principal": float(valor), "indice": float(indice)})

    @staticmethod
    def _liquido_lote(lote: dict, data, indice: float) -> float:
        dias = (pd.Timestamp(data) - lote["data"]).days
        bruto = lote["principal"] * indice / lote["indice"]
        ganho = max(bruto - lote["principal"], 0.0)
        ganho *= 1 - aliquota_iof(dias)
        ganho *= 1 - aliquota_ir_renda_fixa(dias)
        return lote["principal"] + ganho

    def valor_liquido(self, data, indice: float) -> float:
        return sum(self._liquido_lote(l, data, indice) for l in self.lotes)

    def resgatar_liquido(self, data, valor: float, indice: float) -> float:
        """Resgata até `valor` líquido, dos lotes mais antigos para os mais
        novos (menor IR primeiro). Devolve o valor efetivamente resgatado."""
        obtido = 0.0
        restantes = []
        for lote in self.lotes:
            falta = valor - obtido
            if falta <= 1e-9:
                restantes.append(lote)
                continue
            liquido = self._liquido_lote(lote, data, indice)
            if liquido <= falta + 1e-9:
                obtido += liquido
                continue  # lote inteiro resgatado
            fracao = falta / liquido
            obtido += falta
            lote = dict(lote, principal=lote["principal"] * (1 - fracao))
            restantes.append(lote)
        self.lotes = restantes
        return obtido


# ---------------------------------------------------------------------------
# Utilidades comuns
# ---------------------------------------------------------------------------

DIAS_ATE_PAGAMENTO = 10  # pregões entre a data-ex (Yahoo) e o pagamento, aproximado


def _sem_fuso(serie):
    if serie is None:
        return pd.Series(dtype=float)
    if getattr(serie.index, "tz", None) is not None:
        serie = serie.copy()
        serie.index = serie.index.tz_localize(None)
    return serie


def _pagamentos_por_pregao(datas: pd.DatetimeIndex, dividendos) -> dict[int, float]:
    """Mapeia cada provento (data-ex) para o pregão aproximado de pagamento."""
    mapa: dict[int, float] = {}
    d = _sem_fuso(dividendos)
    for data, valor in d.items():
        pos = datas.searchsorted(pd.Timestamp(data))
        if pos < len(datas):
            pos = min(pos + DIAS_ATE_PAGAMENTO, len(datas) - 1)
            mapa[pos] = mapa.get(pos, 0.0) + float(valor)
    return mapa


def _retorno_anual(valor_final: float, aportado: float, data_ini, data_fim) -> float | None:
    anos = (pd.Timestamp(data_fim) - pd.Timestamp(data_ini)).days / 365.25
    if anos <= 0 or aportado <= 0 or valor_final <= 0:
        return None
    return (valor_final / aportado) ** (1 / anos) - 1


ESPACAMENTO_MINIMO_ENTRADAS = 21  # ~1 mês entre datas de entrada


def datas_de_entrada(n_pregoes: int, n_datas: int = 13, historico_minimo: int = 252,
                     horizonte_minimo: int = 252,
                     espacamento_minimo: int = ESPACAMENTO_MINIMO_ENTRADAS) -> list[int]:
    """Índices de pregão das datas de entrada, igualmente espaçados entre
    `historico_minimo` (para haver 12 meses de histórico antes) e o último
    ponto que ainda deixa `horizonte_minimo` pregões até o fim.

    Ativos com histórico curto recebem MENOS datas, para que fiquem ao menos
    `espacamento_minimo` pregões (~1 mês) de distância — datas quase iguais não
    contam como casos independentes."""
    ultimo = n_pregoes - horizonte_minimo
    if ultimo < historico_minimo:
        return []
    n = min(n_datas, 1 + (ultimo - historico_minimo) // espacamento_minimo)
    if n <= 1:
        return [historico_minimo]
    return sorted(set(int(round(x)) for x in np.linspace(historico_minimo, ultimo, n)))


# ---------------------------------------------------------------------------
# Ações e ETFs: cada cenário isolado contra manter
# ---------------------------------------------------------------------------

VARIANTES_ACOES = {
    "Manter": None,
    "Só realizar lucro nas faixas": {"faixas_compra": []},
    "Só aumentar nas quedas": {"faixas_venda": []},
    "Regra completa": {},
}


def simular_cenario_regra(fechamento: pd.Series, dividendos=None, variante: str = "Manter",
                          inicio_idx: int = 252, capital: float = 10000.0,
                          indice_cdi: pd.Series | None = None) -> dict | None:
    """Simula um cenário a partir de `inicio_idx` até o fim da série.

    Compras na queda usam dinheiro novo (somado ao aportado); vendas e
    proventos vão para o caixa em CDB 100% CDI (lote a lote, com IR/IOF).
    Sem `indice_cdi`, o caixa não rende (ex.: ativos em dólar)."""
    serie = _sem_fuso(fechamento).dropna()
    if inicio_idx >= len(serie) - 20:
        return None
    serie = serie.iloc[inicio_idx:]
    datas = serie.index
    precos = serie.values.astype(float)
    indice = (indice_cdi.reindex(datas).ffill().fillna(1.0).values if indice_cdi is not None
              else np.ones(len(datas)))
    pagamentos = _pagamentos_por_pregao(datas, dividendos)
    config = VARIANTES_ACOES[variante]

    estado = novo_estado(capital / precos[0], precos[0])
    caixa = CaixaRendaFixa()
    aportes_no_cdi = CaixaRendaFixa()  # contrafactual: o mesmo dinheiro novo guardado no CDI
    aportado = capital
    n_vendas = n_compras = 0
    # Risco: a carteira (ativo + caixa) é acompanhada como uma "cota" (valor por
    # unidade), para que o dinheiro novo das compras não pareça valorização. O
    # caixa entra pelo valor bruto do CDI no dia (o IR só incide no resgate).
    unidades_caixa = 0.0          # caixa em "unidades de CDI": valor = unidades x índice
    cotas_carteira, nav_max, maior_queda = capital / precos[0], 1.0, 0.0

    def registrar(i):
        nonlocal nav_max, maior_queda
        valor = estado["quantidade"] * precos[i] + unidades_caixa * indice[i]
        nav = valor / cotas_carteira
        nav_max = max(nav_max, nav)
        maior_queda = max(maior_queda, 1 - nav / nav_max)

    for i, preco in enumerate(precos):
        if i in pagamentos and estado["quantidade"] > 0:
            caixa.depositar(datas[i], estado["quantidade"] * pagamentos[i], indice[i])
            unidades_caixa += estado["quantidade"] * pagamentos[i] / indice[i]
        if config is None or i == 0:
            registrar(i)
            continue
        aplicar_resets_por_preco(estado, preco, config)
        decisao = decidir_posicao(preco, estado, config)
        if decisao["acao"] == "comprar":
            q = estado["quantidade"] * decisao["fracao"]
            aportado += q * preco
            nav = (estado["quantidade"] * preco + unidades_caixa * indice[i]) / cotas_carteira
            cotas_carteira += q * preco / nav  # o dinheiro novo "compra cotas" da carteira
            aportes_no_cdi.depositar(datas[i], q * preco, indice[i])
            aplicar_compra(estado, q, preco, eh_reforco=True)
            n_compras += 1
        elif decisao["acao"] == "vender":
            q = estado["quantidade"] * min(decisao["fracao"], 1.0)
            caixa.depositar(datas[i], q * preco, indice[i])
            unidades_caixa += q * preco / indice[i]
            aplicar_venda(estado, q, preco, config)
            n_vendas += 1
        registrar(i)

    valor_final = estado["quantidade"] * precos[-1] + caixa.valor_liquido(datas[-1], indice[-1])
    return {
        "variante": variante, "data_inicio": datas[0], "data_fim": datas[-1],
        "aportado": aportado, "aporte_extra": aportado - capital, "valor_final": valor_final,
        "valor_de_100": 100 * valor_final / aportado,
        "retorno_aa": _retorno_anual(valor_final, aportado, datas[0], datas[-1]),
        "n_compras": n_compras, "n_vendas": n_vendas, "agiu": (n_compras + n_vendas) > 0,
        "maior_queda": maior_queda,  # maior queda da carteira (ativo + caixa) a partir de um topo
        # valor, no fim, do dinheiro novo usado nas compras se tivesse ficado no CDI:
        # é o que a referência ("manter") recebe para a comparação ser justa
        "aportes_no_cdi": aportes_no_cdi.valor_liquido(datas[-1], indice[-1]),
    }


# ---------------------------------------------------------------------------
# FIIs: o que fazer com os rendimentos
# ---------------------------------------------------------------------------

ESTRATEGIAS_FII = {
    "Não reinvestir": {"reinvestir": 0.0},
    "Reinvestir 50%": {"reinvestir": 0.5},
    "Reinvestir 100%": {"reinvestir": 1.0},
    "CDI + compra na queda (preço médio)": {"reinvestir": 0.0, "gatilho": "preco_medio"},
    "CDI + compra na queda (máxima de 12 meses)": {"reinvestir": 0.0, "gatilho": "maxima_12m"},
}

QUEDA_SOBRE_PRECO_MEDIO = 0.15   # mesma faixa de queda da regra do painel
QUEDA_SOBRE_MAXIMA_12M = 0.10    # referência de mercado (variante de comparação)


def simular_rendimentos_fii(fechamento: pd.Series, dividendos=None,
                            estrategia: str = "Não reinvestir", inicio_idx: int = 252,
                            capital: float = 10000.0, indice_cdi: pd.Series | None = None) -> dict | None:
    """Simula o destino dos rendimentos de um FII a partir de `inicio_idx`.

    - Reinvestir: a parcela reinvestida acumula (parada) até completar uma
      cota inteira, comprada ao preço do dia;
    - Não reinvestir: os rendimentos vão para o CDB (lote a lote);
    - CDI + compra na queda: os rendimentos ficam no CDB e, nos dias em que o
      preço está na faixa de queda, o saldo líquido compra cotas inteiras.
    Não há dinheiro novo: o aportado é só o capital inicial."""
    serie_completa = _sem_fuso(fechamento).dropna()
    if inicio_idx >= len(serie_completa) - 20:
        return None
    precos_todos = serie_completa.values.astype(float)
    serie = serie_completa.iloc[inicio_idx:]
    datas = serie.index
    precos = serie.values.astype(float)
    indice = (indice_cdi.reindex(datas).ffill().fillna(1.0).values if indice_cdi is not None
              else np.ones(len(datas)))
    pagamentos = _pagamentos_por_pregao(datas, dividendos)
    params = ESTRATEGIAS_FII[estrategia]
    frac = params["reinvestir"]
    gatilho = params.get("gatilho")

    cotas = capital / precos[0]
    custo = capital
    acumulado = 0.0  # parcela reinvestida aguardando completar uma cota (parada)
    caixa = CaixaRendaFixa()
    n_compras = 0
    dias_gatilho = 0

    def comprar(n, preco):
        nonlocal cotas, custo, n_compras
        cotas += n
        custo += n * preco
        n_compras += 1

    for i, preco in enumerate(precos):
        if i in pagamentos:
            recebido = cotas * pagamentos[i]
            reinvestido = recebido * frac
            caixa.depositar(datas[i], recebido - reinvestido, indice[i])
            acumulado += reinvestido
            n = math.floor(acumulado / preco + 1e-9)
            if n >= 1:
                acumulado -= n * preco
                comprar(n, preco)
        if gatilho:
            if gatilho == "preco_medio":
                na_queda = preco <= (custo / cotas) * (1 - QUEDA_SOBRE_PRECO_MEDIO)
            else:
                j = inicio_idx + i
                maxima = precos_todos[max(0, j - 251):j + 1].max()
                na_queda = preco <= maxima * (1 - QUEDA_SOBRE_MAXIMA_12M)
            if na_queda:
                dias_gatilho += 1
                disponivel = caixa.valor_liquido(datas[i], indice[i])
                n = math.floor(disponivel / preco + 1e-9)
                if n >= 1:
                    caixa.resgatar_liquido(datas[i], n * preco, indice[i])
                    comprar(n, preco)

    caixa_final = caixa.valor_liquido(datas[-1], indice[-1]) + acumulado
    valor_final = cotas * precos[-1] + caixa_final
    d = _sem_fuso(dividendos)
    ult12 = d[(d.index > datas[-1] - pd.Timedelta(days=365)) & (d.index <= datas[-1])].sum() if len(d) else 0.0
    return {
        "estrategia": estrategia, "data_inicio": datas[0], "data_fim": datas[-1],
        "aportado": capital, "valor_final": valor_final,
        "valor_de_100": 100 * valor_final / capital,
        "retorno_aa": _retorno_anual(valor_final, capital, datas[0], datas[-1]),
        "cotas_finais": cotas, "caixa_final": caixa_final,
        "renda_mensal_final": cotas * float(ult12) / 12,
        "n_compras": n_compras, "houve_gatilho": dias_gatilho > 0, "agiu": n_compras > 0,
    }


# ---------------------------------------------------------------------------
# Várias datas de entrada e resumo
# ---------------------------------------------------------------------------

def rodar_em_varias_datas(simular, fechamento, dividendos, nomes, n_datas: int = 13,
                          indice_cdi=None, capital: float = 10000.0) -> dict[str, list[dict]]:
    """Roda `simular` para cada nome de cenário em cada data de entrada.
    Devolve {nome: [resultado por data]}."""
    serie = _sem_fuso(fechamento).dropna()
    entradas = datas_de_entrada(len(serie), n_datas)
    saida: dict[str, list[dict]] = {nome: [] for nome in nomes}
    for idx in entradas:
        for nome in nomes:
            r = simular(serie, dividendos, nome, inicio_idx=idx, capital=capital, indice_cdi=indice_cdi)
            if r is not None:
                saida[nome].append(r)
    return {k: v for k, v in saida.items() if v}


TOLERANCIA_EMPATE = 1e-6  # diferença relativa abaixo disso conta como empate


def _referencia_ajustada(base: dict, cenario: dict) -> float:
    """Valor final da referência na mesma data de entrada, recebendo o MESMO
    dinheiro novo que o cenário usou, guardado no CDI (comparação justa para
    'aumentar nas quedas' e 'regra completa'). Sem aportes, é o próprio
    valor final da referência."""
    return base["valor_final"] + cenario.get("aportes_no_cdi", 0.0)


def resumir_cenarios(resultados: dict[str, dict[str, list[dict]]], tipo_por_ticker: dict[str, str],
                     referencia: str) -> pd.DataFrame:
    """Agrega por tipo de ativo e cenário. Cada caso é um par (ativo, data de
    entrada), comparado com a referência na MESMA data e com o MESMO dinheiro
    total (ver ``_referencia_ajustada``):

    - pct_vence / pct_empata / pct_perde: resultado de todos os casos — empate
      é quando o cenário não chegou a agir (ex.: a faixa nunca foi atingida);
    - pct_agiu: % dos casos em que o cenário de fato comprou ou vendeu;
    - pct_vence_quando_agiu: % de vitórias entre os casos em que agiu;
    - retorno_aa_mediano: retorno anual mediano do cenário;
    - valor_de_100_mediano / valor_de_100_ref_mediano: quanto R$ 100 viraram no
      cenário e na referência ajustada (mediana entre ativos, entrada mais
      antiga, ~4 anos);
    - pct_com_compra_na_queda: % dos casos em que a faixa de queda foi
      atingida (estratégias de FII com gatilho)."""
    linhas = []
    tipos = sorted({tipo_por_ticker[t] for t in resultados if t in tipo_por_ticker})
    for tipo in tipos:
        tickers = [t for t in resultados if tipo_por_ticker.get(t) == tipo and referencia in resultados[t]]
        if not tickers:
            continue
        for nome in list(resultados[tickers[0]].keys()):
            retornos, de100, de100_ref, gatilhos = [], [], [], []
            quedas, quedas_ref, queda_menor = [], [], 0
            vence = empata = perde = casos = agiu = vence_agiu = 0
            for t in tickers:
                lista = resultados[t].get(nome, [])
                ref = {r["data_inicio"]: r for r in resultados[t][referencia]}
                for k, r in enumerate(lista):
                    if r["retorno_aa"] is not None:
                        retornos.append(r["retorno_aa"])
                    if "houve_gatilho" in r:
                        gatilhos.append(r["houve_gatilho"])
                    base = ref.get(r["data_inicio"])
                    if base is None:
                        continue
                    valor_ref = _referencia_ajustada(base, r)
                    if "maior_queda" in r and "maior_queda" in base and nome != referencia and r.get("agiu"):
                        quedas.append(r["maior_queda"])
                        quedas_ref.append(base["maior_queda"])
                        queda_menor += r["maior_queda"] < base["maior_queda"] - 1e-9
                    if k == 0:
                        de100.append(r["valor_de_100"])
                        de100_ref.append(100 * valor_ref / r["aportado"])
                    if nome == referencia:
                        continue
                    casos += 1
                    dif = (r["valor_final"] - valor_ref) / valor_ref
                    if abs(dif) <= TOLERANCIA_EMPATE:
                        empata += 1
                    elif dif > 0:
                        vence += 1
                    else:
                        perde += 1
                    if r.get("agiu"):
                        agiu += 1
                        vence_agiu += dif > TOLERANCIA_EMPATE
            pct = (lambda n: 100 * n / casos) if casos else (lambda n: None)
            linhas.append({
                "tipo": tipo, "cenario": nome, "ativos": len(tickers),
                "casos": sum(len(resultados[t].get(nome, [])) for t in tickers),
                "retorno_aa_mediano": float(np.median(retornos)) if retornos else None,
                "pct_vence": pct(vence), "pct_empata": pct(empata), "pct_perde": pct(perde),
                "pct_agiu": pct(agiu),
                "pct_vence_quando_agiu": (100 * vence_agiu / agiu) if agiu else None,
                "pct_supera_referencia": pct(vence),  # compatibilidade
                # risco, nos casos em que o cenário agiu (mesmo ativo e data na referência)
                "maior_queda_mediana": float(np.median(quedas)) if quedas else None,
                "maior_queda_mediana_ref": float(np.median(quedas_ref)) if quedas_ref else None,
                "pct_queda_menor": (100 * queda_menor / len(quedas)) if quedas else None,
                "retorno_aa_p10": float(np.percentile(retornos, 10)) if retornos else None,
                "valor_de_100_mediano": float(np.median(de100)) if de100 else None,
                "valor_de_100_ref_mediano": float(np.median(de100_ref)) if de100_ref else None,
                "pct_com_compra_na_queda": (100 * float(np.mean(gatilhos))
                                            if gatilhos and "queda" in nome else None),
            })
    return pd.DataFrame(linhas)


def frase_evidencia(df: pd.DataFrame, tipo: str, cenario: str, referencia: str,
                    rotulo_tipo: str, moeda: str = "R$", rotulo_singular: str | None = None,
                    rotulo_referencia: str | None = None) -> str | None:
    """Frase curta para o card ("o que o histórico mostra").

    Usa a referência AJUSTADA (mesmo dinheiro total) e separa os casos em
    que o cenário de fato agiu — empates (faixa não atingida) não contam
    como derrota."""
    sub = df[(df["tipo"] == tipo)].set_index("cenario")
    if cenario not in sub.index or referencia not in sub.index:
        return None
    c = sub.loc[cenario]
    n = int(c["ativos"])
    ativos = f"1 {rotulo_singular or rotulo_tipo}" if n == 1 else f"{n} {rotulo_tipo}"

    def nome(texto):  # inicial em minúscula, exceto quando começa com sigla (ex.: CDI)
        return texto if texto[:2].isupper() else texto[0].lower() + texto[1:]

    ref = rotulo_referencia or nome(referencia)
    frase = (f"em {ativos}, nos últimos 4 anos, cada {moeda} 100 viraram "
             f"{moeda} {c['valor_de_100_mediano']:.0f} com \"{nome(cenario)}\" e {moeda} "
             f"{c['valor_de_100_ref_mediano']:.0f} com \"{ref}\" (mediana).")
    if pd.isna(c["pct_agiu"]) or c["pct_agiu"] == 0:
        return frase + " Nas datas de entrada testadas, a faixa nunca foi atingida."
    frase += f" Começando em datas diferentes, o cenário chegou a agir em {c['pct_agiu']:.0f}% dos casos"
    if not pd.isna(c["pct_vence_quando_agiu"]):
        frase += f"; quando agiu, superou em {c['pct_vence_quando_agiu']:.0f}% deles"
    return frase + "."


def exportar_evidencias(df_acoes: pd.DataFrame, df_fii: pd.DataFrame, periodo: str,
                        gerado_em: str, fonte: str) -> dict:
    """Monta o dicionário de ``carteira_analise/dados/evidencias_cenarios.json``
    (lido pelo painel) a partir das tabelas de ``resumir_cenarios``."""
    def num(v):
        return None if v is None or pd.isna(v) else round(float(v))

    acoes = {}
    for tipo in (df_acoes["tipo"].unique() if not df_acoes.empty else []):
        sub = df_acoes[df_acoes["tipo"] == tipo].set_index("cenario")
        grupo = {"ativos": int(sub["ativos"].iloc[0])}
        for chave, cen in (("realizar", "Só realizar lucro nas faixas"), ("aumentar", "Só aumentar nas quedas")):
            if cen in sub.index:
                l = sub.loc[cen]
                grupo[chave] = {"cenario": num(l["valor_de_100_mediano"]), "referencia": num(l["valor_de_100_ref_mediano"]),
                                "pct_agiu": num(l["pct_agiu"]), "pct_vence_quando_agiu": num(l["pct_vence_quando_agiu"])}
                if "maior_queda_mediana" in l and not pd.isna(l["maior_queda_mediana"]):
                    grupo[chave]["queda"] = num(100 * l["maior_queda_mediana"])
                    grupo[chave]["queda_ref"] = num(100 * l["maior_queda_mediana_ref"])
                    grupo[chave]["pct_queda_menor"] = num(l["pct_queda_menor"])
        acoes[tipo] = grupo
    fiis = None
    if not df_fii.empty:
        sub = df_fii.set_index("cenario")
        r100 = sub.loc["Reinvestir 100%"]
        queda = sub.loc["CDI + compra na queda (preço médio)"]
        fiis = {"ativos": int(r100["ativos"]), "contexto": "período de Selic alta",
                "nao_reinvestir": num(sub.loc["Não reinvestir", "valor_de_100_mediano"]),
                "reinvestir_100": {"cenario": num(r100["valor_de_100_mediano"]), "pct_perde": num(r100["pct_perde"])},
                "compra_na_queda": {"cenario": num(queda["valor_de_100_mediano"]), "pct_agiu": num(queda["pct_agiu"]),
                                    "pct_vence_quando_agiu": num(queda["pct_vence_quando_agiu"])}}
    return {"periodo": periodo, "gerado_em": gerado_em, "fonte": fonte, "acoes": acoes, "fiis": fiis}
