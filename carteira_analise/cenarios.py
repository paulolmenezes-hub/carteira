"""Cenários comparativos por ativo (Sprint 4).

Substitui a "posição sugerida": em vez de dizer o que fazer, cada card
mostra a SITUAÇÃO do ativo em relação às faixas de preço sobre o preço
médio (realização de lucro, queda ou fora das faixas), compara os
caminhos possíveis numa tabela padronizada e cita a evidência histórica
do backtest (Seção 12 do notebook). Para FIIs, mostra também o que teria
acontecido nos últimos 12 meses se os rendimentos tivessem sido
reinvestidos.

A regra de faixas (``regra_posicao``) continua sendo o motor: ela decide
QUAIS cenários mostrar. Os números da comparação não incluem quantidade de
cotas a negociar — só percentuais e valores.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from .backtest_cenarios import DIAS_ATE_PAGAMENTO, CaixaRendaFixa, indice_cdi_acumulado
from .planilha import _TIPO_PARA_GRUPO
from .posicoes import (
    _sem_fuso,
    chave_ordem_card,
    operacoes_por_ticker,
    renda_12m_por_cota,
)
from .contexto import faixa_12m, regua_faixas
from .tecnicos import limpar_outliers_precos, movimento_brusco_recente
from .regra_posicao import (
    CONFIG_REGRA_PADRAO,
    decidir_posicao,
    estado_a_partir_de_operacoes,
    filtro_fundamentos_compra,
    formatar_valor,
    quantidade_para_acao,
)

ARQUIVO_EVIDENCIAS = Path(__file__).parent / "dados" / "evidencias_cenarios.json"

ROTULO_SITUACAO = {
    "realizacao": "Faixa de realização de lucro",
    "queda": "Faixa de queda",
    "fora": "Fora das faixas",
}

# tipo do ativo -> chave das evidências do backtest
_TIPO_EVIDENCIA = {"acao": "Ações B3", "acao_us": "Ações EUA", "etf_br": "ETFs B3", "etf_us": "ETFs EUA"}
_ROTULO_PLURAL = {"Ações B3": "ações B3", "Ações EUA": "ações dos EUA", "ETFs B3": "ETFs B3",
                  "ETFs EUA": "ETFs dos EUA"}
_ROTULO_SINGULAR = {"Ações B3": "ação B3", "Ações EUA": "ação dos EUA", "ETFs B3": "ETF B3",
                    "ETFs EUA": "ETF dos EUA"}


# ---------------------------------------------------------------------------
# Situação, faixas e contexto de mercado
# ---------------------------------------------------------------------------

def situacao_por_variacao(variacao: float, config: dict | None = None) -> str:
    """'realizacao' a partir da menor faixa de venda (+25%), 'queda' a partir
    da faixa de compra mais rasa (−15%), 'fora' entre elas."""
    cfg = CONFIG_REGRA_PADRAO if config is None else {**CONFIG_REGRA_PADRAO, **config}
    if cfg["faixas_venda"] and variacao >= min(f[0] for f in cfg["faixas_venda"]) - 1e-9:
        return "realizacao"
    if cfg["faixas_compra"] and variacao <= max(f[0] for f in cfg["faixas_compra"]) + 1e-9:
        return "queda"
    return "fora"


def precos_das_faixas(preco_medio: float, config: dict | None = None) -> dict:
    """Preço em que começa a faixa de realização e a de queda."""
    cfg = CONFIG_REGRA_PADRAO if config is None else {**CONFIG_REGRA_PADRAO, **config}
    return {
        "realizacao": preco_medio * (1 + min(f[0] for f in cfg["faixas_venda"])),
        "queda": preco_medio * (1 + max(f[0] for f in cfg["faixas_compra"])),
    }


def distancia_da_maxima_12m(precos: pd.Series) -> float | None:
    """Quanto o preço atual está abaixo da máxima dos últimos 12 meses
    (0,06 = 6% abaixo). Contexto de mercado, independente do preço médio."""
    p = _sem_fuso(precos).dropna()
    if p.empty:
        return None
    janela = p[p.index > p.index[-1] - pd.Timedelta(days=365)]
    maxima = float(janela.max())
    return max(0.0, 1 - float(p.iloc[-1]) / maxima) if maxima > 0 else None


# ---------------------------------------------------------------------------
# Tabela padronizada de cenários (8 linhas, mesma ordem em todos os cards)
# ---------------------------------------------------------------------------

LINHAS_CENARIO = [
    ("Valor investido (custo)", "moeda"),
    ("Valor da posição hoje", "moeda"),
    ("Movimento de caixa", "caixa"),
    ("Resultado realizado", "sinal"),
    ("Resultado não realizado", "sinal"),
    ("Preço médio", "moeda"),
    ("Variação sobre o preço médio", "pct"),
    ("Renda estimada/mês", "moeda"),
]


def _coluna(quantidade, custo, preco, caixa=None, realizado=None, renda_cota_ano=None):
    """Uma coluna da tabela de cenários, na ordem de LINHAS_CENARIO."""
    if quantidade <= 1e-9:
        return [None, None, caixa, realizado, None, None, None, None]
    valor = quantidade * preco
    pm = custo / quantidade
    renda = (renda_cota_ano / 12 * quantidade) if renda_cota_ano else None
    return [custo, valor, caixa, realizado, valor - custo, pm, preco / pm - 1, renda]


def tabela_cenario(estado: dict, preco: float, decisao: dict, quantidade: float,
                   renda_12m_por_cota: float | None = None) -> dict | None:
    """Compara 'manter tudo' com o caminho indicado pela faixa atingida
    (vender parte, venda total ou aumentar a posição). Devolve None se não
    houver cenário a comparar (ex.: posição pequena demais para a fração)."""
    if decisao["acao"] not in ("vender", "comprar") or quantidade <= 0:
        return None
    q, pm = estado["quantidade"], estado["preco_medio"]
    fracao = decisao["fracao"]
    manter = _coluna(q, q * pm, preco, renda_cota_ano=renda_12m_por_cota)
    if decisao["acao"] == "vender":
        total = fracao >= 1.0
        restante = q - quantidade
        alternativa = _coluna(restante, restante * pm, preco, caixa=("venda", quantidade * preco),
                              realizado=(preco - pm) * quantidade, renda_cota_ano=renda_12m_por_cota)
        rotulo = "Venda total" if total else f"Realizar {fracao * 100:.0f}%"
        titulo = (f"Cenários na faixa de +{decisao['faixa'] * 100:.0f}%: "
                  + ("venda total" if total else f"realizar {fracao * 100:.0f}% da posição"))
    else:
        nova = q + quantidade
        alternativa = _coluna(nova, q * pm + quantidade * preco, preco, caixa=("aporte", -quantidade * preco),
                              renda_cota_ano=renda_12m_por_cota)
        rotulo = f"Aumentar {fracao * 100:.0f}%"
        titulo = (f"Cenários na faixa de −{abs(decisao['faixa']) * 100:.0f}%: "
                  f"aumentar {fracao * 100:.0f}% da posição")
    linhas = [(nome, fmt, manter[i], alternativa[i]) for i, (nome, fmt) in enumerate(LINHAS_CENARIO)]
    return {"titulo": titulo, "colunas": ["Manter tudo", rotulo], "linhas": linhas}


_NOME_GRUPO_PARTICIPACAO = {
    ("B3", "Ações"): "das Ações B3", ("B3", "FIIs"): "dos FIIs", ("B3", "ETF"): "dos ETFs B3",
    ("EUA", "Ações"): "das Ações EUA", ("EUA", "ETF"): "dos ETFs EUA",
}


def calcular_participacao(cards: list[dict], usd_brl: float | None = None) -> None:
    """Acrescenta a cada card a participação da posição (valor de hoje) na
    carteira e no seu grupo (mercado + tipo de ativo). Posições em dólar
    entram na carteira convertidas por ``usd_brl``; sem a cotação, a
    participação na carteira fica sem valor quando houver ativos dos EUA (a
    do grupo, na mesma moeda, continua valendo). Só informa: a carteira
    ainda não tem metas de alocação para comparar."""
    def valor(c):
        return float(c.get("quantidade") or 0) * float(c.get("preco_atual") or 0)

    tem_dolar = any(c.get("moeda") == "US$" for c in cards)
    total = None
    if not tem_dolar or usd_brl:
        total = sum(valor(c) * (usd_brl if c.get("moeda") == "US$" else 1.0) for c in cards)
    por_grupo: dict[tuple, float] = {}
    for c in cards:
        chave = (c.get("mercado"), c.get("grupo"))
        por_grupo[chave] = por_grupo.get(chave, 0.0) + valor(c)
    for c in cards:
        chave = (c.get("mercado"), c.get("grupo"))
        v = valor(c)
        v_brl = v * (usd_brl if c.get("moeda") == "US$" and usd_brl else 1.0)
        c["participacao"] = {
            "carteira": (v_brl / total) if total else None,
            "grupo": (v / por_grupo[chave]) if por_grupo[chave] else 0.0,
            "nome_grupo": _NOME_GRUPO_PARTICIPACAO.get(chave, "do grupo"),
        }


# ---------------------------------------------------------------------------
# Compras feitas fora da regra de faixas (ciclo atual, últimos 12 meses)
# ---------------------------------------------------------------------------

def compras_fora_da_regra(ops: list[dict], precos: pd.Series | None, data_ref, moeda: str = "R$",
                          dias: int = 365, config=None, maximo: int = 3) -> list[str]:
    """Compras do ciclo atual (desde a última vez que a posição zerou), nos
    últimos ``dias``, que a regra de faixas não teria sugerido: preço acima
    da faixa de aumento (a regra só aumenta a partir de −15% sobre o preço
    médio da época), quantidade maior que a sugerida pela faixa ou aumento
    além do limite por ciclo de queda. A compra que abre o ciclo e as
    bonificações não contam. Devolve frases para o card (as mais recentes
    primeiro, no máximo ``maximo``, e a contagem do restante)."""
    if not ops:
        return []
    ops = sorted(ops, key=lambda o: (pd.Timestamp(o["data"]), 0 if o["tipo"] == "compra" else 1))
    inicio, q = 0, 0.0
    for i, op in enumerate(ops):
        q += op["quantidade"] if op["tipo"] == "compra" else -op["quantidade"]
        if op["tipo"] == "venda" and q <= 1e-9:
            inicio, q = i + 1, 0.0
    limite_data = pd.Timestamp(data_ref) - pd.Timedelta(days=dias)
    p = _sem_fuso(precos).dropna() if precos is not None else pd.Series(dtype=float)
    cfg = {**CONFIG_REGRA_PADRAO, **(config or {})}
    frases = []
    for k in range(inicio + 1, len(ops)):
        op = ops[k]
        data = pd.Timestamp(op["data"])
        if op["tipo"] != "compra" or op.get("evento") or op["preco"] <= 0 or data < limite_data:
            continue
        hist = p[p.index < data] if len(p) else p
        estado = estado_a_partir_de_operacoes(ops[:k], historico_precos=hist if len(hist) else None,
                                              config=config)
        if estado["quantidade"] <= 0 or estado["preco_medio"] <= 0:
            continue
        pm, q_antes = estado["preco_medio"], estado["quantidade"]
        decisao = decidir_posicao(op["preco"], estado, config=config)
        var = op["preco"] / pm - 1
        motivo = None
        if decisao["acao"] == "comprar":
            sugerida = q_antes * decisao["fracao"]
            if op["quantidade"] > max(sugerida * 1.5, sugerida + 1):
                motivo = (f"quantidade de {op['quantidade'] / q_antes * 100:.0f}% da posição, acima dos "
                          f"{decisao['fracao'] * 100:.0f}% sugeridos na faixa de "
                          f"−{abs(decisao['faixa']) * 100:.0f}%")
        elif decisao["acao"] == "manter_nao_aumente":
            motivo = (f"na faixa de aumento, mas além do limite de {cfg['max_reforcos']} aumentos por ciclo de "
                      f"queda")
        elif var >= 0:
            motivo = (f"preço {var * 100:.0f}% acima do preço médio da época "
                      f"({formatar_valor(pm, moeda)}); a regra só aumenta a posição em quedas")
        else:
            limiar = max(f[0] for f in cfg["faixas_compra"])
            motivo = (f"preço {abs(var) * 100:.0f}% abaixo do preço médio da época "
                      f"({formatar_valor(pm, moeda)}); a regra só aumenta a partir de −{abs(limiar) * 100:.0f}%")
        if motivo:
            frases.append((data, f"Compra de {data:%d/%m/%Y} ({op['quantidade']:g} a "
                                 f"{formatar_valor(op['preco'], moeda)}) fora da regra de faixas: {motivo}."))
    frases.sort(key=lambda f: f[0], reverse=True)
    saida = [f for _, f in frases[:maximo]]
    if len(frases) > maximo:
        saida.append(f"… e mais {len(frases) - maximo} compra(s) fora da regra nos últimos 12 meses.")
    return saida


# ---------------------------------------------------------------------------
# FIIs: o que teria acontecido nos últimos 12 meses com reinvestimento
# ---------------------------------------------------------------------------

def _posicao_na_data(ops: list[dict], data) -> tuple[float, float]:
    """(quantidade, preço médio) da posição real em `data`, pelas operações."""
    q, custo = 0.0, 0.0
    for op in sorted(ops, key=lambda o: pd.Timestamp(o["data"])):
        if pd.Timestamp(op["data"]) > pd.Timestamp(data):
            break
        if op["tipo"] == "compra":  # eventos societários já convertidos (aplicar_eventos_societarios)
            q += op["quantidade"]
            custo += op["quantidade"] * op["preco"]
        elif q > 0:
            pm = custo / q
            q = max(q - op["quantidade"], 0.0)
            custo = q * pm
    return q, (custo / q if q > 0 else 0.0)


def retrospecto_reinvestimento(ops: list[dict], precos: pd.Series, dividendos: pd.Series,
                               indice_cdi: pd.Series | None = None, dias: int = 365) -> dict | None:
    """Simula, sobre a posição REAL do usuário nos últimos `dias` (ou desde a
    primeira compra, se for mais recente), o que teria acontecido com os
    rendimentos em três caminhos: não reinvestir (coluna 'Real'), reinvestir
    50% e reinvestir 100% — e o caminho 'CDI + compra na queda' (só compra
    cotas se o preço cair 15% abaixo do preço médio real da época).

    Regras: compra só de cotas inteiras ao preço do dia do pagamento (data-ex
    + ~10 pregões); a sobra aguardando uma cota fica parada; o dinheiro
    guardado é CDB de 100% do CDI, lote a lote, com IR regressivo e IOF,
    avaliado como se resgatado hoje. Rendimentos de FII são isentos."""
    p = _sem_fuso(precos).dropna()
    d = _sem_fuso(dividendos)
    if d is None or d.empty:
        # lista vazia de rendimentos chega com índice numérico: normaliza para datas
        d = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    if p.empty or not ops:
        return None
    fim = p.index[-1]
    primeira = min(pd.Timestamp(o["data"]) for o in ops if o["tipo"] == "compra")
    inicio = max(fim - pd.Timedelta(days=dias), primeira)
    janela = p[p.index >= inicio]
    if len(janela) < 2:
        return None
    datas = janela.index
    valores = janela.values.astype(float)
    if indice_cdi is not None:
        indice = indice_cdi.reindex(datas).ffill().fillna(1.0).values
    else:
        indice = indice_cdi_acumulado(datas, None, 0.0).values
    pagamentos: dict[int, float] = {}
    for data, valor in d[(d.index >= inicio) & (d.index <= fim)].items():
        pos = datas.searchsorted(pd.Timestamp(data))
        if pos < len(datas):
            pos = min(pos + DIAS_ATE_PAGAMENTO, len(datas) - 1)
            pagamentos[pos] = pagamentos.get(pos, 0.0) + float(valor)

    def simular(frac: float, compra_na_queda: bool = False) -> dict:
        extra, acumulado = 0, 0.0
        caixa = CaixaRendaFixa()
        recebido = 0.0
        houve_queda = False
        for i, preco in enumerate(valores):
            q_real, pm_real = _posicao_na_data(ops, datas[i])
            if i in pagamentos:
                r = (q_real + extra) * pagamentos[i]
                recebido += r
                caixa.depositar(datas[i], r * (1 - frac), indice[i])
                acumulado += r * frac
                n = math.floor(acumulado / preco + 1e-9)
                if n >= 1:
                    extra += n
                    acumulado -= n * preco
            if compra_na_queda and pm_real > 0 and preco <= pm_real * (1 - 0.15):
                houve_queda = True
                n = math.floor(caixa.valor_liquido(datas[i], indice[i]) / preco + 1e-9)
                if n >= 1:
                    caixa.resgatar_liquido(datas[i], n * preco, indice[i])
                    extra += n
        q_fim, pm_fim = _posicao_na_data(ops, fim)
        cotas = q_fim + extra
        em_caixa = caixa.valor_liquido(fim, indice[-1]) + acumulado
        renda_cota = renda_12m_por_cota(d, fim) or 0.0
        return {"cotas": cotas, "recebido": recebido, "caixa": em_caixa,
                "renda_mensal": cotas * renda_cota / 12, "patrimonio": cotas * valores[-1] + em_caixa,
                "houve_queda": houve_queda, "pm_fim": pm_fim}

    real, meio, total = simular(0.0), simular(0.5), simular(1.0)
    queda = simular(0.0, compra_na_queda=True)
    return {
        "inicio": datas[0], "fim": fim, "meses": max(1, round((fim - datas[0]).days / 30.4)),
        "colunas": ["Real", "Reinvest. 50%", "Reinvest. 100%"],
        "linhas": [
            ("Cotas hoje", "cotas", real["cotas"], meio["cotas"], total["cotas"]),
            ("Renda mensal hoje", "moeda", real["renda_mensal"], meio["renda_mensal"], total["renda_mensal"]),
            ("Em caixa (CDI líquido)", "moeda", real["caixa"], meio["caixa"], total["caixa"]),
            ("Patrimônio hoje", "moeda", real["patrimonio"], meio["patrimonio"], total["patrimonio"]),
        ],
        "compra_na_queda": {
            "atingida": queda["houve_queda"],
            "preco_faixa": real["pm_fim"] * (1 - 0.15),
            "reserva": queda["caixa"],
            "cotas": queda["cotas"], "renda_mensal": queda["renda_mensal"], "patrimonio": queda["patrimonio"],
        },
    }


# ---------------------------------------------------------------------------
# Evidências do backtest (Seção 12 do notebook)
# ---------------------------------------------------------------------------

def carregar_evidencias(caminho: Path | str | None = None) -> dict | None:
    """Resultados do backtest dos cenários, gerados pela Seção 12 do notebook
    e guardados em ``carteira_analise/dados/evidencias_cenarios.json``."""
    caminho = Path(caminho) if caminho else ARQUIVO_EVIDENCIAS
    try:
        return json.loads(caminho.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def frase_evidencia_card(evidencias: dict | None, tipo: str | None, situacao: str) -> str | None:
    """Frase "o que o histórico mostra" para o card de ações/ETFs, do
    cenário correspondente à situação (realização ou queda)."""
    if not evidencias or situacao not in ("realizacao", "queda"):
        return None
    chave = _TIPO_EVIDENCIA.get(tipo or "")
    grupo = (evidencias.get("acoes") or {}).get(chave)
    if not grupo:
        return None
    cen = grupo.get("realizar" if situacao == "realizacao" else "aumentar")
    if not cen:
        return None
    n = grupo["ativos"]
    ativos = f"1 {_ROTULO_SINGULAR[chave]}" if n == 1 else f"{n} {_ROTULO_PLURAL[chave]}"
    moeda = "US$" if "EUA" in chave else "R$"
    periodo = evidencias.get("periodo", "nos últimos 4 anos")
    inicio = f"em {ativos}, entre {periodo}, "
    if not cen.get("pct_agiu"):
        return inicio + "a faixa nunca foi atingida nas datas de entrada testadas."
    pv = cen["pct_vence_quando_agiu"]
    so = "só " if pv < 50 else ""
    if situacao == "realizacao":
        # Realizar lucro troca parte do ganho possível por proteção: a frase mostra
        # os dois lados (retorno e maior queda), e deixa claro que vale para o período.
        frase = (inicio + f"nas vezes em que o preço chegou à faixa, realizar lucro <b>rendeu mais que manter em "
                 f"{pv:.0f}% delas</b> (mediana em 4 anos: {moeda} {cen['cenario']:.0f} realizando e "
                 f"{moeda} {cen['referencia']:.0f} mantendo, por {moeda} 100)")
        if cen.get("queda") is not None and cen.get("queda_ref") is not None:
            frase += (f"; em compensação, <b>a maior queda da carteira foi de {cen['queda']:.0f}% realizando, contra "
                      f"{cen['queda_ref']:.0f}% mantendo</b>")
        return frase + (". Resultado do período testado, não uma regra: não há como saber de antemão quando uma alta "
                        "termina. Em altas longas, realizar tende a render menos que manter; em compensação, vender "
                        "parte a cada faixa de ganho sobre o preço médio transforma parte do lucro em ganho realizado, "
                        "que não se perde se o preço cair depois.")
    # As medianas incluem os casos em que a faixa não foi atingida (cenário igual à
    # referência), por isso a frase usa a proporção de casos em que a faixa foi atingida.
    return (inicio + f"nas vezes em que o preço chegou à faixa de queda, <b>aumentar a posição superou guardar o "
            f"mesmo dinheiro no CDI em {so}{pv:.0f}% delas</b> (a faixa foi atingida em {cen['pct_agiu']:.0f}% "
            f"dos casos testados, em 4 anos).")


def frase_evidencia_fiis(evidencias: dict | None) -> str | None:
    """Frase do cabeçalho do grupo de FIIs (vale para todos os FIIs)."""
    f = (evidencias or {}).get("fiis")
    if not f:
        return None
    periodo = evidencias.get("periodo", "nos últimos 4 anos")
    contexto = f" ({f['contexto']})" if f.get("contexto") else ""
    r100, queda = f["reinvestir_100"], f["compra_na_queda"]
    pv = queda.get("pct_vence_quando_agiu")
    base = (f"em {f['ativos']} FIIs, entre {periodo}{contexto}, <b>guardar os rendimentos no CDI superou reinvestir "
            f"em {r100['pct_perde']:.0f}% dos casos</b> (R$ {f['nao_reinvestir']:.0f} contra R$ {r100['cenario']:.0f} "
            f"por R$ 100, em 4 anos).")
    if pv is None:
        return base + " A faixa de queda para comprar cotas com a reserva do CDI não foi atingida nas datas testadas."
    # Faixa larga para "sem vantagem clara": entre execuções do teste (datas de
    # entrada e CDI atualizados) a proporção variou de 48% a 62%.
    if 35 <= pv <= 65:
        leitura = "não mostrou vantagem clara (resultado próximo da equivalência, não conclusivo)"
    elif pv > 65:
        leitura = "foi melhor na maioria das vezes"
    else:
        leitura = "foi pior na maioria das vezes"
    return base + (f" Comprar cotas na queda com a reserva do CDI {leitura}: superou em {pv:.0f}% "
                   f"das vezes em que a queda aconteceu.")


# ---------------------------------------------------------------------------
# Montagem dos cards
# ---------------------------------------------------------------------------

def gerar_cenarios(df_resumo, df_operacoes, fonte, tipos: dict[str, str] | None = None,
                   fundamentos: dict[str, dict] | None = None, indice_cdi_por_data=None,
                   evidencias: dict | None = None, periodo: str = "5y") -> tuple[list[dict], list[str]]:
    """Um card por posição aberta, ordenados por mercado e tipo de ativo.

    `indice_cdi_por_data`: função (DatetimeIndex) -> Series com o índice
    acumulado do CDI, usada no retrospecto dos FIIs (sem ela, o caixa não
    rende). `fonte` segue o protocolo do pacote (baixar_precos/dividendos)."""
    tipos = tipos or {}
    fundamentos = fundamentos or {}
    cards, avisos = [], []
    for ticker, ops in sorted(operacoes_por_ticker(df_resumo, df_operacoes).items()):
        # posição já encerrada (ex.: fundo incorporado, sem cotação): sem card e sem aviso
        try:
            if estado_a_partir_de_operacoes(ops)["quantidade"] <= 0:
                continue
        except (ValueError, TypeError, KeyError):
            pass  # o erro é relatado abaixo, com o histórico de preços
        try:
            precos = _sem_fuso(fonte.baixar_precos(ticker, periodo)).dropna()
        except Exception:
            precos = pd.Series(dtype=float)
        if precos.empty:
            avisos.append(f"{ticker}: sem cotação disponível — ativo não avaliado.")
            continue
        # mesma limpeza de dados do resto da ferramenta: um "pico" falso de
        # cotação não pode distorcer a máxima de 12 meses nem o histórico da regra
        precos, _, _ = limpar_outliers_precos(precos)
        preco, data_ref = float(precos.iloc[-1]), precos.index[-1]
        try:
            estado = estado_a_partir_de_operacoes(ops, historico_precos=precos)
        except (ValueError, TypeError, KeyError) as e:
            avisos.append(f"{ticker}: operações inválidas ({e}).")
            continue
        if estado["quantidade"] <= 0:
            continue
        try:
            dividendos = _sem_fuso(fonte.baixar_dividendos(ticker))
        except Exception:
            dividendos = pd.Series(dtype=float)

        tipo = tipos.get(ticker)
        pm = estado["preco_medio"]
        variacao = preco / pm - 1
        situacao = situacao_por_variacao(variacao)
        decisao = decidir_posicao(preco, estado)
        renda_cota = renda_12m_por_cota(dividendos, data_ref)
        moeda = "R$" if ticker.endswith(".SA") else "US$"

        notas, alertas = [], []
        tabela = None
        if decisao["acao"] in ("vender", "comprar"):
            if decisao["acao"] == "vender" and decisao["fracao"] >= 1.0:
                qtd = estado["quantidade"]  # venda total: a posição inteira, inclusive frações
            else:
                # ativos dos EUA aceitam frações nas corretoras; na B3, só cotas inteiras
                qtd = quantidade_para_acao(estado, decisao, permitir_fracionario=(moeda == "US$"))
            tabela = tabela_cenario(estado, preco, decisao, qtd, renda_cota)
            if tabela is None:
                notas.append(f"A posição é pequena demais para simular {decisao['fracao'] * 100:.0f}% "
                             f"(daria menos de 1 cota).")
            if decisao["acao"] == "comprar" and tipo:
                permitido, motivos = filtro_fundamentos_compra(tipo, fundamentos.get(ticker) or {},
                                                               dividendos, data_ref)
                alertas.extend(motivos)
            elif decisao["acao"] == "comprar" and not tipo:
                avisos.append(f"{ticker}: tipo de ativo não confirmado — fundamentos não verificados.")
        elif situacao != "fora":
            # na faixa, mas sem cenário novo: faixa já usada ou limite de aumentos atingido
            notas.append(decisao["motivo"][0].upper() + decisao["motivo"][1:])

        card = {
            "ticker": ticker, "tipo": tipo, "moeda": moeda,
            "mercado": "B3" if ticker.endswith(".SA") else "EUA",
            "grupo": _TIPO_PARA_GRUPO.get(tipo, "Outros"),
            "situacao": situacao, "rotulo_situacao": ROTULO_SITUACAO[situacao],
            "quantidade": estado["quantidade"], "preco_medio": pm, "preco_atual": preco,
            "variacao": variacao, "abaixo_da_maxima_12m": distancia_da_maxima_12m(precos),
            "faixas": precos_das_faixas(pm), "regua": regua_faixas(pm, preco), "faixa_12m": faixa_12m(precos),
            "tabela": tabela, "notas": notas, "alertas": alertas,
            "renda_mensal": (renda_cota / 12 * estado["quantidade"]) if renda_cota else None,
            "evidencia": frase_evidencia_card(evidencias, tipo, situacao) if tabela else None,
            "retrospecto": None,
            "alertas_compras": compras_fora_da_regra(ops, precos, data_ref, moeda),
            "movimento_brusco": movimento_brusco_recente(precos),
        }
        if tipo == "fii":
            if renda_cota:
                indice = indice_cdi_por_data(precos.index) if indice_cdi_por_data else None
                card["retrospecto"] = retrospecto_reinvestimento(ops, precos, dividendos, indice)
            else:
                # sem rendimentos no período não há o que reinvestir (ex.: ETF classificado como FII)
                card["notas"].append("Sem rendimentos pagos nos últimos 12 meses: o retrospecto de reinvestimento "
                                     "não se aplica. Se este ativo não for um FII, corrija o tipo na aba Análise de Carteira.")
        cards.append(card)
    cards.sort(key=chave_ordem_card)
    return cards, avisos
