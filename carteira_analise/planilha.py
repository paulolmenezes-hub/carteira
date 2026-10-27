"""Processamento de carteira combinando duas abas (Resumo + Operações) de
uma planilha — mesmo modelo de dados usado na Seção 9.3 do notebook do
protótipo, agora testado como parte do pacote (Seção 3.2.2 do Relatório:
"o usuário faz upload da planilha e vê o resultado").

Modelo de dados: a aba **Resumo** é o ponto de partida (posição inicial de
cada ativo — o "saldo de abertura"); a aba **Operações** traz as
movimentações feitas a partir dali. As duas são combinadas por ativo: a
posição inicial (se existir) entra como a primeira "compra" sintética, e
cada operação lançada depois se soma em cima dela — igual um extrato. Um
ativo pode ter só Resumo, só Operações, ou as duas (mais comum).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from .ganhos import Operacao, avaliar_operacoes

_PADRAO_TICKER_B3 = re.compile(r"^[A-Z]{4}\d{1,2}$")


def normalizar_ticker(ticker_bruto: str) -> tuple[str, bool]:
    """Limpa espaços e, se o ticker parecer um código B3 (4 letras + 1-2
    dígitos) sem o sufixo '.SA', completa automaticamente — ex: 'POMO4' ->
    'POMO4.SA'. Devolve (ticker_normalizado, foi_corrigido)."""
    t = str(ticker_bruto).strip().upper()
    if t.endswith(".SA") or "." in t or "-" in t:
        return t, False
    if _PADRAO_TICKER_B3.match(t):
        return t + ".SA", True
    return t, False


def _normalizar_nome_coluna(c: str) -> str:
    c = str(c).strip().lower()
    mapa_acentos = (
        ("á", "a"), ("à", "a"), ("â", "a"), ("ã", "a"),
        ("é", "e"), ("ê", "e"),
        ("í", "i"),
        ("ó", "o"), ("ô", "o"), ("õ", "o"),
        ("ú", "u"), ("ü", "u"),
        ("ç", "c"),
    )
    for de, para in mapa_acentos:
        c = c.replace(de, para)
    return c.replace("_", " ")


_MILHAR_SEM_DECIMAL = re.compile(r"^-?\d{1,3}(\.\d{3})+$")
_DATA_ISO = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}")
_EPOCA_EXCEL = pd.Timestamp("1899-12-30")


def ler_numero(valor):
    """Número de uma célula, no formato brasileiro ou não.

    Células digitadas como texto ou coladas do extrato da corretora chegam
    como '6,20', '1.000', 'R$ 4,33' — antes, ``float()`` recusava a vírgula
    (e a linha era ignorada) e lia '1.000' como 1. Vazio vira NaN; texto que
    não é número levanta ValueError."""
    if valor is None or isinstance(valor, bool):
        raise ValueError(f"número ilegível ({valor!r})")
    if isinstance(valor, (int, float)):
        return float(valor)
    texto = str(valor).strip().replace("\xa0", "").replace(" ", "")
    for simbolo in ("R$", "US$", "$"):
        texto = texto.replace(simbolo, "")
    if texto == "" or texto.lower() == "nan":
        return float("nan")
    if "," in texto:
        texto = texto.replace(".", "").replace(",", ".")
    elif _MILHAR_SEM_DECIMAL.match(texto):
        texto = texto.replace(".", "")
    return float(texto)


def ler_data(valor) -> pd.Timestamp:
    """Data de uma célula: data do Excel, texto dd/mm/aaaa (dia primeiro —
    pd.Timestamp('03/07/2024') daria 7 de março) ou número de série do
    Excel (célula sem formato de data). Levanta ValueError se ilegível."""
    if isinstance(valor, bool):
        raise ValueError(f"data ilegível ({valor!r})")
    if isinstance(valor, (int, float)):
        if valor != valor:
            raise ValueError("data vazia")
        if 20000 <= valor <= 80000:  # de 1954 a 2119 como número de série
            return _EPOCA_EXCEL + pd.Timedelta(days=float(valor))
        raise ValueError(f"data ilegível ({valor!r})")
    if isinstance(valor, str):
        texto = valor.strip()
        data = pd.to_datetime(texto, dayfirst=not _DATA_ISO.match(texto), errors="coerce")
    else:
        data = pd.Timestamp(valor)
    if data is None or pd.isna(data):
        raise ValueError(f"data ilegível ({valor!r})")
    return pd.Timestamp(data)


def _converter_coluna(serie: pd.Series, leitor) -> pd.Series:
    """Aplica o leitor célula a célula; o que não der para ler fica como
    estava, para o aviso de linha ignorada mostrar o valor original."""
    def conv(v):
        try:
            return leitor(v)
        except (ValueError, TypeError, OverflowError):
            return v
    return serie.map(conv).astype(object)


# ---------------------------------------------------------------- tipo de ativo
# Coluna opcional nas abas Resumo e Operações (ex.: "Tipo de ativo" com
# "Ação", "FII", "ETF"). O mercado (B3 ou EUA) vem do ticker: .SA é B3.
_NOMES_COLUNA_TIPO_ATIVO = ("tipo ativo", "tipo de ativo", "tipo do ativo", "classe", "classe de ativo",
                            "classe do ativo", "categoria", "tipo de investimento", "ativo tipo")
_ROTULOS_TIPO_ATIVO = {
    "acao": "acao", "acoes": "acao", "acao b3": "acao", "acao (b3)": "acao", "acao br": "acao",
    "stock": "acao", "acao eua": "acao", "acao (eua)": "acao", "acao us": "acao",
    "fii": "fii", "fiis": "fii", "fii (b3)": "fii", "fii b3": "fii", "fundo imobiliario": "fii",
    "fundos imobiliarios": "fii", "fundo de investimento imobiliario": "fii",
    "etf": "etf", "etfs": "etf", "etf b3": "etf", "etf (b3)": "etf", "etf eua": "etf", "etf (eua)": "etf",
    "etf us": "etf",
}


def normalizar_tipo_ativo(valor, ticker: str) -> str | None:
    """'Ação', 'FII', 'ETF', 'Ação (EUA)'... -> 'acao', 'fii', 'acao_us',
    'etf_br' ou 'etf_us' (o mercado sai do ticker). None se vazio ou
    desconhecido."""
    if valor is None or (isinstance(valor, float) and valor != valor):
        return None
    base = _ROTULOS_TIPO_ATIVO.get(" ".join(_normalizar_nome_coluna(str(valor)).split()))
    if base is None:
        return None
    b3 = normalizar_ticker(ticker)[0].endswith(".SA")
    if base == "fii":
        return "fii"
    if base == "acao":
        return "acao" if b3 else "acao_us"
    return "etf_br" if b3 else "etf_us"


def _achar_coluna_tipo_ativo(df: pd.DataFrame, usadas: set) -> object | None:
    """Coluna do tipo de ativo: pelo cabeçalho ou, sem cabeçalho conhecido,
    uma coluna extra cujos valores preenchidos são todos tipos de ativo."""
    for col in df.columns:
        if col not in usadas and " ".join(_normalizar_nome_coluna(col).split()) in _NOMES_COLUNA_TIPO_ATIVO:
            return col
    for col in df.columns:
        if col in usadas:
            continue
        valores = df[col].dropna()
        valores = valores[valores.astype(str).str.strip() != ""]
        if len(valores) and all(normalizar_tipo_ativo(v, "X") is not None for v in valores):
            return col
    return None


def tipos_da_planilha(df_resumo: pd.DataFrame | None,
                      df_operacoes: pd.DataFrame | None) -> tuple[dict[str, str], list[str]]:
    """Tipo de cada ativo informado na planilha (coluna opcional
    ``tipo_ativo`` das abas Resumo e Operações). Devolve (tipos, avisos);
    se o mesmo ticker aparecer com tipos diferentes, vale o da aba Resumo
    (ou o mais frequente nas Operações) e sai um aviso."""
    do_resumo: dict[str, str] = {}
    das_operacoes: dict[str, list[str]] = {}
    for df, destino in ((df_resumo, "resumo"), (df_operacoes, "operacoes")):
        if df is None or "tipo_ativo" not in df.columns:
            continue
        for _, row in df.iterrows():
            ticker = normalizar_ticker(row["ticker"])[0]
            tipo = normalizar_tipo_ativo(row["tipo_ativo"], ticker)
            if not tipo:
                continue
            if destino == "resumo":
                do_resumo.setdefault(ticker, tipo)
            else:
                das_operacoes.setdefault(ticker, []).append(tipo)
    tipos, avisos = {}, []
    for ticker in sorted(set(do_resumo) | set(das_operacoes)):
        lista = ([do_resumo[ticker]] if ticker in do_resumo else []) + das_operacoes.get(ticker, [])
        escolhido = do_resumo.get(ticker) or max(sorted(set(lista)), key=lista.count)
        tipos[ticker] = escolhido
        if len(set(lista)) > 1:
            avisos.append(f"{ticker}: tipo de ativo diferente entre linhas da planilha "
                          f"({', '.join(sorted(set(lista)))}) — usado {escolhido}; confira.")
    return tipos, avisos


def normalizar_aba_resumo(df: pd.DataFrame) -> pd.DataFrame:
    """Colunas esperadas: ticker, quantidade, data_inicio, e preco_medio OU
    valor_investido (pelo menos um dos dois)."""
    mapa = {}
    for col in df.columns:
        c = _normalizar_nome_coluna(col)
        if c in ("ticker", "ativo", "codigo", "papel"):
            mapa[col] = "ticker"
        elif c in ("quantidade", "qtd", "qtde", "quantity"):
            mapa[col] = "quantidade"
        elif c in ("preco medio", "pm", "preco medio r$", "preco", "preco entrada", "preco compra"):
            mapa[col] = "preco_medio"
        elif c in ("valor investido", "valor total", "valor aplicado", "total investido"):
            mapa[col] = "valor_investido"
        elif c in ("data inicio", "data", "data compra", "data aquisicao", "data entrada"):
            mapa[col] = "data_inicio"
    df_novo = df.rename(columns=mapa)
    faltando = [c for c in ("ticker", "quantidade", "data_inicio") if c not in df_novo.columns]
    if faltando:
        raise ValueError(f"Colunas obrigatórias não encontradas na aba Resumo: {faltando}.")
    if "preco_medio" not in df_novo.columns and "valor_investido" not in df_novo.columns:
        raise ValueError("A aba Resumo precisa ter 'preco_medio' OU 'valor_investido' preenchido.")
    for opcional in ("preco_medio", "valor_investido"):
        if opcional not in df_novo.columns:
            df_novo[opcional] = None
    col_tipo = _achar_coluna_tipo_ativo(df_novo, set(mapa) | set(mapa.values()))
    colunas = ["ticker", "quantidade", "preco_medio", "valor_investido", "data_inicio"]
    if col_tipo is not None:
        df_novo = df_novo.rename(columns={col_tipo: "tipo_ativo"})
        colunas.append("tipo_ativo")
    df_novo = df_novo[colunas].copy()
    for col in ("quantidade", "preco_medio", "valor_investido"):
        df_novo[col] = _converter_coluna(df_novo[col], ler_numero)
    df_novo["data_inicio"] = _converter_coluna(df_novo["data_inicio"], ler_data)
    return df_novo


def normalizar_aba_operacoes(df: pd.DataFrame) -> pd.DataFrame:
    """Colunas esperadas: ticker, tipo (compra/venda), quantidade, preco, data."""
    mapa = {}
    for col in df.columns:
        c = _normalizar_nome_coluna(col)
        if c in ("ticker", "ativo", "codigo", "papel"):
            mapa[col] = "ticker"
        elif c in ("tipo", "operacao", "movimento", "tipo de operacao"):
            mapa[col] = "tipo"
        elif c in ("quantidade", "qtd", "qtde", "quantity"):
            mapa[col] = "quantidade"
        elif c in ("preco", "preco unitario", "valor", "valor unitario"):
            mapa[col] = "preco"
        elif c in ("data", "data operacao", "data da operacao"):
            mapa[col] = "data"
    df_novo = df.rename(columns=mapa)
    faltando = [c for c in ("ticker", "tipo", "quantidade", "preco", "data") if c not in df_novo.columns]
    if faltando:
        raise ValueError(f"Colunas obrigatórias não encontradas na aba Operações: {faltando}.")
    col_tipo = _achar_coluna_tipo_ativo(df_novo, set(mapa) | set(mapa.values()))
    colunas = ["ticker", "tipo", "quantidade", "preco", "data"]
    if col_tipo is not None:
        df_novo = df_novo.rename(columns={col_tipo: "tipo_ativo"})
        colunas.append("tipo_ativo")
    df_novo = df_novo[colunas].copy()
    for col in ("quantidade", "preco"):
        df_novo[col] = _converter_coluna(df_novo[col], ler_numero)
    df_novo["data"] = _converter_coluna(df_novo["data"], ler_data)
    return df_novo


def _construir_operacao_da_posicao_inicial(row) -> Operacao:
    quantidade = float(row["quantidade"])
    preco_medio = row.get("preco_medio")
    if preco_medio is None or (isinstance(preco_medio, float) and pd.isna(preco_medio)):
        valor_investido = row.get("valor_investido")
        if valor_investido is None or (isinstance(valor_investido, float) and pd.isna(valor_investido)):
            raise ValueError("informe preco_medio ou valor_investido na aba Resumo")
        preco_medio = float(valor_investido) / quantidade
    return Operacao(data=pd.Timestamp(row["data_inicio"]).date(), tipo="compra",
                     quantidade=quantidade, preco=float(preco_medio))


@dataclass
class LinhaCarteira:
    ticker: str
    moeda: str
    situacao: str  # 'Aberta', 'Encerrada' ou 'erro'
    origem: str
    avisos: list[str] = field(default_factory=list)
    quantidade_aberta: float | None = None
    preco_medio_atual: float | None = None
    preco_atual: float | None = None  # cotação de mercado usada no cálculo do ganho não realizado
    ganho_realizado: float | None = None
    ganho_nao_realizado: float | None = None
    renda_recebida: float | None = None
    ganho_total: float | None = None
    erro: str | None = None
    # desdobramento/grupamento conhecido pela fonte e não lançado na planilha
    avisos_eventos: list[str] = field(default_factory=list)


def interpretar_quantidade_operacao(tipo: str, valor) -> tuple[float | None, str | None]:
    """Quantidade de uma linha da aba Operações, ou (None, motivo) se inválida.

    - venda com quantidade negativa (formato de alguns extratos de corretora):
      aceita, como venda da quantidade em valor absoluto;
    - zero, vazio, texto ou compra com quantidade negativa: inválida."""
    try:
        q = float(valor)
    except (ValueError, TypeError):
        return None, f"quantidade ilegível ({valor!r})"
    if q != q or q == 0:  # NaN (célula vazia) ou zero
        return None, "quantidade vazia ou zero"
    if q < 0:
        if tipo == "venda":
            return -q, None
        return None, f"compra com quantidade negativa ({q:g})"
    return q, None


# Eventos societários aceitos na coluna "tipo" da aba Operações. A
# quantidade é sempre a DIFERENÇA de ações (positiva): as recebidas na
# bonificação/desdobramento, as que deixaram de existir no grupamento.
_TIPOS_OPERACAO = {
    "compra": "compra", "buy": "compra", "c": "compra",
    "venda": "venda", "sell": "venda", "v": "venda",
    "bonificacao": "bonificacao", "bonus": "bonificacao",
    "desdobramento": "desdobramento", "desdobro": "desdobramento", "split": "desdobramento",
    "grupamento": "grupamento", "agrupamento": "grupamento", "inplit": "grupamento",
    "reverse split": "grupamento",
}
EVENTOS_SOCIETARIOS = ("bonificacao", "desdobramento", "grupamento")


def normalizar_tipo_operacao(valor) -> str | None:
    """'Compra', 'Bonificação', 'Desdobro'... -> tipo interno, ou None."""
    texto = _normalizar_nome_coluna(str(valor)).replace("_", " ").strip()
    return _TIPOS_OPERACAO.get(texto)


def aplicar_eventos_societarios(ops: list[dict]) -> list[dict]:
    """Converte a lista de operações (dicts data/tipo/quantidade/preco, já
    validados) para o "padrão de hoje", em que só existem compras e vendas:

    - desdobramento e grupamento mudam o número de ações sem mudar o custo
      total. As operações ANTERIORES ao evento são reescritas pelo fator
      (quantidade x fator, preço / fator) — o mesmo ajuste que as cotações
      históricas já trazem, o que mantém operações e histórico de preços na
      mesma escala. Custo total, ganho realizado e quantidade final não mudam;
    - bonificação vira compra das ações recebidas pelo custo atribuído
      informado pela empresa (coluna preço; 0 se não houver), como manda a
      Receita Federal, marcada como evento (não conta como aumento na queda).

    Compra com preço 0 é tratada como bonificação. Levanta ValueError se o
    evento acontecer sem posição ou se o grupamento consumir a posição toda."""
    ordem = {"compra": 0, "venda": 1}  # no mesmo dia, o evento vem depois das negociações
    ops = sorted(ops, key=lambda o: (pd.Timestamp(o["data"]), ordem.get(o["tipo"], 2)))
    saida: list[dict] = []
    q = 0.0
    for op in ops:
        op = dict(op)
        tipo = op["tipo"]
        if tipo == "compra" and op["preco"] == 0:
            tipo = "bonificacao"
        if tipo in ("desdobramento", "grupamento"):
            data = pd.Timestamp(op["data"])
            if q <= 0:
                raise ValueError(f"{tipo} em {data:%d/%m/%Y} sem posição aberta")
            novo_q = q + op["quantidade"] if tipo == "desdobramento" else q - op["quantidade"]
            if novo_q <= 0:
                raise ValueError(f"grupamento em {data:%d/%m/%Y} de {op['quantidade']:g} ações, mas a "
                                 f"posição era de {q:g}")
            fator = novo_q / q
            for anterior in saida:
                anterior["quantidade"] *= fator
                anterior["preco"] /= fator
            q = novo_q
            continue
        if tipo == "bonificacao":
            if q <= 0:
                raise ValueError(f"bonificação em {pd.Timestamp(op['data']):%d/%m/%Y} sem posição aberta")
            op.update(tipo="compra", evento=True)
        if op["tipo"] == "venda":
            q -= op["quantidade"]
        else:
            q += op["quantidade"]
        saida.append(op)
    return saida


def operacoes_do_ticker(row_resumo, grupo_operacoes: pd.DataFrame | None) -> tuple[list[dict], list[str], int]:
    """Operações válidas de um ativo (posição inicial da aba Resumo + aba
    Operações), já com os eventos societários aplicados. Devolve
    (operações, avisos sobre linhas ignoradas, nº de linhas aproveitadas)."""
    ops: list[dict] = []
    avisos: list[str] = []
    if row_resumo is not None:
        try:
            op = _construir_operacao_da_posicao_inicial(row_resumo)
            ops.append({"data": pd.Timestamp(op.data), "tipo": "compra",
                        "quantidade": float(op.quantidade), "preco": float(op.preco)})
        except (ValueError, TypeError) as e:
            avisos.append(f"posição inicial (aba Resumo) inválida — {e}")
    n = 0
    if grupo_operacoes is not None:
        for _, row in grupo_operacoes.iterrows():
            tipo = normalizar_tipo_operacao(row["tipo"])
            if tipo is None:
                avisos.append(f"tipo inválido {row['tipo']!r} numa operação (use compra, venda, "
                              f"bonificação, desdobramento ou grupamento)")
                continue
            try:
                data_op = pd.Timestamp(row["data"])
                if pd.isna(data_op):
                    raise ValueError
                preco_bruto = row["preco"]
                if tipo in ("desdobramento", "grupamento"):
                    preco_op = 0.0  # não há preço: só muda o número de ações
                elif tipo == "bonificacao" and (preco_bruto is None or pd.isna(preco_bruto)):
                    preco_op = 0.0  # custo atribuído não informado
                else:
                    preco_op = float(preco_bruto)
            except (ValueError, TypeError):
                avisos.append(f"operação com data/preço ilegível (data={row['data']!r}) — linha ignorada")
                continue
            quantidade, motivo = interpretar_quantidade_operacao(
                "venda" if tipo == "grupamento" else tipo, row["quantidade"])
            if quantidade is None:
                avisos.append(f"{tipo} de {data_op:%d/%m/%Y} com {motivo} — linha ignorada")
                continue
            if preco_op < 0 or (preco_op == 0 and tipo == "venda") or preco_op != preco_op:
                avisos.append(f"{tipo} de {data_op:%d/%m/%Y} com preço inválido ({preco_bruto!r}) — linha ignorada")
                continue
            ops.append({"data": data_op, "tipo": tipo, "quantidade": quantidade, "preco": preco_op})
            n += 1
    return ops, avisos, n



def _fator_redondo(fator: float) -> float:
    """10.3 -> 10; 0.098 -> 0.1; 1.1 fica 1.1 (bonificação)."""
    if fator >= 1.5:
        return float(round(fator))
    if fator <= 1 / 1.5:
        return 1 / round(1 / fator)
    return round(fator, 2)


def detectar_saltos_de_patamar(precos: pd.Series, janela: int = 10, fator_minimo: float = 3.0) -> list[tuple]:
    """Mudanças de escala de um dia para o outro no histórico de preços (mesmo
    critério da limpeza em tecnicos.py): sinal de desdobramento ou grupamento
    que a fonte não ajustou. Devolve [(data, fator)], com fator = ações novas
    por ação antiga (10 = desdobramento de 1 para 10)."""
    import numpy as np
    p = precos.dropna()
    if getattr(p.index, "tz", None) is not None:
        p = p.copy()
        p.index = p.index.tz_localize(None)
    if len(p) < 2 * janela:
        return []
    antes = p.rolling(janela).median().shift(1)
    depois = p[::-1].rolling(janela).median()[::-1]
    razao = (antes / depois).values
    marcados = np.where((razao >= fator_minimo) | (razao <= 1 / fator_minimo))[0]
    saltos, grupo = [], []
    for i in marcados:
        if grupo and i != grupo[-1] + 1:
            saltos.append(grupo)
            grupo = []
        grupo.append(i)
    if grupo:
        saltos.append(grupo)
    eventos = []
    for g in saltos:
        # dia do evento: o maior salto de um pregão para o seguinte dentro do trecho
        vals = p.values
        i = max(g, key=lambda k: abs(np.log(vals[k] / vals[k - 1])) if k > 0 else 0.0)
        eventos.append((pd.Timestamp(p.index[i]), _fator_redondo(float(razao[i]))))
    return eventos


def verificar_eventos_nao_registrados(ops: list[dict], eventos_fonte=None, precos: pd.Series | None = None,
                                      tolerancia_dias: int = 15) -> list[str]:
    """Compara os desdobramentos/grupamentos conhecidos pela fonte de mercado
    (lista de eventos e saltos de patamar no histórico de preços) com os
    eventos lançados na planilha. Para cada evento ocorrido com a posição
    aberta e sem lançamento próximo na planilha, devolve um aviso com a linha
    a lançar. `ops` são as operações ANTES de aplicar os eventos (como saem de
    `operacoes_do_ticker`). Não altera nada: a planilha é do usuário, e a
    fonte pode errar."""
    candidatos: list[tuple] = []
    if eventos_fonte is not None and len(eventos_fonte):
        for data, fator in eventos_fonte.items():
            data = pd.Timestamp(data)
            if data.tzinfo is not None:
                data = data.tz_localize(None)
            if fator and float(fator) > 0 and abs(float(fator) - 1) > 1e-9:
                candidatos.append((data.normalize(), float(fator)))
    if precos is not None and len(precos):
        for data, fator in detectar_saltos_de_patamar(precos):
            if not any(abs((data - d).days) <= tolerancia_dias for d, _ in candidatos):
                candidatos.append((data.normalize(), fator))
    if not candidatos or not ops:
        return []
    lancados = [pd.Timestamp(o["data"]) for o in ops
                if o["tipo"] in EVENTOS_SOCIETARIOS or (o["tipo"] == "compra" and o["preco"] == 0)]
    avisos = []
    for data, fator in sorted(candidatos):
        q = 0.0
        for o in ops:
            if pd.Timestamp(o["data"]) >= data:
                continue
            sinal = -1 if o["tipo"] in ("venda", "grupamento") else 1
            q += sinal * o["quantidade"]
        if q <= 0 or any(abs((data - d).days) <= tolerancia_dias for d in lancados):
            continue
        if fator > 1:
            delta = q * (fator - 1)
            if fator >= 1.5:
                nome, tipo = f"desdobramento de 1 para {fator:g}", "desdobramento"
            else:
                nome, tipo = f"bonificação de {(fator - 1) * 100:.0f}%", "bonificação"
        else:
            delta = q * (1 - fator)
            nome, tipo = f"grupamento de {1 / fator:g} para 1", "grupamento"
        avisos.append(
            f"{nome} em {data:%d/%m/%Y} (fonte de mercado) não está registrado na planilha. Com {q:g} cotas antes "
            f"do evento, lance na aba Operações: tipo \"{tipo}\", quantidade {round(delta, 4):g}, data "
            f"{data:%d/%m/%Y}"
            + (" e, no preço, o custo atribuído informado pela empresa (ou 0)" if tipo == "bonificação" else "")
            + ". Se a quantidade da planilha já é a posterior ao evento, ignore este aviso.")
    return avisos


def processar_carteira_combinada(
    df_resumo: pd.DataFrame | None,
    df_operacoes: pd.DataFrame | None,
    fonte,
    periodo: str = "5y",
) -> list[LinhaCarteira]:
    """Uma ``LinhaCarteira`` por ativo, combinando a posição inicial (aba
    Resumo, quando existir) com as movimentações (aba Operações, quando
    existirem). ``fonte`` precisa expor ``baixar_precos(ticker, periodo)``
    e ``baixar_dividendos(ticker)`` (mesmo protocolo de ``fontes.yahoo``,
    usado em ``carteira.analisar_ativo``) — sem I/O de rede direto aqui,
    o que mantém esta função testável com fonte falsa."""
    tickers_resumo: dict[str, object] = {}
    if df_resumo is not None:
        for _, row in df_resumo.iterrows():
            ticker, _ = normalizar_ticker(row["ticker"])
            tickers_resumo[ticker] = row

    tickers_operacoes: dict[str, pd.DataFrame] = {}
    if df_operacoes is not None and not df_operacoes.empty:
        tickers_normalizados = df_operacoes["ticker"].apply(lambda t: normalizar_ticker(t)[0])
        for ticker, grupo in df_operacoes.groupby(tickers_normalizados):
            tickers_operacoes[ticker] = grupo

    todos_tickers = sorted(set(tickers_resumo) | set(tickers_operacoes))
    linhas: list[LinhaCarteira] = []

    for ticker in todos_tickers:
        moeda = "R$" if ticker.endswith(".SA") else "US$"
        ops, avisos, n_operacoes_lancadas = operacoes_do_ticker(
            tickers_resumo.get(ticker), tickers_operacoes.get(ticker))
        ops_brutas = list(ops)
        try:
            ops = aplicar_eventos_societarios(ops)
        except ValueError as e:
            linhas.append(LinhaCarteira(ticker=ticker, moeda=moeda, situacao="erro",
                                         origem="", avisos=avisos, erro=str(e)))
            continue
        operacoes = [Operacao(data=o["data"].date(), tipo=o["tipo"], quantidade=o["quantidade"],
                              preco=o["preco"]) for o in ops]

        if not operacoes:
            linhas.append(LinhaCarteira(ticker=ticker, moeda=moeda, situacao="erro",
                                         origem="", avisos=avisos,
                                         erro="nenhuma posição/operação válida"))
            continue

        try:
            serie_precos = fonte.baixar_precos(ticker, periodo)
            preco_atual = float(serie_precos.iloc[-1])
        except Exception:
            serie_precos, preco_atual = None, None
        try:
            eventos_fonte = fonte.baixar_eventos(ticker) if hasattr(fonte, "baixar_eventos") else None
        except Exception:
            eventos_fonte = None
        avisos_eventos = verificar_eventos_nao_registrados(ops_brutas, eventos_fonte, serie_precos)
        try:
            divs = fonte.baixar_dividendos(ticker)
        except Exception:
            divs = pd.Series(dtype=float)

        try:
            r = avaliar_operacoes(operacoes, preco_atual=preco_atual, dividendos=divs)
        except ValueError as e:
            linhas.append(LinhaCarteira(ticker=ticker, moeda=moeda, situacao="erro",
                                         origem="", avisos=avisos, erro=str(e)))
            continue

        situacao = "Aberta" if r.quantidade_aberta > 0 else "Encerrada"
        origem = []
        if ticker in tickers_resumo:
            origem.append("posição inicial")
        if n_operacoes_lancadas:
            origem.append(f"{n_operacoes_lancadas} operação(ões)")

        linhas.append(LinhaCarteira(
            ticker=ticker, moeda=moeda, situacao=situacao, origem=" + ".join(origem), avisos=avisos,
            avisos_eventos=avisos_eventos,
            quantidade_aberta=r.quantidade_aberta, preco_medio_atual=r.preco_medio_aberto,
            preco_atual=preco_atual,
            ganho_realizado=r.ganho_realizado, ganho_nao_realizado=r.ganho_nao_realizado,
            renda_recebida=r.renda_recebida, ganho_total=r.ganho_total,
        ))

    return linhas


# Nomes aceitos para as duas abas. Os nomes alternativos vieram da validação
# com o usuário-piloto: "Resumo" foi lido como "posição atual", quando é o
# ponto de partida (a primeira compra ou a posição numa data de início).
NOMES_ABA_RESUMO = ["Resumo", "Resumo da carteira", "Posicao", "Posição", "Posição inicial",
                    "1ª compra", "1a compra", "1º compra", "1o compra", "Primeira compra"]
NOMES_ABA_OPERACOES = ["Operações", "Operacoes", "Movimentações", "Movimentacoes",
                       "Compras e vendas posteriores"]


def _normalizar_nome_aba(nome: str) -> str:
    """Como os nomes de coluna, e também: ª/º viram a/o e espaços repetidos somem."""
    texto = _normalizar_nome_coluna(str(nome)).replace("ª", "a").replace("º", "o").replace(".", "")
    return " ".join(texto.split())


def achar_aba(abas: dict[str, pd.DataFrame], nomes_possiveis: list[str]) -> pd.DataFrame | None:
    """Procura uma aba pelo nome, tolerando maiúsculas/minúsculas, acento,
    ª/º e espaços a mais."""
    alvo = {_normalizar_nome_aba(n) for n in nomes_possiveis}
    for nome_real, df in abas.items():
        if _normalizar_nome_aba(nome_real) in alvo:
            return df
    return None


@dataclass
class LinhaDashboardAtivo:
    """Uma linha do dashboard da carteira: posição inicial, compras e
    vendas feitas depois, posição final, ganho e rentabilidade — tudo
    separado por ativo, pra dar uma visão de extrato completo (Seção 4.2
    do Relatório: "dashboard com dados da carteira, por ativo"). Também
    carrega mercado/tipo/situação, usados pra agrupar a exibição (B3 antes
    de EUA, por tipo de ativo, ativas separadas de encerradas)."""
    ticker: str
    moeda: str
    mercado: str = "B3"  # "B3" ou "EUA"
    tipo_grupo: str = "Ações"  # "Ações", "FIIs" ou "ETF" — agrupamento amplo p/ exibição
    situacao: str = "Ativa"  # "Ativa" ou "Encerrada"
    saldo_inicial_valor: float | None = None
    saldo_inicial_data: object = None  # date ou None, se o ativo não tiver aba Resumo
    qtde_inicial: float | None = None
    preco_medio: float | None = None  # posição aberta: custo médio da posição (o mesmo dos cards); encerrada: de tudo que foi comprado
    compras_valor: float = 0.0
    compras_quantidade: float = 0.0
    vendas_valor: float = 0.0
    vendas_quantidade: float = 0.0
    # bonificação/desdobramento (+) e grupamento (−): muda a quantidade sem ser compra ou venda
    eventos_quantidade: float = 0.0
    eventos_custo: float = 0.0  # custo atribuído das bonificações (Receita Federal)
    saldo_final_valor: float | None = None
    saldo_final_qtde: float | None = None
    saldo_final_data: object = None  # date da análise, ou None se ainda não foi calculado
    preco_final: float | None = None
    ganho_realizado: float | None = None
    ganho_nao_realizado: float | None = None
    renda_recebida: float | None = None
    rotulo_renda: str = "Dividendos/JCP"  # ou "Rendimentos", se FII
    rentabilidade_total: float | None = None
    rentabilidade_pct: float | None = None


_TIPO_PARA_GRUPO = {
    "acao": "Ações", "acao_us": "Ações",
    "fii": "FIIs",
    "etf_br": "ETF", "etf_us": "ETF",
}


def construir_dashboard_por_ativo(
    df_resumo: pd.DataFrame | None,
    df_operacoes: pd.DataFrame | None,
    linhas_ganho: "list[LinhaCarteira] | None" = None,
    tipos: dict[str, str] | None = None,
    data_analise=None,
) -> list[LinhaDashboardAtivo]:
    """Monta o dashboard por ativo, combinando as mesmas duas abas usadas em
    ``processar_carteira_combinada`` — mas aqui SEPARANDO posição inicial de
    compras/vendas feitas depois (em vez de tratar tudo como um extrato
    fundido), porque o dashboard quer mostrar cada etapa isolada. Saldo
    final e renda recebida só aparecem se ``linhas_ganho`` for informado
    (resultado já calculado de ``processar_carteira_combinada``) — sem
    isso, o dashboard mostra só o que dá pra saber direto da planilha, sem
    precisar de preço de mercado ao vivo.

    Custo médio ponderado (mesmo método usado pela Receita Federal): uma
    venda parcial não separa a posição em "lote aberto" e "lote fechado"
    — o ativo aparece numa linha só, em "Ativa", com o ganho já realizado
    daquela venda somado ao ganho não realizado do que sobrou."""
    tipos = tipos or {}
    linhas_ganho_por_ticker = {l.ticker: l for l in (linhas_ganho or [])}

    tickers_resumo: dict[str, object] = {}
    if df_resumo is not None:
        for _, row in df_resumo.iterrows():
            ticker, _ = normalizar_ticker(row["ticker"])
            tickers_resumo[ticker] = row

    tickers_operacoes: dict[str, pd.DataFrame] = {}
    if df_operacoes is not None and not df_operacoes.empty:
        tickers_normalizados = df_operacoes["ticker"].apply(lambda t: normalizar_ticker(t)[0])
        for ticker, grupo in df_operacoes.groupby(tickers_normalizados):
            tickers_operacoes[ticker] = grupo

    todos_tickers = sorted(set(tickers_resumo) | set(tickers_operacoes))
    linhas: list[LinhaDashboardAtivo] = []

    for ticker in todos_tickers:
        moeda = "R$" if ticker.endswith(".SA") else "US$"
        tipo_ativo = tipos.get(ticker)
        linha = LinhaDashboardAtivo(
            ticker=ticker, moeda=moeda,
            mercado="B3" if ticker.endswith(".SA") else "EUA",
            tipo_grupo=_TIPO_PARA_GRUPO.get(tipo_ativo, "Ações"),
            rotulo_renda="Rendimentos" if tipo_ativo == "fii" else "Dividendos/JCP",
        )

        if ticker in tickers_resumo:
            row = tickers_resumo[ticker]
            try:
                quantidade = float(row["quantidade"])
                preco_medio_resumo = row.get("preco_medio")
                if preco_medio_resumo is None or (
                    isinstance(preco_medio_resumo, float) and pd.isna(preco_medio_resumo)
                ):
                    preco_medio_resumo = float(row["valor_investido"]) / quantidade
                linha.saldo_inicial_valor = quantidade * float(preco_medio_resumo)
                linha.saldo_inicial_data = pd.Timestamp(row["data_inicio"]).date()
                linha.qtde_inicial = quantidade
            except (ValueError, TypeError, KeyError):
                pass  # posição inicial inválida — dashboard mostra em branco, sem quebrar

        if ticker in tickers_operacoes:
            for _, row in tickers_operacoes[ticker].iterrows():
                tipo = normalizar_tipo_operacao(row["tipo"])
                quantidade_op, _ = interpretar_quantidade_operacao(
                    "venda" if tipo == "grupamento" else (tipo or "compra"), row["quantidade"])
                if tipo is None or quantidade_op is None:
                    continue  # linha inválida — mesma tolerância de processar_carteira_combinada
                if tipo in ("desdobramento", "grupamento"):
                    linha.eventos_quantidade += quantidade_op if tipo == "desdobramento" else -quantidade_op
                    continue
                try:
                    preco_op = float(row["preco"])
                except (ValueError, TypeError):
                    if tipo != "bonificacao":
                        continue
                    preco_op = 0.0
                if preco_op != preco_op:  # vazio
                    if tipo != "bonificacao":
                        continue
                    preco_op = 0.0
                if tipo == "bonificacao" or (tipo == "compra" and preco_op == 0):
                    linha.eventos_quantidade += quantidade_op
                    linha.eventos_custo += quantidade_op * preco_op
                elif tipo == "compra" and preco_op > 0:
                    linha.compras_quantidade += quantidade_op
                    linha.compras_valor += quantidade_op * preco_op
                elif tipo == "venda" and preco_op > 0:
                    linha.vendas_quantidade += quantidade_op
                    linha.vendas_valor += quantidade_op * preco_op

        # Custo médio ponderado de TUDO que já foi comprado (posição inicial
        # + compras), independente do ativo ainda estar aberto ou não —
        # não depende do estado transitório do ganhos.py, então funciona
        # igual pra posição aberta ou já totalmente encerrada.
        qtde_total_comprada = (linha.qtde_inicial or 0.0) + linha.compras_quantidade + linha.eventos_quantidade
        valor_total_comprado = (linha.saldo_inicial_valor or 0.0) + linha.compras_valor + linha.eventos_custo
        if qtde_total_comprada > 0:
            linha.preco_medio = valor_total_comprado / qtde_total_comprada

        resultado_ganho = linhas_ganho_por_ticker.get(ticker)
        if resultado_ganho is not None:
            linha.situacao = "Ativa" if resultado_ganho.quantidade_aberta else "Encerrada"
            linha.saldo_final_qtde = resultado_ganho.quantidade_aberta
            linha.preco_final = resultado_ganho.preco_atual
            if resultado_ganho.quantidade_aberta and resultado_ganho.preco_atual is not None:
                linha.saldo_final_valor = resultado_ganho.quantidade_aberta * resultado_ganho.preco_atual
                linha.saldo_final_data = data_analise
            # posição aberta: mesmo preço médio dos cards (custo médio ponderado
            # da posição em aberto, regra da Receita); encerrada: custo de tudo comprado
            if resultado_ganho.quantidade_aberta and resultado_ganho.preco_medio_atual is not None:
                linha.preco_medio = resultado_ganho.preco_medio_atual
            linha.ganho_realizado = resultado_ganho.ganho_realizado
            linha.ganho_nao_realizado = resultado_ganho.ganho_nao_realizado
            linha.renda_recebida = resultado_ganho.renda_recebida

            partes_rentabilidade = [
                v for v in (resultado_ganho.ganho_realizado, resultado_ganho.ganho_nao_realizado,
                            resultado_ganho.renda_recebida) if v is not None
            ]
            if partes_rentabilidade:
                linha.rentabilidade_total = sum(partes_rentabilidade)
                if valor_total_comprado > 0:
                    linha.rentabilidade_pct = linha.rentabilidade_total / valor_total_comprado
        else:
            # Sem linhas_ganho ainda (usuário não clicou "Analisar carteira") —
            # só dá pra saber se está aberta olhando compra vs. venda acumulada.
            qtde_liquida = qtde_total_comprada - linha.vendas_quantidade
            linha.situacao = "Ativa" if qtde_liquida > 1e-9 else "Encerrada"

        linhas.append(linha)

    return linhas
