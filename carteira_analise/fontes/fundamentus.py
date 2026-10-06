"""Fundamentos de FIIs via Fundamentus (fundamentus.com.br) — complementa o
Yahoo Finance (``fontes/yahoo.py``), que não fornece de forma confiável
P/VP, Dividend Yield ou dados operacionais de FIIs (vacância, cap rate,
segmento, número de imóveis). Ver Seção 3.2.2 do Relatório do Projeto Final
de Curso ("fontes de dados de mercado").

Divisão de responsabilidades entre as duas fontes, para FIIs:

- **Yahoo Finance** continua responsável por: histórico de preços
  (indicadores técnicos) e histórico de proventos pagos (usado em
  ``ganhos.py`` e no cálculo de yield a partir do histórico real).
- **Fundamentus** passa a ser responsável por: P/VP e Dividend Yield
  (substituindo o campo ``priceToBook`` do Yahoo, pouco confiável para
  FIIs), além de dados que o Yahoo simplesmente não tem — Segmento, FFO
  Yield, Vacância Média, Cap Rate, Número de imóveis.

Não é uma API oficial — é uma tabela HTML pública, tratada como scraping,
com as mesmas cautelas já documentadas no projeto (fonte de terceiro,
layout pode mudar, cotação vem com atraso — adequado para triagem
fundamentalista, não para decisão intradiária). Se a busca falhar (rede
indisponível, ticker não encontrado, mudança de layout), a função devolve
``None`` e quem chama deve cair de volta no valor do Yahoo Finance — a
mesma filosofia de robustez já usada no resto do pacote (Seção 6 do Manual
Técnico).
"""
from __future__ import annotations

import io
import re
import urllib.request
from dataclasses import dataclass

import pandas as pd

URL_FII_RESULTADO = "https://www.fundamentus.com.br/fii_resultado.php"
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


@dataclass
class FundamentosFII:
    ticker: str
    segmento: str
    cotacao: float
    ffo_yield: float | None
    dividend_yield: float | None
    pvp: float | None
    valor_mercado: float | None
    liquidez: float | None
    qtd_imoveis: int | None
    cap_rate: float | None
    vacancia_media: float | None


def _parse_numero_br(texto) -> float | None:
    """Converte um número no formato brasileiro (ex: '1.234,56' ou '10,80%')
    para float. Também aceita valores que o pandas já converteu para
    float/int (algumas colunas do Fundamentus são auto-convertidas pelo
    pd.read_html, outras não, dependendo de terem '%' misturado) — nesse
    caso, o valor já está correto e não deve ser reprocessado como string.
    Devolve None se o valor estiver vazio ou não for numérico."""
    if texto is None:
        return None
    if isinstance(texto, (int, float)):
        if isinstance(texto, float) and pd.isna(texto):
            return None
        return float(texto)
    t = str(texto).strip()
    if not t or t in ("-", "N/A", "nan"):
        return None
    eh_percentual = t.endswith("%")
    t = t.rstrip("%").strip()
    t = t.replace(".", "").replace(",", ".")
    try:
        valor = float(t)
    except ValueError:
        return None
    return valor / 100 if eh_percentual else valor


def _ler_tabela_como_texto(html: str) -> pd.DataFrame:
    """Lê a primeira tabela do HTML mantendo TODAS as células como texto,
    no formato brasileiro original, para que a conversão fique inteira a
    cargo de ``_parse_numero_br``.

    Motivo (regressão encontrada na Sprint 3): com ``thousands='.'`` e
    ``decimal=','``, o pandas convertia '0,45' em '0.45' mas mantinha o
    valor como texto quando a coluna tinha algum '-' (bancos, em dívida
    bruta/patrimônio e liquidez corrente) — e o parser brasileiro removia o
    ponto como se fosse milhar, lendo 45 em vez de 0,45. Sem parâmetros, o
    pandas também leria '900.000' (milhar) como 900,0. Forçar texto em todas
    as colunas (duas leituras: a primeira só para contar as colunas) elimina
    as duas situações."""
    n_colunas = len(pd.read_html(io.StringIO(html))[0].columns)
    return pd.read_html(io.StringIO(html), thousands=None,  # padrão do pandas é thousands=","
                        converters={i: str for i in range(n_colunas)})[0]


def _baixar_html_fii_resultado() -> str:
    """Baixa o HTML bruto da tabela de FIIs do Fundamentus. Precisa de um
    User-Agent de navegador — sem isso, o site pode recusar ou degradar a
    resposta (comportamento comum de sites que bloqueiam scraping ingênuo)."""
    req = urllib.request.Request(URL_FII_RESULTADO, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as resp:
        bruto = resp.read()
    return bruto.decode("latin-1")  # Fundamentus publica em ISO-8859-1, não UTF-8


def buscar_todos_fiis_fundamentus() -> dict[str, FundamentosFII]:
    """Busca a tabela completa de FIIs do Fundamentus de uma vez (uma
    chamada de rede para ~500+ fundos, bem mais eficiente do que uma
    chamada por ticker) e devolve um dict {ticker: FundamentosFII}.

    Levanta a exceção original se a rede falhar ou o layout mudar demais
    para ser reconhecido — quem chama decide como tratar (ver
    ``buscar_fundamentos_fii``, que já aplica esse tratamento com fallback
    silencioso)."""
    html = _baixar_html_fii_resultado()
    df = _ler_tabela_como_texto(html)  # ver docstring: evita conversões automáticas do pandas

    # Normaliza nomes de coluna (o Fundamentus não muda muito, mas por
    # segurança comparamos por prefixo em minúsculas, não pelo nome exato)
    colunas = {c: str(c).strip().lower() for c in df.columns}

    def achar_coluna(*pistas: str) -> str | None:
        for col, nome in colunas.items():
            if any(p in nome for p in pistas):
                return col
        return None

    col_papel = achar_coluna("papel")
    col_segmento = achar_coluna("segmento")
    col_cotacao = achar_coluna("cota")
    col_ffo = achar_coluna("ffo")
    col_dy = achar_coluna("dividend")
    col_pvp = achar_coluna("p/vp", "pvp")
    col_valor_mercado = achar_coluna("valor de mercado")
    col_liquidez = achar_coluna("liquidez")
    col_qtd_imoveis = achar_coluna("qtd de im", "imóveis", "imoveis")
    col_cap_rate = achar_coluna("cap rate")
    col_vacancia = achar_coluna("vac")

    resultado: dict[str, FundamentosFII] = {}
    for _, linha in df.iterrows():
        ticker_bruto = linha.get(col_papel)
        if ticker_bruto is None or (isinstance(ticker_bruto, float) and pd.isna(ticker_bruto)):
            continue
        ticker = re.sub(r"\s+", "", str(ticker_bruto)).upper()
        if not ticker:
            continue

        def num(col):
            return _parse_numero_br(linha.get(col)) if col else None

        resultado[ticker] = FundamentosFII(
            ticker=ticker,
            segmento=str(linha.get(col_segmento, "")).strip() if col_segmento else "",
            cotacao=num(col_cotacao) or 0.0,
            ffo_yield=num(col_ffo),
            dividend_yield=num(col_dy),
            pvp=num(col_pvp),
            valor_mercado=num(col_valor_mercado),
            liquidez=num(col_liquidez),
            qtd_imoveis=int(num(col_qtd_imoveis)) if num(col_qtd_imoveis) is not None else None,
            cap_rate=num(col_cap_rate),
            vacancia_media=num(col_vacancia),
        )
    return resultado


# Cache em memória do processo (evita repetir a chamada de ~500 FIIs várias
# vezes na mesma sessão — diferente do cache SQLite de fontes/cache.py, que é
# por ticker; aqui o "recurso" é a tabela inteira, então cacheamos ela toda)
_cache_tabela_completa: dict[str, FundamentosFII] | None = None


def buscar_fundamentos_fii(ticker: str) -> FundamentosFII | None:
    """Busca os fundamentos de um FII específico. Usa cache em memória da
    tabela completa (primeira chamada busca tudo; chamadas seguintes na
    mesma sessão são instantâneas). Devolve None em caso de falha de rede
    ou se o ticker não for encontrado — quem chama deve cair de volta no
    Yahoo Finance nesse caso, sem quebrar o restante da análise."""
    global _cache_tabela_completa
    ticker_normalizado = re.sub(r"\.SA$", "", ticker.strip().upper())
    try:
        if _cache_tabela_completa is None:
            _cache_tabela_completa = buscar_todos_fiis_fundamentus()
        return _cache_tabela_completa.get(ticker_normalizado)
    except Exception:
        return None


# =============================================================================
# AÇÕES B3 — extensão da mesma fonte para a base original do Fundamentus
# (mais consolidada que a de FIIs). Não cobre ETFs (não são empresas com
# balanço patrimonial próprio) nem ações negociadas nos Estados Unidos
# (cobertura do Fundamentus restrita à B3) — ver Seção 3.2.2 do Relatório.
# =============================================================================

URL_ACAO_RESULTADO = "https://www.fundamentus.com.br/resultado.php"


@dataclass
class FundamentosAcao:
    ticker: str
    cotacao: float
    pl: float | None
    pvp: float | None
    roe: float | None
    roic: float | None
    margem_liquida: float | None
    divida_bruta_patrimonio: float | None
    liquidez_corrente: float | None


def _achar_coluna(colunas: dict[str, str], *pistas: str) -> str | None:
    """Localiza uma coluna por substring, tolerando acento (as tabelas de
    ações do Fundamentus têm nomes como 'Mrg. Líq.' e 'Dív.Brut/ Patrim.')."""
    def normalizar(s: str) -> str:
        s = s.lower()
        for de, para in (("í", "i"), ("î", "i"), ("ó", "o"), ("ô", "o"), ("á", "a"), ("â", "a"), ("ã", "a")):
            s = s.replace(de, para)
        return s
    for col, nome in colunas.items():
        nome_norm = normalizar(nome)
        if any(normalizar(p) in nome_norm for p in pistas):
            return col
    return None


def _baixar_html_acao_resultado() -> str:
    """Baixa o HTML bruto da tabela de ações do Fundamentus — mesmo padrão
    de acesso (User-Agent, decode Latin-1) usado para FIIs."""
    req = urllib.request.Request(URL_ACAO_RESULTADO, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as resp:
        bruto = resp.read()
    return bruto.decode("latin-1")


def buscar_todas_acoes_fundamentus() -> dict[str, FundamentosAcao]:
    """Busca a tabela completa de ações da B3 do Fundamentus — sua base
    original e mais consolidada (ver Seção 3.2.2 do Relatório). Mesmo
    padrão de uma única chamada de rede para toda a tabela, com cache em
    memória (ver ``buscar_fundamentos_acao``)."""
    html = _baixar_html_acao_resultado()
    df = _ler_tabela_como_texto(html)  # ver docstring: evita conversões automáticas do pandas
    colunas = {c: str(c).strip().lower() for c in df.columns}

    col_papel = _achar_coluna(colunas, "papel")
    col_cotacao = _achar_coluna(colunas, "cotaç", "cotac")
    col_pl = _achar_coluna(colunas, "p/l")
    col_pvp = _achar_coluna(colunas, "p/vp")
    col_roe = _achar_coluna(colunas, "roe")
    col_roic = _achar_coluna(colunas, "roic")
    col_margem_liq = _achar_coluna(colunas, "mrg.liq", "mrg. liq", "margem liq")
    col_div_patrim = _achar_coluna(colunas, "div.brut", "dív.brut", "divida bruta")
    col_liq_corrente = _achar_coluna(colunas, "liq.corr", "liq. corr")

    resultado: dict[str, FundamentosAcao] = {}
    for _, linha in df.iterrows():
        ticker_bruto = linha.get(col_papel)
        if ticker_bruto is None or (isinstance(ticker_bruto, float) and pd.isna(ticker_bruto)):
            continue
        ticker = re.sub(r"\s+", "", str(ticker_bruto)).upper()
        if not ticker:
            continue

        def num(col):
            return _parse_numero_br(linha.get(col)) if col else None

        resultado[ticker] = FundamentosAcao(
            ticker=ticker,
            cotacao=num(col_cotacao) or 0.0,
            pl=num(col_pl),
            pvp=num(col_pvp),
            roe=num(col_roe),
            roic=num(col_roic),
            margem_liquida=num(col_margem_liq),
            divida_bruta_patrimonio=num(col_div_patrim),
            liquidez_corrente=num(col_liq_corrente),
        )
    return resultado


_cache_tabela_acoes: dict[str, FundamentosAcao] | None = None


def buscar_fundamentos_acao(ticker: str) -> FundamentosAcao | None:
    """Busca os fundamentos de uma ação B3 específica. Mesmo padrão de
    cache em memória e fallback silencioso (devolve None) usado para FIIs
    — quem chama deve cair de volta no Yahoo Finance nesse caso. Não se
    aplica a ETFs nem a ações dos EUA (fora da cobertura do Fundamentus)."""
    global _cache_tabela_acoes
    ticker_normalizado = re.sub(r"\.SA$", "", ticker.strip().upper())
    try:
        if _cache_tabela_acoes is None:
            _cache_tabela_acoes = buscar_todas_acoes_fundamentus()
        return _cache_tabela_acoes.get(ticker_normalizado)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Proventos de FIIs (histórico de rendimentos efetivamente pagos)
# ---------------------------------------------------------------------------
# Página pública "Distribuição de Rendimentos" de cada FII: data-com, tipo,
# data de pagamento e valor por cota. Usada no teste de "preço atrativo"
# (Seção 13 do notebook) porque o histórico de rendimentos de FIIs do Yahoo
# Finance tem lacunas de vários anos. Mesmas cautelas do resto deste módulo.

URL_FII_PROVENTOS = "https://www.fundamentus.com.br/fii_proventos.php?papel={papel}&tipo=2"
_cache_proventos: dict[str, pd.Series] = {}


def _baixar_html_proventos_fii(papel: str) -> str:
    req = urllib.request.Request(URL_FII_PROVENTOS.format(papel=papel), headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.read().decode("latin-1")


def proventos_fii_de_html(html: str) -> pd.Series:
    """Rendimentos por cota indexados pela data-com, a partir do HTML da
    página de proventos. Só linhas do tipo "Rendimento" (amortizações
    devolvem capital e não entram no yield); mais de um rendimento na mesma
    data-com é somado."""
    df = _ler_tabela_como_texto(html)

    def normalizar(nome) -> str:
        import unicodedata
        texto = "".join(c for c in unicodedata.normalize("NFD", str(nome)) if unicodedata.category(c) != "Mn")
        return " ".join(texto.lower().split())

    colunas = {normalizar(c): c for c in df.columns}
    c_data = next((c for n, c in colunas.items() if "data com" in n), None)
    c_tipo = next((c for n, c in colunas.items() if n.startswith("tipo")), None)
    c_valor = next((c for n, c in colunas.items() if n.startswith("valor")), None)
    if not c_data or not c_valor:
        raise ValueError(f"layout inesperado da página de proventos: {list(df.columns)}")
    if c_tipo:
        df = df[df[c_tipo].astype(str).str.lower().str.contains("rendimento")]
    datas = pd.to_datetime(df[c_data], dayfirst=True, errors="coerce")
    valores = df[c_valor].map(_parse_numero_br)
    serie = pd.Series(valores.values, index=datas.values, dtype=float)
    serie = serie[serie.index.notna() & serie.notna() & (serie > 0)]
    return serie.groupby(level=0).sum().sort_index()


def buscar_proventos_fii(ticker: str) -> pd.Series | None:
    """Histórico de rendimentos pagos por cota de um FII (Fundamentus), com
    cache em memória. Devolve None se a busca falhar."""
    papel = ticker.upper().replace(".SA", "")
    if papel in _cache_proventos:
        return _cache_proventos[papel]
    try:
        serie = proventos_fii_de_html(_baixar_html_proventos_fii(papel))
    except Exception:
        return None
    _cache_proventos[papel] = serie
    return serie
