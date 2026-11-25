"""Painel web de análise de carteira (ações, FIIs, ETFs) — Streamlit.

Interface acessível sem conhecimento técnico de programação (Seção 3.2.2
do Relatório do Projeto Final de Curso): o usuário faz upload da planilha
e vê o resultado, sem células de código nem ambiente de notebook.

Toda a lógica de cálculo é reaproveitada do pacote ``carteira_analise``
(motor de indicadores, pontuação, ganhos e planilha), já testado
(175 testes automatizados) — este arquivo é só a camada de interface.

Resultados ficam guardados em ``st.session_state`` (não só dentro do
`if st.button(...)`) para que rodar UMA análise não apague a outra — todo
clique reexecuta o script inteiro do zero, e sem essa persistência o
resultado anterior desapareceria da tela.

Organizado em 4 abas: Carteira (carregar e analisar), Ativo Específico, Dashboard da
Carteira (cenários comparativos por ativo + extrato por ativo) e Guia de
Indicadores (glossário).
"""
import html
import io
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import altair as alt
import pandas as pd
import streamlit as st

from carteira_analise.carteira import analisar_ativo
from carteira_analise.fontes import yahoo as fonte_yahoo
from carteira_analise.fontes.cache import FonteComCache
from carteira_analise.backtest_cenarios import indice_cdi_acumulado
from carteira_analise.cenarios import (calcular_participacao, carregar_evidencias, frase_evidencia_fiis,
                                       gerar_cenarios)
from carteira_analise.fontes.bcb import buscar_cdi_diario
from carteira_analise.fontes.b3_cotahist import carregar_base as carregar_base_b3, completar_precos
from carteira_analise.fontes.cvm_fii import base_recente, serie_rendimentos, serie_valor_patrimonial
from carteira_analise.fontes.fundamentus import (buscar_fundamentos_acao, buscar_grupos_setoriais, buscar_proventos_acao, buscar_tabela_pares, buscar_proventos_acao_detalhe,
                                                 buscar_proventos_fii, buscar_proventos_fii_detalhe,
                                                 buscar_referencias_pl)
from carteira_analise.proventos_ir import a_receber, eh_bdr, proventos_recebidos, resumo_por_ano, sem_data_antigos
from carteira_analise.fontes.rendimentos import (FonteComRendimentosConferidos, ajustar_por_desdobramentos,
                                                  conferir_fontes)
from carteira_analise.contexto import (ATENCAO, BOM, acrescentar_referencias_pl, cdi_liquido_12m,
                                      contexto_do_ativo, faixa_12m)
from carteira_analise.fundamentalistas import classificar_fii, motivo_classificacao, texto_regularidade
from carteira_analise.oportunidades import BAZIN_YIELD, preco_teto_bazin
from carteira_analise.posicoes import fundamentos_de_resultado
from carteira_analise.tabela_indicadores import montar_tabelas
from carteira_analise.raio_x import (pares_fii, posicao_entre_pares, proventos_por_ano, proximo_provento, risco_12m, tendencia_rompida,
                                     selecionar_pares, sinal, sinal_pl, LIMIARES_FII)
from carteira_analise.metas import (CLASSES as CLASSES_META, METAS_PADRAO, TOLERANCIA_PP, classe_do_ativo, comparar,
                                    dividir_aporte, metas_de_texto, metas_para_texto, LIMITES_PADRAO,
                                    limites_de_texto, limites_para_texto, metas_e_limites, pesos_dos_ativos)
from carteira_analise.visao_geral import (cdi_acumulado, linha_do_tempo_ativo, meses_entre, por_ano,
                                          rentabilidade_cotas, resultado_em_reais, resumo_periodo, somar,
                                          tir_anual)
from carteira_analise.regra_posicao import formatar_valor
from carteira_analise.planilha import (
    NOMES_ABA_OPERACOES,
    NOMES_ABA_RESUMO,
    achar_aba,
    construir_dashboard_por_ativo,
    linhas_resumo_por_ticker,
    operacoes_do_ticker,
    normalizar_aba_operacoes,
    normalizar_aba_resumo,
    normalizar_ticker,
    processar_carteira_combinada,
    tipos_da_planilha,
)
from carteira_analise.tecnicos import calcular_rsi, movimento_brusco_recente

st.set_page_config(page_title="Análise de Carteira", page_icon="📊", layout="wide")


@st.cache_resource
def _obter_fonte_com_cache() -> FonteComCache:
    """Uma única instância de cache (SQLite local ao servidor) reaproveitada
    entre execuções do script — evita repetir chamadas de rede pro mesmo
    ticker/período dentro do TTL (12h por padrão). `st.cache_resource` é o
    jeito certo do Streamlit de manter um recurso com estado (aqui, a
    conexão SQLite) vivo entre reruns, em vez de recriar a cada clique."""
    caminho_cache = Path(tempfile.gettempdir()) / "carteira_analise_cache.sqlite"
    return FonteComCache(fonte_yahoo, caminho_db=caminho_cache)


fonte_dados = _obter_fonte_com_cache()


def _usuarios_configurados() -> dict:
    """Acessos definidos em Settings → Secrets do Streamlit Cloud.

    Formato por usuário (um bloco por pessoa; validade opcional, AAAA-MM-DD):

        [usuarios.paulo]
        senha = "senha-do-paulo"

        [usuarios.teste1]
        senha = "senha-provisoria"
        validade = "2026-10-12"

    Para cortar um acesso, apague o bloco da pessoa (ou deixe a validade
    vencer): os demais acessos não mudam. A senha única antiga
    ('senha_painel') continua aceita, como usuário "admin"."""
    try:
        bruto = st.secrets.get("usuarios", {}) or {}
        usuarios = {str(nome).lower(): dict(dados) for nome, dados in dict(bruto).items()}
        senha_antiga = st.secrets.get("senha_painel")
    except Exception:
        # st.secrets lança exceção quando não existe NENHUM secrets.toml
        return {}
    if senha_antiga and "admin" not in usuarios:
        usuarios["admin"] = {"senha": senha_antiga}
    return usuarios


def _acesso_valido(usuarios: dict, nome: str, senha: str, hoje=None) -> tuple[bool, str]:
    """Confere usuário, senha (comparação em tempo constante) e validade."""
    import hmac
    dados = usuarios.get((nome or "").strip().lower())
    if not dados or not hmac.compare_digest(str(dados.get("senha", "")), str(senha or "")):
        return False, "Usuário ou senha incorretos."
    validade = dados.get("validade")
    if validade:
        hoje = hoje or datetime.now(ZoneInfo("America/Sao_Paulo")).date()
        if hoje > pd.Timestamp(str(validade)).date():
            return False, "Este acesso expirou. Fale com o responsável pelo painel."
    return True, ""


def _verificar_senha() -> bool:
    """Bloqueia o painel até um usuário válido entrar. Os acessos ficam nos
    Secrets do Streamlit Cloud, nunca no código (o repositório é público)."""
    usuarios = _usuarios_configurados()
    if st.session_state.get("autenticado"):
        # revalida a cada interação: um acesso apagado ou vencido cai na hora
        nome = st.session_state.get("usuario", "admin")
        dados = usuarios.get(nome)
        ok, _ = _acesso_valido(usuarios, nome, dados.get("senha", "") if dados else "")
        if ok:
            return True
        st.session_state.clear()

    st.title("🔒 Acesso ao Painel")
    if not usuarios:
        st.error(
            "Nenhum acesso configurado ainda. Defina os usuários em Settings → Secrets no "
            "Streamlit Cloud (ou, para testar localmente, no arquivo .streamlit/secrets.toml), "
            'por exemplo: [usuarios.paulo] e, na linha seguinte, senha = "sua-senha".'
        )
        return False
    nome = st.text_input("Usuário")
    senha = st.text_input("Senha", type="password")
    if st.button("Entrar"):
        ok, motivo = _acesso_valido(usuarios, nome, senha)
        if ok:
            st.session_state["autenticado"] = True
            st.session_state["usuario"] = nome.strip().lower()
            st.rerun()
        else:
            st.error(motivo)
    return False


if not _verificar_senha():
    st.stop()

NOMES_TIPO = {
    "acao": "Ação (B3)",
    "fii": "FII (B3)",
    "acao_us": "Ação (EUA)",
    "etf_br": "ETF (B3)",
    "etf_us": "ETF (EUA)",
}
OPCOES_TIPO = list(NOMES_TIPO.keys())

OPCOES_PERIODO = {
    "6mo": "6 meses",
    "1y": "1 ano",
    "2y": "2 anos",
    "5y": "5 anos",
}

_EXPRESSAO_MES_PT = (
    "['JAN','FEV','MAR','ABR','MAI','JUN','JUL','AGO','SET','OUT','NOV','DEZ']"
    "[month(datum.value)] + '/' + (year(datum.value) % 100)"
)


_ETFS_EUA_CONHECIDOS = {
    # Grandes índices e setoriais mais comuns — lista de melhor esforço,
    # não exaustiva (não há um padrão estrutural no ticker americano que
    # distinga ETF de ação, diferente da B3); por isso a confirmação manual
    # continua disponível pra qualquer ticker fora dessa lista.
    "SPY", "QQQ", "IVV", "VOO", "VTI", "VEA", "VWO", "IEF", "TLT", "AGG",
    "BND", "GLD", "SLV", "IAU", "XLE", "XLF", "XLK", "XLV", "XLI", "XLY",
    "XLP", "XLU", "XLB", "XLRE", "XLC", "EEM", "EFA", "IWM", "DIA", "ARKK",
    "EMXC", "REMX", "XME", "IXG", "MCHI", "FXI", "EWZ", "EWJ", "HYG", "LQD",
    "VNQ", "SCHD", "JEPI", "JEPQ", "SPYD", "VIG", "VYM",
}


def _heuristica_tipo(ticker: str) -> str:
    """Chute inicial do tipo de ativo, a partir do formato do ticker —
    o usuário confirma ou corrige antes de rodar a análise. Tickers B3
    terminados em '11' são frequentemente FIIs ou ETFs (ambíguo só pelo
    ticker); o padrão aqui é FII, por ser o caso mais comum. Tickers dos
    EUA não têm um padrão estrutural equivalente pra distinguir ETF de
    ação — usa uma lista de ETFs americanos conhecidos (_ETFS_EUA_CONHECIDOS);
    fora dela, o chute cai pra ação, e o usuário confirma/corrige."""
    if not ticker.endswith(".SA"):
        return "etf_us" if ticker in _ETFS_EUA_CONHECIDOS else "acao_us"
    base = ticker[:-3]
    m = re.match(r"^([A-Z]+)(\d+)$", base)
    if not m:
        return "acao"
    numero = m.group(2)
    return "fii" if numero.startswith("11") else "acao"


def _info_horario_analise() -> tuple[str, bool]:
    """Data/hora da análise (horário de Brasília) e um aviso heurístico de
    pregão possivelmente em andamento — o preço usado é sempre o mais
    recente disponível no momento da consulta, não necessariamente o
    fechamento definitivo do dia (B3: pregão regular ~10h-17h, horário de
    Brasília, dias úteis)."""
    agora = datetime.now(ZoneInfo("America/Sao_Paulo"))
    texto = agora.strftime("%d/%m/%Y às %H:%M (horário de Brasília)")
    eh_dia_util = agora.weekday() < 5
    provavelmente_aberto = eh_dia_util and 10 <= agora.hour < 18
    return texto, provavelmente_aberto


def _grafico_preco_completo(historico: pd.Series, dividendos: pd.Series) -> alt.VConcatChart:
    """Gráfico completo de um ativo: preço + médias móveis (MM50/MM200) +
    faixa de mínima/máxima do período + marcadores de dividendo pago, com
    um painel de RSI(14) logo abaixo — mesmo conteúdo do gráfico do
    notebook (Seção 8), agora no painel web. Meses abreviados em português
    no eixo (ver _EXPRESSAO_MES_PT).

    Largura sempre em 'container' (não um valor fixo em pixels) — sem
    isso, o Vega-Lite (motor por trás do Altair) pode renderizar num
    tamanho fixo próprio, maior que a tela, forçando rolagem horizontal
    mesmo com o `width='stretch'` do lado do Streamlit.

    Normaliza fuso horário antes de comparar as duas séries: o Yahoo
    Finance devolve preço (``yf.download``) sem fuso, mas dividendos
    (``yf.Ticker(...).dividends``) COM fuso — comparar as duas datas
    direto, sem isso, derruba o gráfico com TypeError."""
    historico = historico.copy()
    if historico.index.tz is not None:
        historico.index = historico.index.tz_localize(None)
    if dividendos is not None and not dividendos.empty and dividendos.index.tz is not None:
        dividendos = dividendos.copy()
        dividendos.index = dividendos.index.tz_localize(None)

    df_preco = historico.reset_index()
    df_preco.columns = ["data", "preco"]

    sma50 = historico.rolling(50).mean()
    sma200 = historico.rolling(200).mean()
    df_medias = pd.DataFrame({
        "data": historico.index,
        "MM50": sma50.values,
        "MM200": sma200.values,
    }).melt("data", var_name="média", value_name="valor").dropna()

    minima_periodo = float(historico.quantile(0.05))
    maxima_periodo = float(historico.quantile(0.95))

    df_dividendos = pd.DataFrame(columns=["data", "preco", "valor_dividendo"])
    if dividendos is not None and not dividendos.empty:
        datas_div = dividendos.index[
            (dividendos.index >= historico.index.min()) & (dividendos.index <= historico.index.max())
        ]
        if len(datas_div) > 0:
            pos = historico.index.get_indexer(datas_div, method="nearest")
            df_dividendos = pd.DataFrame({
                "data": datas_div,
                "preco": historico.iloc[pos].values,
                "valor_dividendo": dividendos.loc[datas_div].values,
            })

    eixo_x_topo = alt.Axis(labelExpr=_EXPRESSAO_MES_PT, labelAngle=0)

    linha_preco = alt.Chart(df_preco).mark_line(color="#1f77b4").encode(
        x=alt.X("data:T", title=None, axis=eixo_x_topo),
        y=alt.Y("preco:Q", title=None),
    )
    linhas_medias = alt.Chart(df_medias).mark_line(strokeWidth=1.2).encode(
        x=alt.X("data:T", title=None),
        y=alt.Y("valor:Q", title=None),
        color=alt.Color(
            "média:N", scale=alt.Scale(domain=["MM50", "MM200"], range=["#ff7f0e", "#d62728"]),
            legend=alt.Legend(title=None, orient="bottom", direction="horizontal"),
        ),
    )
    faixa_min_max = alt.Chart(pd.DataFrame({"y": [minima_periodo, maxima_periodo]})).mark_rule(
        strokeDash=[4, 4], color="gray", opacity=0.6
    ).encode(y="y:Q")
    marcadores_dividendo = alt.Chart(df_dividendos).mark_point(
        shape="triangle-up", color="#2ca02c", size=180, filled=True,
        stroke="black", strokeWidth=0.5,
    ).encode(
        x="data:T", y="preco:Q",
        tooltip=[
            alt.Tooltip("data:T", title="Data do pagamento"),
            alt.Tooltip("valor_dividendo:Q", title="Valor pago", format=".2f"),
        ],
    )

    painel_preco = (
        (linha_preco + linhas_medias + faixa_min_max + marcadores_dividendo)
        .properties(height=280, width="container")
    )

    rsi_series = calcular_rsi(historico, 14)
    df_rsi = pd.DataFrame({"data": historico.index, "rsi": rsi_series.values}).dropna()
    linha_rsi = alt.Chart(df_rsi).mark_line(color="#9467bd").encode(
        x=alt.X("data:T", title=None, axis=alt.Axis(labelExpr=_EXPRESSAO_MES_PT, labelAngle=0)),
        y=alt.Y("rsi:Q", title="RSI(14)", scale=alt.Scale(domain=[0, 100])),
    )
    faixa_rsi = alt.Chart(pd.DataFrame({"y": [30, 70]})).mark_rule(
        strokeDash=[4, 4], color="gray", opacity=0.6
    ).encode(y="y:Q")
    painel_rsi = (linha_rsi + faixa_rsi).properties(height=120, width="container")

    return alt.vconcat(painel_preco, painel_rsi).resolve_scale(x="shared").properties(
        autosize=alt.AutoSizeParams(type="fit-x", contains="padding")
    )


def _renderizar_tabela_alinhada(df: pd.DataFrame, colunas_moeda: list[str]) -> None:
    """Renderiza uma tabela em HTML com alinhamento específico — coluna
    'Ticker' à esquerda, colunas de valor monetário (R$/US$) à direita, as
    demais ao centro, cabeçalho sempre centralizado. O `st.dataframe`
    nativo do Streamlit não permite esse nível de controle por coluna, daí
    o HTML próprio em vez do componente nativo (troca: perde ordenação por
    clique no cabeçalho, ganha o alinhamento exato pedido)."""
    styler = df.style.hide(axis="index")
    estilos = [
        {"selector": "table", "props": [("border-collapse", "collapse"), ("width", "100%")]},
        {"selector": "th, td", "props": [
            ("border", "1px solid rgba(255,255,255,0.2)"), ("padding", "6px 10px"),
        ]},
        {"selector": "th", "props": [
            ("text-align", "center"), ("background-color", "rgba(255,255,255,0.08)"),
        ]},
    ]
    for i, col in enumerate(df.columns):
        if col == "Ticker":
            alinhamento = "left"
        elif col in colunas_moeda:
            alinhamento = "right"
        else:
            alinhamento = "center"
        estilos.append({"selector": f"td.col{i}", "props": [("text-align", alinhamento)]})
    styler = styler.set_table_styles(estilos)
    st.markdown(styler.to_html(), unsafe_allow_html=True)


def _fmt_moeda(valor: float | None, moeda: str, forcar_sinal: bool = False) -> str:
    """Formata no padrão de cada moeda: R$ no padrão brasileiro (ponto no
    milhar, vírgula no decimal — ex.: R$ 1.234,56); US$ no padrão
    americano nativo (vírgula no milhar, ponto no decimal — ex.: US$
    1,234.56), sem trocar separadores."""
    if valor is None:
        return "-"
    sinal = "+" if forcar_sinal and valor >= 0 else ""
    texto = f"{abs(valor):,.2f}"
    if moeda == "R$":
        # Padrão americano é o formato nativo do Python — troca pro
        # brasileiro só quando a moeda pede isso.
        texto = texto.replace(",", "_").replace(".", ",").replace("_", ".")
    prefixo = "-" if valor < 0 else sinal
    return f"{moeda} {prefixo}{texto}"


def _fmt_moeda_contabil(valor: float | None, moeda: str) -> str:
    """Formato contábil — sem sinal de +/- na frente; negativo entre
    parênteses (ex.: R$ 1.234,56 / (R$ 1.234,56)) — convenção comum em
    relatório financeiro, usada nas colunas mais estreitas do Dashboard
    (Resultado Obtido, Resultado ñ Realizado, Rentab Total)."""
    if valor is None:
        return "-"
    texto_positivo = _fmt_moeda(abs(valor), moeda)
    return f"({texto_positivo})" if valor < 0 else texto_positivo


def _fmt_pct(valor: float | None) -> str:
    if valor is None:
        return "-"
    texto = f"{valor * 100:,.2f}"
    texto = texto.replace(",", "_").replace(".", ",").replace("_", ".")
    return f"{texto}%"


def _fmt_qtde(valor: float | None) -> str:
    if valor is None:
        return "-"
    if float(valor).is_integer():
        return f"{valor:.0f}"
    texto = f"{valor:,.2f}"
    return texto.replace(",", "_").replace(".", ",").replace("_", ".")


def _fmt_qtde_milhar(valor: float) -> str:
    """1000 -> '1.000' (frações com vírgula)."""
    if float(valor).is_integer():
        return f"{int(valor):,}".replace(",", ".")
    return _fmt_qtde(valor)


def _fmt_data(d) -> str:
    return d.strftime("%d/%m/%y") if d else "-"


_ORDEM_TIPO_GRUPO = {"Ações": 0, "FIIs": 1, "ETF": 2}
_ORDEM_MERCADO = {"B3": 0, "EUA": 1}


def _ordem_agrupada(ticker: str, tipo: str) -> tuple:
    """Chave de ordenação: B3 antes de EUA e, dentro de cada mercado,
    Ações → FIIs → ETF — pedido explícito do orientador, pra manter a
    leitura por classe de ativo (importante inclusive pra validação restrita
    a FIIs, Seção 3.3 do Relatório)."""
    mercado = "B3" if ticker.endswith(".SA") else "EUA"
    grupo = _TIPO_PARA_GRUPO_APP.get(tipo, "Ações")
    return (_ORDEM_MERCADO[mercado], _ORDEM_TIPO_GRUPO[grupo], ticker)


_TIPO_PARA_GRUPO_APP = {
    "acao": "Ações", "acao_us": "Ações",
    "fii": "FIIs",
    "etf_br": "ETF", "etf_us": "ETF",
}


def _renderizar_detalhe(d: str) -> str:
    """Colore o indicador (verde/vermelho) de acordo com o sinal do
    detalhe, mesma convenção da tabela de análise (🟢/🔴) — os detalhes
    sempre começam com '+1' ou '-1' (ver pontuacao.py)."""
    d = re.sub(r"(?<=\d)\.(?=\d)", ",", d)  # 47.4 -> 47,4 (os textos do motor usam ponto)
    if d.startswith("+1"):
        return f"🟢 {d[3:].strip()}"
    if d.startswith("-1"):
        return f"🔴 {d[3:].strip()}"
    return f"🟡 {d}"


_COLUNAS_DASHBOARD = [
    "Ticker", "Saldo Inicial (data)", "Saldo Inicial (valor)", "Preço Médio (PM)",
    "Qtde Inicial", "Compras (qtde)", "Compras (valor)", "Vendas (qtde)", "Vendas (valor)",
    "Eventos (qtde)", "Saldo Final (valor)", "Saldo Final (qtde)", "Preço Final",
    "Ganho Realizado", "Ganho/Prejuízo não Realizado", "Dividendos/Rendimentos", "Rentabilidade Total",
    "Rentabilidade %",
]
# Cabeçalho quebrado em 2 linhas na exibição (colunas estreitas + texto
# longo forçavam a tabela a ultrapassar a largura da tela). Só afeta o
# rótulo mostrado — as chaves internas dos dicts continuam as de cima.
_CABECALHOS_DUAS_LINHAS = {
    "Saldo Inicial (data)": "Saldo Inicial<br>(data)",
    "Saldo Inicial (valor)": "Saldo Inicial<br>(valor)",
    "Preço Médio (PM)": "Preço Médio<br>(PM)",
    "Qtde Inicial": "Qtde<br>Inicial",
    "Compras (qtde)": "Compras<br>(qtde)",
    "Compras (valor)": "Compras<br>(valor)",
    "Vendas (qtde)": "Vendas<br>(qtde)",
    "Vendas (valor)": "Vendas<br>(valor)",
    "Eventos (qtde)": "Eventos<br>(qtde)",
    "Saldo Final (valor)": "Saldo Final<br>(valor)",
    "Saldo Final (qtde)": "Saldo Final<br>(qtde)",
    "Saldo Final (data)": "Saldo Final<br>(data)",
    "Preço Final": "Preço<br>Final",
    "Ganho Realizado": "Resultado<br>Obtido",
    "Ganho/Prejuízo não Realizado": "Resultado não<br>Realizado",
    "Dividendos/Rendimentos": "Divid/Rend",
    "Rentabilidade Total": "Rentab<br>Total",
    "Rentabilidade %": "Rentab%",
}
_COLUNAS_DASHBOARD_MOEDA = {
    "Saldo Inicial (valor)", "Preço Médio (PM)", "Compras (valor)", "Vendas (valor)",
    "Saldo Final (valor)", "Preço Final", "Ganho Realizado", "Ganho/Prejuízo não Realizado",
    "Dividendos/Rendimentos", "Rentabilidade Total",
}


def _linha_dashboard_para_dict(l) -> dict:
    return {
        "Ticker": getattr(l, "rotulo", "") or l.ticker,
        "Saldo Inicial (data)": _fmt_data(l.saldo_inicial_data),
        "Saldo Inicial (valor)": _fmt_moeda(l.saldo_inicial_valor, l.moeda),
        "Preço Médio (PM)": _fmt_moeda(l.preco_medio, l.moeda),
        "Qtde Inicial": _fmt_qtde(l.qtde_inicial),
        "Compras (qtde)": _fmt_qtde(l.compras_quantidade) if l.compras_quantidade else "-",
        "Compras (valor)": _fmt_moeda(l.compras_valor, l.moeda) if l.compras_valor else "-",
        "Vendas (qtde)": _fmt_qtde(l.vendas_quantidade) if l.vendas_quantidade else "-",
        "Vendas (valor)": _fmt_moeda(l.vendas_valor, l.moeda) if l.vendas_valor else "-",
        "Eventos (qtde)": (f"{'+' if l.eventos_quantidade > 0 else '−'}{_fmt_qtde(abs(l.eventos_quantidade))}"
                           if l.eventos_quantidade else "-"),
        "Saldo Final (valor)": _fmt_moeda(l.saldo_final_valor, l.moeda),
        "Saldo Final (qtde)": _fmt_qtde(l.saldo_final_qtde),
        "Saldo Final (data)": _fmt_data(l.saldo_final_data),
        "Preço Final": _fmt_moeda(l.preco_final, l.moeda),
        "Ganho Realizado": _fmt_moeda_contabil(l.ganho_realizado, l.moeda),
        "Ganho/Prejuízo não Realizado": _fmt_moeda_contabil(l.ganho_nao_realizado, l.moeda),
        "Dividendos/Rendimentos": _fmt_moeda(l.renda_recebida, l.moeda),
        "Rentabilidade Total": _fmt_moeda_contabil(l.rentabilidade_total, l.moeda),
        "Rentabilidade %": _fmt_pct(l.rentabilidade_pct),
    }


def _linha_subtotal_para_dict(rotulo: str, linhas_grupo: list, moeda: str) -> dict:
    """Soma as colunas de valor (não as de data/preço unitário/percentual,
    que não fazem sentido somados) pra uma linha 'Total X'."""
    def soma(attr):
        valores = [getattr(l, attr) for l in linhas_grupo if getattr(l, attr) is not None]
        return sum(valores) if valores else None

    saldo_inicial = soma("saldo_inicial_valor")
    compras = soma("compras_valor")
    vendas = soma("vendas_valor")
    saldo_final = soma("saldo_final_valor")
    ganho_realizado = soma("ganho_realizado")
    ganho_nao_realizado = soma("ganho_nao_realizado")
    renda = soma("renda_recebida")
    rentabilidade = soma("rentabilidade_total")
    base_pct = (saldo_inicial or 0.0) + (compras or 0.0)
    rentabilidade_pct = rentabilidade / base_pct if rentabilidade is not None and base_pct > 0 else None

    return {
        "Ticker": rotulo,  # destaque vem do CSS por tipo de linha (sem marcador Markdown)
        "Saldo Inicial (data)": "", "Preço Médio (PM)": "", "Qtde Inicial": "",
        "Saldo Inicial (valor)": _fmt_moeda(saldo_inicial, moeda),
        "Compras (qtde)": "", "Compras (valor)": _fmt_moeda(compras, moeda) if compras else "-",
        "Vendas (qtde)": "", "Vendas (valor)": _fmt_moeda(vendas, moeda) if vendas else "-",
        "Eventos (qtde)": "", "Saldo Final (valor)": _fmt_moeda(saldo_final, moeda),
        "Saldo Final (qtde)": "", "Saldo Final (data)": "", "Preço Final": "",
        "Ganho Realizado": _fmt_moeda_contabil(ganho_realizado, moeda),
        "Ganho/Prejuízo não Realizado": _fmt_moeda_contabil(ganho_nao_realizado, moeda),
        "Dividendos/Rendimentos": _fmt_moeda(renda, moeda),
        "Rentabilidade Total": _fmt_moeda_contabil(rentabilidade, moeda),
        "Rentabilidade %": _fmt_pct(rentabilidade_pct),
    }


def _renderizar_dashboard_agrupado(linhas_dashboard: list) -> None:
    """Renderiza o Dashboard da Carteira nos 4 blocos pedidos — Ativas B3,
    Ativas EUA, Encerradas B3, Encerradas EUA —, cada um agrupado por tipo
    de ativo (Ações, FIIs, ETF) com linha de subtotal, e total geral por
    mercado ao final do bloco. Blocos e grupos sem nenhum ativo são
    omitidos (não mostra 'Total FIIs: -' quando não há FII nenhum)."""
    linhas_html: list[dict] = []
    tipos_marcador: list[str] = []  # 'secao' | 'subtotal' | 'total_geral' | 'dado', paralelo a linhas_html

    for situacao, rotulo_situacao in [("Ativa", "OP. ATIVAS"), ("Encerrada", "OP. ENCERRADAS")]:
        for mercado in ("B3", "EUA"):
            linhas_bloco = [l for l in linhas_dashboard if l.situacao == situacao and l.mercado == mercado]
            if not linhas_bloco:
                continue

            linhas_html.append({"Ticker": f"{rotulo_situacao} {mercado}"})  # destaque via CSS
            tipos_marcador.append("secao")

            moeda_bloco = linhas_bloco[0].moeda
            for tipo_grupo in ("Ações", "FIIs", "ETF"):
                linhas_grupo = [l for l in linhas_bloco if l.tipo_grupo == tipo_grupo]
                if not linhas_grupo:
                    continue
                for l in sorted(linhas_grupo, key=lambda x: (x.ticker, getattr(x, "ciclo", 1))):
                    linhas_html.append(_linha_dashboard_para_dict(l))
                    tipos_marcador.append("dado")
                linhas_html.append(_linha_subtotal_para_dict(f"Total {tipo_grupo}", linhas_grupo, moeda_bloco))
                tipos_marcador.append("subtotal")

            linhas_html.append(_linha_subtotal_para_dict(f"Total {mercado}", linhas_bloco, moeda_bloco))
            tipos_marcador.append("total_geral")

    if not linhas_html:
        st.info("Nenhum ativo encontrado na planilha.")
        return

    df = pd.DataFrame(linhas_html).reindex(columns=_COLUNAS_DASHBOARD).fillna("")
    df = df.rename(columns=_CABECALHOS_DUAS_LINHAS)
    styler = df.style.hide(axis="index")
    estilos = [
        # Sem "width: 100%" na tabela de propósito — a tabela precisa poder
        # crescer além do container (17 colunas com texto sem quebra) pra
        # disparar a rolagem do <div> em volta; travar a tabela em 100%
        # espremia as células em vez de rolar.
        {"selector": "table", "props": [("border-collapse", "collapse"), ("font-size", "0.82em"),
                                         ("background-color", "#ffffff"), ("color", "#111827")]},
        {"selector": "td", "props": [
            ("border", "1px solid #e5e7eb"), ("padding", "4px 6px"), ("white-space", "nowrap"),
            ("color", "#111827 !important"),  # o tema escuro do Streamlit pinta o texto das células de claro
        ]},
        {"selector": "th", "props": [
            ("border", "1px solid #e5e7eb"), ("padding", "4px 6px"), ("color", "#374151"),
            ("text-align", "center"), ("background-color", "#f3f4f6"),
            ("white-space", "normal"), ("max-width", "80px"), ("line-height", "1.2"),
        ]},
    ]
    for i, col in enumerate(_COLUNAS_DASHBOARD):
        alinhamento = "right" if col in _COLUNAS_DASHBOARD_MOEDA else "center"  # valores à direita; o resto ao centro
        estilos.append({"selector": f"td.col{i}", "props": [("text-align", alinhamento)]})

    # Linhas de seção e subtotal em negrito/destaque — pela posição da linha,
    # já que o Styler não sabe de conteúdo "de negócio", só índice.
    for row_idx, tipo in enumerate(tipos_marcador):
        if tipo == "secao":
            estilos.append({
                "selector": f"tr:nth-child({row_idx + 1})",
                "props": [("background-color", "#dbeafe"), ("font-weight", "bold"), ("color", "#1e3a8a")],
            })
        elif tipo == "subtotal":
            estilos.append({
                "selector": f"tr:nth-child({row_idx + 1})",
                "props": [("background-color", "#f8fafc"), ("font-style", "italic")],
            })
        elif tipo == "total_geral":
            estilos.append({
                "selector": f"tr:nth-child({row_idx + 1})",
                "props": [("background-color", "#e5e7eb"), ("font-weight", "bold")],
            })

    styler = styler.set_table_styles(estilos)
    # Rolagem contida só nesta tabela (não na página inteira) — com 17
    # colunas, é esperado que a tabela seja mais larga que a tela em
    # monitores comuns; o padrão usual pra tabela financeira larga é
    # rolar só ela, com uma barra visível logo abaixo dela mesma.
    st.markdown(
        f'<div style="overflow-x: auto; max-width: 100%; display: block; background:#ffffff; border-radius:12px; '
        f'padding:10px">{styler.to_html()}</div>',
        unsafe_allow_html=True,
    )
    # A data do saldo final é a mesma para todos os ativos (a da análise): sai
    # da tabela, que ganhou a coluna de eventos, e vai para a legenda.
    data_final = next((l.saldo_final_data for l in linhas_dashboard if l.saldo_final_data), None)
    if data_final:
        st.caption(f"Saldo final e preço final em {_fmt_data(data_final)}.")


st.title("📊 Análise de Carteira — Ações, FIIs e ETFs")
tab_carteira, tab_visao, tab_dashboard, tab_ativo, tab_metas, tab_ir, tab_guia = st.tabs([
    "📂 Carteira", "🧭 Visão geral", "💼 Dashboard da Carteira", "🔍 Raio-X do ativo", "🎯 Metas",
    "🧾 Proventos e IR", "📖 Guia de Indicadores",
])

with tab_carteira:
    CAMINHO_MODELO = Path(__file__).resolve().parent / "modelos" / "modelo_carteira.xlsx"

    st.markdown("**Como carregar a sua carteira:** baixe o modelo, preencha com os seus ativos e envie o "
                "arquivo abaixo. A planilha fica só nesta sessão do navegador; não é gravada no servidor.")
    if CAMINHO_MODELO.exists():
        st.download_button("⬇️ Baixar o modelo (Excel)", CAMINHO_MODELO.read_bytes(), file_name="modelo_carteira.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with st.expander("Como preencher a planilha", expanded=st.session_state.get("planilha_bytes") is None):
        st.markdown(
            "**Aba Resumo** — o ponto de partida de cada ativo: Ticker, Quantidade, Preço (médio), Data e Tipo Ativo.\n"
            "- Com a aba Operações: lance a **primeira compra**; a posição de hoje o painel calcula.\n"
            "- Só com o Resumo: lance a **posição de hoje**, com o preço médio e a data de entrada.\n\n"
            "**Aba Operações** (opcional) — compras e vendas depois da data do Resumo: Ticker, Tipo "
            "(Compra/Venda), Quantidade, Preço, Data e Tipo Ativo. Eventos da empresa também entram aqui, com a "
            "**diferença** de ações: Bonificação (ações recebidas; preço = custo atribuído ou 0), Desdobramento "
            "(ações recebidas; preço em branco) e Grupamento (ações que deixaram de existir; preço em branco).\n\n"
            "**Uma ou duas abas?** Com as duas, o painel mostra também vendas, ganho realizado e posições "
            "encerradas. Só com o Resumo, a análise começa na data de entrada.\n\n"
            "**Ticker:** B3 com ou sem .SA (VALE3 ou VALE3.SA); EUA como no Yahoo Finance (BRK-B, NOBL). "
            "**Tipo Ativo:** Ação, FII ou ETF."
        )

    # A planilha fica guardada SÓ na sessão do navegador de quem a enviou
    # (st.session_state). Uma versão anterior a gravava num arquivo do servidor,
    # compartilhado por todos os acessos ao painel: um segundo usuário veria a
    # carteira do primeiro. Custo da correção: ao recarregar a página (F5), é
    # preciso enviar a planilha de novo.
    arquivo = st.file_uploader("Escolha o arquivo Excel (.xlsx) da sua carteira", type=["xlsx"])

    df_resumo = None
    df_operacoes = None
    tickers_encontrados: list[str] = []
    tipos_planilha: dict[str, str] = {}
    avisos_tipos_planilha: list[str] = []
    bytes_planilha = None

    if arquivo is not None:
        bytes_planilha = arquivo.getvalue()
        if st.session_state.get("planilha_bytes") not in (None, bytes_planilha):
            # planilha nova: cards e extrato calculados com a anterior deixam de valer
            for chave in ("carteira_cenarios", "carteira_linhas_ganho", "carteira_linhas_analise"):
                st.session_state.pop(chave, None)
        st.session_state["planilha_bytes"] = bytes_planilha
        st.session_state["planilha_nome"] = arquivo.name
    elif st.session_state.get("planilha_bytes") is not None:
        # Reaproveita a planilha já enviada NESTA sessão (ex.: ao trocar de aba)
        bytes_planilha = st.session_state["planilha_bytes"]
        col_info, col_limpar = st.columns([5, 1])
        with col_info:
            st.info(f"📎 Usando a planilha enviada nesta sessão ({st.session_state.get('planilha_nome', 'carteira')}). "
                    "Envie um novo arquivo acima para substituir.")
        with col_limpar:
            if st.button("🗑️ Limpar"):
                for chave in ("planilha_bytes", "planilha_nome", "carteira_cenarios"):
                    st.session_state.pop(chave, None)
                st.rerun()

    if bytes_planilha is not None:
        try:
            abas = pd.read_excel(io.BytesIO(bytes_planilha), sheet_name=None)
        except Exception as e:
            st.error(f"Não consegui ler o arquivo: {e}")
            st.stop()

        df_resumo_bruto = achar_aba(abas, NOMES_ABA_RESUMO)
        df_operacoes_bruto = achar_aba(abas, NOMES_ABA_OPERACOES)

        if df_resumo_bruto is None and df_operacoes_bruto is None:
            st.warning(
                "Não encontrei as abas da carteira. Nomes aceitos: 'Resumo', '1ª Compra' ou 'Posição inicial' "
                "(ponto de partida) e 'Operações', 'Movimentações' ou 'Compras e vendas posteriores'. "
                "Renomeie as abas do arquivo e tente de novo."
            )
            st.stop()

        try:
            df_resumo = normalizar_aba_resumo(df_resumo_bruto) if df_resumo_bruto is not None else None
            df_operacoes = normalizar_aba_operacoes(df_operacoes_bruto) if df_operacoes_bruto is not None else None
        except ValueError as e:
            st.error(f"Problema nas colunas da planilha: {e}")
            st.stop()

        st.success(
            f"Planilha lida — "
            f"{'sem aba Resumo' if df_resumo is None else f'{len(df_resumo)} ativo(s) na aba Resumo'}, "
            f"{'sem aba Operações' if df_operacoes is None else f'{len(df_operacoes)} operação(ões)'}."
        )

        if df_resumo is not None:
            tickers_encontrados += [normalizar_ticker(t)[0] for t in df_resumo["ticker"]]
        if df_operacoes is not None:
            tickers_encontrados += [normalizar_ticker(t)[0] for t in df_operacoes["ticker"]]
        tickers_encontrados = sorted(set(tickers_encontrados))
        tipos_planilha, avisos_tipos_planilha = tipos_da_planilha(df_resumo, df_operacoes)


def _sem_formula(texto: str) -> str:
    """Escapa o cifrão para o Markdown do Streamlit: dois '$' na mesma linha
    (ex.: 'R$ 45,83 · hoje R$ 57,44') viram fórmula matemática e somem."""
    return texto.replace("$", "\\$")


# ---------------------------------------------------------------------------
# Cenários comparativos por ativo (aba Dashboard)
# ---------------------------------------------------------------------------

CDI_RESERVA_AA = 0.14  # usado só se a API do Banco Central estiver fora do ar


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def _cdi_bcb_em_cache(inicio: str, fim: str):
    return buscar_cdi_diario(inicio, fim)


def _cdi_do_banco_central():
    """CDI diário dos últimos ~2 anos (série 12 do Banco Central), em cache
    por 12h. Devolve (série ou None, aviso ou None)."""
    hoje = datetime.now(ZoneInfo("America/Sao_Paulo")).date()
    serie = _cdi_bcb_em_cache(str(hoje - pd.Timedelta(days=800)), str(hoje))
    if serie is None:
        return None, (f"CDI do Banco Central indisponível no momento — o retrospecto dos FIIs usa "
                      f"{CDI_RESERVA_AA * 100:.0f}% ao ano para o dinheiro em caixa.")
    return serie, None


def _cdi_liquido_atual() -> float:
    """CDI dos últimos 12 meses, líquido de IR: a régua do dividend yield no
    "Contexto do ativo". Sem o Banco Central, usa a taxa de reserva."""
    serie, _ = _cdi_do_banco_central()
    valor = cdi_liquido_12m(serie)
    return valor if valor else CDI_RESERVA_AA * (1 - 0.175)


# Cor de fundo (gradiente), ícone e cor de destaque de cada situação. Sem cinza
# e sem verde/vermelho puros (que seriam lidos como "compre"/"venda").
_ESTILO_SITUACAO = {
    "realizacao": ("linear-gradient(135deg,#4f6bff,#2f45e0)", "📈", "#2f45e0"),
    "queda": ("linear-gradient(135deg,#f0a93b,#d9831c)", "📉", "#d9831c"),
    "fora": ("linear-gradient(135deg,#2bb3a3,#0f7f73)", "↔️", "#0f7f73"),
}
_NOME_GRUPO = {"Ações": "Ações", "FIIs": "Fundos Imobiliários (FIIs)", "ETF": "ETFs", "Outros": "Outros ativos"}


def _fmt_celula(fmt: str, valor, moeda: str) -> str:
    """Formata uma célula das tabelas de cenário."""
    if valor is None:
        return "—"
    if fmt == "caixa":
        tipo, v = valor
        return ("+" if v > 0 else "−") + formatar_valor(abs(v), moeda) + f" ({tipo})"
    if fmt == "sinal":
        return ("+" if valor > 0 else "−" if valor < 0 else "") + formatar_valor(abs(valor), moeda)
    if fmt == "pct":
        return f"{valor * 100:+.0f}%".replace("-", "−")
    if fmt == "cotas":
        return f"{valor:.0f}" if abs(valor - round(valor)) < 1e-9 else f"{valor:.2f}"
    return formatar_valor(valor, moeda)


def _html_tabela(colunas: list, linhas: list, moeda: str) -> str:
    th = "".join(f"<th style='text-align:right;padding:3px 6px;font-weight:600;white-space:nowrap'>"
                 f"{html.escape(c)}</th>" for c in colunas)
    trs = ""
    for nome, fmt, *valores in linhas:
        tds = "".join(f"<td style='text-align:right;padding:3px 6px;white-space:nowrap'>"
                      f"{html.escape(_fmt_celula(fmt, v, moeda))}</td>" for v in valores)
        trs += f"<tr><td style='padding:3px 6px;opacity:.9;white-space:nowrap'>{html.escape(nome)}</td>{tds}</tr>"
    return ("<table style='width:100%;border-collapse:collapse;font-size:.8em;margin-top:6px;"
            f"background:rgba(0,0,0,.12);border-radius:8px'><tr><th></th>{th}</tr>{trs}</table>")


# ---------------------------------------------------------------------------
# Régua de faixas, faixa de 12 meses e contexto do ativo (cards)
# ---------------------------------------------------------------------------

_COR_TRECHO = {"queda": "#f0a93b", "fora": "#2bb3a3", "realizacao": "#7b8fff"}
_COR_NIVEL = {BOM: "#2bb3a3", ATENCAO: "#f0a93b"}
_ROTULO_FAIXA = {"queda": "queda", "fora": "fora das faixas", "realizacao": "realização de lucro"}
FRASE_EVIDENCIA_TECNICA = ("considera só fundamentos e renda. Os sinais técnicos não mostraram vantagem no "
                           "backtest e ficam na pontuação completa. Use como contexto, não como sinal.")



def _pct_br(v: float, casas: int = 0, sinal: bool = False) -> str:
    txt = f"{v * 100:{'+' if sinal else ''}.{casas}f}%".replace(".", ",")
    return txt.replace("-", "−")


def _html_regua(regua: dict | None, moeda: str, titulo: str = "Faixa atual", mostrar_preco_medio: bool = False,
                mostrar_rotulo: bool = True) -> str:
    """Régua de três cores (aumentar | fora das faixas | realizar) com o
    marcador no preço de hoje e os preços em que cada faixa começa."""
    if not regua:
        return ""
    sit = regua["situacao"]
    rotulo = _ROTULO_FAIXA[sit]
    if regua.get("degrau") is not None:
        rotulo += f" ({_pct_br(regua['degrau'], sinal=True)})"
    pq, pr = formatar_valor(regua["preco_queda"], moeda), formatar_valor(regua["preco_realizacao"], moeda)
    hoje = formatar_valor(regua["preco"], moeda)
    prox = regua.get("proximo")
    if sit == "fora":
        rodape = f"Hoje {hoje}: sem cenário novo enquanto o preço estiver entre {pq} e {pr}."
    elif prox:
        quando = "a partir de" if sit == "realizacao" else "abaixo de"
        extra = " (venda total)" if prox.get("venda_total") else ""
        rodape = (f"Hoje {hoje}, na faixa de {_pct_br(regua['degrau'], sinal=True)}. Próxima: "
                  f"<b>{_pct_br(prox['faixa'], sinal=True)}{extra} {quando} {formatar_valor(prox['preco'], moeda)}</b>.")
    else:
        rodape = f"Hoje {hoje}, na última faixa ({_pct_br(regua['degrau'], sinal=True)})."
    pos = regua["posicao"] * 100
    return (
        "<div style='margin-top:10px;background:rgba(0,0,0,.28);border-radius:12px;padding:10px 12px'>"
        "<div style='display:flex;justify-content:space-between;gap:8px;align-items:baseline;flex-wrap:wrap'>"
        f"<span style='font-weight:700;font-size:.9em'>{html.escape(titulo)}"
        f"{(': ' + html.escape(rotulo)) if mostrar_rotulo else ''}</span>"
        + (f"<span style='font-size:.78em;opacity:.9'>preço médio {formatar_valor(regua['preco_medio'], moeda)} · "
           f"{_pct_br(regua['variacao'], sinal=True)}</span>" if mostrar_preco_medio else "") + "</div>"
        "<div style='position:relative;display:grid;grid-template-columns:1fr 2fr 1fr;gap:3px;height:11px;margin:9px 0 6px'>"
        f"<div style='background:{_COR_TRECHO['queda']};border-radius:999px 0 0 999px'></div>"
        f"<div style='background:{_COR_TRECHO['fora']}'></div>"
        f"<div style='background:{_COR_TRECHO['realizacao']};border-radius:0 999px 999px 0'></div>"
        f"<div style='position:absolute;left:{pos:.1f}%;top:-5px;width:4px;height:21px;margin-left:-2px;"
        "border-radius:2px;background:#ffffff;box-shadow:0 0 0 1px rgba(0,0,0,.35)'></div></div>"
        "<div style='display:grid;grid-template-columns:1fr 2fr 1fr;gap:3px;font-size:.74em;opacity:.95;line-height:1.3'>"
        f"<span>Aumentar<br><b>abaixo de {pq}</b></span>"
        f"<span style='text-align:center'>Fora das faixas<br><b>{pq} a {pr}</b></span>"
        f"<span style='text-align:right'>Realizar<br><b>a partir de {pr}</b></span></div>"
        f"<div style='font-size:.78em;margin-top:6px'>{rodape}</div></div>")


def _html_faixa_12m(f: dict | None, moeda: str) -> str:
    """Mínima e máxima de 12 meses, com o preço de hoje entre elas."""
    if not f:
        return ""
    if f["abaixo_da_maxima"] < 0.005:
        frase = "<b>na máxima de 12 meses</b>"
    elif f["acima_da_minima"] < 0.005:
        frase = "<b>na mínima de 12 meses</b>"
    else:
        frase = (f"<b>{_pct_br(f['abaixo_da_maxima'])} abaixo da máxima</b> e "
                 f"<b>{_pct_br(f['acima_da_minima'])} acima da mínima</b>")
    pos = f["posicao"] * 100
    return (
        "<div style='margin-top:8px;background:rgba(0,0,0,.2);border-radius:12px;padding:10px 12px'>"
        "<div style='display:flex;justify-content:space-between;font-size:.76em;opacity:.92'>"
        f"<span>Mínima 12 meses<br><b style='font-size:1.12em'>{formatar_valor(f['minima'], moeda)}</b></span>"
        f"<span style='text-align:right'>Máxima 12 meses<br><b style='font-size:1.12em'>{formatar_valor(f['maxima'], moeda)}</b></span></div>"
        "<div style='position:relative;height:9px;border-radius:999px;background:rgba(255,255,255,.22);margin:8px 0 6px'>"
        f"<div style='position:absolute;left:0;top:0;bottom:0;width:{pos:.1f}%;border-radius:999px;background:rgba(255,255,255,.5)'></div>"
        f"<div style='position:absolute;left:{pos:.1f}%;top:-5px;width:4px;height:19px;margin-left:-2px;border-radius:2px;"
        "background:#ffffff'></div></div>"
        f"<div style='font-size:.8em'>Hoje {formatar_valor(f['preco'], moeda)} — {frase}</div></div>")


_SELO_PRECO = {
    "forte": ("Sim — yield alto e com desconto", "background:#0f7f73;color:#ffffff"),
    "sim": ("Sim — yield alto", "background:#2bb3a3;color:#ffffff"),
    "desconto": ("Com desconto", "background:#fde7c7;color:#92400e"),
    "nao": ("Não, no momento", "background:#f1f5f9;color:#334155"),
}
_EVIDENCIA_PRECO = {
    "forte": ("com yield alto e desconto juntos, superou o CDI nos 12 meses seguintes em 79% das vezes, contra 44% "
              "(juros altos, 2021–2025). Com juros mais baixos, quase não ocorreu."),
    "sim": ("superou o CDI nos 12 meses seguintes em 62% das vezes, contra 42% (2019–2025); a vantagem apareceu com "
            "juros intermediários e altos."),
    "desconto": ("com juros altos (2021–2025), cotas com desconto superaram o CDI em 63% a 70% das vezes, contra 45%. "
                 "Com juros intermediários, não deu para concluir."),
    "nao": ("o critério de yield superou o CDI em 62% das vezes, contra 42% (2019–2025); com yield alto e desconto "
            "juntos, em 79% (juros altos)."),
}


# FIIs de papel (CRI): no teste restrito a eles (Seção 13 do notebook, rodada de 05/11/2026 com o MXRF11 já
# como papel; 12 FIIs com casos), os três critérios tiveram vantagem com margem estatística: 4c +35 p.p.,
# 4a +21 p.p. e, no limite, o caminho 2 (yield alto sozinho), +17 p.p. (IC de +3 a +32) — inconclusivo na
# rodada anterior, com 11 FIIs. Por isso o selo do yield alto é o mesmo dos demais FIIs, e a evidência
# avisa que essa vantagem é a mais frágil. (_SELO_PAPEL_YIELD fica para o caso de o resultado voltar a
# ficar sem margem numa rodada futura.)
_SELO_PAPEL_YIELD = ("Yield alto · sem vantagem comprovada", "background:#dbeafe;color:#1e40af")
_PAPEL_YIELD_SEM_VANTAGEM = False
_EVIDENCIA_PAPEL = {
    "forte": ("nos FIIs de papel, yield alto e desconto juntos superaram o CDI em 84% das vezes, contra 48%, com "
              "margem estatística."),
    "sim": ("nos FIIs de papel, superou o CDI em 66% das vezes, contra 48% — vantagem no limite da margem "
            "estatística, menor e mais incerta que a do desconto no P/VP."),
    "desconto": ("nos FIIs de papel, P/VP abaixo de 0,95 com renda estável superou o CDI em 68% das vezes, contra "
                 "47%, com margem estatística."),
    "nao": ("nos FIIs de papel, o sinal mais forte no teste é o desconto no P/VP (68% contra 47%), que hoje não "
            "está presente."),
}


def _evidencia_preco(nivel: str, papel: bool, pa: dict) -> str:
    """Evidência do teste para o nível do card. No FII de papel em "Não, no
    momento", diz o que falta: o desconto (se não há) ou a renda estável (se
    há desconto, mas os rendimentos caíram — casos KNIP11, KNSC11, RECR11)."""
    if not papel:
        return _EVIDENCIA_PRECO[nivel]
    if nivel == "nao" and pa.get("desconto") and pa.get("estavel") is False:
        return ("nos FIIs de papel, o sinal mais forte no teste é o desconto no P/VP com renda estável (superou o CDI "
                "em 68% das vezes, contra 47%); aqui o desconto existe, mas os rendimentos caíram.")
    return _EVIDENCIA_PAPEL[nivel]


def _html_preco_atrativo(ctx: dict) -> str:
    """Bloco "O preço está atrativo?" dos FIIs: selo em quatro níveis (yield
    alto e com desconto, yield alto, com desconto, não), três caixinhas com o
    que foi checado e a evidência do teste de cada nível (Seção 13)."""
    pa = ctx.get("preco_atrativo")
    conf = ctx.get("conferencia_rendimentos")
    nota_fonte = ""
    if conf:
        cor = {"confirmado": "#6b7280", "divergente": "#b45309"}.get(conf["selo"], "#6b7280")
        nota_fonte = (f"<div style='font-size:.74em;color:{cor};margin-top:4px'>{html.escape(conf['texto'])}.</div>"
                      + "".join(f"<div style='font-size:.72em;color:#6b7280'>🔍 {html.escape(o)}.</div>"
                                for o in conf.get("observacoes", [])))
    if not pa:
        return ("<div style='border-top:1px solid #e5e7eb;padding-top:10px'>"
                "<div style='font-weight:700;color:#0f5f56'>O preço está atrativo?</div>"
                "<div style='font-size:.86em;margin-top:4px'>Sem histórico suficiente para avaliar. "
                + html.escape(ctx.get("preco_atrativo_motivo") or "São precisos 3 anos de pagamentos, sem lacunas.")
                + "</div>" + nota_fonte + "</div>")
    if ctx.get("motivo_classe"):
        nota_fonte += f"<div style='font-size:.72em;color:#6b7280'>ℹ️ {html.escape(ctx['motivo_classe'])}</div>"
    if ctx.get("precos_b3_ate") is not None:
        nota_fonte += (f"<div style='font-size:.72em;color:#6b7280'>ℹ️ Cotações antes de "
                       f"{ctx['precos_b3_ate']:%m/%Y}: fechamento mensal oficial da B3 (o Yahoo só tem a partir "
                       "dessa data).</div>")
    nivel = pa.get("nivel") or ("sim" if pa.get("atende") else "nao")
    papel = ctx.get("classe_fii") == "papel"
    etiqueta, estilo = (_SELO_PAPEL_YIELD if (papel and nivel == "sim" and _PAPEL_YIELD_SEM_VANTAGEM)
                        else _SELO_PRECO[nivel])
    y, m, lim = _pct_br(pa["yield"], 1), _pct_br(pa["yield_medio_24m"], 1), _pct_br(pa["limiar"], 1)
    pvp, pvp_m, estavel = pa.get("pvp"), pa.get("pvp_medio_24m"), pa.get("estavel")

    def caixa(rotulo, valor, ok):
        cor = {True: "#0f7f73", False: "#b45309", None: "#475569"}[ok]
        return ("<div style='border:1px solid #e5e7eb;border-radius:8px;padding:5px 7px;line-height:1.3'>"
                f"<div style='font-size:.68em;color:#6b7280'>{rotulo}</div>"
                f"<div style='font-size:.76em;font-weight:700;color:{cor}'>{valor}</div></div>")
    if pvp is None:
        txt_pvp, ok_pvp = "—", None
    elif pa.get("desconto"):
        txt_pvp = f"{pvp:.2f}".replace(".", ",") + " · desconto" + (
            f" (média {pvp_m:.2f})".replace(".", ",") if pvp_m else "")
        ok_pvp = True
    else:
        txt_pvp, ok_pvp = f"{pvp:.2f}".replace(".", ",") + " · sem desconto", None
    caixas = ("<div style='display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px;margin-top:8px'>"
              + caixa("Yield × própria média", f"{y} · {'acima' if pa['yield_alto'] else 'abaixo'} de {lim}",
                      bool(pa["yield_alto"]))
              + caixa("P/VP", txt_pvp, ok_pvp)
              + caixa("Rendimentos", "—" if estavel is None else ("estáveis" if estavel else "em queda"),
                      None if estavel is None else bool(estavel)) + "</div>")
    if nivel == "forte":
        texto = "O fundo rende mais que o normal dele e a cota está abaixo do patrimônio."
    elif nivel == "sim" and papel:
        texto = ("O fundo rende mais que o normal dele, embora a cota não esteja com desconto. Nos FIIs de papel, "
                 "esse sinal teve vantagem menor que a do desconto no P/VP.")
    elif nivel == "sim":
        texto = "O fundo rende mais que o normal dele, embora a cota não esteja com desconto."
    elif nivel == "desconto" and papel:
        texto = ("A cota está abaixo do patrimônio" + (" e do próprio histórico" if pvp_m and pvp <= 0.9 * pvp_m else "")
                 + ", com renda estável. Nos FIIs de papel, esse foi o sinal mais consistente no teste.")
    elif nivel == "desconto":
        texto = ("A cota está abaixo do patrimônio" + (" e do próprio histórico" if pvp_m and pvp <= 0.9 * pvp_m else "")
                 + f", mas o yield ainda está no nível normal do fundo (precisaria render {lim} para estar alto).")
    elif estavel is False and (pa["yield_alto"] or pa.get("desconto")):
        texto = (("O yield está alto" if pa["yield_alto"] else "A cota tem desconto")
                 + ", mas os rendimentos caíram mais de 10%: todos os critérios exigem renda estável, para não "
                 "confundir preço baixo com um fundo em deterioração.")
    else:
        texto = (f"Nos últimos 2 anos o fundo rendeu, em média, {m}. Para estar barato em relação a ele mesmo, "
                 f"precisaria render pelo menos <b>{lim}</b>.")
    return ("<div style='border-top:1px solid #e5e7eb;padding-top:10px'>"
            "<div style='display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap'>"
            "<span style='font-weight:700;color:#0f5f56'>O preço está atrativo?</span>"
            f"<span style='font-size:.78em;font-weight:700;padding:3px 10px;border-radius:999px;{estilo}'>{etiqueta}</span></div>"
            f"{caixas}<div style='font-size:.86em;line-height:1.45;margin-top:8px'>{texto}</div>"
            f"<div style='font-size:.76em;color:#4b5563;margin-top:6px;line-height:1.4'><b>Evidência:</b> "
            f"{_evidencia_preco(nivel, papel, pa)}</div>{nota_fonte}</div>")


_SELO_REGULARIDADE = {
    "regular": ("Regular", "background:#0f7f73;color:#ffffff"),
    "oscilante": ("Oscilante", "background:#fde7c7;color:#92400e"),
    "irregular": ("Irregular", "background:#fee2e2;color:#991b1b"),
    "incompleto": ("Dados incompletos", "background:#f1f5f9;color:#334155"),
}


def _html_regularidade(reg: dict | None) -> str:
    """Bloco "Os dividendos são regulares?" das ações: DY de cada um dos 5
    anos fechados em barras (laranja = ano fora da curva; traço vermelho =
    sem pagamento; cinza com "?" = sem registro no Yahoo, mas com pagamento
    no Fundamentus; linha tracejada = mediana), três caixinhas e a frase."""
    if not reg:
        return ""
    etiqueta, estilo = _SELO_REGULARIDADE[reg["selo"]]
    valores = [a["dy"] for a in reg["anos"] if a["dy"] is not None]
    mediana, topo = reg["mediana"], max(max(valores, default=0), 1e-9)
    barras = ""
    for a in reg["anos"]:
        dy = a["dy"]
        if a.get("sem_registro"):
            barra = ("<div style='width:100%;height:30px;border:1.5px dashed #94a3b8;border-bottom:none;"
                     "border-radius:4px 4px 0 0'></div>")
            rotulo = "?"
        elif dy <= 0:
            barra = "<div style='width:100%;border-top:2px dashed #ef4444'></div>"
            rotulo = "0"
        else:
            cor = "#f0a93b" if a["ano"] in reg["extraordinarios"] else "#2bb3a3"
            barra = f"<div style='width:100%;height:{dy / topo * 62:.0f}px;border-radius:4px 4px 0 0;background:{cor}'></div>"
            rotulo = _pct_br(dy, 1)
        barras += ("<div style='flex:1;display:flex;flex-direction:column;align-items:center;justify-content:flex-end'>"
                   f"<span style='font-size:.68em;font-weight:600;color:#374151'>{rotulo}</span>{barra}"
                   f"<span style='font-size:.68em;color:#6b7280;margin-top:2px'>{a['ano']}</span></div>")
    linha_mediana = (f"<div style='position:absolute;left:0;right:0;bottom:{mediana / topo * 62 + 16:.0f}px;"
                     "border-top:1.5px dashed #64748b'></div>") if mediana else ""
    grafico = ("<div style='position:relative;display:flex;gap:8px;align-items:flex-end;height:96px;margin-top:8px'>"
               f"{linha_mediana}{barras}</div>")

    def caixa(rotulo, valor, ok):
        cor = "#0f7f73" if ok else "#b45309"
        return ("<div style='border:1px solid #e5e7eb;border-radius:8px;padding:5px 7px;line-height:1.3'>"
                f"<div style='font-size:.68em;color:#6b7280'>{rotulo}</div>"
                f"<div style='font-size:.76em;font-weight:700;color:{cor}'>{valor}</div></div>")
    caixas = evidencia = ""
    if reg["selo"] != "incompleto":
        completo = reg["pagou"] == reg["n_anos"]
        caixas = ("<div style='display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px;margin-top:8px'>"
                  + caixa("Anos com pagamento", f"{reg['pagou']} de {reg['n_anos']}"
                          + (" com dado" if reg.get("lacunas") else ""), completo)
                  + caixa("Pior ano", f"{_pct_br(reg['pior'], 1)} ({reg['ano_pior']})",
                          completo and reg["pior"] >= 0.5 * mediana)
                  + caixa("Mediana × média", f"{_pct_br(mediana, 1)} · {_pct_br(reg['media'], 1)}",
                          not reg["extraordinarios"] and reg["selo"] != "irregular") + "</div>")
        evidencia = ("<div style='font-size:.76em;color:#4b5563;margin-top:6px;line-height:1.4'><b>Evidência:</b> com "
                     "o selo Regular, a renda se manteve no ano seguinte em 99% dos casos, contra 82% nas ações "
                     "Oscilantes ou Irregulares (cerca de 40 ações, 2022–2025).</div>")
    nota_b3 = ""
    if reg.get("anos_b3"):
        anos = ", ".join(str(a) for a in reg["anos_b3"])
        nota_b3 = f" Preço médio de {anos}: B3 (o Yahoo não tem cotações suficientes nesse período)."
    corrente = reg.get("ano_corrente")
    if corrente and corrente.get("sem_registro"):
        nota_corrente = (f" Em {corrente['ano']}, o Yahoo ainda não registra proventos, mas o Fundamentus registra "
                         "pagamento.")
    elif corrente and corrente["dy"] == 0:
        # nenhuma data-com no ano (caso KLBN11 em 2026: o dividendo pago em novembro tem data-com em
        # dezembro do ano anterior e conta naquele ano)
        nota_corrente = (f" Em {corrente['ano']}, nenhuma data-com até agora"
                         + (" (o Fundamentus também não registra)" if reg.get("conferencia") == "ok" else "")
                         + f"; os pagos em {corrente['ano']} foram declarados em {corrente['ano'] - 1} e contam lá.")
    elif corrente:
        nota_corrente = f" Em {corrente['ano']}, até agora: {_pct_br(corrente['dy'], 1)} (ano incompleto, fora da conta)."
    else:
        nota_corrente = ""
    zerados = [str(a["ano"]) for a in reg["anos"] if a["dy"] == 0]
    if reg.get("conferencia") == "falhou" and zerados:
        nota_corrente += (" ⚠️ Os anos sem proventos no Yahoo (" + ", ".join(zerados) + ") não puderam ser conferidos "
                          "no Fundamentus agora: podem ser lacuna da fonte, e não falta de pagamento."
                          + (f" (motivo técnico: {html.escape(reg['conferencia_motivo'])})"
                             if reg.get("conferencia_motivo") else ""))
    elif reg.get("conferencia") == "ok" and zerados:
        nota_corrente += " O Fundamentus também não registra pagamento em " + ", ".join(zerados) + "."
    return ("<div style='border-top:1px solid #e5e7eb;padding-top:10px'>"
            "<div style='display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap'>"
            "<span style='font-weight:700;color:#0f5f56'>Os dividendos são regulares?</span>"
            f"<span style='font-size:.78em;font-weight:700;padding:3px 10px;border-radius:999px;{estilo}'>{etiqueta}</span></div>"
            f"{grafico}{caixas}<div style='font-size:.86em;line-height:1.45;margin-top:8px'>"
            f"{html.escape(texto_regularidade(reg))}</div>{evidencia}"
            "<div style='font-size:.74em;color:#4b5563;margin-top:4px;line-height:1.4'>DY de cada ano = proventos "
            "do ano (dividendos e JCP brutos, Yahoo, completados com o Fundamentus quando faltam) ÷ preço médio do ano, nos 5 anos fechados. Regular: pagou todos "
            f"os anos e o pior rendeu ao menos metade da mediana.{nota_b3}{nota_corrente}</div></div>")


def _html_preco_teto(pt: dict | None) -> str:
    """Caixa "Preço-teto (Bazin)" das ações da B3: dividendos dos últimos 12 meses
    ÷ 6%, como o mercado calcula, com o motivo em uma frase e a conferência com a
    média de 5 anos. Referência de máximo, não altera a pontuação."""
    if not pt:
        return ""
    pct = lambda v: f"{v * 100:.1f}%".replace(".", ",")
    preco, teto = pt["preco"], pt["teto"]
    if preco <= teto:
        cor, fundo, borda = "#15803d", "#f0fdf4", "#86efac"
        frase = f"preço {(1 - preco / teto) * 100:.0f}% abaixo do teto"
    else:
        cor, fundo, borda = "#a16207", "#fefce8", "#fde68a"
        frase = f"preço {(preco / teto - 1) * 100:.0f}% acima do teto"
    rende = "acima" if pt["yield_12m"] >= BAZIN_YIELD else "abaixo"
    motivo = (f"Os proventos dos últimos 12 meses ({formatar_valor(pt['div12'], 'R$')} por ação, JCP bruto) rendem "
              f"{pct(pt['yield_12m'])} ao preço de hoje, {rende} dos 6% do teto.")
    nota = ""
    if pt.get("dividendos_fora_da_media") and pt.get("teto_5a"):
        direcao = "bem acima" if pt["dividendos_fora_da_media"] == "acima" else "bem abaixo"
        nota = (f"<div style='font-size:.8em;color:#374151;margin-top:4px;line-height:1.4'>⚠️ Esses proventos estão "
                f"{direcao} da média dos últimos 5 anos ({formatar_valor(pt['div_medio_5a'], 'R$')} por ação ao ano): "
                f"pela média, o teto seria {formatar_valor(pt['teto_5a'], 'R$')}.</div>")
    return (f"<div style='border:1px solid {borda};background:{fundo};border-radius:10px;padding:9px 11px'>"
            "<div style='display:flex;justify-content:space-between;align-items:baseline;gap:8px'>"
            "<span style='font-weight:700;color:#1e3a8a;font-size:.92em'>Preço-teto (Bazin)</span>"
            f"<span style='font-weight:700;font-size:1.05em;color:{cor}'>{formatar_valor(teto, 'R$')}</span></div>"
            f"<div style='font-size:.86em;font-weight:600;color:{cor};margin-top:2px'>Hoje "
            f"{formatar_valor(preco, 'R$')}: {frase}</div>"
            f"<div style='font-size:.8em;color:#374151;margin-top:5px;line-height:1.4'>{motivo}</div>{nota}"
            "<details style='font-size:.76em;color:#4b5563;margin-top:4px'><summary style='cursor:pointer;"
            "color:#1e3a8a;font-weight:600'>Como é calculado?</summary><div style='margin-top:4px;line-height:1.4'>"
            "Preço-teto = proventos por ação dos últimos 12 meses (dividendos e JCP) ÷ 6%, o yield mínimo de Décio "
            "Bazin — a conta usada pelos sites e calculadoras de mercado. É o preço máximo para os proventos renderem "
            f"6% ao ano. Com o yield mínimo do painel ({pct(pt['limite_regua'])}, 80% do CDI líquido), o teto seria "
            f"{formatar_valor(pt['teto_regua'], 'R$')}. O método olha só os proventos: não considera o crescimento do "
            "lucro, e empresa que distribui pouco do que lucra fica acima do teto. No teste do notebook (Seção 16, "
            "2017–2025), comprar ações bem acima do teto (1,5 vez ou mais, pela régua do painel e média de 5 anos) "
            "superou o CDI em 12 meses em 42% dos casos, contra 55% a 57% até esse nível. Referência de máximo, não "
            "recomendação.</div></details></div>")


def _html_contexto(ctx: dict | None, tipo: str | None, aviso_cenarios: bool = True) -> str:
    """Bloco branco "Contexto do ativo": leitura combinada, indicadores e,
    para FIIs, o preço atrativo. Não altera os cenários do card."""
    if not ctx:
        return ""
    chips = ""
    for ind in ctx["indicadores"]:
        cor = _COR_NIVEL.get(ind["nivel"], "#9ca3af")
        chips += ("<div style='border:1px solid #e5e7eb;border-radius:10px;padding:7px 9px;display:flex;gap:8px;"
                  "align-items:center'>"
                  f"<span style='width:10px;height:10px;border-radius:999px;background:{cor};flex-shrink:0'></span>"
                  f"<div style='display:flex;flex-direction:column;line-height:1.25'><span style='font-size:.74em;"
                  f"color:#4b5563'>{html.escape(ind['nome'])}</span><span style='font-size:.86em;font-weight:600'>"
                  f"{html.escape(ind['valor'])} · {html.escape(ind['leitura'])}</span></div></div>")
    def _lista(titulo, itens):
        if not itens:
            return ""
        return (f"<li style='list-style:none;margin:6px 0 2px -18px;font-weight:700;color:#374151'>{titulo}</li>"
                + "".join(f"<li>{html.escape(_renderizar_detalhe(d))}</li>" for d in itens))
    # duas partes: os mesmos dados (ex.: o yield) podem pontuar nas duas, cada uma com o seu propósito
    detalhes = (_lista("Sinal do momento (preço, tendência e fundamentos)", ctx.get("detalhes_timing", []))
                + _lista("Qualidade da renda (proventos pagos)", ctx.get("detalhes_renda", [])))
    if tipo == "fii" and ctx.get("leitura") and detalhes:
        # nos FIIs, a resposta é o "O preço está atrativo?" (com evidência); a leitura do
        # motor de pontuação, sem vantagem no backtest, fica só dentro da pontuação completa
        detalhes = (f"<li style='list-style:none;margin-left:-18px;color:#4b5563'><i>Leitura do motor de pontuação "
                    f"(regras sem vantagem comprovada no backtest): {html.escape(ctx['leitura'])}</i></li>" + detalhes)
    bloco_detalhes = (f"<details style='font-size:.8em'><summary style='cursor:pointer;color:#1e3a8a;font-weight:600'>"
                      f"Ver pontuação completa</summary><ul style='margin:6px 0 0 0;padding-left:18px'>{detalhes}</ul>"
                      "</details>") if detalhes else ""
    leitura = (f"<div style='font-size:.88em'><b>Leitura combinada:</b> {html.escape(ctx['leitura'])}</div>"
               if ctx.get("leitura") and not (tipo == "fii" and ctx.get("preco_atrativo")) else "")
    subtitulo = ("<span style='font-size:.74em;color:#4b5563'>não altera os cenários acima</span>"
                 if aviso_cenarios else "")
    evidencia = ("" if tipo == "fii" else
                 f"<div style='font-size:.76em;color:#4b5563;line-height:1.4'><b>Sobre a leitura "
                 f"combinada:</b> {FRASE_EVIDENCIA_TECNICA}</div>")
    if ctx.get("regua_pl"):
        evidencia += (f"<div style='font-size:.76em;color:#4b5563;line-height:1.4'><b>Sobre o P/L:</b> "
                      f"{html.escape(ctx['regua_pl'])}</div>")
    preco_atrativo = (_html_preco_atrativo(ctx) if tipo == "fii"
                      else _html_regularidade(ctx.get("regularidade")) + _html_preco_teto(ctx.get("preco_teto")))
    return ("<div style='margin-top:12px;background:#ffffff;color:#1f2937;border-radius:14px;padding:12px 14px;"
            "display:flex;flex-direction:column;gap:9px'>"
            "<div style='display:flex;justify-content:space-between;align-items:baseline;gap:8px'>"
            f"<span style='font-weight:700;color:#1e3a8a'>Contexto do ativo</span>{subtitulo}</div>"
            f"{leitura}<div style='display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:7px'>{chips}</div>"
            f"{evidencia}{preco_atrativo}{bloco_detalhes}</div>")


def _html_card_ativo(ticker: str, tipo: str, preco: float, f12: dict | None, ctx: dict | None,
                     regua: dict | None, movimento: dict | None = None) -> str:
    """Card neutro da aba Ativo Específico: só dados do ativo; a régua de
    faixas aparece apenas quando o ticker está na carteira enviada."""
    moeda = "R$" if ticker.endswith(".SA") else "US$"
    partes = [_html_movimento_brusco(movimento, moeda), _html_faixa_12m(f12, moeda)]
    if regua:
        partes.append(_html_regua(regua, moeda, titulo="Na sua carteira", mostrar_preco_medio=True))
        partes.append("<div style='font-size:.76em;opacity:.85;margin-top:4px'>Os cenários completos estão no card da "
                      "aba Dashboard da Carteira.</div>")
    partes.append(_html_contexto(ctx, tipo, aviso_cenarios=False))
    partes.append("<div style='margin-top:8px;font-size:.72em;opacity:.85;font-style:italic'>Informação para "
                  "investigar, não recomendação. A decisão é sempre sua.</div>")
    return ("<div style='background:#1f2633;border:1px solid #334155;color:#f3f4f6;border-radius:16px;"
            "padding:16px 18px;max-width:640px;box-shadow:0 6px 18px rgba(0,0,0,.15);font-family:inherit'>"
            "<div style='display:flex;justify-content:space-between;align-items:center'>"
            f"<div><div style='font-weight:700;font-size:1.1em'>{html.escape(ticker)}</div>"
            f"<div style='font-size:.8em;color:#9ca3af'>{html.escape(NOMES_TIPO.get(tipo, tipo))}</div></div>"
            f"<div style='font-weight:700;font-size:1.25em'>{formatar_valor(preco, moeda)}</div></div>"
            f"{''.join(partes)}</div>")


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def _proventos_fii_em_cache(ticker: str):
    return buscar_proventos_fii(ticker)


class _ReferenciaIncompleta(Exception):
    """Resultado parcial (falta o segmento/setor ou a bolsa): não vai para o cache."""


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _referencias_pl_completas(ticker: str):
    ref = buscar_referencias_pl(ticker)
    if not ref or not ref.get("completa"):
        raise _ReferenciaIncompleta(ref)  # exceção não é guardada: a próxima análise tenta de novo
    return ref


def _referencias_pl_em_cache(ticker: str):
    """Mediana do P/L do segmento (ou setor) e da bolsa, no Fundamentus. Só o
    resultado completo fica no cache (um dia); o parcial é usado nesta análise
    e buscado de novo na próxima."""
    try:
        return _referencias_pl_completas(ticker)
    except _ReferenciaIncompleta as e:
        return e.args[0] if e.args else None


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def _proventos_acao_em_cache(ticker: str):
    """Proventos de uma ação no Fundamentus: completam as lacunas do Yahoo em tudo
    que usa proventos (yield, renda, preço-teto, extrato) — regra de 17/11/2026."""
    return buscar_proventos_acao(ticker)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _base_cvm_recente():
    """Informes mensais de FIIs da CVM dos últimos 3 anos, baixados uma vez
    por dia: servem para conferir os rendimentos do Fundamentus."""
    try:
        return base_recente(anos=3)
    except Exception:
        return pd.DataFrame()


def _contexto_para(ticker: str, resultado, dividendos_yahoo=None):
    """Contexto do card de um ativo. Para FIIs, o preço atrativo usa 10 anos de
    preços e os rendimentos conferidos entre Fundamentus, CVM e Yahoo
    (``conferir_fontes``): a primeira fonte que responde é usada, as outras
    conferem, e desdobramentos não ajustados são corrigidos pelos eventos
    registrados no Yahoo, a mesma fonte dos preços."""
    if resultado is None:
        return None
    precos = divs = conf = None
    if getattr(resultado, "tipo", None) == "fii":
        try:
            precos = fonte_dados.baixar_precos(ticker, "10y")  # 10 anos, como o notebook (caso AFHI11)
        except Exception:
            precos = None
        base = _base_cvm_recente()
        cvm = serie_rendimentos(base, ticker, "dy") if base is not None and not base.empty else None
        try:
            eventos = fonte_dados.baixar_eventos(ticker)
        except Exception:
            eventos = None
        # Yahoo com poucos anos de cotação (caso AFHI11: só desde 01/2024): completa com o fechamento
        # mensal oficial da B3 (base gerada pelo notebook), na escala do Yahoo
        precos_b3_ate = None
        try:
            precos, precos_b3_ate = completar_precos(precos, carregar_base_b3(), ticker, eventos)
        except Exception:
            precos_b3_ate = None
        conf = conferir_fontes({"fundamentus": _proventos_fii_em_cache(ticker), "cvm": cvm,
                                "yahoo": dividendos_yahoo}, eventos=eventos)
        divs = conf["serie"] if conf["usada"] else None
        # valor patrimonial por cota (CVM), na escala dos preços: P/VP de hoje e da média de 24 meses
        vp = serie_valor_patrimonial(base, ticker) if base is not None and not base.empty else None
        if vp is not None and len(vp):
            vp, _ = ajustar_por_desdobramentos(vp, eventos)
    else:
        vp = None
    ctx = contexto_do_ativo(resultado, precos, divs, cdi_liquido_aa=_cdi_liquido_atual(), valor_patrimonial=vp)
    if ctx is not None:
        ctx["regularidade"] = getattr(resultado, "regularidade", None)
        if getattr(resultado, "tipo", None) == "acao":
            try:
                acrescentar_referencias_pl(ctx, _referencias_pl_em_cache(ticker))
            except Exception:
                pass
        if getattr(resultado, "tipo", None) == "acao" and getattr(resultado, "tec", None) is not None:
            # preço-teto de Bazin (6% sobre os proventos de 12 meses); com lacuna no Yahoo
            # conferida no Fundamentus (regularidade), usa a série completa
            reg = ctx["regularidade"] or {}
            serie = reg.get("proventos_conferidos")
            serie = serie if serie is not None and len(serie) else dividendos_yahoo
            try:
                ctx["preco_teto"] = preco_teto_bazin(serie, resultado.tec.preco_atual,
                                                     hoje=resultado.tec.historico.index[-1],
                                                     cdi_liquido_aa=_cdi_liquido_atual())
            except Exception:
                ctx["preco_teto"] = None
        if getattr(resultado, "tipo", None) == "fii":
            ctx["classe_fii"] = classificar_fii(getattr(resultado, "fund", None) or {})
            ctx["motivo_classe"] = motivo_classificacao(getattr(resultado, "fund", None) or {})
    if ctx is not None and conf is not None:
        ctx["conferencia_rendimentos"] = {"texto": conf["texto"], "selo": conf["selo"],
                                          "observacoes": conf.get("observacoes", [])}
        if precos_b3_ate is not None:
            ctx["precos_b3_ate"] = precos_b3_ate
    return ctx


def _html_caixas_preco(pm: float, preco: float, var: float | None, moeda: str) -> str:
    """Preço médio, preço de hoje e variação em destaque: três caixas brancas,
    valores em negrito; a variação em vermelho quando negativa e em verde
    quando positiva (fundo branco para o vermelho ser legível em qualquer card)."""
    cor = "#b91c1c" if (var or 0) < 0 else ("#15803d" if (var or 0) > 0 else "#111827")
    texto_var = "—" if var is None else f"{'+' if var > 0 else '−' if var < 0 else ''}{abs(var) * 100:.1f}%".replace(".", ",")

    def caixa(rotulo, valor, cor_valor="#111827"):
        return ("<div style='flex:1;min-width:0;background:#ffffff;border-radius:10px;padding:6px 10px'>"
                f"<div style='font-size:.72em;color:#4b5563;font-weight:600'>{rotulo}</div>"
                f"<div style='font-size:1.12em;font-weight:800;color:{cor_valor};white-space:nowrap'>{valor}</div></div>")
    return ("<div style='display:flex;gap:8px;margin-top:10px'>"
            + caixa("Preço médio", formatar_valor(pm, moeda)) + caixa("Hoje", formatar_valor(preco, moeda))
            + caixa("Variação", texto_var, cor) + "</div>")


def _html_caixas_posicao(card: dict) -> str:
    """Segunda linha de caixas: quantidade, valor da posição hoje e resultado
    (ganho ou perda ainda não realizado), com a participação na carteira logo
    abaixo."""
    moeda = card["moeda"]
    q = float(card.get("quantidade") or 0)
    preco, pm = float(card["preco_atual"]), float(card["preco_medio"])
    unidade = "ações" if card.get("grupo") == "Ações" else "cotas"
    if q.is_integer():
        texto_q = f"{int(q):,}".replace(",", ".") + f" {unidade}"
    else:
        texto_q = f"{q:,.4f}".rstrip("0").rstrip(".").replace(",", "_").replace(".", ",").replace("_", ".") + f" {unidade}"
    resultado = q * (preco - pm)
    cor = "#b91c1c" if resultado < 0 else ("#15803d" if resultado > 0 else "#111827")
    sinal = "+" if resultado > 0 else ("−" if resultado < 0 else "")

    def caixa(rotulo, valor, cor_valor="#111827"):
        return ("<div style='flex:1;min-width:0;background:#ffffff;border-radius:10px;padding:6px 10px'>"
                f"<div style='font-size:.72em;color:#4b5563;font-weight:600'>{rotulo}</div>"
                f"<div style='font-size:1.02em;font-weight:800;color:{cor_valor};white-space:nowrap'>{valor}</div></div>")
    linha = ("<div style='display:flex;gap:8px;margin-top:8px'>"
             + caixa("Quantidade", texto_q) + caixa("Valor hoje", formatar_valor(q * preco, moeda))
             + caixa("Ganho/perda", sinal + formatar_valor(abs(resultado), moeda), cor) + "</div>")
    part = card.get("participacao")
    if part:
        txt = (f"{part['carteira'] * 100:.1f}% da carteira · " if part.get("carteira") is not None else "")
        txt += f"{part['grupo'] * 100:.1f}% {part['nome_grupo']}"
        linha += f"<div style='font-size:.8em;margin-top:6px;opacity:.95'>📊 {txt.replace('.', ',')}</div>"
    if card.get("renda_recebida") is not None:
        # terceira linha: o que a posição rendeu desde o início (proventos, resultado e rentabilidade)
        nome = {"FIIs": "Rendimentos", "Ações": "Dividendos e JCP"}.get(card.get("grupo"), "Proventos")
        if nome == "Dividendos e JCP" and moeda != "R$":
            nome = "Dividendos"  # JCP só existe no Brasil
        # Resultado total = ganho/perda da caixa acima (mesma cotação) + vendas já feitas
        # no ciclo atual (ganho realizado) + proventos recebidos.
        realizado = float(card.get("ganho_realizado_ciclo") or 0.0)
        total = resultado + realizado + float(card["renda_recebida"] or 0.0)
        custo = card.get("custo_total_comprado")
        rent = total / custo if custo else None

        def cor_de(v):
            return "#b91c1c" if (v or 0) < 0 else ("#15803d" if (v or 0) > 0 else "#111827")

        def com_sinal(v, texto):
            return ("+" if v > 0 else ("−" if v < 0 else "")) + texto
        desde = f"desde {card['desde']:%d/%m/%Y}" if card.get("desde") else "desde o início da posição"
        linha += (f"<div style='font-size:.74em;margin-top:8px;opacity:.95'>Desde o início da posição ({desde.replace('desde ', '')}): "
                  "preço, vendas e proventos</div>"
                  "<div style='display:flex;gap:8px;margin-top:4px'>"
                  + caixa(nome, formatar_valor(card["renda_recebida"], moeda), "#15803d"
                          if card["renda_recebida"] > 0 else "#111827")
                  + (caixa("Ganho nas vendas" if realizado > 0 else "Perda nas vendas",
                           com_sinal(realizado, formatar_valor(abs(realizado), moeda)), cor_de(realizado))
                     if abs(realizado) >= 0.005 else "")
                  + caixa("Resultado total", "—" if total is None else com_sinal(total, formatar_valor(abs(total), moeda)),
                          cor_de(total))
                  + caixa("Rentabilidade total", "—" if rent is None else
                          com_sinal(rent, f"{abs(rent) * 100:.1f}%".replace(".", ",")), cor_de(rent))
                  + "</div>")
        if abs(realizado) >= 0.005:
            linha += ("<div style='font-size:.74em;margin-top:4px;opacity:.95'>Resultado total = ganho/perda + "
                      f"{'ganho' if realizado > 0 else 'perda'} nas vendas feitas desde o início da posição + "
                      f"{nome.lower() if nome != 'Dividendos e JCP' else 'dividendos e JCP'}.</div>")
    return linha


def _html_movimento_brusco(mov: dict | None, moeda: str) -> str:
    if not mov:
        return ""
    direcao = "Queda" if mov["variacao"] < 0 else "Alta"
    if mov.get("tipo") == "mercado":
        # sem salto de um dia com fator redondo: movimento de mercado (ex.: após a eleição de 10/2026)
        return ("<div style='margin-top:10px;padding:8px 10px;background:#ffffff;color:#374151;"
                "border-left:5px solid #9ca3af;border-radius:8px;font-size:.8em;line-height:1.4'>"
                f"{'📉' if mov['variacao'] < 0 else '📈'} {direcao} de {abs(mov['variacao']) * 100:.0f}% nas últimas "
                f"semanas ({formatar_valor(mov['referencia'], moeda)} → {formatar_valor(mov['preco'], moeda)}): "
                "movimento de mercado, sem sinal de desdobramento ou grupamento.</div>")
    return ("<div style='margin-top:10px;padding:8px 10px;background:#ffffff;color:#92400e;"
            "border-left:5px solid #d97706;border-radius:8px;font-size:.82em;line-height:1.4'>"
            f"<b>{'📉' if mov['variacao'] < 0 else '📈'} {direcao} de {abs(mov['variacao']) * 100:.0f}% em poucos pregões</b> "
            f"({formatar_valor(mov['referencia'], moeda)} → {formatar_valor(mov['preco'], moeda)}). Pode ser movimento "
            "de mercado, mas o salto num único pregão sugere um evento societário (desdobramento, grupamento) "
            "ainda não ajustado pela fonte; confira antes de usar os cenários.</div>")


def _html_tendencia(t: dict | None, compacto: bool = False) -> str:
    """Aviso de tendência de médio prazo rompida (raio_x.tendencia_rompida)."""
    if not t:
        return ""
    partes = []
    if t.get("abaixo_desde") is not None and t["pregoes_abaixo"] >= 20:
        partes.append(f"preço abaixo da média de 200 dias desde {pd.Timestamp(t['abaixo_desde']):%d/%m/%Y} "
                      f"({t['pregoes_abaixo']} pregões; hoje {abs(t['distancia']) * 100:.0f}% abaixo dela)")
    if t.get("cruzamento") is not None:
        partes.append(f"média de 50 dias cruzou para baixo da de 200 em {pd.Timestamp(t['cruzamento']):%d/%m/%Y}")
    if not partes:
        return ""
    texto = "; ".join(partes)
    return ("<div style='margin-top:10px;padding:8px 10px;background:#ffffff;color:#374151;"
            "border-left:5px solid #64748b;border-radius:8px;font-size:.8em;line-height:1.4'>"
            f"<b>📉 Tendência de médio prazo rompida</b>: {texto}."
            + ("" if compacto else " Aviso para reavaliar a posição com as suas regras; não é sinal de venda.")
            + "</div>")


def _html_alertas_compras(alertas: dict | None, moeda: str = "R$") -> str:
    """Compras fora da regra: uma linha curta por compra e a regra uma vez só."""
    if not alertas or not alertas.get("compras"):
        return ""
    itens = "".join(
        f"<li><b>{c['data']:%d/%m/%Y}</b> · {_fmt_qtde_milhar(c['quantidade'])} a {formatar_valor(c['preco'], moeda)} — "
        f"{html.escape(c['motivo'])}</li>" for c in alertas["compras"])
    if alertas.get("restantes"):
        itens += f"<li>… e mais {alertas['restantes']}</li>"
    return ("<div style='margin-top:10px;padding:8px 10px;background:#ffffff;color:#92400e;"
            "border-left:5px solid #d97706;border-radius:8px;font-size:.8em;line-height:1.4'>"
            "<b>🛒 Compras fora da regra de faixas (últimos 12 meses)</b>"
            f"<ul style='margin:4px 0 4px 18px;padding:0'>{itens}</ul>"
            f"<div style='font-size:.92em;color:#78350f'>Regra: {html.escape(alertas['regra'])}</div></div>")


def _html_card_cenario(card: dict) -> str:
    """Card de um ativo: situação, preços e contexto de mercado; tabela de
    cenários (ações/ETFs na faixa) ou próximas faixas; para FIIs, o
    retrospecto de reinvestimento dos últimos 12 meses."""
    fundo, icone, _ = _ESTILO_SITUACAO[card["situacao"]]
    moeda = card["moeda"]
    var = card["variacao"]
    partes = [
        f"<div style='font-size:1.3em;font-weight:700;margin-top:10px'>{html.escape(card['rotulo_situacao'])}</div>",
        _html_caixas_preco(card["preco_medio"], card["preco_atual"], var, moeda),
        _html_caixas_posicao(card),
    ]
    partes.append(_html_movimento_brusco(card.get("movimento_brusco"), moeda))
    partes.append(_html_tendencia(card.get("tendencia"), compacto=True))
    # Evento societário não lançado: o preço médio do card fica errado, então o
    # aviso vem logo abaixo dos preços, em destaque (fundo branco, texto vermelho).
    for aviso in card.get("alertas_evento", []):
        partes.append("<div style='margin-top:10px;padding:8px 10px;background:#ffffff;color:#b91c1c;"
                      "border-left:5px solid #b91c1c;border-radius:8px;font-size:.84em;font-weight:700;"
                      f"line-height:1.4'>⚠️ {'Possível evento não lançado: ' if 'não está registrado' in aviso else 'Confira o evento: '}"
                      f"{html.escape(aviso)}</div>")
    # Linha da aba Operações que não entrou no cálculo: o preço médio e a
    # quantidade do card ficam errados (caso POMO4), então o aviso fica no card.
    linhas_fora = card.get("alertas_planilha", [])
    if linhas_fora:
        itens = "".join(f"<li>{html.escape(a)}</li>" for a in linhas_fora[:6])
        resto = f"<li>… e mais {len(linhas_fora) - 6}</li>" if len(linhas_fora) > 6 else ""
        partes.append("<div style='margin-top:10px;padding:8px 10px;background:#ffffff;color:#b91c1c;"
                      "border-left:5px solid #b91c1c;border-radius:8px;font-size:.84em;line-height:1.4'>"
                      f"<b>⚠️ {len(linhas_fora)} linha(s) da planilha não entraram no cálculo — preço médio e "
                      f"quantidade deste card podem estar errados:</b><ul style='margin:4px 0 0 18px;padding:0'>"
                      f"{itens}{resto}</ul></div>")
    partes.append(_html_alertas_compras(card.get("alertas_compras"), moeda))
    # a situação já está no título do card: a régua não a repete
    partes.append(_html_regua(card.get("regua"), moeda, titulo="Régua de faixas", mostrar_rotulo=False))
    partes.append(_html_faixa_12m(card.get("faixa_12m"), moeda))
    tabela = card.get("tabela")
    if tabela:
        partes.append(f"<div style='margin-top:10px;font-weight:600;font-size:.9em'>{html.escape(tabela['titulo'])}</div>")
        partes.append(_html_tabela(tabela["colunas"], tabela["linhas"], moeda))
    else:
        if not card.get("regua"):
            f = card["faixas"]
            partes.append(f"<div style='margin-top:10px;font-size:.85em'>Próximas faixas: realização a partir de "
                          f"{formatar_valor(f['realizacao'], moeda)} (+25%) · aumento abaixo de "
                          f"{formatar_valor(f['queda'], moeda)} (−15%).</div>")
        if card.get("renda_mensal") and not card.get("retrospecto"):
            partes.append(f"<div style='margin-top:6px;font-size:.85em'>💵 Renda estimada: cerca de "
                          f"{formatar_valor(card['renda_mensal'], moeda)} por mês.</div>")
    for nota in card.get("notas", []):
        partes.append(f"<div style='margin-top:6px;font-size:.82em'>ℹ️ {html.escape(nota)}</div>")
    for alerta in card.get("alertas", []):
        partes.append(f"<div style='margin-top:6px;font-size:.82em'>⚠️ Atenção: {html.escape(alerta)}.</div>")
    retro = card.get("retrospecto")
    if retro:
        periodo = "nos últimos 12 meses" if retro["meses"] >= 12 else f"desde a compra ({retro['meses']} meses)"
        partes.append("<div style='margin-top:10px;padding-top:8px;border-top:1px solid rgba(255,255,255,.3);"
                      f"font-weight:600;font-size:.9em'>Rendimentos {html.escape(periodo)}: e se tivesse reinvestido?</div>")
        partes.append(_html_tabela(retro["colunas"], retro["linhas"], moeda))
        q = retro["compra_na_queda"]
        if q["atingida"]:
            texto_queda = (f"Compra na queda: a faixa (abaixo de {formatar_valor(q['preco_faixa'], moeda)}) foi atingida; "
                           f"comprando com a reserva do CDI, hoje seriam {q['cotas']:.0f} cotas, renda de "
                           f"{formatar_valor(q['renda_mensal'], moeda)}/mês e patrimônio de "
                           f"{formatar_valor(q['patrimonio'], moeda)}.")
        else:
            texto_queda = (f"Compra na queda: faixa não atingida no período (abaixo de "
                           f"{formatar_valor(q['preco_faixa'], moeda)}). Reserva no CDI: {formatar_valor(q['reserva'], moeda)}.")
        partes.append(f"<div style='font-size:.78em;margin-top:6px'>{html.escape(texto_queda)}</div>")
    if card.get("evidencia"):
        partes.append("<div style='margin-top:10px;padding:6px 8px;background:rgba(255,255,255,.15);"
                      f"border-radius:8px;font-size:.8em'>📊 <b>O que o histórico mostra:</b> {card['evidencia']}</div>")
    partes.append(_html_contexto(card.get("contexto"), card.get("tipo")))
    partes.append("<div style='margin-top:8px;font-size:.72em;opacity:.85;font-style:italic'>"
                  "Cenários para comparação, pela regra de faixas. A decisão é sempre sua.</div>")
    return (f"<div style='background:{fundo};color:white;border-radius:16px;padding:16px 18px;"
            f"box-shadow:0 6px 18px rgba(0,0,0,.15)'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center'>"
            f"<div style='display:flex;gap:10px;align-items:center'>"
            f"<div style='background:rgba(255,255,255,.22);border-radius:10px;width:36px;height:36px;display:flex;"
            f"align-items:center;justify-content:center;font-size:19px'>{icone}</div>"
            f"<div style='font-weight:700;font-size:1.1em'>{html.escape(card['ticker'])}</div></div>"
            f"</div>{''.join(partes)}</div>")


def _html_resumo_cenarios(cards: list) -> str:
    """Resumo no topo: quantos ativos em cada situação."""
    blocos = ""
    for situacao, rotulo in (("realizacao", "Faixa de realização de lucro"), ("queda", "Faixa de queda"),
                             ("fora", "Fora das faixas")):
        _, icone, cor = _ESTILO_SITUACAO[situacao]
        n = sum(1 for c in cards if c["situacao"] == situacao)
        blocos += (f"<div style='background:white;color:#1f2937;border-radius:16px;padding:14px 16px;"
                   f"box-shadow:0 4px 14px rgba(0,0,0,.1);border-top:4px solid {cor}'>"
                   f"<div style='background:{cor};border-radius:10px;width:34px;height:34px;display:flex;"
                   f"align-items:center;justify-content:center'>{icone}</div>"
                   f"<div style='font-size:1.9em;font-weight:700;margin-top:8px'>{n}</div>"
                   f"<div style='font-size:.9em;color:#4b5563'>{rotulo}</div></div>")
    return f"<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:18px'>{blocos}</div>"


def _html_cabecalho_fiis(evidencias) -> str:
    """Bloco único no topo do grupo de FIIs, com o que vale para todos."""
    frase = frase_evidencia_fiis(evidencias)
    historico = f"📊 <b>O que o histórico mostra:</b> {frase}<br>" if frase else ""
    return ("<div style='background:#2a2f3a;color:#f3f4f6;border-radius:12px;padding:10px 14px;margin:0 0 12px;"
            f"font-size:.82em;line-height:1.45'>{historico}"
            "📖 <b>Como ler a tabela de cada FII:</b> a coluna <i>Real</i> é o que aconteceu com a sua posição nos "
            "últimos 12 meses, sem reinvestir; as outras simulam o reinvestimento dos rendimentos, com os preços e "
            "rendimentos reais do período. Só cotas inteiras, compradas na data aproximada de pagamento (a sobra "
            "fica parada). O dinheiro em caixa rende o CDI, como CDB de 100% do CDI, com IR regressivo (22,5% a 15%) "
            "e IOF, como se resgatado hoje; rendimentos de FII são isentos. <i>Compra na queda</i>: os rendimentos "
            "ficam no CDI e só compram cotas se o preço cair 15% abaixo do seu preço médio.</div>")


def _html_cabecalho_evidencias(cards_do_grupo: list) -> str:
    """Bloco único no topo de um grupo de ações/ETFs com o que o histórico
    mostra para cada faixa: a frase é a mesma para todos os ativos do grupo,
    por isso aparece uma vez só (e não repetida em cada card)."""
    rotulos = {"realizacao": "📈 Faixa de realização de lucro", "queda": "📉 Faixa de queda"}
    blocos = []
    for situacao in ("realizacao", "queda"):
        frase = next((c["evidencia"] for c in cards_do_grupo
                      if c["situacao"] == situacao and c.get("evidencia")), None)
        if frase:
            blocos.append(f"<b>{rotulos[situacao]}</b> — {frase}")
    if not blocos:
        return ""
    return ("<div style='background:#2a2f3a;color:#f3f4f6;border-radius:12px;padding:10px 14px;margin:0 0 12px;"
            "font-size:.82em;line-height:1.45'>📊 <b>O que o histórico mostra</b> (vale para os cards deste grupo)"
            "<br>" + "<br>".join(blocos) + "</div>")


def _html_grupos_cenarios(cards: list, evidencias, colunas: int | None = None) -> str:
    """Cards agrupados por mercado e tipo de ativo, com um cabeçalho único
    no topo de cada grupo (FIIs: evidências e como ler a tabela; demais:
    evidências por faixa)."""
    grupos: dict = {}
    for card in cards:
        grupos.setdefault((card["mercado"], card["grupo"]), []).append(card)
    partes = []
    for chave, cards_do_grupo in grupos.items():
        partes.append(f"<div style='font-weight:700;font-size:1.05em;margin:6px 0 10px 2px'>"
                      f"{html.escape(chave[0])} · {html.escape(_NOME_GRUPO.get(chave[1], chave[1]))}</div>")
        partes.append(_html_cabecalho_fiis(evidencias) if chave[1] == "FIIs"
                      else _html_cabecalho_evidencias(cards_do_grupo))
        grade = [_html_card_cenario({**c, "evidencia": None}) for c in cards_do_grupo]
        grade_css = (f"repeat({colunas},minmax(0,1fr))" if colunas else "repeat(auto-fill,minmax(460px,1fr))")
        partes.append(f"<div style='display:grid;grid-template-columns:{grade_css};"
                      "gap:16px;margin-bottom:22px;align-items:start'>" + "".join(grade) + "</div>")
    return "".join(partes)


_COLUNAS_DINHEIRO = ("Preço", "Preço-teto")


def _colunas_texto_centralizadas(df: pd.DataFrame) -> dict:
    """Tudo ao centro, menos os valores em dinheiro (preço e preço-teto), à
    direita. ``alignment`` existe a partir do Streamlit 1.47."""
    import inspect
    if "alignment" not in inspect.signature(st.column_config.TextColumn).parameters:
        return {}
    config = {}
    for c in df.columns:
        if c in _COLUNAS_DINHEIRO:
            config[c] = st.column_config.Column(alignment="right")
        elif pd.api.types.is_numeric_dtype(df[c]):
            config[c] = st.column_config.NumberColumn(alignment="center")
        else:
            config[c] = st.column_config.TextColumn(alignment="center")
    return config


def _renderizar_tabela_indicadores(cenarios) -> None:
    """Tabela de indicadores (uma linha por posição aberta, por grupo) no
    alto do Dashboard; clicar numa linha abre o card do ativo logo abaixo."""
    st.markdown("#### 📋 Indicadores da carteira")
    if not cenarios or not cenarios[0]:
        st.caption("Clique em 'Analisar carteira' na aba 'Carteira' para ver a tabela.")
        return
    cards = cenarios[0]
    fundamentos = {t: fundamentos_de_resultado(r)
                   for t, _, r in st.session_state.get("carteira_linhas_analise") or []}
    for t, fund in fundamentos.items():
        if t.endswith(".SA") and "ev_ebitda" not in fund:
            try:
                fa = buscar_fundamentos_acao(t)
                fund["ev_ebitda"] = fa.ev_ebitda if fa else None
                if fa and fund.get("roic") is None:
                    fund["roic"] = fa.roic
            except Exception:
                fund["ev_ebitda"] = None
    hoje = datetime.now(ZoneInfo("America/Sao_Paulo")).date()
    tabelas = montar_tabelas(cards, fundamentos, st.session_state.get("carteira_dividendos") or {}, hoje)
    if not tabelas:
        return
    pesos = _pesos_atuais()
    if pesos:  # coluna "Peso (%)" logo depois da faixa, em vermelho acima do limite
        for df in tabelas.values():
            df.insert(list(df.columns).index("Faixa") + 1 if "Faixa" in df.columns else 1, "Peso (%)",
                      [pesos[t]["peso"] if t in pesos else None for t in df.index])
    st.caption("Uma linha por posição aberta. **Para ordenar, clique no título da coluna** (clique de novo para "
               "inverter). Marque até 4 linhas (inclusive de abas "
               "diferentes) para ver os cards lado a lado e comparar. Números iguais aos dos cards. Preço × teto "
               "negativo = preço abaixo do preço-teto. Meses: os da data-com dos proventos nos últimos 24 meses.")
    abas = st.tabs([f"{g} ({len(df)})" for g, df in tabelas.items()])
    casas = {"P/VP": 2, "P/VP médio 24m": 2, "Preço": 2, "Preço-teto": 2, "Preço × teto (%)": 0}
    cor_faixa = {"📈": "#2f45e0", "📉": "#d9831c", "↔️": "#0f7f73"}

    def estilo_faixa(valor):
        cor = next((c for icone, c in cor_faixa.items() if str(valor).startswith(icone)), None)
        return f"background-color:{cor};color:#ffffff;font-weight:600" if cor else ""

    def estilizar(df, moeda):
        numericas = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        def numero_br(casas_dec):
            return lambda v: ("—" if pd.isna(v) else
                              f"{v:,.{casas_dec}f}".replace(",", "X").replace(".", ",").replace("X", "."))
        formatos = {c: numero_br(casas.get(c, 1)) for c in numericas if c not in _COLUNAS_DINHEIRO}
        formatos.update({c: (lambda v, m=moeda: formatar_valor(v, m) if pd.notna(v) else "—")
                         for c in _COLUNAS_DINHEIRO if c in df.columns})
        # negativos em vermelho; "Preço × teto" fica de fora: lá negativo é preço ABAIXO do teto
        negativas = [c for c in numericas if c not in ("Preço × teto (%)",)]
        return (df.style.set_properties(**{"background-color": "#ffffff", "color": "#000000"})
                .map(lambda v: "color:#b91c1c" if isinstance(v, (int, float)) and pd.notna(v) and v < 0 else "",
                     subset=negativas)
                .map(estilo_faixa, subset=["Faixa"])
                .apply(lambda col: ["color:#b91c1c;font-weight:600" if pesos.get(t, {}).get("situacao") == "acima"
                                    else "" for t in col.index], subset=[c for c in ("Peso (%)",) if c in df.columns])
                .format(formatos))

    selecionados = []
    for aba, (grupo, df) in zip(abas, tabelas.items()):
        with aba:
            evento = st.dataframe(estilizar(df, "US$" if grupo == "EUA" else "R$"), hide_index=True, width="stretch", on_select="rerun",
                                  height=min(36 * (len(df) + 1) + 3, 640),
                                  selection_mode="multi-row", key=f"tab_ind_{grupo}",
                                  column_config=_colunas_texto_centralizadas(df))
            linhas_sel = getattr(getattr(evento, "selection", None), "rows", None) or []
            selecionados += [df.index[i] for i in linhas_sel]
    if selecionados:
        if len(selecionados) > 4:
            st.caption("Mostrando os 4 primeiros ativos selecionados.")
        escolhidos = [c for t in selecionados[:4] for c in cards if c["ticker"] == t]
        grade = "".join(_html_card_cenario({**c, "evidencia": None}) for c in escolhidos)
        n = max(len(escolhidos), 2)
        st.html(f"<div style='display:grid;grid-template-columns:repeat({n},minmax(0,1fr));gap:14px;"
                f"margin:6px 0 16px;align-items:start'>{grade}</div>")


def _renderizar_cenarios(cenarios) -> None:
    """Cenários comparativos por ativo (regra de faixas de preço sobre o
    preço médio). ``cenarios`` é o par (cards, avisos) de ``gerar_cenarios``,
    guardado na sessão ao clicar em 'Analisar carteira'. Usa ``st.html``
    (HTML puro, sem Markdown), o que evita o cifrão virar fórmula."""
    st.markdown("#### 🧭 Cenários para cada ativo")
    if cenarios is None:
        st.caption("Clique em 'Analisar carteira' na aba 'Carteira' para ver os cenários de cada ativo.")
        return
    cards, avisos = cenarios
    st.caption("Compare o que acontece em cada caminho. Os números são calculados pela regra de faixas e não "
               "constituem recomendação: a decisão é sempre sua. Histórico de preço completo na aba Raio-X do ativo.")
    if cards:
        cont = {k: sum(1 for c in cards if c.get("situacao") == k) for k in ("realizacao", "queda", "fora")}
        with st.expander(f"Ver os {len(cards)} cards ({cont['realizacao']} na faixa de realização · "
                         f"{cont['queda']} na de queda · {cont['fora']} fora das faixas)"):
            st.html(_html_resumo_cenarios(cards) + _html_grupos_cenarios(cards, carregar_evidencias()))
    else:
        st.info("Nenhuma posição aberta encontrada na planilha.")
    for aviso in avisos:
        st.caption(f"⚠️ {_sem_formula(aviso)}")



# =============================================================================
# ABA 1 — Carteira (carregar e analisar)
# =============================================================================
with tab_carteira:
    if bytes_planilha is None:
        st.info("Envie um arquivo Excel acima para começar a análise.")
    else:
        st.subheader("Confirme o tipo de cada ativo")
        sem_tipo = [t for t in tickers_encontrados if t not in tipos_planilha]
        if tipos_planilha and not sem_tipo:
            st.caption("Tipos lidos da coluna de tipo de ativo da planilha. Para corrigir algum, mude aqui "
                       "ou na planilha.")
        elif tipos_planilha:
            st.caption(f"Tipos lidos da planilha. Sem tipo na planilha ({', '.join(sem_tipo)}): adivinhamos "
                       f"pelo ticker (⚠️) — confira, principalmente os terminados em '11' (FII ou ETF).")
        else:
            st.caption(
                "Adivinhamos o tipo pelo formato do ticker — confira se está certo antes de rodar "
                "a análise, principalmente para tickers terminados em '11' (podem ser FII ou ETF). "
                "Dica: uma coluna 'Tipo de ativo' (Ação, FII ou ETF) na planilha dispensa esta etapa."
            )
        for aviso in avisos_tipos_planilha:
            st.caption(f"⚠️ {aviso}")

        tipos_confirmados: dict[str, str] = {}
        colunas = st.columns(3)
        for i, ticker in enumerate(tickers_encontrados):
            chute = tipos_planilha.get(ticker) or _heuristica_tipo(ticker)
            # a chave muda com o tipo da planilha: planilha corrigida não fica presa à escolha anterior
            chave = f"tipo_{ticker}_{tipos_planilha.get(ticker, 'auto')}"
            with colunas[i % 3]:
                escolha = st.selectbox(
                    ticker if ticker in tipos_planilha or not tipos_planilha else f"{ticker} ⚠️",
                    options=OPCOES_TIPO, index=OPCOES_TIPO.index(chute),
                    format_func=lambda t: NOMES_TIPO[t], key=chave,
                )
                tipos_confirmados[ticker] = escolha

        periodo_carteira = st.selectbox(
            "Histórico para análise técnica", options=list(OPCOES_PERIODO.keys()),
            format_func=lambda p: OPCOES_PERIODO[p], index=list(OPCOES_PERIODO.keys()).index("2y"),
            key="periodo_carteira",
        )

        if st.button("📊 Analisar carteira", type="primary"):
            texto_horario, pregao_provavelmente_aberto = _info_horario_analise()

            with st.spinner("Buscando dados de mercado e calculando... isso pode levar um tempo."):
                # FIIs: rendimentos conferidos entre Fundamentus, CVM e Yahoo (o Yahoo sozinho
                # tem lacunas de anos) na renda recebida do extrato e na renda dos cards
                base_cvm = _base_cvm_recente()
                fonte_rendimentos = FonteComRendimentosConferidos(
                    fonte_dados, tipos_confirmados,
                    lambda t: {"fundamentus": _proventos_fii_em_cache(t),
                               "cvm": (serie_rendimentos(base_cvm, t, "dy")
                                       if base_cvm is not None and not base_cvm.empty else None)},
                    proventos_da_acao=_proventos_acao_em_cache)
                _tempos = {"inicio": time.perf_counter()}  # tempo de cada etapa (mostrado abaixo do resultado)
                # Busca de uma vez, em paralelo, o que o Yahoo ainda não tem em cache (preços em
                # lote; dividendos, eventos e dados cadastrais em até 8 buscas simultâneas). A
                # análise abaixo, ativo por ativo, encontra tudo pronto. Cache compartilhado
                # entre usuários (dados públicos de mercado; a planilha fica só na sessão).
                try:
                    fonte_dados.aquecer(tickers_encontrados, periodos=(periodo_carteira, "5y", "10y"),
                                        info=True)
                except Exception:
                    pass
                _tempos["Yahoo em lote"] = time.perf_counter()
                linhas_ganho = processar_carteira_combinada(df_resumo, df_operacoes, fonte_rendimentos,
                                                            periodo_carteira)
                _tempos["extrato"] = time.perf_counter()
                linhas_analise = []
                dividendos_por_ticker = {}
                # posição encerrada (vendida, ou fundo extinto e liquidado, como o BCFF11): fica só no
                # extrato; não é analisada nem gera aviso de "sem dados de mercado" (ativo deslistado)
                encerrados = sorted(
                    {l.ticker for l in linhas_ganho if l.situacao == "Encerrada"}
                    - {l.ticker for l in linhas_ganho if l.situacao != "Encerrada"})
                st.session_state["carteira_encerrados"] = encerrados
                for ticker in tickers_encontrados:
                    if ticker in encerrados:
                        continue
                    tipo = tipos_confirmados[ticker]
                    # FIIs: o motor usa os mesmos rendimentos conferidos do "O preço está atrativo?"
                    resultado = analisar_ativo(ticker, tipo, periodo_carteira, fonte_rendimentos,
                                               cdi_liquido_aa=_cdi_liquido_atual())
                    linhas_analise.append((ticker, tipo, resultado))
                    try:
                        dividendos_por_ticker[ticker] = fonte_rendimentos.baixar_dividendos(ticker)
                    except Exception:
                        dividendos_por_ticker[ticker] = pd.Series(dtype=float)

                # Cenários comparativos por ativo (regra de faixas de preço) — mostrados
                # na aba Dashboard. Falha aqui não pode derrubar o resto da análise.
                _tempos["análise de cada ativo"] = time.perf_counter()
                try:
                    fundamentos_por_ticker = {
                        t: fundamentos_de_resultado(r) for t, tp, r in linhas_analise
                    }
                    cdi, aviso_cdi = _cdi_do_banco_central()
                    cards, avisos = gerar_cenarios(
                        df_resumo, df_operacoes, fonte_rendimentos, tipos_confirmados, fundamentos_por_ticker,
                        indice_cdi_por_data=lambda idx: indice_cdi_acumulado(idx, cdi, cdi_aa=CDI_RESERVA_AA),
                        evidencias=carregar_evidencias())
                    if aviso_cdi and any(c.get("retrospecto") for c in cards):
                        avisos.append(aviso_cdi)
                    # Desdobramento/grupamento conhecido pela fonte e não lançado na
                    # planilha: o preço médio do card fica errado (caso RBVA11 na
                    # validação), então o aviso vai no próprio card e no topo.
                    eventos = {l.ticker: l.avisos_eventos for l in linhas_ganho if l.avisos_eventos}
                    resultado_por_ticker = {t: r for t, tp, r in linhas_analise}
                    linhas_fora = {l.ticker: l.avisos for l in linhas_ganho if l.avisos}
                    for c in cards:
                        c["alertas_evento"] = list(eventos.get(c["ticker"], []))
                        c["alertas_planilha"] = list(linhas_fora.get(c["ticker"], []))
                        try:
                            c["contexto"] = _contexto_para(c["ticker"], resultado_por_ticker.get(c["ticker"]),
                                                           dividendos_por_ticker.get(c["ticker"]))
                            pt = (c["contexto"] or {}).get("preco_teto")
                            if pt and c.get("preco_atual"):
                                # mesma cotação do card (a análise baixa o preço em outro momento do pregão)
                                pt.update(preco=float(c["preco_atual"]), yield_12m=pt["div12"] / float(c["preco_atual"]))
                        except Exception:
                            c["contexto"] = None  # sem contexto, o card segue com os cenários
                    avisos[:0] = [f"{t}: {a}" for t, lista in eventos.items() for a in lista]
                    try:  # cotação do dólar para somar ativos dos EUA à carteira
                        usd_brl = float(fonte_dados.baixar_precos("BRL=X", "5d").dropna().iloc[-1])
                    except Exception:
                        usd_brl = None
                    calcular_participacao(cards, usd_brl)
                    # proventos recebidos e resultado total da posição atual (ciclo aberto, como no extrato)
                    ciclo_aberto = {l.ticker: l for l in linhas_ganho if l.situacao == "Aberta"}
                    for c in cards:
                        l = ciclo_aberto.get(c["ticker"])
                        if l is not None:
                            c["renda_recebida"] = l.renda_recebida
                            c["desde"] = l.ciclo_inicio
                            # O resultado total é montado no card (``_html_caixas_posicao``) a
                            # partir do ganho/perda mostrado nele: somar aqui o ganho não
                            # realizado da planilha usava outra cotação (baixada em outro
                            # momento do pregão) e não batia com a caixa ao lado.
                            c["ganho_realizado_ciclo"] = l.ganho_realizado or 0.0
                            c["custo_total_comprado"] = getattr(l, "custo_total_comprado", None)
                    cenarios = (cards, avisos)
                    _tempos["cards e contexto (Fundamentus)"] = time.perf_counter()
                except Exception as e:
                    cenarios = ([], [f"não foi possível calcular os cenários ({e})"])

            # Guarda tudo na sessão — sobrevive a um clique posterior noutra aba/botão
            # (ver docstring do arquivo) e alimenta também a aba Dashboard.
            st.session_state["carteira_horario"] = texto_horario
            try:
                marcas = list(_tempos.items())
                st.session_state["carteira_tempos"] = [(nome, fim - ini) for (_, ini), (nome, fim)
                                                       in zip(marcas, marcas[1:])]
            except Exception:
                st.session_state["carteira_tempos"] = []
            st.session_state["carteira_pregao_aberto"] = pregao_provavelmente_aberto
            st.session_state["carteira_linhas_ganho"] = linhas_ganho
            st.session_state["carteira_linhas_analise"] = linhas_analise
            st.session_state["carteira_dividendos"] = dividendos_por_ticker
            st.session_state["carteira_fonte_rendimentos"] = fonte_rendimentos
            st.session_state.pop("visao_geral_dados", None)  # recalcula a aba Visão geral
            st.session_state.pop("ir_recebidos", None)
            st.session_state["carteira_tipos_confirmados"] = tipos_confirmados
            st.session_state["carteira_data_analise"] = datetime.now(ZoneInfo("America/Sao_Paulo")).date()
            st.session_state["carteira_cenarios"] = cenarios

        if "carteira_linhas_analise" in st.session_state:
            # Aba só de entrada: o resultado fica nos cards e no extrato do Dashboard.
            # A tabela do motor de pontuação, os detalhes e o gráfico saíram daqui:
            # repetiam o card (e, nos FIIs, contradiziam o "O preço está atrativo?");
            # o gráfico continua na aba Ativo Específico.
            n_ativos = len(st.session_state["carteira_linhas_analise"])
            sem_dados = [t for t, _, r in st.session_state["carteira_linhas_analise"] if r is None]
            n_dados = len(sem_dados)
            st.success(f"✅ Análise de {n_ativos} ativo(s) concluída em {st.session_state['carteira_horario']}. "
                       "Os cards de cada ativo e o extrato estão na aba **💼 Dashboard da Carteira**.")
            tempos = st.session_state.get("carteira_tempos") or []
            if tempos:
                st.caption("⏱️ Tempo da análise: " + " · ".join(f"{n} {seg:.0f}s" for n, seg in tempos)
                           + f" · total {sum(seg for _, seg in tempos):.0f}s")
            encerrados = st.session_state.get("carteira_encerrados") or []
            if encerrados:
                st.caption(f"Posições encerradas ({', '.join(encerrados)}): só no extrato do Dashboard, "
                           "sem análise de mercado.")
            if n_dados:
                st.warning(f"⚠️ Sem dados de mercado suficientes: {', '.join(sem_dados)} — confira o ticker e o tipo "
                           "(ativo deslistado ou com outro código não tem cotação).")
            if st.session_state["carteira_pregao_aberto"]:
                st.caption(
                    "📊 Pregão provavelmente em andamento — os preços usados são os mais "
                    "recentes disponíveis agora, não necessariamente o fechamento definitivo "
                    "do dia. Rodar a análise de novo após o fechamento (~18h, horário de "
                    "Brasília) pode trazer números diferentes."
                )
            st.caption("O gráfico de preço de cada ativo está na aba **🔍 Raio-X do ativo**. Este painel não é uma "
                       "recomendação de investimento nem substitui um assessor ou consultor licenciado (CVM).")

# =============================================================================
# ABA 2 — Visão geral (a carteira mês a mês, com filtros)
# =============================================================================
_NOME_TIPO_VG = {"acao": "Ações", "acao_us": "Ações", "fii": "FIIs", "etf_br": "ETFs", "etf_us": "ETFs"}
_MES_PT = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]


def _rotulo_mes(p: pd.Period) -> str:
    return f"{_MES_PT[p.month - 1]}/{p.year}"


def _serie_mensal(serie: pd.Series | None) -> pd.Series | None:
    if serie is None or len(serie) == 0:
        return None
    s = pd.Series(serie, dtype=float).dropna()
    idx = pd.DatetimeIndex(s.index)
    s.index = (idx.tz_localize(None) if idx.tz is not None else idx).to_period("M")
    return s.groupby(level=0).last()


def _dados_visao_geral(df_resumo, df_operacoes, tipos: dict) -> dict:
    """Cada ativo da planilha (abertos e encerrados) mês a mês, do primeiro
    mês com operação até hoje. Preços: base mensal da B3 (escala da época)
    completada pela cotação de hoje; sem a base (EUA, ativos fora dela),
    Yahoo ajustado por desdobramento, convertido pela escala dos eventos."""
    resumo = linhas_resumo_por_ticker(df_resumo)
    grupos = {}
    if df_operacoes is not None and not df_operacoes.empty:
        chaves = df_operacoes["ticker"].apply(lambda t: normalizar_ticker(t)[0])
        grupos = {t: g for t, g in df_operacoes.groupby(chaves)}
    ops_por = {}
    for t in sorted(set(resumo) | set(grupos)):
        ops, _, _ = operacoes_do_ticker(resumo.get(t), grupos.get(t))
        if ops:
            ops_por[t] = ops
    if not ops_por:
        return {}
    hoje = pd.Timestamp(datetime.now(ZoneInfo("America/Sao_Paulo")).date())
    inicio = min(pd.Timestamp(o["data"]) for ops in ops_por.values() for o in ops)
    meses = meses_entre(inicio, hoje)
    try:
        base = carregar_base_b3()
    except Exception:
        base = None
    try:
        fonte_dados.aquecer([t for t in ops_por if not t.endswith(".SA")] + ["BRL=X"], periodos=("max",),
                            dividendos=False, eventos=False)
        fonte_dados.aquecer(list(ops_por), periodos=("5d",))
    except Exception:
        pass
    fonte_rend = st.session_state.get("carteira_fonte_rendimentos")
    dividendos_sessao = st.session_state.get("carteira_dividendos") or {}
    linhas, meta, sem_preco = {}, {}, []
    for t, ops in ops_por.items():
        b3 = t.endswith(".SA")
        try:
            eventos = fonte_dados.baixar_eventos(t)
        except Exception:
            eventos = None
        precos, ajustados = None, True
        if b3 and base is not None and not base.empty:
            b = base[base["ticker"] == t[:-3]]
            if len(b):
                precos = pd.Series(b["fechamento"].values, index=pd.to_datetime(b["data"]))
                ajustados = False
                try:  # cotação de hoje (escala de hoje = escala da época no mês atual)
                    ult = fonte_dados.baixar_precos(t, "5d").dropna()
                    if len(ult):
                        precos.loc[hoje] = float(ult.iloc[-1])
                except Exception:
                    pass
        if precos is None:
            try:
                precos = fonte_dados.baixar_precos(t, "max")
            except Exception:
                precos = None
        if precos is None or len(precos) == 0:
            sem_preco.append(t)
            continue
        divs = dividendos_sessao.get(t)
        if divs is None:
            try:
                divs = (fonte_rend or fonte_dados).baixar_dividendos(t)
            except Exception:
                divs = None
        linhas[t] = linha_do_tempo_ativo(ops, meses, precos, divs, eventos, precos_ajustados=ajustados)
        tipo = tipos.get(t) or ("acao" if b3 else "etf_us")
        meta[t] = {"mercado": "B3" if b3 else "EUA", "tipo": _NOME_TIPO_VG.get(tipo, "Ações")}
    try:
        cambio = _serie_mensal(fonte_dados.baixar_precos("BRL=X", "max"))
    except Exception:
        cambio = None
    return {"linhas": linhas, "meta": meta, "sem_preco": sem_preco, "meses": meses, "cambio": cambio,
            "ops": ops_por}


def _kpi_vg(rotulo, valor, sub, cor="#111827"):
    return (f"<div style='flex:1;min-width:170px;background:#fff;border-radius:12px;padding:10px 12px'>"
            f"<div style='font-size:.75em;color:#4b5563'>{rotulo}</div>"
            f"<div style='font-size:1.25em;font-weight:700;color:{cor}'>{valor}</div>"
            f"<div style='font-size:.74em;color:#4b5563'>{sub}</div></div>")


def _pct_vg(v):
    return ("+" if v > 0 else ("−" if v < 0 else "")) + f"{abs(v) * 100:.1f}%".replace(".", ",")


_EIXO_MIL = "replace(format(datum.value / 1000, ',.0f'), ',', '.') + ' mil'"
_EIXO_PCT = "replace(format(datum.value, '.0%'), '.', ',')"


def _eixo_meses(indice) -> tuple[list[str], alt.X]:
    """Eixo X por mês (rótulos "mm/aaaa"), com no máximo ~12 rótulos visíveis.
    Ordinal: barras centradas no mês e sem rótulos repetidos."""
    rot = [f"{d:%m/%Y}" for d in pd.DatetimeIndex(indice)]
    passo = max(1, -(-len(rot) // 12))
    eixo = alt.X("Rótulo:O", title=None, sort=None,
                 axis=alt.Axis(values=rot[::passo], labelAngle=0, labelOverlap=True))
    return rot, eixo


def _grafico_linhas(df: pd.DataFrame, eixo_y: str, formato: str, cores: dict, tracejadas=()):
    rot, eixo_x = _eixo_meses(df.index)
    dados = df.assign(Rótulo=rot).reset_index(drop=True).melt("Rótulo", var_name="Série", value_name="valor")
    return (alt.Chart(dados).mark_line(strokeWidth=2.4).encode(
        x=eixo_x,
        y=alt.Y("valor:Q", title=eixo_y, axis=alt.Axis(labelExpr=_EIXO_PCT if "%" in formato else _EIXO_MIL)),
        color=alt.Color("Série:N", scale=alt.Scale(domain=list(cores), range=list(cores.values())),
                        legend=alt.Legend(orient="top", title=None)),
        strokeDash=alt.condition(alt.FieldOneOfPredicate("Série", list(tracejadas)), alt.value([6, 4]),
                                 alt.value([1, 0])),
        tooltip=[alt.Tooltip("Rótulo:O", title="Mês"), "Série:N", alt.Tooltip("valor:Q", format=formato)],
    ).properties(height=280))


def _faixas(escolhidos) -> list:
    """Faixas das posições abertas, clicáveis. Devolve os cards da faixa
    escolhida (desenhados por ``_grade_cards``, na largura toda da página)."""
    cards = (st.session_state.get("carteira_cenarios") or ([], []))[0]
    cards = [c for c in cards if c.get("ticker") in escolhidos]
    if cards:
        rotulos = {"realizacao": "📈 {n} na faixa de realização", "queda": "📉 {n} na faixa de queda",
                   "fora": "↔️ {n} fora das faixas"}
        opcoes = {rot.format(n=sum(1 for c in cards if c.get("situacao") == k)): k for k, rot in rotulos.items()}
        # botões com as cores das faixas (as mesmas dos cards), letra branca; o escolhido ganha contorno
        st.html("<style>"
                ".st-key-vg_faixa button{color:#fff !important;border:none !important;padding:6px 16px}"
                ".st-key-vg_faixa button p{font-size:1.08rem;font-weight:600;color:#fff !important}"
                ".st-key-vg_faixa button:nth-of-type(1){background:#2f45e0 !important}"
                ".st-key-vg_faixa button:nth-of-type(2){background:#d9831c !important}"
                ".st-key-vg_faixa button:nth-of-type(3){background:#0f7f73 !important}"
                ".st-key-vg_faixa button[aria-checked='true']{box-shadow:0 0 0 3px #ffffff !important}"
                "</style>")
        escolha = st.pills("Clique numa faixa para ver os cards dos ativos", list(opcoes), selection_mode="single",
                           key="vg_faixa")
        if escolha:
            return [c for c in cards if c.get("situacao") == opcoes[escolha]] or [None]
    return []


def _grade_cards(selecionados: list) -> None:
    """Cards da faixa escolhida: uma grade só, 4 por linha, sem separar por
    mercado/tipo (o card já mostra o ticker e a moeda)."""
    if not selecionados:
        return
    if selecionados == [None]:
        st.caption("Nenhum ativo nesta faixa.")
        return
    grade = "".join(_html_card_cenario({**c, "evidencia": None}) for c in selecionados)
    st.html("<div style='display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;"
            f"margin:4px 0 18px;align-items:start'>{grade}</div>")


def _destaques(escolhidos, linhas, i0, i1) -> None:
    """Maiores altas e quedas no período (posições abertas)."""
    abertos = [(t, linhas[t]) for t in escolhidos if linhas[t]["quantidade"].iloc[-1] > 1e-9]
    if abertos:
        rent_ativo = []
        for t, df_t in abertos:
            p = df_t.iloc[i0:i1 + 1]
            v0 = float(df_t["patrimonio"].iloc[i0 - 1]) if i0 > 0 else 0.0
            rent_ativo.append((t.replace(".SA", ""), float(rentabilidade_cotas(p, v0).iloc[-1])))
        rent_ativo.sort(key=lambda x: -x[1])
        fmt = lambda lst: " · ".join(f"{t} {_pct_vg(v)}" for t, v in lst)
        st.html("<div style='background:#fff;color:#111827;border-radius:10px;padding:8px 12px;font-size:.88em;"
                "margin:4px 0 8px'>"
                f"<b style='color:#2f45e0'>Maiores altas no período:</b> {fmt(rent_ativo[:3])}<br>"
                f"<b style='color:#d9831c'>Maiores quedas no período:</b> {fmt(rent_ativo[::-1][:3])}</div>")


_CORES_TIPO = {"Ações": "#2563eb", "FIIs": "#0f766e", "ETFs": "#a855f7"}


def _concentracao_html(escolhidos, linhas, cambio, i1) -> str:
    """Maior posição e fatia dos 5 maiores no último mês do período."""
    valores = []
    for t in escolhidos:
        v = float(linhas[t]["patrimonio"].iloc[i1])
        if cambio and t in cambio and cambio[t] is not None:
            serie = cambio[t].reindex(linhas[t].index).ffill().bfill()
            v *= float(serie.iloc[i1])
        if v > 0:
            valores.append((t.replace(".SA", ""), v))
    total = sum(v for _, v in valores)
    if not total:
        return ""
    valores.sort(key=lambda x: -x[1])
    maior, top5 = valores[0], valores[:5]
    pct = lambda v: f"{v / total * 100:.1f}%".replace(".", ",")
    return (f"<div style='background:#fff;border-radius:12px;padding:10px 12px;color:#111827'>"
            f"<div style='font-size:.75em;color:#4b5563'>Concentração</div>"
            f"<div style='font-size:1.02em;font-weight:700'>{maior[0]} {pct(maior[1])} · "
            f"{len(top5)} maiores {pct(sum(v for _, v in top5))}</div>"
            f"<div style='font-size:.74em;color:#4b5563'>{', '.join(t for t, _ in top5)}</div></div>")


def _tabela_html(cabecalho: list[str], linhas_tab: list[list[str]]) -> str:
    th = "padding:5px 10px;font-size:.76em;color:#4b5563;border-bottom:1px solid #e5e7eb;text-align:right"
    td = "padding:5px 10px;font-size:.84em;border-bottom:1px solid #f1f5f9;text-align:right"
    cab = "".join(f"<th style='{th}{';text-align:left' if i == 0 else ''}'>{c}</th>" for i, c in enumerate(cabecalho))
    corpo = "".join("<tr>" + "".join(f"<td style='{td}{';text-align:left' if i == 0 else ''}'>{c}</td>"
                                     for i, c in enumerate(l)) + "</tr>" for l in linhas_tab)
    return f"<table style='width:100%;border-collapse:collapse'><tr>{cab}</tr>{corpo}</table>"


_COR_SINAL = {"bom": "#1d4ed8", "ruim": "#b91c1c"}


def _grupos_html(grupos: list[tuple[str, list[tuple]]], legenda: bool = False) -> str:
    """Blocos lado a lado (título + linhas "rótulo ... valor", com nota
    opcional embaixo). Cada item: (rótulo, valor, nota[, sinal]); sinal
    "bom" pinta o valor de azul, "ruim" de vermelho."""
    blocos = []
    for titulo, itens in grupos:
        linhas = ""
        for item in itens:
            r, v, n = item[:3]
            cor = _COR_SINAL.get(item[3] if len(item) > 3 else None, "#111827")
            linhas += ("<div style='display:flex;justify-content:space-between;gap:10px;padding:5px 0;"
                       "border-bottom:1px solid #f1f5f9'>"
                       f"<span style='color:#4b5563;font-size:.84em'>{r}</span>"
                       f"<span style='text-align:right'><b style='font-size:.92em;color:{cor}'>{v}</b>"
                       + (f"<div style='font-size:.72em;color:#6b7280'>{n}</div>" if n else "") + "</span></div>")
        blocos.append("<div style='background:#f8fafc;border:1px solid #e5e7eb;border-radius:10px;padding:8px 12px'>"
                      f"<div style='font-size:.78em;font-weight:700;color:#1e3a8a;text-transform:uppercase;"
                      f"letter-spacing:.04em;margin-bottom:2px'>{titulo}</div>{linhas}</div>")
    saida = ("<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px'>"
             + "".join(blocos) + "</div>")
    if legenda:
        saida += ("<div style='font-size:.72em;color:#4b5563;margin-top:6px'>"
                  f"<b style='color:{_COR_SINAL['bom']}'>azul</b>: leitura favorável · "
                  f"<b style='color:{_COR_SINAL['ruim']}'>vermelho</b>: ponto de atenção · preto: neutro ou sem "
                  "referência. Limiares do Guia de Indicadores, olhando cada número isolado; não é recomendação.</div>")
    return saida


def _caixa_branca(titulo: str, corpo: str) -> str:
    return (f"<div style='background:#fff;border-radius:12px;padding:10px 12px;color:#111827'>"
            f"<div style='font-size:.9em;font-weight:700;color:#1e3a8a;margin-bottom:4px'>{titulo}</div>{corpo}</div>")


def _pp(v: float) -> str:
    return f"{abs(v) * 100:.1f} p.p.".replace(".", ",", 1)


def _claro(grafico):
    """Gráfico em fundo branco (dentro dos cartões da Visão geral)."""
    return (grafico.configure(background="#ffffff")
            .configure_axis(labelColor="#374151", titleColor="#4b5563", gridColor="#eef2f7", domainColor="#d1d5db",
                            tickColor="#d1d5db")
            .configure_legend(labelColor="#111827", titleColor="#111827")
            .configure_view(strokeWidth=0))


def _titulo_cartao(texto: str) -> None:
    st.html(f"<div style='font-size:.95em;font-weight:700;color:#1e3a8a'>{texto}</div>")


_CSS_CARTOES = ("<style>div[class*='st-key-vgc_']{background:#ffffff;border-radius:12px;padding:10px 12px 4px;"
                "color:#111827}</style>")


def _renderizar_visao_geral() -> None:
    dados = st.session_state.get("visao_geral_dados")
    if dados is None:
        with st.spinner("Montando a carteira mês a mês (preços, proventos e câmbio)..."):
            dados = _dados_visao_geral(df_resumo, df_operacoes,
                                       st.session_state.get("carteira_tipos_confirmados", {}))
        st.session_state["visao_geral_dados"] = dados
    if not dados or not dados.get("linhas"):
        st.info("Sem operações com cotação para montar a visão geral.")
        return
    st.html(_CSS_CARTOES)
    linhas, meta, meses = dados["linhas"], dados["meta"], dados["meses"]
    aberto = {t: float(df["quantidade"].iloc[-1]) > 1e-9 for t, df in linhas.items()}

    # filtros: abre nos últimos 12 meses (carteira inteira) ou desde o 1º aporte (com filtro)
    c1, c2, c3, c4, c5, c6 = st.columns([1, 1, 1.1, 1.5, 1.4, 1.6])
    mercados = c3.multiselect("Mercado", ["B3", "EUA"], default=["B3", "EUA"], key="vg_mercado")
    tipos_sel = c4.multiselect("Tipo", ["Ações", "FIIs", "ETFs"], default=["Ações", "FIIs", "ETFs"], key="vg_tipo")
    situacoes = c5.multiselect("Situação", ["Abertas", "Encerradas"], default=["Abertas", "Encerradas"],
                               key="vg_situacao")
    candidatos = sorted(t for t, m in meta.items() if m["mercado"] in mercados and m["tipo"] in tipos_sel
                        and ("Abertas" if aberto[t] else "Encerradas") in situacoes)
    ativos = c6.multiselect("Ativo", candidatos, default=[], placeholder="Todos", key="vg_ativo",
                            format_func=lambda t: t.replace(".SA", ""))
    escolhidos = ativos or candidatos
    if not escolhidos:
        st.info("Nenhum ativo com esses filtros.")
        return
    com_posicao = [i for i in range(len(meses))
                   if any(linhas[t]["aportes"].iloc[i] > 0 or linhas[t]["quantidade"].iloc[i] > 1e-9
                          for t in escolhidos)]
    primeiro = com_posicao[0] if com_posicao else 0
    rotulos = [_rotulo_mes(m) for m in meses][primeiro:]
    filtro_padrao = (set(mercados) == {"B3", "EUA"} and set(tipos_sel) == {"Ações", "FIIs", "ETFs"}
                     and set(situacoes) == {"Abertas", "Encerradas"} and not ativos)
    ini = c1.selectbox("De", rotulos, index=max(0, len(rotulos) - 12) if filtro_padrao else 0,
                       key=f"vg_de_{primeiro}_{filtro_padrao}")
    fim = c2.selectbox("Até", rotulos, index=len(rotulos) - 1, key=f"vg_ate_{primeiro}_{filtro_padrao}")
    i0, i1 = primeiro + rotulos.index(ini), primeiro + rotulos.index(fim)
    if i0 > i1:
        st.warning("O mês inicial é depois do final.")
        return
    so_eua = all(meta[t]["mercado"] == "EUA" for t in escolhidos)
    moeda = "US$" if so_eua else "R$"
    cambio = None if so_eua else {t: dados["cambio"] for t in escolhidos if meta[t]["mercado"] == "EUA"}
    if cambio and dados["cambio"] is None:
        st.caption("⚠️ Cotação do dólar indisponível: ativos dos EUA ficam de fora da soma em reais.")
        escolhidos = [t for t in escolhidos if meta[t]["mercado"] == "B3"]
        cambio = None

    def recorte(tickers):
        tot = somar(linhas, tickers, cambio)
        v0 = float(tot["patrimonio"].iloc[i0 - 1]) if i0 > 0 else 0.0
        return tot, tot.iloc[i0:i1 + 1], v0

    total, periodo, inicial = recorte(escolhidos)
    r = resumo_periodo(periodo, inicial)
    rent = rentabilidade_cotas(periodo, inicial)
    st.caption(f"Período: {ini} a {fim} · valores em {moeda}"
               + (" (EUA convertidos pelo dólar de cada mês)" if cambio else ""))

    cdi, serie_cdi = None, None
    try:
        serie_cdi = _cdi_bcb_em_cache(str(periodo.index[0].start_time.date()), str(periodo.index[-1].end_time.date()))
        cdi = cdi_acumulado(serie_cdi, periodo.index)
    except Exception:
        cdi = None
    fv = lambda v: formatar_valor(v, moeda)
    fs = lambda v: ("+" if v > 0 else ("−" if v < 0 else "")) + fv(abs(v))

    # patrimônio: de onde veio a diferença
    aportes_liq = r["aportes"] - r["vendas"]
    dif = r["patrimonio_final"] - r["patrimonio_inicial"]
    sub_pat = f"{fs(dif)} no período = aportes {fs(aportes_liq)} e preço/câmbio {fs(dif - aportes_liq)}"
    # resultado em dinheiro, com abertas e encerradas
    res = resultado_em_reais(periodo, inicial)
    partes = []
    for nome, grupo in (("abertas", [t for t in escolhidos if aberto[t]]),
                        ("encerradas", [t for t in escolhidos if not aberto[t]])):
        if grupo and len(grupo) < len(escolhidos):
            _, p_g, v_g = recorte(grupo)
            partes.append(f"{nome} {fs(resultado_em_reais(p_g, v_g))}")
    sub_res = "patrimônio − aportes + proventos" + ("<br>" + " · ".join(partes) if partes else "")
    # rentabilidade × CDI e TIR
    sub_rent = "CDI indisponível"
    if cdi is not None and len(cdi):
        c_fim = float(cdi.iloc[-1])
        if r["rentabilidade"] >= c_fim and c_fim > 0:
            sub_rent = f"CDI {_pct_vg(c_fim)} · {r['rentabilidade'] / c_fim * 100:.0f}% do CDI"
        else:
            sub_rent = f"CDI {_pct_vg(c_fim)} · {_pp(c_fim - r['rentabilidade'])} abaixo do CDI"
    tir = tir_anual(periodo, inicial)
    if tir is not None:
        sub_rent += f"<br>TIR (pelo dinheiro): {_pct_vg(tir)} ao ano"
    st.html("<div style='display:flex;gap:8px;flex-wrap:wrap'>"
            + _kpi_vg(f"Patrimônio em {fim}", fv(r["patrimonio_final"]), sub_pat)
            + _kpi_vg("Aportes líquidos", fv(aportes_liq), f"compras {fv(r['aportes'])} − vendas {fv(r['vendas'])}")
            + _kpi_vg("Proventos", fv(r["proventos"]), f"média de {fv(r['proventos_mes'])} por mês", "#0f7f73")
            + _kpi_vg("Resultado no período", fs(res), sub_res, "#15803d" if res >= 0 else "#b91c1c")
            + _kpi_vg("Rentabilidade (cotas)", _pct_vg(r["rentabilidade"]), sub_rent, "#1e3a8a")
            + "</div>")

    col_conc, col_faixas = st.columns([1, 2])
    with col_conc:
        conc = _concentracao_html(escolhidos, linhas, cambio, i1)
        if conc:
            st.html(conc)
    with col_faixas:
        cards_faixa = _faixas(escolhidos)
    _grade_cards(cards_faixa)

    idx = periodo.index.to_timestamp()
    g1, g2 = st.columns(2)
    with g1, st.container(key="vgc_patrimonio"):
        _titulo_cartao("Patrimônio × valor investido")
        df1 = pd.DataFrame({"Patrimônio": periodo["patrimonio"].values, "Valor investido": periodo["custo"].values},
                           index=pd.Index(idx, name="Mês"))
        st.altair_chart(_claro(_grafico_linhas(df1, moeda, ",.0f", {"Patrimônio": "#1e3a8a", "Valor investido": "#9ca3af"},
                                        tracejadas=["Valor investido"])), width="stretch")
    por_tipo_df = {}
    for nome in ("Ações", "FIIs", "ETFs"):
        ts = [t for t in escolhidos if meta[t]["tipo"] == nome]
        if ts:
            por_tipo_df[nome] = somar(linhas, ts, cambio).iloc[i0:i1 + 1]
    with g2, st.container(key="vgc_tipo"):
        _titulo_cartao("Patrimônio por tipo")
        rot2, eixo2 = _eixo_meses(idx)
        df2 = (pd.DataFrame({k: v["patrimonio"].values for k, v in por_tipo_df.items()})
               .assign(Rótulo=rot2).melt("Rótulo", var_name="Tipo", value_name="valor"))
        st.altair_chart(_claro(alt.Chart(df2).mark_area(opacity=.85).encode(
            x=eixo2,
            y=alt.Y("valor:Q", stack=True, title=moeda, axis=alt.Axis(labelExpr=_EIXO_MIL)),
            color=alt.Color("Tipo:N", scale=alt.Scale(domain=list(_CORES_TIPO), range=list(_CORES_TIPO.values())),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=[alt.Tooltip("Rótulo:O", title="Mês"), "Tipo:N", alt.Tooltip("valor:Q", format=",.2f")],
        ).properties(height=280)), width="stretch")

    _destaques(escolhidos, linhas, i0, i1)

    df3 = pd.DataFrame({"Carteira": rent.values}, index=pd.Index(idx, name="Mês"))
    if cdi is not None:
        df3["CDI"] = cdi.values
    with st.container(key="vgc_rentabilidade"):
        _titulo_cartao("Rentabilidade da carteira × CDI")
        st.altair_chart(_claro(_grafico_linhas(df3, "acumulado no período", ".1%",
                                               {"Carteira": "#1e3a8a", "CDI": "#d9831c"}, tracejadas=["CDI"])),
                        width="stretch")
    # tabelas por ano: últimos 5 anos completos + o ano atual, independentes do período escolhido
    # (seguem os filtros de mercado, tipo, situação e ativo)
    historico = total.iloc[primeiro:]
    ano_atual = int(historico.index[-1].year)
    try:
        cdi_hist = _cdi_bcb_em_cache(str(historico.index[0].start_time.date()),
                                     str(historico.index[-1].end_time.date()))
    except Exception:
        cdi_hist = None
    anos = por_ano(historico, 0.0, cdi_hist)
    anos = anos[anos["ano"] >= ano_atual - 5].reset_index(drop=True)
    if len(anos):
        linhas_tab = []
        for _, a in anos.iterrows():
            rotulo = f"{int(a['ano'])}" + (f" ({int(a['meses'])} {'mês' if a['meses'] == 1 else 'meses'})" if a["meses"] < 12 else "")
            if a["cdi"] is None or pd.isna(a["cdi"]):
                cdi_txt = dif_txt = "—"
            else:
                d = a["rentabilidade"] - a["cdi"]
                cdi_txt = _pct_vg(a["cdi"])
                dif_txt = (f"<b style='color:{'#15803d' if d >= 0 else '#b91c1c'}'>"
                           f"{'+' if d >= 0 else '−'}{_pp(d)}</b>")
            linhas_tab.append([rotulo, _pct_vg(a["rentabilidade"]), cdi_txt, dif_txt, fs(a["resultado"])])
        st.html(_caixa_branca("Rentabilidade por ano <span style='font-weight:400;font-size:.8em;color:#6b7280'>"
                              "(últimos 5 anos e o atual; não depende do período escolhido)</span>",
                              _tabela_html(["Ano", "Carteira", "CDI", "Diferença", "Resultado"], linhas_tab[::-1])))
    st.caption("Rentabilidade pelo método das cotas: preço + proventos, sem contar aportes e vendas como ganho "
               "(mede o desempenho dos ativos). TIR: pondera pelo dinheiro investido em cada mês (mede o seu "
               "resultado). Ativos vendidos contam enquanto estavam na carteira; a venda sai como resgate.")

    # proventos: barras por classe, anel por classe e tabela por ano
    rot3, eixo3 = _eixo_meses(idx)
    df4 = (pd.DataFrame({k: v["proventos"].values for k, v in por_tipo_df.items()})
           .assign(Rótulo=rot3).melt("Rótulo", var_name="Tipo", value_name="valor"))
    media = pd.DataFrame({"Rótulo": rot3,
                          "Média de 12 meses": total["proventos"].rolling(12, min_periods=1).mean().iloc[i0:i1 + 1].values})
    with st.container(key="vgc_proventos"):
        _titulo_cartao("Proventos por mês")
        st.altair_chart(_claro(
        (alt.Chart(df4).mark_bar(opacity=.9).encode(
            x=eixo3,
            y=alt.Y("valor:Q", stack=True, title=moeda,
                    axis=alt.Axis(labelExpr="replace(format(datum.value, ',.0f'), ',', '.')")),
            color=alt.Color("Tipo:N", scale=alt.Scale(domain=list(_CORES_TIPO), range=list(_CORES_TIPO.values())),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=[alt.Tooltip("Rótulo:O", title="Mês"), "Tipo:N", alt.Tooltip("valor:Q", format=",.2f")])
         + alt.Chart(media).mark_line(color="#d9831c", strokeWidth=2).encode(x=eixo3, y="Média de 12 meses:Q")
         ).properties(height=240)), width="stretch")
    st.caption("Proventos recebidos no mês (dividendos e JCP brutos, rendimentos), pela quantidade na data-ex. "
               "Linha laranja: média dos últimos 12 meses.")
    p1, p2 = st.columns(2)
    soma_tipo = {k: float(v["proventos"].sum()) for k, v in por_tipo_df.items() if float(v["proventos"].sum()) > 0}
    with p1:
        if soma_tipo:
            dfp = pd.DataFrame({"Tipo": list(soma_tipo), "valor": list(soma_tipo.values())})
            tot_p = sum(soma_tipo.values())
            with st.container(key="vgc_prov_tipo"):
                _titulo_cartao("Proventos por tipo de ativo")
                st.altair_chart(_claro(alt.Chart(dfp).mark_arc(innerRadius=55).encode(
                theta="valor:Q",
                color=alt.Color("Tipo:N", scale=alt.Scale(domain=list(_CORES_TIPO), range=list(_CORES_TIPO.values())),
                                legend=alt.Legend(orient="right", title=None)),
                tooltip=["Tipo:N", alt.Tooltip("valor:Q", format=",.2f")]).properties(height=200)), width="stretch")
                st.html("<div style='font-size:.84em;color:#374151'>" + " · ".join(
                    f"<b>{k}</b> {fv(v)} ({v / tot_p * 100:.0f}%)" for k, v in soma_tipo.items()) + "</div>")
    with p2:
        if len(anos):
            linhas_p, anterior = [], None
            for _, a in anos.iterrows():
                var = "—"
                if anterior is not None and a["meses"] == 12 and anterior["meses"] == 12 and anterior["proventos"] > 0:
                    var = _pct_vg(a["proventos"] / anterior["proventos"] - 1)
                rotulo = f"{int(a['ano'])}" + (f" ({int(a['meses'])} {'mês' if a['meses'] == 1 else 'meses'})" if a["meses"] < 12 else "")
                linhas_p.append([rotulo, fv(a["proventos"]), fv(a["proventos_mes"]), var])
                anterior = a
            st.html(_caixa_branca("Proventos por ano <span style='font-weight:400;font-size:.8em;color:#6b7280'>"
                                  "(últimos 5 anos e o atual)</span>",
                                  _tabela_html(["Ano", "Total", "Média por mês", "Variação"], linhas_p[::-1])))

    if dados.get("sem_preco"):
        st.caption("Sem cotação, fora da conta: " + ", ".join(t.replace(".SA", "") for t in dados["sem_preco"]) + ".")

@st.cache_data(ttl=12 * 3600, show_spinner=False)
def _proventos_detalhe_em_cache(ticker: str):
    """Tipo (dividendo/JCP) e data de pagamento dos proventos (Fundamentus),
    compartilhado entre usuários por 12h (dado público)."""
    return buscar_proventos_acao_detalhe(ticker)


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def _proventos_fii_detalhe_em_cache(ticker: str):
    """Tipo (rendimento/amortização) e data de pagamento dos proventos de um
    FII (Fundamentus), compartilhado entre usuários por 12h."""
    return buscar_proventos_fii_detalhe(ticker)


def _renderizar_proventos_ir(dados: dict) -> None:
    """Dividendos e JCP (ações) e rendimentos (FIIs) da B3 por ano de
    pagamento, para conferir a declaração do IR."""
    st.caption("Pelo ano do PAGAMENTO, que é o que vale na declaração. Ações: dividendos isentos; JCP com 15% "
               "retidos na fonte (declara-se o líquido, tributação exclusiva). FIIs: rendimentos isentos para pessoa "
               "física nas condições da lei; amortização é devolução de capital (reduz o custo, não é rendimento). "
               "Quantidade da sua planilha na data-com × valor por cota informado (Fundamentus). Para conferir: o "
               "documento oficial é o informe de rendimentos da corretora ou do emissor.")
    b3 = {t: m for t, m in dados["meta"].items() if m["mercado"] == "B3"}
    acoes = sorted(t for t, m in b3.items() if m["tipo"] == "Ações" and not eh_bdr(t))
    fiis = sorted(t for t, m in b3.items() if m["tipo"] == "FIIs")
    bdrs = sorted(t for t in b3 if eh_bdr(t))
    if not acoes and not fiis:
        st.info("Sem ações ou FIIs da B3 na planilha.")
        return
    if "ir_recebidos" not in st.session_state:
        if not st.button("Montar a tabela de proventos para o IR", key="ir_montar"):
            return
        partes, falhas = [], []
        with st.spinner("Buscando tipo e data de pagamento dos proventos no Fundamentus..."):
            for t in acoes + fiis:
                detalhe = (_proventos_detalhe_em_cache(t) if t in acoes else _proventos_fii_detalhe_em_cache(t))
                if detalhe is None:
                    falhas.append(t.replace(".SA", ""))
                    continue
                partes.append(proventos_recebidos(t, dados["ops"].get(t, []), detalhe))
        st.session_state["ir_recebidos"] = (pd.concat(partes, ignore_index=True) if partes else pd.DataFrame(),
                                            falhas)
    recebidos, falhas = st.session_state["ir_recebidos"]
    if recebidos is None or recebidos.empty:
        st.info("Nenhum provento encontrado para os ativos da planilha.")
        return
    anos = sorted({int(a) for a in pd.to_datetime(recebidos["pagamento"]).dropna().dt.year}, reverse=True)
    hoje = datetime.now(ZoneInfo("America/Sao_Paulo")).date()
    ano = st.selectbox("Ano do pagamento (ano-base da declaração)", anos,
                       index=anos.index(hoje.year - 1) if hoje.year - 1 in anos else 0, key="ir_ano")
    tab = resumo_por_ano(recebidos, ano)
    fv = lambda v: formatar_valor(v, "R$") if v else "—"
    nome = lambda t: t.replace(".SA", "")

    def tabela(ativos, colunas, rotulos, titulo):
        parte = tab[tab["ticker"].isin(ativos)]
        if parte.empty:
            return
        linhas_tab = [[nome(r["ticker"])] + [fv(r[c]) for c in colunas] for _, r in parte.iterrows()]
        linhas_tab.append(["<b>Total</b>"] + [f"<b>{fv(parte[c].sum())}</b>" for c in colunas])
        st.html(_caixa_branca(titulo, _tabela_html(["Ativo"] + rotulos, linhas_tab)))

    if tab.empty:
        st.info(f"Nenhum pagamento em {ano}.")
    else:
        tabela(acoes, ["dividendos", "jcp_bruto", "ir_retido", "jcp_liquido"],
               ["Dividendos (isentos)", "JCP bruto", "IR retido (15%)", "JCP líquido"], f"Ações — pagos em {ano}")
        tabela(fiis, ["rendimentos", "amortizacao"], ["Rendimentos (isentos)", "Amortização (devolução de capital)"],
               f"FIIs — pagos em {ano}")
        planilha_ir = io.BytesIO()
        renomear = {"ticker": "Ativo", "dividendos": "Dividendos (isentos)", "jcp_bruto": "JCP bruto",
                    "ir_retido": "IR retido (15%)", "jcp_liquido": "JCP líquido",
                    "rendimentos": "Rendimentos FII (isentos)", "amortizacao": "Amortização FII",
                    "total_liquido": "Total líquido"}
        with pd.ExcelWriter(planilha_ir, engine="openpyxl") as w:
            for rotulo, ativos, cols in (("Ações", acoes, ["ticker", "dividendos", "jcp_bruto", "ir_retido",
                                                            "jcp_liquido"]),
                                         ("FIIs", fiis, ["ticker", "rendimentos", "amortizacao"])):
                parte = tab[tab["ticker"].isin(ativos)][cols]
                if len(parte):
                    parte.assign(ticker=parte["ticker"].map(nome)).rename(columns=renomear).to_excel(
                        w, sheet_name=f"{rotulo} {ano}", index=False)
            det = recebidos[pd.to_datetime(recebidos["pagamento"]).dt.year == ano].copy()
            det["ticker"] = det["ticker"].map(nome)
            det.rename(columns={"ticker": "Ativo", "data_com": "Data-com", "pagamento": "Pagamento", "tipo": "Tipo",
                                "quantidade": "Quantidade", "bruto": "Bruto", "ir_retido": "IR retido",
                                "liquido": "Líquido"}).to_excel(w, sheet_name="Detalhe", index=False)
        st.download_button(f"⬇️ Baixar {ano} em Excel", planilha_ir.getvalue(), file_name=f"proventos_ir_{ano}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    pend = a_receber(recebidos, hoje)
    if pend is not None and len(pend):
        tipo_txt = {"jcp": "JCP", "dividendo": "Dividendo", "rendimento": "Rendimento", "amortizacao": "Amortização",
                    "outro": "Outro"}
        linhas_p = [[nome(r.ticker), tipo_txt.get(r.tipo, r.tipo), f"{pd.Timestamp(r.data_com):%d/%m/%Y}",
                     f"{pd.Timestamp(r.pagamento):%d/%m/%Y}" if pd.notna(r.pagamento) else "sem data",
                     formatar_valor(r.bruto, "R$")] for r in pend.itertuples()]
        with st.expander(f"A receber ({len(pend)}): data-com passada, pagamento futuro ou ainda sem data"):
            st.html(_tabela_html(["Ativo", "Tipo", "Data-com", "Pagamento", "Valor bruto"], linhas_p))
    antigos = sem_data_antigos(recebidos, hoje)
    if antigos is not None and len(antigos):
        st.caption(f"{len(antigos)} provento(s) antigo(s) sem data de pagamento no Fundamentus não entram em nenhum "
                   "ano (" + ", ".join(sorted({nome(t) for t in antigos["ticker"]})) + "): confira no informe.")
    if bdrs:
        st.caption("BDRs (" + ", ".join(nome(t) for t in bdrs) + ") ficam de fora: seguem a regra de rendimentos "
                   "do exterior.")
    if falhas:
        st.caption("Sem dados do Fundamentus agora: " + ", ".join(falhas) + ". Tente de novo mais tarde.")


with tab_visao:
    st.subheader("🧭 Visão geral da carteira")
    if bytes_planilha is None:
        st.info("Envie um arquivo Excel na aba 'Carteira' para ver a visão geral.")
    elif "carteira_linhas_ganho" not in st.session_state:
        st.info("Clique em 'Analisar carteira' na aba 'Carteira' para montar a visão geral.")
    else:
        try:
            _renderizar_visao_geral()
        except Exception as e:
            st.error(f"Não foi possível montar a visão geral ({e}).")



# =============================================================================
# ABA — Metas de alocação
# =============================================================================
def _valores_por_ativo(dados: dict) -> dict[str, float]:
    """Patrimônio de hoje (último mês) de cada ativo, em reais (EUA pelo dólar)."""
    valores = {}
    cambio = dados.get("cambio")
    for t, df in dados["linhas"].items():
        v = float(df["patrimonio"].iloc[-1])
        if v <= 0:
            continue
        if dados["meta"][t]["mercado"] == "EUA":
            if cambio is None or not len(cambio):
                continue
            v *= float(cambio.reindex(df.index).ffill().bfill().iloc[-1])
        valores[t] = v
    return valores


def _valores_por_classe(dados: dict) -> dict[str, float]:
    """Patrimônio de hoje (último mês) por classe de meta, em reais."""
    valores = {c: 0.0 for c in CLASSES_META}
    for t, v in _valores_por_ativo(dados).items():
        m = dados["meta"][t]
        valores[classe_do_ativo(m["mercado"], m["tipo"])] += v
    return valores


def _limites_salvos() -> dict[str, float]:
    """Limites dos campos da aba Metas (se já mexidos nesta sessão) ou do endereço da página."""
    mx, mn = st.session_state.get("limite_max"), st.session_state.get("limite_min")
    if mx is not None and mn is not None and mn < mx:
        return {"maximo": float(mx), "minimo": float(mn)}
    return limites_de_texto(st.query_params.get("limites")) or dict(LIMITES_PADRAO)


def _pesos_atuais() -> dict[str, dict]:
    """ticker -> {peso, situacao, diferenca, valor}, com os limites salvos.
    Vazio se a carteira de hoje ainda não foi montada."""
    dados = st.session_state.get("visao_geral_dados")
    if not dados:
        return {}
    try:
        return {l["ticker"]: l for l in pesos_dos_ativos(_valores_por_ativo(dados), _limites_salvos())}
    except Exception:
        return {}


def _renderizar_limites(dados: dict) -> None:
    st.markdown("#### ⚖️ Limite por ativo")
    st.caption("Peso máximo de cada ativo na carteira e, se quiser, o mínimo para a posição valer a pena. O painel "
               "aponta quem passou do limite; não sugere compra nem venda. Fica guardado no endereço da página.")
    salvos = _limites_salvos()
    c1, c2, _ = st.columns([1, 1, 2])
    maximo = c1.number_input("Peso máximo por ativo (%)", min_value=1.0, max_value=100.0, value=float(salvos["maximo"]),
                             step=1.0, format="%.0f", key="limite_max")
    minimo = c2.number_input("Peso mínimo (%) · 0 = sem mínimo", min_value=0.0, max_value=50.0,
                             value=float(salvos["minimo"]), step=0.5, format="%.1f", key="limite_min")
    if minimo >= maximo:
        st.warning("O mínimo precisa ser menor que o máximo.")
        return
    limites = {"maximo": maximo, "minimo": minimo}
    st.query_params["limites"] = limites_para_texto(limites)
    linhas = pesos_dos_ativos(_valores_por_ativo(dados), limites)
    if not linhas:
        return
    fv = lambda v: formatar_valor(v, "R$")
    pct = lambda v: f"{v:.1f}%".replace(".", ",")
    acima = [l for l in linhas if l["situacao"] == "acima"]
    abaixo = [l for l in linhas if l["situacao"] == "abaixo"]
    maior = linhas[0]
    plural = lambda n: f"{n} ativo{'s' if n != 1 else ''}"
    st.html("<div style='display:flex;gap:8px;flex-wrap:wrap'>"
            + _kpi_vg("Acima do limite", plural(len(acima)), f"peso maior que {maximo:g}%")
            + _kpi_vg("Abaixo do mínimo", plural(len(abaixo)) if minimo else "—",
                      f"peso menor que {pct(minimo)}" if minimo else "sem mínimo definido")
            + _kpi_vg("Maior peso", f"{maior['ticker'].replace('.SA', '')} · {pct(maior['peso'])}", fv(maior["valor"]))
            + "</div>")
    escala = max(1.5, max(l["peso"] for l in linhas) / maximo)  # barra: o traço marca o limite
    largura = 160
    cores = {"acima": "#b91c1c", "abaixo": "#d9831c", "dentro": "#0f7f73"}
    tab = []
    for l in linhas:
        cor = cores[l["situacao"]]
        sit = {"acima": f"acima do limite · {fv(l['diferenca'])} além dos {maximo:g}%",
               "abaixo": f"abaixo do mínimo · faltam {fv(l['diferenca'])}",
               "dentro": "dentro dos limites"}[l["situacao"]]
        barra = (f"<div style='position:relative;background:#eef2f7;border-radius:6px;height:8px;width:{largura}px'>"
                 f"<div style='width:{min(l['peso'] / maximo / escala, 1) * largura:.0f}px;height:8px;border-radius:6px;"
                 f"background:{cor}'></div><div style='position:absolute;left:{largura / escala:.0f}px;top:-3px;"
                 f"width:2px;height:14px;background:#111827'></div></div>")
        m = dados["meta"][l["ticker"]]
        tab.append([f"<b>{html.escape(l['ticker'].replace('.SA', ''))}</b>", classe_do_ativo(m["mercado"], m["tipo"]),
                    pct(l["peso"]), fv(l["valor"]), barra, f"<span style='color:{cor};font-weight:600'>{sit}</span>"])
    st.html(_caixa_branca("Peso de cada ativo (posições abertas)",
                          _tabela_html(["Ativo", "Classe", "Peso", "Valor", f"Peso × limite (traço = {maximo:g}%)",
                                        "Situação"], tab)))
    if acima:
        st.caption("Ao simular um aporte acima, lembre que " + ", ".join(l["ticker"].replace(".SA", "") for l in acima)
                   + (" já está" if len(acima) == 1 else " já estão") + " acima do seu limite.")


def _renderizar_metas(dados: dict) -> None:
    st.caption("Escolha o percentual de cada classe. O painel compara com a carteira de hoje e mostra quanto falta "
               "ou sobra em cada uma. Não sugere ativos nem compra ou venda: a escolha continua sua. As metas ficam "
               "guardadas no endereço desta página (nada é gravado no servidor): ao recarregar, continuam; salve o "
               "link nos favoritos para usar depois.")
    salvas = metas_de_texto(st.query_params.get("metas")) or METAS_PADRAO
    colunas = st.columns(len(CLASSES_META))
    metas = {c: col.number_input(f"{c} (%)", min_value=0.0, max_value=100.0, value=float(salvas[c]), step=1.0,
                                 format="%.0f", key=f"meta_{i}")
             for i, (c, col) in enumerate(zip(CLASSES_META, colunas))}
    soma = sum(metas.values())
    if abs(soma - 100) > 0.01:
        st.warning(f"As metas somam {soma:.0f}%: ajuste para 100%.")
        return
    st.query_params["metas"] = metas_para_texto(metas)
    valores = _valores_por_classe(dados)
    linhas = comparar(valores, metas)
    total = sum(valores.values())
    fv = lambda v: formatar_valor(v, "R$")
    exterior = valores["EUA (ações e ETFs)"] / total * 100 if total else 0
    meta_ext = metas["EUA (ações e ETFs)"]
    dif_br = total * (100 - meta_ext) / 100 - (total - valores["EUA (ações e ETFs)"])
    fora = [l for l in linhas if not l["na_meta"]]
    st.html("<div style='display:flex;gap:8px;flex-wrap:wrap'>"
            + _kpi_vg("Patrimônio", fv(total), "posições abertas, EUA pelo dólar de hoje")
            + _kpi_vg("Brasil × Exterior", f"{100 - exterior:.0f}% · {exterior:.0f}%",
                      f"meta {100 - meta_ext:.0f}% · {meta_ext:.0f}% → "
                      + (f"{'+' if dif_br > 0 else '−'}{fv(abs(dif_br))} no Brasil" if abs(dif_br) >= 1 else "na meta"))
            + _kpi_vg("Fora da meta", f"{len(fora)} classe{'s' if len(fora) != 1 else ''}",
                      f"diferença maior que {TOLERANCIA_PP:.0f} p.p.")
            + "</div>")
    aporte = st.number_input("Simular um aporte (R$): como dividi-lo entre as classes abaixo da meta", min_value=0,
                             value=0, step=1000, key="meta_aporte")
    divisao = dividir_aporte(linhas, aporte) if aporte > 0 else {}
    limites = _limites_salvos()
    por_classe = {c: [] for c in CLASSES_META}
    for t, v in _valores_por_ativo(dados).items():
        m = dados["meta"][t]
        por_classe[classe_do_ativo(m["mercado"], m["tipo"])].append(v)
    cap, avisos = metas_e_limites(linhas, por_classe, limites)
    linhas_tab = []
    for l in linhas:
        at = l["atingido"]
        cor_barra = "#0f7f73" if l["na_meta"] else "#d9831c"
        barra = ("—" if at is None else
                 f"<div style='display:flex;align-items:center;gap:6px;justify-content:flex-start'>"
                 f"<div style='background:#eef2f7;border-radius:6px;height:8px;width:80px'>"
                 f"<div style='width:{min(at, 100) * 0.8:.0f}px;height:8px;border-radius:6px;background:{cor_barra}'>"
                 f"</div></div><span>{at:.0f}%</span></div>")
        falta = l["falta"]
        falta_txt = (f"<b style='color:{'#15803d' if falta > 0 else '#b91c1c'}'>"
                     f"{'+' if falta > 0 else '−'}{fv(abs(falta))}</b>") if abs(falta) >= 1 else "—"
        k = cap[l["classe"]]
        ativos_txt = (f"{k['n']}" + (f" · <span style='color:#b91c1c'>{k['acima']} acima</span>" if k["acima"] else "")
                      if k["n"] else "—")
        leitura = {"na_meta": f"<span style='color:#0f7f73'>na meta (±{TOLERANCIA_PP:g} p.p.)</span>",
                   "acima_da_meta": "<span style='color:#4b5563'>classe acima da meta</span>",
                   "sem_ativos": "<span style='color:#d9831c;font-weight:600'>sem ativos na classe</span>",
                   "cabe": "<span style='color:#0f7f73;font-weight:600'>cabe nos ativos atuais</span>",
                   "nao_cabe": f"<span style='color:#d9831c;font-weight:600'>não cabe: faltam {fv(k['falta_espaco'])} "
                               f"de espaço</span>"}[k["leitura"]]
        nw = lambda t: f"<span style='white-space:nowrap'>{t}</span>"
        linha = [nw(f"<b>{l['classe']}</b>"), f"{l['meta']:.0f}%", f"{l['hoje']:.1f}%".replace(".", ","), nw(fv(l["tem"])),
                 nw(falta_txt), barra, nw(ativos_txt), nw(fv(k["espaco"])) if k["n"] else "—", leitura]
        if divisao:
            linha.append(fv(divisao[l["classe"]]) if divisao[l["classe"]] >= 0.01 else "—")
        linhas_tab.append(linha)
    cab = ["Classe", "Meta", "Hoje", "Tem", "Para chegar à meta", "Atingido", "Ativos",
           f"Espaço até o limite ({limites['maximo']:g}%)", "Leitura"]
    if divisao:
        cab.append(f"Do aporte de {fv(aporte)}")
    st.html(_caixa_branca("Alocação por classe", _tabela_html(cab, linhas_tab)))
    if avisos:
        st.html("<div style='background:#fff7ed;border:1px solid #fed7aa;border-radius:10px;padding:8px 12px;"
                "color:#7c2d12;font-size:.86em'><b>⚠️ Metas e limites não fecham</b><ul style='margin:4px 0 0 18px;"
                "padding:0'>" + "".join(f"<li>{html.escape(a)}</li>" for a in avisos) + "</ul></div>")
    st.caption("Para chegar à meta: positivo = quanto aportar; negativo = quanto a classe está acima da meta. "
               "O aporte simulado vai só para as classes abaixo da meta, na proporção do que falta a cada uma. "
               "Espaço até o limite: quanto ainda cabe nos ativos que você já tem sem nenhum passar do peso máximo "
               "(limite por ativo, logo abaixo); se for menor do que falta, a meta só fecha aumentando o limite, "
               "revendo a meta ou com mais ativos na classe.")


with tab_metas:
    st.subheader("🎯 Metas de alocação")
    if bytes_planilha is None:
        st.info("Envie um arquivo Excel na aba 'Carteira' para usar as metas.")
    elif "carteira_linhas_ganho" not in st.session_state:
        st.info("Clique em 'Analisar carteira' na aba 'Carteira' para comparar com as metas.")
    else:
        try:
            dados_m = st.session_state.get("visao_geral_dados")
            if dados_m is None:
                with st.spinner("Montando a carteira de hoje..."):
                    dados_m = _dados_visao_geral(df_resumo, df_operacoes,
                                                 st.session_state.get("carteira_tipos_confirmados", {}))
                st.session_state["visao_geral_dados"] = dados_m
            if dados_m:
                _renderizar_metas(dados_m)
                _renderizar_limites(dados_m)
        except Exception as e:
            st.error(f"Não foi possível montar as metas ({e}).")


# =============================================================================
# ABA 3 — Raio-X do ativo
# =============================================================================
@st.cache_data(ttl=12 * 3600, show_spinner=False)
def _todos_fiis_em_cache():
    from carteira_analise.fontes.fundamentus import buscar_todos_fiis_fundamentus
    return buscar_todos_fiis_fundamentus()


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def _pares_acao_cache(tipo: str, ident: str):
    t = buscar_tabela_pares(tipo, ident)
    if t is None:
        raise RuntimeError("tabela de pares não baixada")  # falha não fica no cache
    return t


def _pares_acao_em_cache(tipo: str, ident: str):
    try:
        return _pares_acao_cache(tipo, ident)
    except Exception:
        return None


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _grupos_setoriais_cache(ticker: str):
    g = buscar_grupos_setoriais(ticker)
    if not g:
        raise RuntimeError("segmento não encontrado")
    return g


def _grupos_setoriais_em_cache(ticker: str) -> list[dict]:
    try:
        return _grupos_setoriais_cache(ticker)
    except Exception:
        return []


def _kv(rotulo: str, valor: str, sub: str = "", cor: str = "#111827") -> str:
    return (f"<div style='flex:1;min-width:0;background:#f8fafc;border:1px solid #eef2f7;border-radius:10px;"
            f"padding:6px 9px'><div style='font-size:.72em;color:#4b5563'>{rotulo}</div>"
            f"<div style='font-weight:700;font-size:.98em;color:{cor}'>{valor}</div>"
            + (f"<div style='font-size:.7em;color:#6b7280'>{sub}</div>" if sub else "") + "</div>")


def _linha_kv(*caixas) -> str:
    return "<div style='display:flex;gap:6px;margin-top:6px;flex-wrap:wrap'>" + "".join(c for c in caixas if c) + "</div>"


def _pct_txt(v, casas=1, sinal=False) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    s = f"{abs(v) * 100:.{casas}f}%".replace(".", ",")
    return (("+" if v > 0 else "−" if v < 0 else "") + s) if sinal else ("−" + s if v < 0 else s)


def _num_txt(v, casas=1) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return f"{v:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _mi_txt(v, moeda="R$") -> str:
    """Valor grande abreviado: R$ 200 mi, R$ 1,2 bi."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    if abs(v) >= 1e9:
        return f"{moeda} {_num_txt(v / 1e9, 1)} bi"
    if abs(v) >= 1e6:
        return f"{moeda} {_num_txt(v / 1e6, 0 if abs(v) >= 1e8 else 1)} mi"
    return f"{moeda} {_num_txt(v / 1e3, 0)} mil"


def _ops_do_ticker(ticker: str) -> list[dict]:
    """Operações da planilha (escala da época) para marcar no gráfico."""
    if df_resumo is None and df_operacoes is None:
        return []
    resumo = linhas_resumo_por_ticker(df_resumo)
    grupo = None
    if df_operacoes is not None and not df_operacoes.empty:
        chaves = df_operacoes["ticker"].apply(lambda t: normalizar_ticker(t)[0])
        grupo = df_operacoes[chaves == ticker]
    ops, _, _ = operacoes_do_ticker(resumo.get(ticker), grupo if grupo is not None and len(grupo) else None)
    return ops


_COR_MAXIMA, _COR_MINIMA = "#1d4ed8", "#b91c1c"


def _grafico_raio_x(historico: pd.Series, moeda: str, pm=None, faixas=None, teto=None, ops=None, f12=None,
                    tendencia=None):
    s = historico.dropna()
    idx = pd.DatetimeIndex(s.index)
    s.index = idx.tz_localize(None) if idx.tz is not None else idx
    df = pd.DataFrame({"Data": s.index, "Preço": s.values, "Média de 200 dias": s.rolling(200).mean().values})
    dados = df.melt("Data", var_name="Série", value_name="valor").dropna()
    linhas = alt.Chart(dados).mark_line().encode(
        x=alt.X("Data:T", title=None, axis=alt.Axis(format="%m/%Y")),
        y=alt.Y("valor:Q", title=moeda, scale=alt.Scale(zero=False)),
        color=alt.Color("Série:N", scale=alt.Scale(domain=["Preço", "Média de 200 dias"],
                                                    range=["#1e3a8a", "#9ca3af"]),
                        legend=alt.Legend(orient="top", title=None)),
        strokeDash=alt.condition(alt.datum["Série"] == "Média de 200 dias", alt.value([6, 4]), alt.value([1, 0])),
        tooltip=[alt.Tooltip("Data:T", format="%d/%m/%Y"), "Série:N", alt.Tooltip("valor:Q", format=",.2f")])
    camadas = [linhas]
    refs = []
    if pm:
        refs.append(("Seu preço médio", pm, "#111827", [2, 2]))
    if faixas:
        refs.append(("Faixa de aumento", faixas["queda"], "#d9831c", [1, 0]))
        refs.append(("Faixa de realização", faixas["realizacao"], "#2f45e0", [1, 0]))
    if teto:
        refs.append(("Preço-teto", teto, "#15803d", [8, 3, 2, 3]))
    for nome, valor, cor, traco in refs:
        camadas.append(alt.Chart(pd.DataFrame({"y": [valor], "nome": [nome]})).mark_rule(
            color=cor, strokeDash=traco, strokeWidth=1.4).encode(
            y="y:Q", tooltip=[alt.Tooltip("nome:N", title=""), alt.Tooltip("y:Q", format=",.2f", title=moeda)]))
        camadas.append(alt.Chart(pd.DataFrame({"y": [valor], "t": [nome], "x": [s.index[0]]})).mark_text(
            align="left", dy=-6, fontSize=10, color=cor).encode(x="x:T", y="y:Q", text="t:N"))
    if ops:
        inicio = s.index[0]
        marcas = pd.DataFrame([{"Data": pd.Timestamp(o["data"]), "preço": float(o["preco"]),
                                "op": "Compra" if o["tipo"] == "compra" else "Venda",
                                "qtd": float(o["quantidade"])}
                               for o in ops if o["tipo"] in ("compra", "venda") and o.get("preco")
                               and pd.Timestamp(o["data"]) >= inicio])
        if len(marcas):
            camadas.append(alt.Chart(marcas).mark_point(filled=True, size=90).encode(
                x="Data:T", y="preço:Q",
                shape=alt.Shape("op:N", scale=alt.Scale(domain=["Compra", "Venda"],
                                                        range=["triangle-up", "triangle-down"]),
                                legend=alt.Legend(orient="top", title=None)),
                color=alt.Color("op:N", scale=alt.Scale(domain=["Compra", "Venda"], range=["#0f7f73", "#b91c1c"]),
                                legend=None),
                tooltip=[alt.Tooltip("Data:T", format="%d/%m/%Y"), "op:N", alt.Tooltip("qtd:Q", title="quantidade"),
                         alt.Tooltip("preço:Q", format=",.2f")]))
    if tendencia:  # linha vertical no dia em que a tendência rompeu (o mesmo do aviso acima do gráfico)
        marcos = []
        if tendencia.get("abaixo_desde") is not None and tendencia.get("pregoes_abaixo", 0) >= 20:
            marcos.append((pd.Timestamp(tendencia["abaixo_desde"]), "Abaixo da média de 200 dias"))
        if tendencia.get("cruzamento") is not None:
            marcos.append((pd.Timestamp(tendencia["cruzamento"]), "Média de 50 cruzou a de 200"))
        marcos = [(d.tz_localize(None) if d.tz is not None else d, t) for d, t in marcos]
        marcos = [(d, t) for d, t in marcos if d >= s.index[0]]
        if marcos:
            df_m = pd.DataFrame({"Data": [d for d, _ in marcos], "t": [t for _, t in marcos],
                                 "y": [float(s.max())] * len(marcos)})
            camadas.append(alt.Chart(df_m).mark_rule(color="#64748b", strokeDash=[4, 3], strokeWidth=1.4).encode(
                x="Data:T", tooltip=[alt.Tooltip("t:N", title="Tendência rompida"),
                                     alt.Tooltip("Data:T", format="%d/%m/%Y")]))
            camadas.append(alt.Chart(df_m).mark_text(align="left", dx=4, dy=-4, fontSize=10, color="#475569").encode(
                x="Data:T", y="y:Q", text=alt.value("📉 tendência rompida")))
    if f12:  # máxima (azul) e mínima (vermelha) dos 12 meses, as mesmas dos boxes abaixo
        pontos = []
        for nome, chave_d, chave_v, cor in (("Máxima 12 meses", "data_maxima", "maxima", _COR_MAXIMA),
                                            ("Mínima 12 meses", "data_minima", "minima", _COR_MINIMA)):
            d = f12.get(chave_d)
            if d is None or pd.isna(d):
                continue
            d = pd.Timestamp(d).tz_localize(None) if pd.Timestamp(d).tz is not None else pd.Timestamp(d)
            if d < s.index[0]:
                continue
            perto = s[s.index <= d]
            y = float(perto.iloc[-1]) if len(perto) else float(f12[chave_v])
            pontos.append({"Data": d, "y": y, "valor": float(f12[chave_v]), "nome": nome, "cor": cor})
        if pontos:
            camadas.append(alt.Chart(pd.DataFrame(pontos)).mark_circle(size=130, opacity=1, stroke="#ffffff",
                                                                       strokeWidth=2).encode(
                x="Data:T", y="y:Q", color=alt.Color("cor:N", scale=None),
                tooltip=[alt.Tooltip("nome:N", title=""), alt.Tooltip("Data:T", format="%d/%m/%Y"),
                         alt.Tooltip("valor:Q", format=",.2f", title=moeda)]))
    return alt.layer(*camadas).properties(height=320).resolve_scale(color="independent", shape="independent")


def _renderizar_raio_x(ticker: str, tipo: str, resultado, ctx: dict | None, dividendos, f12) -> None:
    moeda = "R$" if ticker.endswith(".SA") else "US$"
    fv = lambda v: formatar_valor(v, moeda) if v is not None and not pd.isna(v) else "—"
    papel = ticker.replace(".SA", "")
    hoje = datetime.now(ZoneInfo("America/Sao_Paulo")).date()
    preco = float(resultado.tec.preco_atual)
    historico = resultado.tec.historico
    ctx = ctx or {}
    fund = fundamentos_de_resultado(resultado)
    b3 = ticker.endswith(".SA")
    eh_fii = tipo == "fii"

    cenarios_sessao = st.session_state.get("carteira_cenarios")
    card = next((c for c in (cenarios_sessao[0] if cenarios_sessao else []) if c["ticker"] == ticker), None)

    # dados extras (Fundamentus), todos com cache e sem derrubar a aba se falharem
    fa = ref = pares = detalhe = None
    grupo_pares = {}
    try:
        if b3 and tipo == "acao":
            fa = buscar_fundamentos_acao(ticker)
            ref = _referencias_pl_em_cache(ticker)
            # pares: segmento primeiro, setor se o segmento não tiver com quem comparar
            for g in _grupos_setoriais_em_cache(ticker):
                pares = selecionar_pares(_pares_acao_em_cache(g["tipo"], str(g["id"])), ticker)
                if pares is not None:
                    grupo_pares = g
                    break
            detalhe = _proventos_detalhe_em_cache(ticker)
        elif eh_fii:
            pares = pares_fii(_todos_fiis_em_cache(), ticker)
            detalhe = _proventos_fii_detalhe_em_cache(ticker)
    except Exception:
        pass
    if fa is not None:
        fund.setdefault("ev_ebitda", fa.ev_ebitda)
        fund["roic"] = fund.get("roic") if fund.get("roic") is not None else fa.roic
    try:
        info = fonte_dados.baixar_info(ticker) or {}
    except Exception:
        info = {}
    nome = info.get("longName") or info.get("shortName") or ""
    nomes = (ref or {}).get("nomes") or {}
    classe = ("FII" if eh_fii else "ETF" if tipo in ("etf_br", "etf_us") else "Ação") + (" · B3" if b3 else " · EUA")
    setor = " › ".join(n for n in (nomes.get("setor"), nomes.get("segmento")) if n) or (fund.get("segmento") or "")

    # cabeçalho: blocos Preço / Rentabilidade e saúde (ou O fundo) / Proventos
    pseudo = card or {"ticker": ticker, "mercado": "B3" if b3 else "EUA",
                      "grupo": {"acao": "Ações", "fii": "FIIs"}.get(tipo, "ETF"), "situacao": None,
                      "preco_atual": preco, "variacao": None}
    pseudo = {**pseudo, "contexto": ctx}
    tabelas = montar_tabelas([pseudo], {ticker: fund}, {ticker: dividendos}, hoje)
    pt = ctx.get("preco_teto") or {}
    df_l = next(iter(tabelas.values())) if tabelas else None

    def col(c):
        if df_l is None or c not in df_l.columns:
            return None
        v = df_l.iloc[0][c]
        return None if v is None or (isinstance(v, float) and pd.isna(v)) else v

    def txt(v):
        return "—" if v is None else str(v)

    nota_teto = ""
    if pt.get("teto"):
        nota_teto = f"preço {_pct_txt(abs(preco / pt['teto'] - 1), 0)} {'abaixo' if preco <= pt['teto'] else 'acima'}"
    dy_nivel = next((i.get("nivel") for i in ctx.get("indicadores") or []
                     if str(i.get("nome", "")).startswith(("Dividend", "Yield"))), None)
    sinal_dy = {"bom": "bom", "atencao": "ruim"}.get(dy_nivel)
    sinal_reg = {"Regular": "bom", "Irregular": "ruim"}.get(str(col("Dividendos")))
    proventos_g = ("Proventos", [("DY 12 meses", _pct_txt(col("DY 12m (%)") / 100) if col("DY 12m (%)") is not None else "—",
                                  "", sinal_dy),
                                 ("Meses de data-com", txt(col("Meses (data-com)")), "")])
    grupos = []
    if eh_fii:
        ff = None
        try:
            ff = (_todos_fiis_em_cache() or {}).get(papel)
        except Exception:
            pass
        pa = ctx.get("preco_atrativo") or {}
        grupos.append(("Preço", [("P/VP", _num_txt(col("P/VP"), 2), "", sinal("pvp", col("P/VP"), LIMIARES_FII)),
                                 ("P/VP médio 24 meses", _num_txt(col("P/VP médio 24m"), 2), ""),
                                 ("Preço atrativo?", txt(col("Preço atrativo?")), "",
                                  {"forte": "bom", "sim": "bom", "nao": "ruim"}.get(pa.get("nivel")))]))
        grupos.append(("O fundo", [
            ("Tipo", ("Tijolo" if (ff.qtd_imoveis or 0) > 0 else "Papel (recebíveis)") if ff else "—", ""),
            ("Segmento", html.escape(txt(col("Segmento"))), ""),
            ("Valor de mercado", _mi_txt(getattr(ff, "valor_mercado", None)), ""),
            ("Liquidez diária", _mi_txt(getattr(ff, "liquidez", None)), ""),
            ("Imóveis", str(ff.qtd_imoveis) if ff and ff.qtd_imoveis else "—", ""),
            ("Vacância média", _pct_txt(getattr(ff, "vacancia_media", None)), "",
             sinal("vacancia_media", getattr(ff, "vacancia_media", None), LIMIARES_FII)),
            ("Cap rate", _pct_txt(getattr(ff, "cap_rate", None)), "")]))
        nome_g, itens_g = proventos_g
        grupos.append((nome_g, [itens_g[0], ("Yield × média 24 meses",
                                             f"{_pct_txt(pa.get('yield'))} · {_pct_txt(pa.get('yield_medio_24m'))}", ""),
                                itens_g[1]]))
    elif b3 and tipo == "acao":
        grupos.append(("Preço", [("P/L", _num_txt(col("P/L")), txt(col("P/L × referência")) if col("P/L × referência") else "",
                                  sinal_pl(col("P/L"), col("P/L × referência"))),
                                 ("P/VP", _num_txt(col("P/VP"), 2), "", sinal("pvp", col("P/VP"))),
                                 ("EV/EBITDA", _num_txt(fund.get("ev_ebitda") or None), ""),
                                 ("Preço-teto (Bazin)", fv(pt.get("teto")), nota_teto,
                                  ("bom" if preco <= pt["teto"] else "ruim") if pt.get("teto") else None),
                                 ("Teto pela média 5 anos", fv(pt.get("teto_5a")), "")]))
        grupos.append(("Rentabilidade e saúde", [
            ("ROE", _pct_txt(fund.get("roe")), "", sinal("roe", fund.get("roe"))),
            ("ROIC", _pct_txt(fund.get("roic") or None), "", sinal("roic", fund.get("roic") or None)),
            ("Margem líquida", _pct_txt(getattr(fa, "margem_liquida", None)), "",
             sinal("margem_liquida", getattr(fa, "margem_liquida", None))),
            ("Receita em 5 anos", _pct_txt(getattr(fa, "crescimento_receita_5a", None), 1, True), "ao ano",
             sinal("crescimento_receita_5a", getattr(fa, "crescimento_receita_5a", None))),
            ("Dívida bruta / patrimônio", _num_txt(getattr(fa, "divida_bruta_patrimonio", None), 2), "",
             sinal("divida_bruta_patrimonio", getattr(fa, "divida_bruta_patrimonio", None))),
            ("Liquidez corrente", _num_txt(getattr(fa, "liquidez_corrente", None), 2), "",
             sinal("liquidez_corrente", getattr(fa, "liquidez_corrente", None)))]))
        nome_g, itens_g = proventos_g
        grupos.append((nome_g, [itens_g[0], ("Regularidade", txt(col("Dividendos")), "", sinal_reg), itens_g[1]]))
    elif df_l is not None:  # ETFs e EUA: o que a tabela do Dashboard tiver
        fora = ("Ativo", "Preço", "Variação s/ PM (%)", "Faixa", "DY 12m (%)", "Meses (data-com)")
        itens = []
        for c in df_l.columns:
            if c in fora:
                continue
            v = col(c)
            itens.append((c.replace(" (%)", ""), _num_txt(v, 2 if "P/VP" in c else 1) if isinstance(v, (int, float)) else txt(v), ""))
        grupos += [("Indicadores", itens), proventos_g]
    linha_html = _grupos_html(grupos, legenda=bool(grupos))

    sit = (card or {}).get("situacao")
    cores_fx = {"realizacao": ("📈 Faixa de realização", "#2f45e0"), "queda": ("📉 Faixa de queda", "#d9831c"),
                "fora": ("↔️ Fora das faixas", "#0f7f73")}
    if sit in cores_fx:
        selo_fx = (f"<span style='background:{cores_fx[sit][1]};color:#fff;border-radius:999px;padding:3px 10px;"
                   f"font-size:.8em;font-weight:600'>{cores_fx[sit][0]}</span>")
    elif not card:
        selo_fx = ("<span style='background:#f3f4f6;color:#4b5563;border-radius:999px;padding:3px 10px;font-size:.75em'>"
                   "Faixas: só para ativos da sua carteira (dependem do seu preço médio)</span>")
    else:
        selo_fx = ""
    var_pm = f"{_pct_txt(card['variacao'], 1, True)} sobre o seu preço médio" if card else "fora da sua carteira"
    peso = _pesos_atuais().get(ticker) if card else None
    selo_peso = ""
    if peso:
        lim = _limites_salvos()
        acima = peso["situacao"] == "acima"
        selo_peso = (f"<div style='margin-top:4px'><span style='background:{'#fee2e2' if acima else '#f3f4f6'};"
                     f"color:{'#b91c1c' if acima else '#374151'};border-radius:999px;padding:3px 10px;font-size:.78em;"
                     f"font-weight:600'>⚖️ peso {_num_txt(peso['peso'], 1)}%"
                     + (f" · acima do seu limite de {lim['maximo']:g}%" if acima else
                        f" · abaixo do seu mínimo de {_num_txt(lim['minimo'], 1)}%" if peso["situacao"] == "abaixo" else
                        f" · limite {lim['maximo']:g}%") + "</span></div>")
    st.html(f"<div style='background:#fff;color:#111827;border-radius:12px;padding:12px 14px'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;gap:10px'>"
            f"<div><span style='font-size:1.45em;font-weight:800'>{html.escape(papel)}</span> "
            f"<span style='color:#4b5563'>{html.escape(nome)}</span>"
            f"<div style='font-size:.82em;color:#6b7280'>{html.escape(classe)}{(' · ' + html.escape(setor)) if setor else ''}</div></div>"
            f"<div style='text-align:right'><div style='font-size:1.45em;font-weight:800'>{fv(preco)}</div>"
            f"<div style='font-size:.82em;color:#4b5563'>{var_pm}</div><div style='margin-top:4px'>{selo_fx}</div>{selo_peso}</div></div>"
            f"<div style='margin-top:8px'>{linha_html}</div></div>")

    tendencia = tendencia_rompida(historico)
    st.html(_html_tendencia(tendencia))

    # preço
    risco = risco_12m(historico)
    with st.container(key="vgc_rx_preco"):
        _titulo_cartao(f"Preço — {OPCOES_PERIODO.get(st.session_state.get('individual_periodo'), '')}")
        st.altair_chart(_claro(_grafico_raio_x(historico, moeda, pm=card.get("preco_medio") if card else None,
                                               faixas=card.get("faixas") if card else None, teto=pt.get("teto"),
                                               ops=_ops_do_ticker(ticker) if card else None, f12=f12,
                                               tendencia=tendencia)),
                        width="stretch")
        f12 = f12 or {}
        dt = lambda d: pd.Timestamp(d).strftime("%d/%m/%Y") if d is not None and not pd.isna(d) else ""
        em = lambda d: f"em {dt(d)}" if dt(d) else ""
        pxm = f12.get("preco_x_media")
        sub_media = ("" if pxm is None else "preço de hoje na média" if abs(pxm) < 0.005 else
                     f"preço de hoje {_pct_txt(abs(pxm), 0)} {'acima' if pxm > 0 else 'abaixo'}")
        topo, fundo = (risco or {}).get("data_topo"), (risco or {}).get("data_fundo")
        sub_queda = (f"de {pd.Timestamp(topo).strftime('%d/%m/%y')} a {pd.Timestamp(fundo).strftime('%d/%m/%y')}"
                     if topo is not None and fundo is not None else "do topo ao fundo")
        st.html(_linha_kv(
            _kv(f"<span style='color:{_COR_MINIMA}'>●</span> Mínima 12 meses", fv(f12.get("minima")),
                em(f12.get("data_minima")), _COR_MINIMA),
            _kv(f"<span style='color:{_COR_MAXIMA}'>●</span> Máxima 12 meses", fv(f12.get("maxima")),
                em(f12.get("data_maxima")), _COR_MAXIMA),
            _kv("Preço médio 12 meses", fv(f12.get("media")), sub_media),
            _kv("Maior queda em 12 meses", _pct_txt((risco or {}).get("maior_queda"), 0), sub_queda),
            _kv("Volatilidade 12 meses", _pct_txt((risco or {}).get("volatilidade"), 0) + " ao ano", "oscilação diária anualizada"),
            _html_movimento_brusco(movimento_brusco_recente(historico), moeda) and _kv(
                "Movimento recente", "queda ou alta brusca", "veja o aviso no card do Dashboard")))

    # proventos
    reg = ctx.get("regularidade") or {}
    dy_ind = next((i for i in ctx.get("indicadores") or [] if str(i.get("nome", "")).startswith(("Dividend", "Yield"))),
                  None)
    div12 = pt.get("div12")
    if div12 is None and dividendos is not None and len(dividendos):
        d = dividendos.copy()
        d.index = pd.DatetimeIndex(d.index).tz_localize(None) if pd.DatetimeIndex(d.index).tz is not None else d.index
        div12 = float(d[d.index >= pd.Timestamp(hoje) - pd.DateOffset(months=12)].sum())
    yoc = (div12 / card["preco_medio"]) if card and div12 and card.get("preco_medio") else None
    corpo = _linha_kv(
        _kv("DY 12 meses", (dy_ind or {}).get("valor", "—"), (dy_ind or {}).get("leitura", "")),
        _kv("Selo", {"regular": "Regular", "oscilante": "Oscilante", "irregular": "Irregular"}.get(reg.get("selo"), "—"),
            f"pagou {reg.get('pagou')} de {reg.get('n_anos')} anos" if reg.get("n_anos") else ""),
        _kv("Yield sobre o seu preço médio", _pct_txt(yoc), f"{fv(div12)} ÷ {fv(card['preco_medio'])}" if yoc else
            "só para ativos da sua carteira"))
    anos = [a["ano"] for a in reg.get("anos") or []]
    if not anos:
        anos = list(range(hoje.year - 5, hoje.year))
    anos = sorted(set(anos) | {hoje.year})  # o ano atual também (em andamento)
    por = proventos_por_ano(detalhe, anos)
    dy_ano = {a["ano"]: a.get("dy") for a in reg.get("anos") or []}
    sem_det = "<span style='color:#a16207'>sem detalhe no Fundamentus</span>"
    def _cel(a, chave):
        v = por[a][chave]
        if v:
            return fv(v)
        pagou = (dy_ano.get(a) or 0) > 0 and not any(por[a].values())
        return sem_det if pagou else "—"
    if eh_fii:
        cab = ["Ano", "Yield", "Rendimentos por cota", "Amortização por cota"]
        linhas_t = [[str(a), _pct_txt(dy_ano.get(a)), _cel(a, "rendimentos"),
                     "—" if not por[a]["amortizacao"] else fv(por[a]["amortizacao"])] for a in sorted(anos, reverse=True)]
    else:
        cab = ["Ano", "DY", "Dividendos por ação", "JCP por ação"]
        linhas_t = [[str(a), _pct_txt(dy_ano.get(a)), _cel(a, "dividendos"),
                     "—" if not por[a]["jcp"] else fv(por[a]["jcp"])] for a in sorted(anos, reverse=True)]
    corpo += "<div style='margin-top:8px'>" + _tabela_html(cab, linhas_t) + "</div>"
    prox = proximo_provento(detalhe, hoje)
    if prox:
        tipo_txt = {"jcp": "JCP", "dividendo": "Dividendo", "rendimento": "Rendimento",
                    "amortizacao": "Amortização"}.get(prox["tipo"], "Provento")
        corpo += _linha_kv(_kv("Próximo provento", f"{tipo_txt} {fv(prox['valor'])} por {'cota' if eh_fii else 'ação'}",
                               f"data-com {prox['data_com']:%d/%m/%Y} · pagamento "
                               + (f"{prox['pagamento']:%d/%m/%Y}" if prox["pagamento"] else "ainda sem data")))
    corpo += ("<div style='font-size:.7em;color:#6b7280;margin-top:6px'>Valores por " + ("cota" if eh_fii else "ação")
              + " na data-com, da época (Fundamentus); DY pelo preço médio de cada ano.</div>")
    st.html(_caixa_branca("Proventos", corpo))
    # comparação com os pares (ordem fixa: liquidez diária, da maior para a menor)
    if pares is not None:
        if eh_fii:
            cab = ["Ativo", "DY 12m", "P/VP", "Liquidez diária ▼"]
            lin = [[r.papel, _pct_txt(r.dy), _num_txt(r.pvp, 2), _mi_txt(r.liquidez)] for r in pares.itertuples()]
            titulo = f"Comparação com FIIs do segmento {html.escape(getattr((_todos_fiis_em_cache() or {}).get(papel), 'segmento', '') or '')}"
            rodape = "FIIs do mesmo segmento no Fundamentus, ordenados pela liquidez diária (R$ 1 milhão/dia ou mais)."
        else:
            cab = ["Ativo", "P/L", "P/VP", "ROE", "DY 12m", "Liquidez diária ▼"]
            lin = [[r.papel, _num_txt(r.pl) if pd.notna(r.pl) and r.pl > 0 else ("prejuízo" if pd.notna(r.pl) else "—"),
                    _num_txt(r.pvp, 2), _pct_txt(r.roe), _pct_txt(r.dy), _mi_txt(r.liquidez)]
                   for r in pares.itertuples()]
            g = (ref or {}).get("grupo") or {}
            titulo = f"Comparação com o {grupo_pares.get('tipo', 'segmento')} {html.escape(grupo_pares.get('nome', ''))}"
            pos = posicao_entre_pares(pares)
            mesma_ref = g.get("tipo") == grupo_pares.get("tipo") and g.get("mediana")
            rodape = ((f"{papel}: {pos}. " if pos else "")
                      + "Empresas com liquidez de R$ 1 milhão/dia ou mais, ordenadas pela liquidez diária"
                      + (f" · mediana do P/L {_num_txt(g.get('mediana'))}" if mesma_ref else "") + ".")
        proprio = [i for i, r in enumerate(pares.itertuples()) if r.proprio]
        tabela = _tabela_html(cab, [[f"<b style='color:#1e3a8a'>▶ {c}</b>" if j == 0 and i in proprio else
                                     (f"<b>{c}</b>" if i in proprio else c) for j, c in enumerate(l)]
                                    for i, l in enumerate(lin)])
        st.html(_caixa_branca(titulo, tabela + f"<div style='font-size:.72em;color:#4b5563;margin-top:6px'>{rodape}</div>"))
    elif b3 and tipo in ("acao", "fii"):
        st.html(_caixa_branca("Comparação com os pares",
                              "<div style='font-size:.85em;color:#4b5563'>Sem pares disponíveis agora no "
                              "Fundamentus (tente de novo mais tarde).</div>"))

    if card:
        q = float(card.get("quantidade") or 0)
        resultado_pos = q * (preco - float(card["preco_medio"]))
        realizado = float(card.get("ganho_realizado_ciclo") or 0.0)
        renda = float(card.get("renda_recebida") or 0.0)
        total = resultado_pos + realizado + renda
        ops = _ops_do_ticker(ticker)
        compras = [o for o in ops if o["tipo"] == "compra"]
        vendas = [o for o in ops if o["tipo"] == "venda"]
        rent_txt, rent_sub = "—", ""
        dados_vg = st.session_state.get("visao_geral_dados")
        if dados_vg and ticker in dados_vg["linhas"]:
            df_t = dados_vg["linhas"][ticker]
            ativos = [i for i in range(len(df_t)) if df_t["quantidade"].iloc[i] > 1e-9 or df_t["aportes"].iloc[i] > 0]
            if ativos:
                parte = df_t.iloc[ativos[0]:]
                r = float(rentabilidade_cotas(parte, 0.0).iloc[-1])
                try:
                    cdi = cdi_acumulado(_cdi_bcb_em_cache(str(parte.index[0].start_time.date()),
                                                          str(parte.index[-1].end_time.date())), parte.index)
                    c_fim = float(cdi.iloc[-1]) if cdi is not None else None
                except Exception:
                    c_fim = None
                rent_txt = _pct_txt(r, 1, True) + (f" · CDI {_pct_txt(c_fim, 1, True)}" if c_fim is not None else "")
                rent_sub = f"desde {_rotulo_mes(parte.index[0])} (método das cotas)"
        else:
            rent_sub = "abra a Visão geral uma vez para calcular"
        regua = card.get("faixas") or {}
        corpo = _linha_kv(
            _kv("Posição", f"{_num_txt(q, 0 if q.is_integer() else 4)} {'cotas' if eh_fii or tipo.startswith('etf') else 'ações'}",
                fv(q * preco)),
            _kv("Preço médio", fv(card["preco_medio"])),
            _kv("Resultado total", ("+" if total >= 0 else "−") + fv(abs(total)), "preço + vendas + proventos",
                "#15803d" if total >= 0 else "#b91c1c"),
            _kv("Proventos recebidos", fv(renda), f"desde {card['desde']:%d/%m/%Y}" if card.get("desde") else ""))
        corpo += _linha_kv(
            _kv("Rentabilidade × CDI", rent_txt, rent_sub),
            _kv("Régua de faixas", f"aumentar abaixo de {fv(regua.get('queda'))}",
                f"realizar a partir de {fv(regua.get('realizacao'))}"),
            _kv("Operações", f"{len(compras)} compra(s) · {len(vendas)} venda(s)",
                (f"última: {pd.Timestamp(ops[-1]['data']):%d/%m/%Y}" if ops else "")))
        st.html(_caixa_branca("Na sua carteira", corpo))
    st.caption("Informação para investigar, não recomendação. A decisão é sempre sua.")


def _limpar_raio_x(origem: str) -> None:
    """Trocou o ativo: apaga o raio-X anterior e o outro campo de escolha
    (lista da carteira × ticker digitado), para não misturar os dois."""
    if origem == "lista":
        st.session_state["ticker_individual"] = ""
    else:
        st.session_state["rx_lista"] = "—"
    for chave in ("individual_resultado", "individual_ticker", "individual_contexto", "individual_dividendos",
                  "individual_faixa_12m", "individual_horario"):
        st.session_state.pop(chave, None)


with tab_ativo:
    st.html(_CSS_CARTOES)
    st.subheader("🔍 Raio-X do ativo")
    st.caption("Escolha um ativo da sua carteira ou digite qualquer outro (o .SA da B3 é opcional).")
    da_carteira = sorted(set(tickers_encontrados)) if tickers_encontrados else []
    col_lista, col_ticker, col_tipo, col_per = st.columns([1.4, 1.4, 1, 1])
    with col_lista:
        escolha_lista = st.selectbox("Ativo da carteira", ["—"] + da_carteira, key="rx_lista",
                                     on_change=_limpar_raio_x, args=("lista",),
                                     format_func=lambda t: t.replace(".SA", "") if t != "—" else "—")
    with col_ticker:
        ticker_individual = st.text_input("ou digite outro ticker", placeholder="ex: PETR4, MXRF11, AAPL",
                                          key="ticker_individual", on_change=_limpar_raio_x, args=("texto",))
    ticker_normalizado = (normalizar_ticker(ticker_individual)[0] if ticker_individual.strip()
                          else (escolha_lista if escolha_lista != "—" else ""))
    with col_tipo:
        chute_individual = (tipos_planilha.get(ticker_normalizado) or _heuristica_tipo(ticker_normalizado)
                            if ticker_normalizado else "acao")
        tipo_individual = st.selectbox(
            "Tipo", options=OPCOES_TIPO, index=OPCOES_TIPO.index(chute_individual),
            format_func=lambda t: NOMES_TIPO[t], key=f"tipo_individual_{ticker_normalizado}_{chute_individual}")
    with col_per:
        periodo_individual = st.selectbox(
            "Histórico", options=list(OPCOES_PERIODO.keys()), format_func=lambda p: OPCOES_PERIODO[p],
            index=list(OPCOES_PERIODO.keys()).index("2y"), key="periodo_individual")

    if st.button("🔍 Ver o raio-X", type="secondary"):
        ticker_limpo = ticker_normalizado
        if not ticker_limpo:
            st.warning("Escolha um ativo da lista ou digite um ticker.")
        else:
            texto_horario, _ = _info_horario_analise()
            with st.spinner(f"Buscando dados de {ticker_limpo}..."):
                base_cvm_ind = _base_cvm_recente() if tipo_individual == "fii" else None
                fonte_individual = FonteComRendimentosConferidos(
                    fonte_dados, {ticker_limpo: tipo_individual},
                    lambda t: {"fundamentus": _proventos_fii_em_cache(t),
                               "cvm": (serie_rendimentos(base_cvm_ind, t, "dy")
                                       if base_cvm_ind is not None and not base_cvm_ind.empty else None)},
                    proventos_da_acao=_proventos_acao_em_cache)
                resultado = analisar_ativo(ticker_limpo, tipo_individual, periodo_individual, fonte_individual,
                                           cdi_liquido_aa=_cdi_liquido_atual())
                try:
                    dividendos_individual = fonte_individual.baixar_dividendos(ticker_limpo)
                except Exception:
                    dividendos_individual = pd.Series(dtype=float)
            st.session_state["individual_ticker"] = ticker_limpo
            st.session_state["individual_tipo"] = tipo_individual
            st.session_state["individual_periodo"] = periodo_individual
            st.session_state["individual_horario"] = texto_horario
            st.session_state["individual_resultado"] = resultado
            st.session_state["individual_dividendos"] = dividendos_individual
            try:
                st.session_state["individual_contexto"] = _contexto_para(ticker_limpo, resultado, dividendos_individual)
            except Exception:
                st.session_state["individual_contexto"] = None
            try:
                st.session_state["individual_faixa_12m"] = faixa_12m(fonte_dados.baixar_precos(ticker_limpo, "1y"))
            except Exception:
                st.session_state["individual_faixa_12m"] = None

    if "individual_resultado" in st.session_state:
        ticker_limpo = st.session_state["individual_ticker"]
        resultado = st.session_state["individual_resultado"]
        st.caption(f"🕒 Dados de {st.session_state['individual_horario']}")
        if resultado is None:
            st.error(f"Não foi possível obter dados suficientes para {ticker_limpo}. Confira o ticker e o tipo.")
        else:
            try:
                _renderizar_raio_x(ticker_limpo, st.session_state["individual_tipo"], resultado,
                                   st.session_state.get("individual_contexto"),
                                   st.session_state.get("individual_dividendos"),
                                   st.session_state.get("individual_faixa_12m"))
            except Exception as e:
                st.error(f"Não foi possível montar o raio-X ({e}).")
            with st.expander("Gráfico técnico (MM50, MM200, RSI e dividendos)"):
                st.altair_chart(_grafico_preco_completo(resultado.tec.historico,
                                                        st.session_state["individual_dividendos"]), width='stretch')

# =============================================================================
# ABA 4 — Dashboard da Carteira (extrato por ativo)
# =============================================================================
with tab_dashboard:
    st.subheader("💼 Dashboard da Carteira")
    st.caption(
        "Extrato por ativo, agrupado por situação (ativa/encerrada), mercado (B3/EUA) e "
        "tipo de ativo (Ações, FIIs, ETF), com subtotal por grupo e total por mercado."
    )

    if bytes_planilha is None:
        st.info("Envie um arquivo Excel na aba 'Carteira' para ver o dashboard.")
    else:
        tem_saldo_final = "carteira_linhas_ganho" in st.session_state
        if not tem_saldo_final:
            st.caption(
                "⚠️ Saldo final, ganho e rentabilidade dependem de preço de mercado atual — "
                "clique em 'Analisar carteira' na aba 'Carteira' pra completar essas "
                "colunas. Posição inicial, compras e vendas já aparecem abaixo, direto da planilha."
            )

        try:
            _renderizar_tabela_indicadores(st.session_state.get("carteira_cenarios"))
        except Exception as e:
            st.caption(f"Não foi possível montar a tabela de indicadores ({e}).")
        _renderizar_cenarios(st.session_state.get("carteira_cenarios"))
        st.divider()
        st.markdown("#### 📒 Extrato por ativo")

        linhas_dashboard = construir_dashboard_por_ativo(
            df_resumo, df_operacoes,
            linhas_ganho=st.session_state.get("carteira_linhas_ganho"),
            tipos=st.session_state.get("carteira_tipos_confirmados", {}),
            data_analise=st.session_state.get("carteira_data_analise"),
        )

        # Linhas da planilha que não entraram no cálculo, eventos da fonte não
        # lançados e conferência do totalizador (inicial + compras − vendas ±
        # eventos = saldo final), para que nenhuma divergência passe em silêncio.
        avisos_planilha = [f"{l.ticker}: {a}" for l in (st.session_state.get("carteira_linhas_ganho") or [])
                           for a in list(getattr(l, "avisos_eventos", [])) + l.avisos]
        # posição (ou ciclo encerrado) cujo resultado não pôde ser calculado: o extrato
        # mostraria só "-" nas colunas; o motivo vem aqui, se ainda não foi avisado
        _ja_avisados = {l.ticker for l in st.session_state.get("carteira_linhas_ganho") or []
                        if any("maior que a posição" in a for a in l.avisos)}
        for l in st.session_state.get("carteira_linhas_ganho") or []:
            if l.situacao == "erro" and l.erro and l.ticker not in _ja_avisados:
                avisos_planilha.append(f"{l.ticker}: resultado não calculado — {l.erro}")
        for l in linhas_dashboard:
            if l.saldo_final_qtde is None:
                continue
            esperado = (l.qtde_inicial or 0.0) + l.compras_quantidade - l.vendas_quantidade + l.eventos_quantidade
            if abs(esperado - l.saldo_final_qtde) > 1e-6:
                avisos_planilha.append(
                    f"{getattr(l, 'rotulo', '') or l.ticker}: a planilha soma {esperado:g} cotas (inicial + compras − vendas ± eventos), mas o "
                    f"saldo calculado é {l.saldo_final_qtde:g} — confira as linhas desse ativo.")
        if avisos_planilha:
            with st.expander(f"⚠️ {len(avisos_planilha)} aviso(s) sobre a planilha", expanded=True):
                for a in avisos_planilha:
                    st.markdown(f"- {_sem_formula(a)}")

        _renderizar_dashboard_agrupado(linhas_dashboard)
        st.caption("Eventos (qtde): ações recebidas em bonificação ou desdobramento (+) e retiradas em "
                   "grupamento (−). Desdobramento e grupamento não mudam o custo total, só a quantidade e o "
                   "preço médio; bonificação soma ao custo o valor atribuído pela empresa. Compra com preço 0 "
                   "é tratada como bonificação sem custo.")

# =============================================================================
# ABA — Proventos e IR
# =============================================================================
with tab_ir:
    st.subheader("🧾 Proventos e IR")
    if bytes_planilha is None:
        st.info("Envie um arquivo Excel na aba 'Carteira' para ver os proventos para o IR.")
    elif "carteira_linhas_ganho" not in st.session_state:
        st.info("Clique em 'Analisar carteira' na aba 'Carteira' para montar a tabela.")
    else:
        try:
            dados_ir = st.session_state.get("visao_geral_dados")
            if dados_ir is None:
                with st.spinner("Lendo as operações da planilha..."):
                    dados_ir = _dados_visao_geral(df_resumo, df_operacoes,
                                                  st.session_state.get("carteira_tipos_confirmados", {}))
                st.session_state["visao_geral_dados"] = dados_ir
            if dados_ir:
                _renderizar_proventos_ir(dados_ir)
        except Exception as e:
            st.error(f"Não foi possível montar a tabela do IR ({e}).")

# =============================================================================
# ABA 6 — Guia de Indicadores
# =============================================================================
with tab_guia:
    st.subheader("📖 Guia de Indicadores")
    st.caption(
        "O que cada indicador significa, e como o painel interpreta o resultado (os "
        "limiares abaixo são os mesmos usados de verdade no motor de pontuação)."
    )

    with st.expander("📈 Indicadores técnicos (todos os tipos de ativo)", expanded=True):
        st.markdown(
            "**MM200 (Média Móvel de 200 dias)** — a média do preço nos últimos 200 "
            "pregões. Preço acima da MM200 costuma indicar tendência de alta no médio "
            "prazo; abaixo, tendência de baixa. É a mesma lógica usada por boa parte do "
            "mercado como referência de tendência de longo prazo.\n\n"
            "**MM50 (Média Móvel de 50 dias)** — mesma ideia, mas numa janela mais curta "
            "(reage mais rápido a mudanças recentes de preço). Aparece no gráfico junto "
            "com a MM200 pra dar contexto de tendência de curto vs. médio prazo.\n\n"
            "**RSI(14) — Índice de Força Relativa** — mede a velocidade e magnitude das "
            "variações de preço recentes, numa escala de 0 a 100. **Acima de 70** costuma "
            "indicar sobrecompra (o ativo subiu rápido demais, pode estar 'esticado'); "
            "**abaixo de 30-40** indica sobrevenda (pode estar descontado, mas também pode "
            "ser sinal de queda estrutural — o painel não distingue as duas coisas).\n\n"
            "**Piso e topo usuais do período** — compara o preço atual com o percentil 5% "
            "(piso usual) e 95% (topo usual) do histórico buscado: só 5% dos fechamentos "
            "ficaram abaixo do piso, e 95% abaixo do topo. Não são o menor e o maior preço "
            "— por isso a cota pode ficar acima do topo usual sem ter batido um recorde; o "
            "maior e o menor preço dos últimos 12 meses aparecem na faixa de 12 meses do card. "
            "Perto do piso pode ser oportunidade ou queda continuada; perto do topo, o oposto."
        )

    with st.expander("🏢 Indicadores fundamentalistas — Ações (B3 e EUA)"):
        st.markdown(
            "**P/L (Preço/Lucro)** — quantos anos de lucro atual seriam necessários pra "
            "'pagar' o preço da ação, no ritmo de hoje. Mais baixo costuma indicar ação "
            "mais barata relativa ao lucro que gera — mas P/L muito baixo também pode "
            "significar que o mercado espera o lucro cair. Limiar do painel: B3 < 15 é "
            "considerado baixo, > 25 é considerado alto (para ações dos EUA, os limiares "
            "são mais altos — 25 e 40 — porque o mercado americano historicamente negocia "
            "com múltiplos mais elevados).\n\n"
            "**P/VP (Preço/Valor Patrimonial)** — compara o preço da ação com o patrimônio "
            "líquido por ação da empresa. Abaixo de 1 significa que a ação vale menos que "
            "o patrimônio contábil da empresa. Limiar do painel: B3 < 1,5 é baixo, > 4 é "
            "alto (EUA: 4 e 10).\n\n"
            "**ROE (Retorno sobre o Patrimônio)** — quanto lucro a empresa gera em relação "
            "ao patrimônio dos acionistas, em %. Maior costuma ser melhor (empresa mais "
            "eficiente usando o capital próprio). Limiar: > 12% (B3) ou > 15% (EUA).\n\n"
            "**ROIC (Retorno sobre o Capital Investido)** — parecido com o ROE, mas "
            "considera todo o capital investido (próprio + dívida), não só o patrimônio. "
            "Só disponível para ações da B3 (via Fundamentus). Limiar: > 15%.\n\n"
            "**Margem Líquida** — quanto do faturamento vira lucro, em %. Só disponível "
            "para ações da B3. Limiar: > 10%.\n\n"
            "**Dívida Bruta/Patrimônio** — o quanto a empresa deve em relação ao próprio "
            "patrimônio. Mais baixo costuma indicar menor risco financeiro. Limiar: < 0,5 "
            "é baixo endividamento, > 1,5 é alto. Não se aplica a bancos e seguradoras "
            "(estrutura de balanço diferente — o painel trata isso automaticamente).\n\n"
            "**Liquidez Corrente** — capacidade de pagar dívidas de curto prazo com ativos "
            "de curto prazo. Acima de 1 significa que a empresa tem mais ativo do que "
            "dívida no curto prazo. Limiar: > 1,5 é saudável, < 1,0 é apertado (mesma "
            "exceção de bancos/seguradoras do item anterior)."
        )

    with st.expander("🏠 Indicadores fundamentalistas — FIIs"):
        st.markdown(
            "**P/VP (Preço/Valor Patrimonial)** — compara o preço da cota com o valor "
            "patrimonial por cota do fundo. Abaixo de 1 (a cota 'com desconto') costuma "
            "ser visto como mais atrativo; acima de 1 ('com ágio'), o oposto. Limiar do "
            "painel: < 0,95 desconto, > 1,10 ágio.\n\n"
            "**Vacância Média** — % dos imóveis do fundo que estão desocupados, sem "
            "gerar aluguel. Mais baixo é melhor. Limiar: < 5% é baixa, > 15% é alta. Só "
            "se aplica a FIIs 'de tijolo' (que têm imóveis físicos) — FIIs 'de papel' "
            "(que investem em CRIs/recebíveis) não têm esse conceito, e o painel não "
            "pontua vacância pra eles.\n\n"
            "**Cap Rate** — retorno anual gerado pelos aluguéis em relação ao valor dos "
            "imóveis do fundo. Mais alto costuma ser melhor. Limiar: > 8% atrativo, < 5% "
            "baixo. Mesma exceção de FIIs de papel do item anterior."
        )

    with st.expander("💰 Qualidade da renda (dividendos/rendimentos) — todos os tipos"):
        st.markdown(
            "**Yield efetivo (12 meses)** — soma de todos os proventos pagos nos últimos "
            "12 meses, dividida pelo preço atual — calculado a partir do histórico real "
            "de pagamentos (não usa o campo 'dividend yield' bruto de nenhuma fonte, que "
            "costuma vir com escala inconsistente). Limiar de yield 'atrativo' varia por "
            "tipo de ativo: FII > 8%, ação B3 > 6%, ação EUA > 2%, ETF B3 > 4%, ETF "
            "EUA > 1,5% (o mercado americano historicamente distribui menos que o "
            "brasileiro).\n\n"
            "**Crescimento dos dividendos (ano vs. ano anterior)** — compara o total pago "
            "nos últimos 12 meses com os 12 meses anteriores a esses. Crescendo (> 5%) é "
            "positivo; caindo mais de 15% é sinal de atenção.\n\n"
            "**Regularidade dos pagamentos** — quantas vezes o ativo pagou provento nos "
            "últimos 12 meses. FIIs costumam pagar mensalmente (limiar: 10+ pagamentos "
            "pra ser considerado regular); ações costumam pagar com menos frequência "
            "(limiar: 2+)."
        )

    with st.expander("🧭 Cenários por ativo (regra de faixas de preço)"):
        st.markdown(
            "O painel compara o preço de hoje com o **seu preço médio** e mostra em que **situação** cada "
            "ativo está — e os cenários possíveis, lado a lado, para você comparar. Não há recomendação: "
            "a decisão é sempre sua.\n\n"
            "| Variação sobre o preço médio | Situação | Cenários comparados |\n"
            "|---|---|---|\n"
            "| Alta de 25% ou mais | Faixa de realização de lucro | Manter tudo × realizar 10% / 20% / 30% / 40% "
            "(nas altas de 25% / 35% / 45% / 60%) ou venda total (a partir de +100%) |\n"
            "| Entre −15% e +25% | Fora das faixas | — (o card mostra a que preço cada faixa começa) |\n"
            "| Queda de 15% ou mais | Faixa de queda | Manter tudo × aumentar a posição em 10% (−15%) ou 25% (−25%) |\n\n"
            "**Como ler a tabela de cenários:** as mesmas 8 linhas, na mesma ordem, em todos os cards — valor "
            "investido, valor da posição, movimento de caixa (+ venda / − aporte), resultado realizado e não "
            "realizado, preço médio, variação e renda estimada por mês.\n\n"
            "**O que o histórico mostra:** cada card cita o resultado do backtest do seu próprio cenário (Seção 12 "
            "do notebook): ativos testados entre 2022 e 2026, a partir de até 13 datas de entrada, com o dinheiro "
            "guardado rendendo o CDI real, líquido de IR e IOF. Quando o cenário usa dinheiro novo (aumentar a "
            "posição), a comparação é com o mesmo dinheiro guardado no CDI.\n\n"
            "**FIIs:** além da situação, o card mostra o que teria acontecido nos últimos 12 meses com os "
            "rendimentos — sem reinvestir, reinvestindo 50% ou 100%, ou guardando no CDI para comprar cotas na "
            "queda —, com os preços, rendimentos e CDI reais do período.\n\n"
            "**Proteções que viram avisos no card:** no máximo 2 aumentos de posição na queda por ativo (o "
            "contador zera quando o preço volta a ficar 5% acima do preço médio); e alertas de saúde quando a "
            "empresa dá prejuízo, tem dívida alta (acima de 1,5 vez o patrimônio) ou pouca liquidez, quando o FII "
            "tem P/VP acima de 1,10 ou vacância acima de 15%, ou quando os rendimentos caíram mais de 15% no "
            "último ano. Cada faixa de realização vale uma vez; volta a valer depois de um novo aumento de "
            "posição ou se o preço voltar ao seu preço médio.\n\n"
            "O preço médio segue a regra da Receita Federal: compras mudam o preço médio, vendas não. A renda "
            "por mês é estimada pelos proventos pagos nos últimos 12 meses e, no backtest, errou em média cerca "
            "de 2,5 pontos percentuais de yield — ela não antecipa mudanças de rendimento."
        )

    st.info(
        "Nenhum desses indicadores, isolado ou em conjunto, garante retorno — são regras "
        "de bolso genéricas de mercado, não uma recomendação de investimento. Este painel "
        "não substitui um assessor/consultor licenciado (CVM)."
    )
