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

Organizado em 4 abas: Análise de Carteira, Ativo Específico, Dashboard da
Carteira (extrato por ativo) e Guia de Indicadores (glossário).
"""
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import altair as alt
import pandas as pd
import streamlit as st

from carteira_analise.carteira import analisar_ativo
from carteira_analise.fontes import yahoo as fonte_yahoo
from carteira_analise.planilha import (
    achar_aba,
    construir_dashboard_por_ativo,
    normalizar_aba_operacoes,
    normalizar_aba_resumo,
    normalizar_ticker,
    processar_carteira_combinada,
)
from carteira_analise.tecnicos import calcular_rsi

st.set_page_config(page_title="Análise de Carteira", page_icon="📊", layout="wide")


def _verificar_senha() -> bool:
    """Bloqueia o acesso ao painel até a senha correta ser informada. A
    senha fica no gerenciador de Secrets do Streamlit Cloud (Settings →
    Secrets), nunca no código — mesmo com o repositório público no GitHub,
    a senha não fica exposta."""
    if st.session_state.get("autenticado"):
        return True

    try:
        senha_esperada = st.secrets.get("senha_painel")
    except Exception:
        # st.secrets lança exceção (em vez de devolver None) quando não existe
        # NENHUM secrets.toml configurado ainda — trata como "sem senha".
        senha_esperada = None
    if not senha_esperada:
        st.title("🔒 Acesso ao Painel")
        st.error(
            "Nenhuma senha configurada ainda. Antes de publicar, defina "
            "'senha_painel' em Settings → Secrets no Streamlit Cloud "
            "(ou, pra testar localmente, crie um arquivo "
            ".streamlit/secrets.toml com o conteúdo: "
            'senha_painel = "sua-senha-aqui").'
        )
        return False

    st.title("🔒 Acesso ao Painel")
    senha_digitada = st.text_input("Senha de acesso", type="password")
    if st.button("Entrar"):
        if senha_digitada == senha_esperada:
            st.session_state["autenticado"] = True
            st.rerun()
        else:
            st.error("Senha incorreta.")
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

    df_dividendos = pd.DataFrame(columns=["data", "preco"])
    if dividendos is not None and not dividendos.empty:
        datas_div = dividendos.index[
            (dividendos.index >= historico.index.min()) & (dividendos.index <= historico.index.max())
        ]
        if len(datas_div) > 0:
            pos = historico.index.get_indexer(datas_div, method="nearest")
            df_dividendos = pd.DataFrame({
                "data": datas_div,
                "preco": historico.iloc[pos].values,
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
            "média:N", scale=alt.Scale(range=["#ff7f0e", "#d62728"]),
            legend=alt.Legend(title=None, orient="bottom", direction="horizontal"),
        ),
    )
    faixa_min_max = alt.Chart(pd.DataFrame({"y": [minima_periodo, maxima_periodo]})).mark_rule(
        strokeDash=[4, 4], color="gray", opacity=0.6
    ).encode(y="y:Q")
    marcadores_dividendo = alt.Chart(df_dividendos).mark_point(
        shape="triangle-up", color="#2ca02c", size=60, filled=True
    ).encode(x="data:T", y="preco:Q", tooltip=["data:T", "preco:Q"])

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


def _linha_ganho_para_dict(l) -> dict:
    return {
        "Ticker": l.ticker, "Origem": l.origem,
        "Quantidade aberta": f"{l.quantidade_aberta:.0f}" if l.quantidade_aberta is not None else "-",
        "Preço médio": f"{l.moeda} {l.preco_medio_atual:.2f}" if l.preco_medio_atual else "-",
        "Preço atual": f"{l.moeda} {l.preco_atual:.2f}" if l.preco_atual is not None else "N/D",
        "Ganho realizado": f"{l.moeda} {l.ganho_realizado:+.2f}" if l.ganho_realizado is not None else "-",
        "Ganho não realizado": (
            f"{l.moeda} {l.ganho_nao_realizado:+.2f}" if l.ganho_nao_realizado is not None else "N/A"
        ),
        "Renda recebida": f"{l.moeda} {l.renda_recebida:.2f}" if l.renda_recebida is not None else "-",
        "Ganho total": f"{l.moeda} {l.ganho_total:+.2f}" if l.ganho_total is not None else "-",
    }


def _renderizar_detalhe(d: str) -> str:
    """Colore o indicador (verde/vermelho) de acordo com o sinal do
    detalhe, mesma convenção da tabela de análise (🟢/🔴) — os detalhes
    sempre começam com '+1' ou '-1' (ver pontuacao.py)."""
    if d.startswith("+1"):
        return f"🟢 {d[3:].strip()}"
    if d.startswith("-1"):
        return f"🔴 {d[3:].strip()}"
    return f"🟡 {d}"


st.title("📊 Análise de Carteira — Ações, FIIs e ETFs")
st.markdown(
    "Faça upload da sua planilha (Excel) com uma ou duas abas:\n\n"
    "- **Resumo** — posição inicial de cada ativo: `ticker`, `quantidade`, "
    "`preco_medio` (ou `valor_investido`), `data_inicio`.\n"
    "- **Operações** — movimentações feitas a partir dali: `ticker`, `tipo` "
    "(compra/venda), `quantidade`, `preco`, `data`.\n\n"
    "Você pode ter só uma das abas, ou as duas — quando tiver as duas, a "
    "posição inicial entra como o primeiro registro, e cada operação "
    "lançada depois ajusta quantidade, preço médio e ganho automaticamente, "
    "igual um extrato."
)

arquivo = st.file_uploader("Escolha o arquivo Excel (.xlsx) da sua carteira", type=["xlsx"])

df_resumo = None
df_operacoes = None
tickers_encontrados: list[str] = []

if arquivo is not None:
    try:
        abas = pd.read_excel(arquivo, sheet_name=None)
    except Exception as e:
        st.error(f"Não consegui ler o arquivo: {e}")
        st.stop()

    df_resumo_bruto = achar_aba(abas, ["Resumo", "Resumo da carteira", "Posicao", "Posição"])
    df_operacoes_bruto = achar_aba(abas, ["Operações", "Operacoes", "Movimentações", "Movimentacoes"])

    if df_resumo_bruto is None and df_operacoes_bruto is None:
        st.warning(
            "Não encontrei nenhuma aba chamada 'Resumo' ou 'Operações' (nem variações). "
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

tab_carteira, tab_ativo, tab_dashboard, tab_guia = st.tabs([
    "📊 Análise de Carteira", "🔍 Ativo Específico", "💼 Dashboard da Carteira", "📖 Guia de Indicadores",
])

# =============================================================================
# ABA 1 — Análise de Carteira
# =============================================================================
with tab_carteira:
    if arquivo is None:
        st.info("Envie um arquivo Excel acima para começar a análise.")
    else:
        st.subheader("Confirme o tipo de cada ativo")
        st.caption(
            "Adivinhamos o tipo pelo formato do ticker — confira se está certo antes de rodar "
            "a análise, principalmente para tickers terminados em '11' (podem ser FII ou ETF)."
        )

        tipos_confirmados: dict[str, str] = {}
        colunas = st.columns(3)
        for i, ticker in enumerate(tickers_encontrados):
            chute = _heuristica_tipo(ticker)
            with colunas[i % 3]:
                escolha = st.selectbox(
                    ticker, options=OPCOES_TIPO, index=OPCOES_TIPO.index(chute),
                    format_func=lambda t: NOMES_TIPO[t], key=f"tipo_{ticker}",
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
                linhas_ganho = processar_carteira_combinada(df_resumo, df_operacoes, fonte_yahoo, periodo_carteira)
                linhas_analise = []
                dividendos_por_ticker = {}
                for ticker in tickers_encontrados:
                    tipo = tipos_confirmados[ticker]
                    resultado = analisar_ativo(ticker, tipo, periodo_carteira, fonte_yahoo)
                    linhas_analise.append((ticker, tipo, resultado))
                    try:
                        dividendos_por_ticker[ticker] = fonte_yahoo.baixar_dividendos(ticker)
                    except Exception:
                        dividendos_por_ticker[ticker] = pd.Series(dtype=float)

            # Guarda tudo na sessão — sobrevive a um clique posterior noutra aba/botão
            # (ver docstring do arquivo) e alimenta também a aba Dashboard.
            st.session_state["carteira_horario"] = texto_horario
            st.session_state["carteira_pregao_aberto"] = pregao_provavelmente_aberto
            st.session_state["carteira_linhas_ganho"] = linhas_ganho
            st.session_state["carteira_linhas_analise"] = linhas_analise
            st.session_state["carteira_dividendos"] = dividendos_por_ticker
            st.session_state["carteira_tipos_confirmados"] = tipos_confirmados
            st.session_state["carteira_data_analise"] = datetime.now(ZoneInfo("America/Sao_Paulo")).date()

        if "carteira_linhas_analise" in st.session_state:
            st.caption(f"🕒 Análise executada em {st.session_state['carteira_horario']}")
            if st.session_state["carteira_pregao_aberto"]:
                st.caption(
                    "📊 Pregão provavelmente em andamento — os preços usados são os mais "
                    "recentes disponíveis agora, não necessariamente o fechamento definitivo "
                    "do dia. Rodar a análise de novo após o fechamento (~18h, horário de "
                    "Brasília) pode trazer números diferentes."
                )

            linhas_analise = st.session_state["carteira_linhas_analise"]
            linhas_ganho = st.session_state["carteira_linhas_ganho"]

            st.subheader("📈 Análise de cada ativo")
            linhas_tabela = []
            for ticker, tipo, resultado in linhas_analise:
                if resultado is None:
                    linhas_tabela.append({
                        "Ticker": ticker, "Tipo": NOMES_TIPO[tipo],
                        "Sinal de timing": "⚠️ dados insuficientes", "Qualidade da renda": "-",
                        "Leitura combinada": "-",
                    })
                    continue
                linhas_tabela.append({
                    "Ticker": ticker, "Tipo": NOMES_TIPO[tipo],
                    "Sinal de timing": resultado.sinal_timing,
                    "Qualidade da renda": resultado.qualidade_renda,
                    "Leitura combinada": resultado.leitura_combinada_texto,
                })

            _renderizar_tabela_alinhada(pd.DataFrame(linhas_tabela), colunas_moeda=[])

            with st.expander("Ver detalhes da pontuação de cada ativo"):
                ativos_com_resultado = [(t, tp, r) for t, tp, r in linhas_analise if r is not None]
                n_colunas = min(4, len(ativos_com_resultado)) or 1
                for inicio in range(0, len(ativos_com_resultado), n_colunas):
                    bloco = ativos_com_resultado[inicio:inicio + n_colunas]
                    colunas_detalhe = st.columns(len(bloco))
                    for coluna, (ticker, tipo, resultado) in zip(colunas_detalhe, bloco):
                        with coluna:
                            st.markdown(f"**{ticker}** ({NOMES_TIPO[tipo]})")
                            if resultado.detalhes_timing:
                                st.markdown("*Timing:*")
                                for d in resultado.detalhes_timing:
                                    st.markdown(f"{_renderizar_detalhe(d)}")
                            if resultado.detalhes_renda:
                                st.markdown("*Qualidade da renda:*")
                                for d in resultado.detalhes_renda:
                                    st.markdown(f"{_renderizar_detalhe(d)}")

            ativos_com_resultado = [(t, tp, r) for t, tp, r in linhas_analise if r is not None]
            if ativos_com_resultado:
                st.markdown("**Ver gráfico de preço de um ativo**")
                ticker_grafico = st.selectbox(
                    "Escolha o ativo", options=[t for t, tp, r in ativos_com_resultado],
                    key="ticker_grafico_carteira",
                )
                resultado_grafico = next(r for t, tp, r in ativos_com_resultado if t == ticker_grafico)
                dividendos_grafico = st.session_state["carteira_dividendos"].get(
                    ticker_grafico, pd.Series(dtype=float)
                )
                st.caption(
                    "Linha azul = preço · laranja = MM50 · vermelho = MM200 · linhas tracejadas = "
                    "mínima/máxima do período (percentil 5%/95%) · triângulos verdes = dividendo "
                    "pago. Painel de baixo: RSI(14), com faixas de referência em 30 e 70."
                )
                grafico = _grafico_preco_completo(resultado_grafico.tec.historico, dividendos_grafico)
                st.altair_chart(grafico, width='stretch')

            st.subheader("💰 Ganho da carteira")
            abertas = [l for l in linhas_ganho if l.situacao == "Aberta"]
            encerradas = [l for l in linhas_ganho if l.situacao == "Encerrada"]
            com_erro = [l for l in linhas_ganho if l.situacao == "erro"]

            colunas_moeda_ganho = [
                "Preço médio", "Preço atual", "Ganho realizado",
                "Ganho não realizado", "Renda recebida", "Ganho total",
            ]

            if abertas:
                st.markdown(f"**Carteira atual — {len(abertas)} posição(ões) em aberto**")
                _renderizar_tabela_alinhada(
                    pd.DataFrame([_linha_ganho_para_dict(l) for l in abertas]), colunas_moeda_ganho
                )

            if encerradas:
                st.markdown(f"**Posições encerradas — {len(encerradas)} ativo(s)**")
                _renderizar_tabela_alinhada(
                    pd.DataFrame([_linha_ganho_para_dict(l) for l in encerradas]), colunas_moeda_ganho
                )

            if com_erro:
                st.warning(f"{len(com_erro)} ativo(s) com problema ao calcular o ganho:")
                for l in com_erro:
                    st.markdown(f"- **{l.ticker}**: {l.erro}")
                    for a in l.avisos:
                        st.caption(f"  ⚠️ {a}")

            totais_por_moeda: dict[str, float] = {}
            for l in linhas_ganho:
                if l.ganho_total is not None:
                    totais_por_moeda[l.moeda] = totais_por_moeda.get(l.moeda, 0.0) + l.ganho_total
            if totais_por_moeda:
                st.markdown("**Totais por moeda (sem conversão cambial entre elas):**")
                for moeda, total in totais_por_moeda.items():
                    st.metric(f"Ganho total ({moeda})", f"{moeda} {total:+,.2f}")

            st.caption(
                "Este painel não é uma recomendação de investimento nem substitui um "
                "assessor/consultor licenciado (CVM). Os critérios usados são regras de "
                "bolso genéricas de mercado, não garantem retorno."
            )

# =============================================================================
# ABA 2 — Ativo Específico
# =============================================================================
with tab_ativo:
    st.subheader("🔍 Analisar um ativo específico")
    st.caption(
        "Quer olhar um ativo que não está na sua carteira, ou só conferir um antes de "
        "decidir? Digite o ticker abaixo — não precisa estar em nenhuma planilha."
    )

    col_ticker, col_tipo = st.columns([2, 1])
    with col_ticker:
        ticker_individual = st.text_input(
            "Ticker", placeholder="ex: PETR4.SA, MXRF11.SA, AAPL", key="ticker_individual"
        )
    with col_tipo:
        chute_individual = _heuristica_tipo(ticker_individual.strip().upper()) if ticker_individual.strip() else "acao"
        tipo_individual = st.selectbox(
            "Tipo", options=OPCOES_TIPO, index=OPCOES_TIPO.index(chute_individual),
            format_func=lambda t: NOMES_TIPO[t], key="tipo_individual",
        )

    periodo_individual = st.selectbox(
        "Histórico para análise técnica", options=list(OPCOES_PERIODO.keys()),
        format_func=lambda p: OPCOES_PERIODO[p], index=list(OPCOES_PERIODO.keys()).index("2y"),
        key="periodo_individual",
    )

    if st.button("🔍 Analisar ativo", type="secondary"):
        ticker_limpo = ticker_individual.strip().upper()
        if not ticker_limpo:
            st.warning("Digite um ticker antes de analisar.")
        else:
            texto_horario, _ = _info_horario_analise()
            with st.spinner(f"Buscando dados de {ticker_limpo}..."):
                resultado = analisar_ativo(ticker_limpo, tipo_individual, periodo_individual, fonte_yahoo)
                try:
                    dividendos_individual = fonte_yahoo.baixar_dividendos(ticker_limpo)
                except Exception:
                    dividendos_individual = pd.Series(dtype=float)

            st.session_state["individual_ticker"] = ticker_limpo
            st.session_state["individual_tipo"] = tipo_individual
            st.session_state["individual_periodo"] = periodo_individual
            st.session_state["individual_horario"] = texto_horario
            st.session_state["individual_resultado"] = resultado
            st.session_state["individual_dividendos"] = dividendos_individual

    if "individual_resultado" in st.session_state:
        ticker_limpo = st.session_state["individual_ticker"]
        resultado = st.session_state["individual_resultado"]

        st.caption(f"🕒 Análise executada em {st.session_state['individual_horario']}")

        if resultado is None:
            st.error(
                f"Não foi possível obter dados suficientes para {ticker_limpo}. "
                "Confira se o ticker está certo (tickers da B3 precisam do sufixo '.SA')."
            )
        else:
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Sinal de timing", resultado.sinal_timing)
            col_b.metric("Qualidade da renda", resultado.qualidade_renda)
            col_c.metric("Preço atual", f"{resultado.tec.preco_atual:.2f}")

            st.markdown(f"**Leitura combinada:** {resultado.leitura_combinada_texto}")

            col_det1, col_det2 = st.columns(2)
            with col_det1:
                if resultado.detalhes_timing:
                    st.markdown("**Timing (compra/venda):**")
                    for d in resultado.detalhes_timing:
                        st.markdown(f"{_renderizar_detalhe(d)}")
            with col_det2:
                if resultado.detalhes_renda:
                    st.markdown("**Qualidade da renda:**")
                    for d in resultado.detalhes_renda:
                        st.markdown(f"{_renderizar_detalhe(d)}")

            st.markdown(f"**Histórico de preço ({OPCOES_PERIODO[st.session_state['individual_periodo']]}):**")
            st.caption(
                "Linha azul = preço · laranja = MM50 · vermelho = MM200 · linhas tracejadas = "
                "mínima/máxima do período (percentil 5%/95%) · triângulos verdes = dividendo "
                "pago. Painel de baixo: RSI(14), com faixas de referência em 30 e 70."
            )
            grafico = _grafico_preco_completo(resultado.tec.historico, st.session_state["individual_dividendos"])
            st.altair_chart(grafico, width='stretch')

# =============================================================================
# ABA 3 — Dashboard da Carteira (extrato por ativo)
# =============================================================================
with tab_dashboard:
    st.subheader("💼 Dashboard da Carteira")
    st.caption(
        "Extrato por ativo: posição inicial (planilha), compras e vendas feitas depois, "
        "posição final (com preço de mercado atual) e renda recebida."
    )

    if arquivo is None:
        st.info("Envie um arquivo Excel na aba 'Análise de Carteira' para ver o dashboard.")
    else:
        tem_saldo_final = "carteira_linhas_ganho" in st.session_state
        if not tem_saldo_final:
            st.caption(
                "⚠️ Saldo final e renda recebida dependem de preço de mercado atual — clique em "
                "'Analisar carteira' na aba 'Análise de Carteira' pra completar essas colunas. "
                "Posição inicial, compras e vendas já aparecem abaixo, direto da planilha."
            )

        linhas_dashboard = construir_dashboard_por_ativo(
            df_resumo, df_operacoes,
            linhas_ganho=st.session_state.get("carteira_linhas_ganho"),
            tipos=st.session_state.get("carteira_tipos_confirmados", {}),
            data_analise=st.session_state.get("carteira_data_analise"),
        )

        linhas_tabela_dash = []
        for l in linhas_dashboard:
            linhas_tabela_dash.append({
                "Ticker": l.ticker,
                "Saldo Inicial (valor)": (
                    f"{l.moeda} {l.saldo_inicial_valor:,.2f}" if l.saldo_inicial_valor is not None else "-"
                ),
                "Saldo Inicial (data)": l.saldo_inicial_data.strftime("%d/%m/%Y") if l.saldo_inicial_data else "-",
                "Compras (qtde)": f"{l.compras_quantidade:.0f}" if l.compras_quantidade else "-",
                "Compras (valor)": f"{l.moeda} {l.compras_valor:,.2f}" if l.compras_valor else "-",
                "Vendas (qtde)": f"{l.vendas_quantidade:.0f}" if l.vendas_quantidade else "-",
                "Vendas (valor)": f"{l.moeda} {l.vendas_valor:,.2f}" if l.vendas_valor else "-",
                "Saldo Final (valor)": (
                    f"{l.moeda} {l.saldo_final_valor:,.2f}" if l.saldo_final_valor is not None else "-"
                ),
                "Saldo Final (data)": l.saldo_final_data.strftime("%d/%m/%Y") if l.saldo_final_data else "-",
                f"{linhas_dashboard[0].rotulo_renda if linhas_dashboard else 'Renda'}": (
                    f"{l.moeda} {l.renda_recebida:,.2f}" if l.renda_recebida is not None else "-"
                ),
            })

        if linhas_tabela_dash:
            nome_coluna_renda = linhas_dashboard[0].rotulo_renda if linhas_dashboard else "Renda"
            colunas_moeda_dash = [
                "Saldo Inicial (valor)", "Compras (valor)", "Vendas (valor)",
                "Saldo Final (valor)", nome_coluna_renda,
            ]
            _renderizar_tabela_alinhada(pd.DataFrame(linhas_tabela_dash), colunas_moeda_dash)
        else:
            st.info("Nenhum ativo encontrado na planilha.")

# =============================================================================
# ABA 4 — Guia de Indicadores
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
            "**Distância da mínima/máxima do período** — compara o preço atual com o "
            "percentil 5% (mínima) e 95% (máxima) de todo o histórico buscado. Perto da "
            "mínima pode ser oportunidade ou pode ser queda continuada; perto da máxima, "
            "o oposto."
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

    st.info(
        "Nenhum desses indicadores, isolado ou em conjunto, garante retorno — são regras "
        "de bolso genéricas de mercado, não uma recomendação de investimento. Este painel "
        "não substitui um assessor/consultor licenciado (CVM)."
    )
