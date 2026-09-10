"""Painel web de análise de carteira (ações, FIIs, ETFs) — Streamlit.

Interface acessível sem conhecimento técnico de programação (Seção 3.2.2
do Relatório do Projeto Final de Curso): o usuário faz upload da planilha
e vê o resultado, sem células de código nem ambiente de notebook.

Toda a lógica de cálculo é reaproveitada do pacote ``carteira_analise``
(motor de indicadores, pontuação, ganhos e planilha), já testado
(169 testes automatizados) — este arquivo é só a camada de interface.

Resultados ficam guardados em ``st.session_state`` (não só dentro do
`if st.button(...)`) para que rodar UMA análise não apague a outra — todo
clique reexecuta o script inteiro do zero, e sem essa persistência o
resultado anterior desapareceria da tela.
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
    normalizar_aba_operacoes,
    normalizar_aba_resumo,
    normalizar_ticker,
    processar_carteira_combinada,
)

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


def _heuristica_tipo(ticker: str) -> str:
    """Chute inicial do tipo de ativo, a partir do formato do ticker —
    o usuário confirma ou corrige antes de rodar a análise. Tickers B3
    terminados em '11' são frequentemente FIIs ou ETFs (ambíguo só pelo
    ticker); o padrão aqui é FII, por ser o caso mais comum."""
    if not ticker.endswith(".SA"):
        return "acao_us"
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


def _grafico_preco_pt_br(historico: pd.Series) -> alt.Chart:
    """Gráfico de preço com meses abreviados em português (JAN, FEV, MAR...)
    no eixo — o Vega-Lite (motor por trás do Altair) não tem locale pt-BR
    embutido, então a tradução é feita via expressão direta no eixo, sem
    depender de configuração de locale."""
    df = historico.reset_index()
    df.columns = ["data", "preco"]
    expressao_mes_pt = (
        "['JAN','FEV','MAR','ABR','MAI','JUN','JUL','AGO','SET','OUT','NOV','DEZ']"
        "[month(datum.value)] + '/' + (year(datum.value) % 100)"
    )
    return (
        alt.Chart(df)
        .mark_line(color="#1f77b4")
        .encode(
            x=alt.X("data:T", title=None, axis=alt.Axis(labelExpr=expressao_mes_pt, labelAngle=0)),
            y=alt.Y("preco:Q", title=None),
        )
        .properties(height=300)
    )


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

    tickers_encontrados: list[str] = []
    if df_resumo is not None:
        tickers_encontrados += [normalizar_ticker(t)[0] for t in df_resumo["ticker"]]
    if df_operacoes is not None:
        tickers_encontrados += [normalizar_ticker(t)[0] for t in df_operacoes["ticker"]]
    tickers_encontrados = sorted(set(tickers_encontrados))

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

    if st.button("📊 Analisar carteira", type="primary"):
        texto_horario, pregao_provavelmente_aberto = _info_horario_analise()

        with st.spinner("Buscando dados de mercado e calculando... isso pode levar um tempo."):
            linhas_ganho = processar_carteira_combinada(df_resumo, df_operacoes, fonte_yahoo)
            linhas_analise = []
            for ticker in tickers_encontrados:
                tipo = tipos_confirmados[ticker]
                resultado = analisar_ativo(ticker, tipo, "2y", fonte_yahoo)
                linhas_analise.append((ticker, tipo, resultado))

        # Guarda tudo na sessão — é isso que faz o resultado sobreviver a um
        # clique posterior no botão "Analisar ativo" (ver docstring do arquivo).
        st.session_state["carteira_horario"] = texto_horario
        st.session_state["carteira_pregao_aberto"] = pregao_provavelmente_aberto
        st.session_state["carteira_linhas_ganho"] = linhas_ganho
        st.session_state["carteira_linhas_analise"] = linhas_analise

    # ---- Exibição: roda sempre que houver resultado guardado, mesmo que o
    # rerun atual tenha sido disparado pelo outro botão (ativo específico). ----
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

        st.dataframe(pd.DataFrame(linhas_tabela), width='stretch', hide_index=True)

        with st.expander("Ver detalhes da pontuação de cada ativo"):
            for ticker, tipo, resultado in linhas_analise:
                if resultado is None:
                    continue
                st.markdown(f"**{ticker}** ({NOMES_TIPO[tipo]})")
                if resultado.detalhes_timing:
                    st.markdown("Timing (compra/venda):")
                    for d in resultado.detalhes_timing:
                        st.markdown(f"- {d}")
                if resultado.detalhes_renda:
                    st.markdown("Qualidade da renda:")
                    for d in resultado.detalhes_renda:
                        st.markdown(f"- {d}")
                st.divider()

        st.subheader("💰 Ganho da carteira")
        abertas = [l for l in linhas_ganho if l.situacao == "Aberta"]
        encerradas = [l for l in linhas_ganho if l.situacao == "Encerrada"]
        com_erro = [l for l in linhas_ganho if l.situacao == "erro"]

        if abertas:
            st.markdown(f"**Carteira atual — {len(abertas)} posição(ões) em aberto**")
            st.dataframe(pd.DataFrame([_linha_ganho_para_dict(l) for l in abertas]),
                         width='stretch', hide_index=True)

        if encerradas:
            st.markdown(f"**Posições encerradas — {len(encerradas)} ativo(s)**")
            st.dataframe(pd.DataFrame([_linha_ganho_para_dict(l) for l in encerradas]),
                         width='stretch', hide_index=True)

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
else:
    st.info("Envie um arquivo Excel para começar a análise.")

st.divider()
st.subheader("🔍 Analisar um ativo específico")
st.caption(
    "Quer olhar um ativo que não está na sua carteira, ou só conferir um antes de "
    "decidir? Digite o ticker abaixo — não precisa estar em nenhuma planilha."
)

col_ticker, col_tipo = st.columns([2, 1])
with col_ticker:
    ticker_individual = st.text_input("Ticker", placeholder="ex: PETR4.SA, MXRF11.SA, AAPL", key="ticker_individual")
with col_tipo:
    chute_individual = _heuristica_tipo(ticker_individual.strip().upper()) if ticker_individual.strip() else "acao"
    tipo_individual = st.selectbox(
        "Tipo", options=OPCOES_TIPO, index=OPCOES_TIPO.index(chute_individual),
        format_func=lambda t: NOMES_TIPO[t], key="tipo_individual",
    )

if st.button("🔍 Analisar ativo", type="secondary"):
    ticker_limpo = ticker_individual.strip().upper()
    if not ticker_limpo:
        st.warning("Digite um ticker antes de analisar.")
    else:
        texto_horario, _ = _info_horario_analise()
        with st.spinner(f"Buscando dados de {ticker_limpo}..."):
            resultado = analisar_ativo(ticker_limpo, tipo_individual, "2y", fonte_yahoo)

        # Guarda na sessão — mesmo motivo do bloco da carteira acima.
        st.session_state["individual_ticker"] = ticker_limpo
        st.session_state["individual_tipo"] = tipo_individual
        st.session_state["individual_horario"] = texto_horario
        st.session_state["individual_resultado"] = resultado

if "individual_resultado" in st.session_state:
    ticker_limpo = st.session_state["individual_ticker"]
    tipo_individual_salvo = st.session_state["individual_tipo"]
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
                    st.markdown(f"- {d}")
        with col_det2:
            if resultado.detalhes_renda:
                st.markdown("**Qualidade da renda:**")
                for d in resultado.detalhes_renda:
                    st.markdown(f"- {d}")

        st.markdown("**Histórico de preço (últimos 2 anos):**")
        st.altair_chart(_grafico_preco_pt_br(resultado.tec.historico), width='stretch')
