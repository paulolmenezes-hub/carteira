"""Relatório da carteira em PDF: um resumo para ler fora do painel, com o que
as abas Visão geral, Dashboard, Metas e Proventos mostram.

Não faz I/O de rede nem usa Streamlit: recebe um dicionário já montado pelo
app (ver ``gerar_relatorio``) e devolve os bytes do PDF. Gráficos com
matplotlib (backend Agg), páginas com reportlab.

Informação para investigar, não recomendação: as seções de "pontos de
atenção" repetem as regras que o próprio usuário definiu (faixas sobre o
preço médio, metas, limites) e os avisos do painel; não sugerem ativos.
"""
from __future__ import annotations

import io
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,  # noqa: E402
                                Table, TableStyle)

AZUL, LARANJA, VERDE, VERMELHO, CINZA = "#1e3a8a", "#d9831c", "#0f7f73", "#b91c1c", "#6b7280"
COR_FAIXA = {"realizacao": "#2f45e0", "queda": LARANJA, "fora": VERDE}
NOME_FAIXA = {"realizacao": "Realização", "queda": "Queda", "fora": "Fora"}
_MES = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]
AVISO = ("Informação para investigar, não recomendação de investimento. A decisão é sempre sua. Este relatório "
         "não substitui um assessor ou consultor registrado na CVM.")


# ---------------------------------------------------------------- formatação
def reais(v, moeda: str = "R$", sinal: bool = False) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    s = f"{abs(v):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    pre = ("+" if v > 0 else "−" if v < 0 else "") if sinal else ("−" if v < 0 else "")
    return f"{pre}{moeda} {s}"


def pct(v, casas: int = 1, sinal: bool = True) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    s = f"{abs(v) * 100:.{casas}f}".replace(".", ",")
    pre = ("+" if v > 0 else "−" if v < 0 else "") if sinal else ("−" if v < 0 else "")
    return f"{pre}{s}%"


def pp(v) -> str:
    return f"{abs(v) * 100:.1f}".replace(".", ",") + " p.p."


def rotulo_mes(p) -> str:
    p = pd.Period(p, "M")
    return f"{_MES[p.month - 1]}/{str(p.year)[2:]}"


# ---------------------------------------------------------------- gráficos
def _png(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _eixo(ax, n):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#d1d5db")
    ax.tick_params(colors="#374151", labelsize=7)
    ax.grid(axis="y", color="#eef2f7", linewidth=.8)
    passo = max(1, n // 8)
    return passo


def grafico_patrimonio(serie: pd.DataFrame) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    x = list(range(len(serie)))
    ax.plot(x, serie["patrimonio"], color=AZUL, lw=2, label="Patrimônio")
    ax.plot(x, serie["custo"], color="#9ca3af", lw=1.6, ls="--", label="Valor investido")
    passo = _eixo(ax, len(x))
    ax.set_xticks(x[::passo], [rotulo_mes(p) for p in serie.index][::passo])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f} mil"))
    ax.legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    return _png(fig)


def grafico_rentabilidade(serie: pd.DataFrame) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    x = list(range(len(serie)))
    ax.plot(x, serie["rentabilidade"] * 100, color=AZUL, lw=2, label="Carteira")
    if "cdi" in serie and serie["cdi"].notna().any():
        ax.plot(x, serie["cdi"] * 100, color=LARANJA, lw=1.6, ls="--", label="CDI")
    if "ibov" in serie and serie["ibov"].notna().any():
        ax.plot(x, serie["ibov"] * 100, color=VERDE, lw=1.6, ls="--", label="Ibovespa")
    ax.axhline(0, color="#9ca3af", lw=.6)
    passo = _eixo(ax, len(x))
    ax.set_xticks(x[::passo], [rotulo_mes(p) for p in serie.index][::passo])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
    return _png(fig)


def grafico_proventos(por_mes: pd.DataFrame) -> io.BytesIO:
    """``por_mes``: índice = meses, colunas = classes (Ações, FIIs, ETFs...)."""
    fig, ax = plt.subplots(figsize=(7.2, 2.3))
    x = list(range(len(por_mes)))
    base = [0.0] * len(x)
    cores = {"Ações": AZUL, "FIIs": VERDE, "ETFs": LARANJA, "EUA": "#7c3aed"}
    for col in por_mes.columns:
        vals = por_mes[col].fillna(0).tolist()
        ax.bar(x, vals, bottom=base, color=cores.get(col, CINZA), width=.7, label=col)
        base = [b + v for b, v in zip(base, vals)]
    passo = _eixo(ax, len(x))
    ax.set_xticks(x[::passo], [rotulo_mes(p) for p in por_mes.index][::passo])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"R$ {v:,.0f}".replace(",", ".")))
    ax.legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4)
    return _png(fig)


# ---------------------------------------------------------------- estilos
def _estilos():
    base = getSampleStyleSheet()
    return {
        "titulo": ParagraphStyle("t", parent=base["Title"], fontSize=18, textColor=colors.HexColor(AZUL),
                                 alignment=TA_LEFT, spaceAfter=2),
        "sub": ParagraphStyle("s", parent=base["Normal"], fontSize=8.5, textColor=colors.HexColor(CINZA)),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=12, textColor=colors.HexColor(AZUL),
                             spaceBefore=10, spaceAfter=4),
        "txt": ParagraphStyle("x", parent=base["Normal"], fontSize=8.5, leading=11.5),
        "nota": ParagraphStyle("n", parent=base["Normal"], fontSize=7, leading=9, textColor=colors.HexColor(CINZA)),
        "cel": ParagraphStyle("c", parent=base["Normal"], fontSize=7.5, leading=9),
        "kpi_r": ParagraphStyle("kr", parent=base["Normal"], fontSize=7, textColor=colors.HexColor(CINZA)),
        "kpi_v": ParagraphStyle("kv", parent=base["Normal"], fontSize=12.5, leading=15, fontName="Helvetica-Bold"),
    }


def _tabela(cab, linhas, larguras, alinhar_dir=(), est=None, cores_linha=None):
    dados = [[Paragraph(f"<b>{c}</b>", est["cel"]) for c in cab]]
    for ln in linhas:
        dados.append([Paragraph(str(v), est["cel"]) for v in ln])
    t = Table(dados, colWidths=larguras, repeatRows=1)
    estilo = [("LINEBELOW", (0, 0), (-1, 0), .6, colors.HexColor("#d1d5db")),
              ("LINEBELOW", (0, 1), (-1, -1), .3, colors.HexColor("#eef2f7")),
              ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
              ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    for i in alinhar_dir:
        estilo.append(("ALIGN", (i, 0), (i, -1), "RIGHT"))
    for (li, cor) in (cores_linha or []):
        estilo.append(("BACKGROUND", (0, li), (-1, li), colors.HexColor(cor)))
    t.setStyle(TableStyle(estilo))
    for linha in dados:
        for j, p in enumerate(linha):
            if j in alinhar_dir:
                p.style = ParagraphStyle("cr", parent=p.style, alignment=2)
    return t


def _kpis(itens, est):
    """``itens``: [(rótulo, valor, detalhe, cor)] em uma linha de caixas."""
    celulas = []
    for rot, val, det, cor in itens:
        celulas.append([Paragraph(rot, est["kpi_r"]),
                        Paragraph(f"<font color='{cor or '#111827'}'>{val}</font>", est["kpi_v"]),
                        Paragraph(det or "", est["nota"])])
    t = Table([celulas], colWidths=[18 * cm / len(itens)] * len(itens))
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), .4, colors.HexColor("#e5e7eb")),
                           ("INNERGRID", (0, 0), (-1, -1), .4, colors.HexColor("#e5e7eb")),
                           ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    return t


# ---------------------------------------------------------------- relatório
def gerar_relatorio(d: dict) -> bytes:
    """Monta o PDF. Chaves de ``d`` (todas opcionais, exceto ``resumo``):

    - ``data`` (date), ``periodo`` (texto), ``marca_dagua`` (texto, p. ex. "AMOSTRA")
    - ``resumo``: patrimonio, aportes_liq, proventos, resultado, rentabilidade, cdi, ibov, tir
    - ``serie``: DataFrame mensal (PeriodIndex) com patrimonio, custo, rentabilidade, cdi, ibov
    - ``anos``: lista de dicts ano, rentabilidade, cdi, ibov, resultado, proventos
    - ``alocacao``: lista de dicts classe, meta, hoje, falta, leitura; ``limites``: {maximo, minimo}
    - ``ativos``: lista de dicts ticker, classe, peso, quantidade, preco_medio, preco, variacao, faixa,
      dy, avisos (lista de textos), moeda
    - ``proventos_mes``: DataFrame (meses × classes); ``proventos_ativos``: lista (ticker, valor 12m)
    """
    est = _estilos()
    buf = io.BytesIO()
    marca = d.get("marca_dagua")
    hoje = d.get("data") or date.today()

    def rodape(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 6.5)
        canvas.setFillColor(colors.HexColor(CINZA))
        canvas.drawString(1.5 * cm, 1 * cm, f"Relatório da carteira · {hoje:%d/%m/%Y} · informação para investigar, "
                                            "não recomendação de investimento")
        canvas.drawRightString(A4[0] - 1.5 * cm, 1 * cm, f"{doc.page}")
        if marca:
            canvas.setFont("Helvetica-Bold", 60)
            canvas.setFillColor(colors.Color(.9, .3, .1, alpha=.12))
            canvas.translate(A4[0] / 2, A4[1] / 2)
            canvas.rotate(40)
            canvas.drawCentredString(0, 0, marca)
        canvas.restoreState()

    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.4 * cm,
                            bottomMargin=1.6 * cm, title="Relatório da carteira")
    h = []
    r = d["resumo"]
    h.append(Paragraph("Relatório da carteira", est["titulo"]))
    h.append(Paragraph(f"Gerado em {hoje:%d/%m/%Y} · período {d.get('periodo', '')} · valores em reais "
                       "(EUA pelo dólar de cada mês)", est["sub"]))
    h.append(Spacer(1, 8))

    # 1. resumo
    h.append(Paragraph("1. Resumo", est["h2"]))
    rent_det = []
    if r.get("cdi") is not None:
        dif = r["rentabilidade"] - r["cdi"]
        rent_det.append(f"CDI {pct(r['cdi'])} ({pp(dif)} {'acima' if dif >= 0 else 'abaixo'})")
    if r.get("ibov") is not None:
        dif = r["rentabilidade"] - r["ibov"]
        rent_det.append(f"Ibovespa {pct(r['ibov'])} ({pp(dif)} {'acima' if dif >= 0 else 'abaixo'})")
    if r.get("tir") is not None:
        rent_det.append(f"TIR {pct(r['tir'])} ao ano")
    h.append(_kpis([
        ("Patrimônio", reais(r["patrimonio"]), d.get("periodo_fim", ""), None),
        ("Aportes líquidos", reais(r.get("aportes_liq")), "compras − vendas no período", None),
        ("Proventos", reais(r.get("proventos")), "recebidos no período", VERDE),
        ("Resultado", reais(r.get("resultado"), sinal=True), "patrimônio − aportes + proventos",
         VERDE if (r.get("resultado") or 0) >= 0 else VERMELHO),
        ("Rentabilidade", pct(r["rentabilidade"]), "<br/>".join(rent_det), AZUL),
    ], est))
    h.append(Spacer(1, 6))
    destaques = d.get("destaques") or []
    if destaques:
        h.append(Paragraph("<b>Em poucas palavras</b>", est["txt"]))
        for t in destaques:
            h.append(Paragraph(f"• {t}", est["txt"]))

    # 2. evolução
    serie = d.get("serie")
    if serie is not None and len(serie) > 1:
        h.append(Paragraph("2. Evolução", est["h2"]))
        h.append(Paragraph("Patrimônio × valor investido", est["txt"]))
        h.append(Image(grafico_patrimonio(serie), width=18 * cm, height=6.1 * cm))
        h.append(Paragraph("Rentabilidade acumulada (método das cotas) × CDI"
                           + (" × Ibovespa" if "ibov" in serie and serie["ibov"].notna().any() else ""), est["txt"]))
        h.append(Image(grafico_rentabilidade(serie), width=18 * cm, height=6.1 * cm))
    anos = d.get("anos") or []
    if anos:
        tem_ibov = any(a.get("ibov") is not None for a in anos)
        cab = ["Ano", "Carteira", "CDI", "Diferença"] + (["Ibovespa"] if tem_ibov else []) + ["Resultado", "Proventos"]
        linhas = []
        for a in anos:
            dif = (a["rentabilidade"] - a["cdi"]) if a.get("cdi") is not None else None
            cor = VERDE if (dif or 0) >= 0 else VERMELHO
            ln = [a["rotulo"], pct(a["rentabilidade"]), pct(a.get("cdi")),
                  f"<font color='{cor}'>{('+' if dif >= 0 else '−') + pp(dif)}</font>" if dif is not None else "—"]
            if tem_ibov:
                ln.append(pct(a.get("ibov")))
            ln += [reais(a.get("resultado"), sinal=True), reais(a.get("proventos"))]
            linhas.append(ln)
        n = len(cab)
        h.append(KeepTogether([Paragraph("Rentabilidade por ano", est["txt"]), Spacer(1, 2),
                               _tabela(cab, linhas, [18 * cm / n] * n, alinhar_dir=range(1, n), est=est)]))

    # 3. alocação e metas
    aloc = d.get("alocacao") or []
    if aloc:
        h.append(PageBreak())
        h.append(Paragraph("3. Alocação × metas", est["h2"]))
        cores_leit = {"na meta": VERDE, "cabe nos ativos atuais": VERDE, "classe acima da meta": CINZA}
        linhas = [[f"<b>{a['classe']}</b>", f"{a['meta']:.0f}%", pct(a["hoje"] / 100, 1, False),
                   f"<font color='{VERDE if a['falta'] > 0 else VERMELHO}'>{reais(a['falta'], sinal=True)}</font>"
                   if abs(a["falta"]) >= 1 else "—",
                   f"<font color='{cores_leit.get(a['leitura'], LARANJA)}'>{a['leitura']}</font>"]
                  for a in aloc]
        h.append(_tabela(["Classe", "Meta", "Hoje", "Para chegar à meta", "Leitura"], linhas,
                         [4.2 * cm, 1.8 * cm, 1.8 * cm, 3.6 * cm, 6.6 * cm], alinhar_dir=(1, 2, 3), est=est))
        lim = d.get("limites")
        if lim:
            h.append(Spacer(1, 3))
            h.append(Paragraph(f"Limite por ativo: máximo {lim['maximo']:g}%"
                               + (f", mínimo {lim['minimo']:g}%" if lim.get("minimo") else "")
                               + ". Positivo = quanto aportar na classe para chegar à meta.", est["nota"]))

    # 4. pontos de atenção (as regras do próprio usuário)
    ativos = d.get("ativos") or []
    if ativos:
        h.append(Paragraph("4. Pontos de atenção", est["h2"]))
        h.append(Paragraph("O que as suas regras e os avisos do painel apontam hoje. Para cada ativo, os cenários "
                           "(manter, realizar parte, aumentar) estão no Dashboard do painel.", est["nota"]))
        grupos = [
            ("Faixa de realização (alta de 25% ou mais sobre o preço médio)",
             [a for a in ativos if a.get("faixa") == "realizacao"], "#2f45e0"),
            ("Faixa de queda (queda de 15% ou mais sobre o preço médio)",
             [a for a in ativos if a.get("faixa") == "queda"], LARANJA),
            ("Com avisos (limite por ativo, tendência, saúde)", [a for a in ativos if a.get("avisos")], VERMELHO),
        ]
        for titulo, lista, cor in grupos:
            h.append(Paragraph(f"<font color='{cor}'><b>{titulo}</b></font>: "
                               + (", ".join(f"{a['ticker']} ({pct(a['variacao'], 0)})" if "avisos" not in titulo.lower()
                                            else f"{a['ticker']} ({'; '.join(a['avisos'])})" for a in lista)
                                  if lista else "nenhum"), est["txt"]))
            h.append(Spacer(1, 2))

        # 5. ativos
        h.append(Paragraph("5. Ativos", est["h2"]))
        linhas, fundos = [], []
        for i, a in enumerate(ativos, start=1):
            fx = a.get("faixa")
            moeda = a.get("moeda", "R$")
            linhas.append([f"<b>{a['ticker']}</b>", a["classe"], pct(a["peso"] / 100, 1, False),
                           reais(a.get("preco_medio"), moeda), reais(a.get("preco"), moeda),
                           f"<font color='{VERDE if a['variacao'] >= 0 else VERMELHO}'>{pct(a['variacao'])}</font>",
                           f"<font color='{COR_FAIXA.get(fx, CINZA)}'><b>{NOME_FAIXA.get(fx, '—')}</b></font>",
                           pct(a.get("dy"), 1, False) if a.get("dy") is not None else "—"])
        h.append(_tabela(["Ativo", "Classe", "Peso", "Preço médio", "Preço", "Variação", "Faixa", "DY 12m"], linhas,
                         [2.1 * cm, 2.6 * cm, 1.6 * cm, 2.6 * cm, 2.4 * cm, 2.0 * cm, 2.3 * cm, 2.4 * cm],
                         alinhar_dir=(2, 3, 4, 5, 7), est=est))

    # 6. proventos
    pm = d.get("proventos_mes")
    if pm is not None and len(pm):
        h.append(PageBreak())
        h.append(Paragraph("6. Proventos", est["h2"]))
        total = float(pm.sum().sum())
        h.append(Paragraph(f"Últimos {len(pm)} meses: <b>{reais(total)}</b> (média de {reais(total / len(pm))} por "
                           "mês). Pelo mês da data-ex, com a quantidade da sua planilha.", est["txt"]))
        h.append(Image(grafico_proventos(pm), width=18 * cm, height=5.6 * cm))
        pa = d.get("proventos_ativos") or []
        if pa:
            linhas = [[f"<b>{t}</b>", reais(v), pct(v / total if total else None, 1, False)] for t, v in pa[:12]]
            h.append(KeepTogether([Paragraph("Quem mais pagou", est["txt"]), Spacer(1, 2),
                                   _tabela(["Ativo", "Recebido", "% do total"], linhas,
                                           [4 * cm, 4 * cm, 3 * cm], alinhar_dir=(1, 2), est=est)]))
        h.append(Spacer(1, 4))
        h.append(Paragraph("Para a declaração do IR (dividendos, JCP e rendimentos por ano de pagamento), use a aba "
                           "Proventos e IR do painel e o informe da corretora ou da B3.", est["nota"]))

    h.append(Spacer(1, 10))
    h.append(Paragraph(AVISO, est["nota"]))
    doc.build(h, onFirstPage=rodape, onLaterPages=rodape)
    return buf.getvalue()
