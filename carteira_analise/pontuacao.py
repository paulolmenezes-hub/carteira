"""Motor de pontuação: transforma indicadores em sinais interpretáveis.

Todas as regras (limiares, pesos) ficam centralizadas neste módulo, à
parte do cálculo dos indicadores em si — facilita auditar, ajustar ou (no
Horizonte 3 do roadmap) substituir por um modelo calibrado estatisticamente
sem tocar nos módulos de indicadores.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .fundamentalistas import HistoricoDividendos, tem_imoveis
from .tecnicos import IndicadoresTecnicos

TipoAtivo = str  # "acao" | "fii"

# Dividend yield "atrativo" (revisão de 04/11/2026). No mercado brasileiro, a
# régua acompanha os juros: um yield de 6% é muito com a Selic a 2% e pouco
# com a Selic a 15%. O limite é uma fração do CDI dos últimos 12 meses,
# líquido de IR (proventos são isentos); as frações foram calibradas para dar
# os limites fixos antigos (6% ações, 8% FIIs, 4% ETFs B3) com o CDI de
# nov/2026, ~12,3% líquido — hoje o motor se comporta como antes, mas se
# ajusta quando os juros mudarem. O piso evita uma régua irrisória com juros
# muito baixos. Nos EUA (sem CDI), os limites continuam fixos.
FATOR_DY_CDI = {"acao": 0.50, "fii": 0.65, "etf_br": 0.33}
PISO_DY = {"acao": 0.03, "fii": 0.04, "etf_br": 0.02}
LIMITE_DY_FIXO = {"acao": 0.06, "fii": 0.08, "etf_br": 0.04, "acao_us": 0.02, "etf_us": 0.015}


def limite_dy(tipo: TipoAtivo, cdi_liquido_aa: float | None = None) -> float:
    """Yield mínimo para o "+1 yield atrativo": fração do CDI líquido (B3) ou
    limite fixo (EUA, ou sem o CDI do Banco Central)."""
    if cdi_liquido_aa and tipo in FATOR_DY_CDI:
        return max(FATOR_DY_CDI[tipo] * cdi_liquido_aa, PISO_DY[tipo])
    return LIMITE_DY_FIXO.get(tipo, 0.06)


def _texto_limite(tipo: TipoAtivo, cdi_liquido_aa: float | None) -> str:
    lim = limite_dy(tipo, cdi_liquido_aa)
    if cdi_liquido_aa and tipo in FATOR_DY_CDI:
        return (f"régua {lim * 100:.1f}% = {FATOR_DY_CDI[tipo] * 100:.0f}% do CDI líquido de "
                f"{cdi_liquido_aa * 100:.1f}%").replace(".", ",")
    return f"régua {lim * 100:.1f}%".replace(".", ",")


@dataclass
class ResultadoPontuacao:
    pontos: int | None
    detalhes: list[str] = field(default_factory=list)


def _valores_extremos_periodo(tec: IndicadoresTecnicos) -> tuple[float, float]:
    """Recalcula o valor de preço da mínima e da máxima do período (não só
    a distância percentual já guardada em IndicadoresTecnicos) — pra poder
    mostrar os números reais nos detalhes de pontuação, não só o sinal de
    '+1/-1 próximo da mínima/máxima', que sozinho pode parecer contraditório
    quando os dois disparam ao mesmo tempo (ativo pouco volátil, intervalo
    estreito entre mínima e máxima — ver Seção 4.2 do Relatório)."""
    minima = tec.preco_atual / (1 + tec.dist_minima_pct / 100)
    maxima = tec.preco_atual / (1 + tec.dist_maxima_pct / 100)
    return minima, maxima


def _detalhes_extremos(tec: IndicadoresTecnicos) -> tuple[str | None, str | None]:
    """Monta as duas mensagens de detalhe (mínima e máxima), com os valores
    reais entre parênteses — devolve (None, None) pra quem não deveria
    pontuar. Compartilhada pelas quatro funções de pontuação (ação, ação
    EUA, FII, ETF), que usam o mesmo critério técnico de distância do
    período.

    "Mínima"/"máxima" aqui são, na verdade, o percentil 5%/95% do período
    (ver tecnicos.py), não o menor/maior preço já visto — por isso o preço
    atual pode ficar ABAIXO da "mínima" (ou ACIMA da "máxima"), e não só
    perto dela. Nesse caso o texto muda de "próximo de" para "abaixo de"/
    "acima de", pra não sugerir que o preço está só se aproximando de um
    patamar que na verdade ele já ultrapassou."""
    minima, maxima = _valores_extremos_periodo(tec)
    detalhe_minima = detalhe_maxima = None

    def br(v: float) -> str:
        return f"{v:.2f}".replace(".", ",")

    # "Topo/piso usual" deixa claro que é o percentil 95%/5% do período, e não o
    # maior/menor preço: ficar acima do topo usual não é bater um recorde.
    if tec.dist_minima_pct < 15:
        verbo = "abaixo do" if tec.preco_atual < minima else "perto do"
        detalhe_minima = (
            f"+1 {verbo} piso usual do período (só 5% dos fechamentos ficaram abaixo de "
            f"{br(minima)}; hoje {br(tec.preco_atual)})"
        )
    if tec.dist_maxima_pct > -5:
        verbo = "acima do" if tec.preco_atual > maxima else "perto do"
        detalhe_maxima = (
            f"-1 {verbo} topo usual do período (95% dos fechamentos ficaram abaixo de "
            f"{br(maxima)}; hoje {br(tec.preco_atual)})"
        )
    return detalhe_minima, detalhe_maxima


def pontuar_acao(
    tec: IndicadoresTecnicos, fund: dict, div: HistoricoDividendos | None,
    cdi_liquido_aa: float | None = None,
) -> ResultadoPontuacao:
    pontos = 0
    detalhes: list[str] = []

    if tec.sma200 is not None:
        if tec.preco_atual > tec.sma200:
            pontos += 1
            detalhes.append("+1 preço acima da MM200 (tendência de alta)")
        else:
            pontos -= 1
            detalhes.append("-1 preço abaixo da MM200 (tendência de baixa)")

    if tec.rsi14 is not None:
        if tec.rsi14 < 40:
            pontos += 1
            detalhes.append("+1 RSI < 40 (não sobrecomprado)")
        elif tec.rsi14 > 70:
            pontos -= 1
            detalhes.append("-1 RSI > 70 (sobrecomprado)")

    detalhe_minima, detalhe_maxima = _detalhes_extremos(tec)
    if detalhe_minima:
        pontos += 1
        detalhes.append(detalhe_minima)
    if detalhe_maxima:
        pontos -= 1
        detalhes.append(detalhe_maxima)

    pl = fund.get("pl")
    if pl is not None and pl > 0:
        if pl < 15:
            pontos += 1
            detalhes.append(f"+1 P/L baixo ({pl:.1f})")
        elif pl > 25:
            pontos -= 1
            detalhes.append(f"-1 P/L alto ({pl:.1f})")

    pvp = fund.get("pvp")
    if pvp is not None:
        if pvp < 1.5:
            pontos += 1
            detalhes.append(f"+1 P/VP baixo ({pvp:.2f})")
        elif pvp > 4:
            pontos -= 1
            detalhes.append(f"-1 P/VP alto ({pvp:.2f})")

    roe = fund.get("roe")
    if roe is not None and roe > 0.12:
        pontos += 1
        detalhes.append(f"+1 ROE saudável ({roe * 100:.1f}%)")

    # Indicadores exclusivos do Fundamentus (só B3, não disponíveis via Yahoo
    # Finance) — ver Seção 3.2.2 do Relatório.
    roic = fund.get("roic")
    if roic is not None and roic > 0.15:
        pontos += 1
        detalhes.append(f"+1 ROIC saudável ({roic * 100:.1f}%)")

    margem_liquida = fund.get("margem_liquida")
    if margem_liquida is not None and margem_liquida > 0.10:
        pontos += 1
        detalhes.append(f"+1 margem líquida saudável ({margem_liquida * 100:.1f}%)")

    # Setor financeiro (bancos, seguradoras) tem estrutura de balanço
    # diferente de empresas comuns — o Fundamentus mostra "-" (não
    # aplicável) para dívida bruta/patrimônio e liquidez corrente nesses
    # casos, o que nossa tabela recebe como 0.00, não como ausente. Trata
    # exatamente 0.0 como dado ausente aqui: nenhuma empresa real tem
    # endividamento ou liquidez corrente exatamente zero (mesmo padrão do
    # caso RECR11/FII de papel — ver Seção 4.2 do Relatório).
    divida_patrim = fund.get("divida_bruta_patrimonio")
    if divida_patrim is not None and divida_patrim > 0:
        if divida_patrim < 0.5:
            pontos += 1
            detalhes.append(f"+1 baixo endividamento (dívida/patrimônio {divida_patrim:.2f})")
        elif divida_patrim > 1.5:
            pontos -= 1
            detalhes.append(f"-1 alto endividamento (dívida/patrimônio {divida_patrim:.2f})")

    liquidez_corrente = fund.get("liquidez_corrente")
    if liquidez_corrente is not None and liquidez_corrente > 0:
        if liquidez_corrente > 1.5:
            pontos += 1
            detalhes.append(f"+1 liquidez de curto prazo saudável ({liquidez_corrente:.2f})")
        elif liquidez_corrente < 1.0:
            pontos -= 1
            detalhes.append(f"-1 liquidez de curto prazo apertada ({liquidez_corrente:.2f})")

    dy = div.dy_12m_real if div else None
    if dy is not None and dy > limite_dy("acao", cdi_liquido_aa):
        pontos += 1
        detalhes.append(f"+1 yield efetivo (12m) atrativo ({dy * 100:.1f}%; {_texto_limite('acao', cdi_liquido_aa)})")

    return ResultadoPontuacao(pontos, detalhes)


def pontuar_acao_us(
    tec: IndicadoresTecnicos, fund: dict, div: HistoricoDividendos | None
) -> ResultadoPontuacao:
    """Mesma lógica técnica de pontuar_acao, mas com limiares fundamentalistas
    próprios para o mercado americano — historicamente os múltiplos de P/L e
    P/VP do S&P 500 rodam mais altos que os do Ibovespa."""
    pontos = 0
    detalhes: list[str] = []

    if tec.sma200 is not None:
        if tec.preco_atual > tec.sma200:
            pontos += 1
            detalhes.append("+1 preço acima da MM200 (tendência de alta)")
        else:
            pontos -= 1
            detalhes.append("-1 preço abaixo da MM200 (tendência de baixa)")

    if tec.rsi14 is not None:
        if tec.rsi14 < 40:
            pontos += 1
            detalhes.append("+1 RSI < 40 (não sobrecomprado)")
        elif tec.rsi14 > 70:
            pontos -= 1
            detalhes.append("-1 RSI > 70 (sobrecomprado)")

    detalhe_minima, detalhe_maxima = _detalhes_extremos(tec)
    if detalhe_minima:
        pontos += 1
        detalhes.append(detalhe_minima)
    if detalhe_maxima:
        pontos -= 1
        detalhes.append(detalhe_maxima)

    pl = fund.get("pl")
    if pl is not None and pl > 0:
        if pl < 25:
            pontos += 1
            detalhes.append(f"+1 P/L baixo p/ padrão americano ({pl:.1f})")
        elif pl > 40:
            pontos -= 1
            detalhes.append(f"-1 P/L alto mesmo p/ padrão americano ({pl:.1f})")

    pvp = fund.get("pvp")
    if pvp is not None:
        if pvp < 4:
            pontos += 1
            detalhes.append(f"+1 P/VP baixo p/ padrão americano ({pvp:.2f})")
        elif pvp > 10:
            pontos -= 1
            detalhes.append(f"-1 P/VP alto mesmo p/ padrão americano ({pvp:.2f})")

    roe = fund.get("roe")
    if roe is not None and roe > 0.15:
        pontos += 1
        detalhes.append(f"+1 ROE saudável ({roe * 100:.1f}%)")

    dy = div.dy_12m_real if div else None
    if dy is not None and dy > 0.02:
        pontos += 1
        detalhes.append(f"+1 yield efetivo (12m) atrativo p/ padrão americano ({dy * 100:.1f}%)")

    return ResultadoPontuacao(pontos, detalhes)


def pontuar_etf(
    tec: IndicadoresTecnicos, div: HistoricoDividendos | None, tipo: TipoAtivo,
    cdi_liquido_aa: float | None = None,
) -> ResultadoPontuacao:
    """ETFs não têm P/L, P/VP nem ROE (não são empresas, são cestas de ativos que
    seguem um índice) — critério deliberadamente mais simples: só técnico + yield,
    com limiar de yield próprio por mercado (B3 costuma distribuir mais que EUA)."""
    pontos = 0
    detalhes: list[str] = []

    if tec.sma200 is not None:
        if tec.preco_atual > tec.sma200:
            pontos += 1
            detalhes.append("+1 preço acima da MM200 (tendência de alta)")
        else:
            pontos -= 1
            detalhes.append("-1 preço abaixo da MM200 (tendência de baixa)")

    if tec.rsi14 is not None:
        if tec.rsi14 < 40:
            pontos += 1
            detalhes.append("+1 RSI < 40 (não sobrecomprado)")
        elif tec.rsi14 > 70:
            pontos -= 1
            detalhes.append("-1 RSI > 70 (sobrecomprado)")

    detalhe_minima, detalhe_maxima = _detalhes_extremos(tec)
    if detalhe_minima:
        pontos += 1
        detalhes.append(detalhe_minima)
    if detalhe_maxima:
        pontos -= 1
        detalhes.append(detalhe_maxima)

    dy = div.dy_12m_real if div else None
    if dy is not None and dy > limite_dy(tipo, cdi_liquido_aa):
        pontos += 1
        detalhes.append(f"+1 yield efetivo (12m) atrativo p/ ETF ({dy * 100:.1f}%; {_texto_limite(tipo, cdi_liquido_aa)})")

    return ResultadoPontuacao(pontos, detalhes)


def pontuar_fii(
    tec: IndicadoresTecnicos, fund: dict, div: HistoricoDividendos | None,
    cdi_liquido_aa: float | None = None,
) -> ResultadoPontuacao:
    pontos = 0
    detalhes: list[str] = []

    if tec.sma200 is not None:
        if tec.preco_atual > tec.sma200:
            pontos += 1
            detalhes.append("+1 cota acima da MM200")
        else:
            pontos -= 1
            detalhes.append("-1 cota abaixo da MM200")

    if tec.rsi14 is not None:
        if tec.rsi14 < 40:
            pontos += 1
            detalhes.append("+1 RSI < 40 (não sobrecomprado)")
        elif tec.rsi14 > 70:
            pontos -= 1
            detalhes.append("-1 RSI > 70 (sobrecomprado)")

    detalhe_minima, detalhe_maxima = _detalhes_extremos(tec)
    if detalhe_minima:
        pontos += 1
        detalhes.append(detalhe_minima)
    if detalhe_maxima:
        pontos -= 1
        detalhes.append(detalhe_maxima)

    pvp = fund.get("pvp")
    if pvp is not None:
        if pvp < 0.95:
            pontos += 1
            detalhes.append(f"+1 cota com desconto (P/VP {pvp:.2f})")
        elif pvp > 1.10:
            pontos -= 1
            detalhes.append(f"-1 cota com ágio (P/VP {pvp:.2f})")

    # Indicadores operacionais exclusivos do Fundamentus (Yahoo Finance não
    # fornece dados de vacância/cap rate de FIIs) — ver Seção 3.2.2 do
    # Relatório. Não se aplica a FIIs "de papel" (recebíveis/CRIs), que não
    # possuem imóveis físicos — vacância e cap rate são conceitos exclusivos
    # de FIIs "de tijolo" (e híbridos). A classificação usa o segmento do
    # Fundamentus (``classificar_fii``); a quantidade de imóveis só vale sem
    # segmento — um fundo de recebíveis com um imóvel recebido em pagamento
    # de dívida (RECR11) não é FII de tijolo.
    eh_fii_de_tijolo = tem_imoveis(fund)

    vacancia = fund.get("vacancia_media")
    if eh_fii_de_tijolo and vacancia is not None:
        if vacancia < 0.05:
            pontos += 1
            detalhes.append(f"+1 vacância baixa ({vacancia * 100:.1f}%)")
        elif vacancia > 0.15:
            pontos -= 1
            detalhes.append(f"-1 vacância alta ({vacancia * 100:.1f}%)")

    cap_rate = fund.get("cap_rate")
    if eh_fii_de_tijolo and cap_rate is not None:
        if cap_rate > 0.08:
            pontos += 1
            detalhes.append(f"+1 cap rate atrativo ({cap_rate * 100:.1f}%)")
        elif cap_rate < 0.05:
            pontos -= 1
            detalhes.append(f"-1 cap rate baixo ({cap_rate * 100:.1f}%)")

    dy = div.dy_12m_real if div else None
    if dy is not None and dy > limite_dy("fii", cdi_liquido_aa):
        pontos += 1
        detalhes.append(f"+1 yield efetivo (12m) atrativo ({dy * 100:.1f}%; {_texto_limite('fii', cdi_liquido_aa)})")

    return ResultadoPontuacao(pontos, detalhes)


def classificar(pontos: int) -> str:
    """Descreve o que os indicadores técnicos mostram — não é uma
    recomendação de compra/venda, é uma leitura do conjunto de sinais.
    Cabe ao usuário decidir o que fazer com essa informação."""
    if pontos >= 3:
        return "🟢 Indicadores técnicos majoritariamente favoráveis"
    elif pontos >= 1:
        return "🟡 Indicadores técnicos mistos, leve viés favorável"
    elif pontos >= -1:
        return "🟡 Indicadores técnicos mistos, sem direção clara"
    return "🔴 Indicadores técnicos majoritariamente desfavoráveis"


def avaliar_renda(
    div: HistoricoDividendos | None, tipo: TipoAtivo, cdi_liquido_aa: float | None = None,
    regularidade: dict | None = None,
) -> ResultadoPontuacao:
    """Avalia a QUALIDADE da renda passiva de um ativo, com base no
    histórico real de dividendos pagos — independente de o momento ser bom
    pra comprar ou vender. ``cdi_liquido_aa``: régua do yield ligada aos
    juros (``limite_dy``). ``regularidade`` (ações, ver
    ``regularidade_dividendos``): com 5 anos fechados de histórico, substitui
    a contagem de pagamentos dos últimos 12 meses — Regular +1, Irregular -1."""
    if div is None:
        if tipo == "acao_us":
            # Comum em ações de crescimento americanas (reinvestem lucro em
            # vez de distribuir) — "dados insuficientes" seria enganoso.
            return ResultadoPontuacao(
                pontos="NA",
                detalhes=[
                    "ativo não tem histórico de dividendos na fonte de dados — comum em "
                    "ações de crescimento que reinvestem lucro em vez de distribuir"
                ],
            )
        if tipo in ("etf_br", "etf_us"):
            return ResultadoPontuacao(
                pontos="NA",
                detalhes=[
                    "ETF não tem histórico de distribuição de proventos na fonte de dados — "
                    "pode ser um ETF de acumulação, um provento não capturado como 'dividendo' "
                    "pela fonte de dados (comum em ETFs de renda fixa), ou dado indisponível"
                ],
            )
        return ResultadoPontuacao(
            pontos=None, detalhes=["sem histórico de dividendos suficiente na fonte de dados"]
        )

    pontos = 0
    detalhes: list[str] = []
    limite = limite_dy(tipo, cdi_liquido_aa)
    if div.dy_12m_real is not None:
        dy = div.dy_12m_real
        if dy > limite:
            pontos += 1
            detalhes.append(f"+1 yield efetivo dos últimos 12m atrativo ({dy * 100:.1f}%; "
                            f"{_texto_limite(tipo, cdi_liquido_aa)})")
        elif dy < limite / 2:
            pontos -= 1
            detalhes.append(f"-1 yield efetivo dos últimos 12m baixo ({dy * 100:.1f}%; menos da metade da "
                            f"{_texto_limite(tipo, cdi_liquido_aa)})")

    if div.crescimento_yoy is not None:
        cresc = div.crescimento_yoy
        if cresc > 0.05:
            pontos += 1
            detalhes.append(f"+1 dividendos crescendo vs. ano anterior ({cresc * 100:+.1f}%)")
        elif cresc < -0.15:
            pontos -= 1
            detalhes.append(f"-1 dividendos caindo vs. ano anterior ({cresc * 100:+.1f}%)")

    if regularidade and tipo == "acao":
        reg = regularidade
        med = f"{reg['mediana'] * 100:.1f}%".replace(".", ",")
        if reg["selo"] == "regular":
            pontos += 1
            detalhes.append(f"+1 dividendos regulares: pagou nos {reg['n_anos']} últimos anos, sem ano fraco "
                            f"(mediana {med})")
        elif reg["selo"] == "irregular":
            pontos -= 1
            detalhes.append(f"-1 dividendos irregulares: {reg['n_anos'] - reg['pagou']} ano(s) sem pagar nos "
                            f"{reg['n_anos']} últimos")
        else:
            detalhes.append(f"dividendos oscilantes: pagou nos {reg['n_anos']} últimos anos, mas "
                            f"{reg['ano_pior']} rendeu menos da metade da mediana ({med})")
        return ResultadoPontuacao(pontos, detalhes)

    pagamentos_esperados = 10 if tipo == "fii" else 2
    if div.n_pagamentos_12m >= pagamentos_esperados:
        pontos += 1
        detalhes.append(f"+1 pagamentos regulares ({div.n_pagamentos_12m} nos últimos 12m)")
    elif div.n_pagamentos_12m == 0:
        pontos -= 1
        detalhes.append("-1 nenhum pagamento nos últimos 12 meses")

    return ResultadoPontuacao(pontos, detalhes)


def classificar_renda(pontos: int | str | None) -> str:
    if pontos is None:
        return "⚪ dados de dividendos insuficientes"
    if pontos == "NA":
        return "⚪ Não aplicável — foco em crescimento, não renda"
    if pontos >= 2:
        return "🟢 renda saudável e crescente"
    elif pontos >= 0:
        return "🟡 renda estável"
    return "🔴 atenção: renda em deterioração"


def leitura_combinada(pontos_timing: int, pontos_renda: int | str | None) -> str:
    """Combina o sinal técnico com a qualidade da renda numa LEITURA
    descritiva do conjunto de indicadores — não é uma recomendação de
    compra/venda/manutenção; descreve o que os números mostram, cabendo ao
    usuário decidir o que fazer com essa informação. (Renomeada de
    sugerir_estrategia(): o nome antigo sugeria uma ação; o conteúdo sempre
    foi, e continua sendo, só leitura de dado — ver Seção 3.2.1 do
    Relatório do Projeto Final de Curso, sobre comunicação estritamente
    informativa como mitigação de risco regulatório.)"""
    if pontos_renda == "NA":
        pontos_renda = None  # "não aplicável" é neutro pra fins desta leitura

    if pontos_renda is not None and pontos_renda < 0:
        return "🔴 Renda em deterioração — indicador mais relevante que o preço neste caso"

    if pontos_timing <= -2:
        if pontos_renda is None or pontos_renda >= 0:
            return "🔵 Preço no percentil superior do período, com renda estável"
        return "🔴 Preço no percentil superior do período, sem histórico de renda confiável"

    if pontos_timing >= 3:
        return "🟢 Indicadores técnicos e de renda apontam na mesma direção favorável"

    if pontos_renda is not None and pontos_renda >= 2:
        return "🟡 Renda saudável, sem sinal técnico forte no momento"

    return "🟡 Sem sinal claro — monitorar"
