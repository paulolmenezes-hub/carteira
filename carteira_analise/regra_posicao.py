import numpy as np
import pandas as pd

# ============================================================================
# REGRA DE POSIÇÃO POR FAIXAS DE PREÇO
# Compara o preço atual com o PREÇO MÉDIO da posição (no momento da análise)
# e devolve uma posição clara: comprar mais, manter, manter sem aumentar,
# vender parte ou vender tudo.
#   - Compras na queda ("reforços") limitadas a 2 por ciclo; o contador zera
#     quando o preço volta a ficar pelo menos 5% acima do preço médio.
#   - Cada faixa de venda dispara uma vez; as faixas de venda voltam a valer
#     depois de uma nova compra (o preço médio muda) ou quando o preço volta
#     a ficar abaixo do preço médio.
#   - Distância mínima entre reforços: DESLIGADA por padrão (decisão de
#     projeto); existe só como variante de comparação no backtest.
# ============================================================================

CONFIG_REGRA_PADRAO = {
    # (variação mínima de queda, fração da posição a comprar)
    'faixas_compra': [(-0.15, 0.10), (-0.25, 0.25)],
    # (variação mínima de alta, fração da posição a vender)
    'faixas_venda': [(0.25, 0.10), (0.35, 0.20), (0.45, 0.30), (0.60, 0.40), (1.00, 1.00)],
    'max_reforcos': 2,
    'limiar_reset_reforcos': 0.05,   # volta à faixa de "Manter" em alta
    'distancia_minima_nova_compra': None,  # ex: 0.10 = só reforça 10% abaixo do último reforço
}


_EPS = 1e-9  # tolerância de arredondamento (ex: 145/100 - 1 = 0,4499999...)


def novo_estado(quantidade, preco_medio):
    """Estado de uma posição para a regra de faixas."""
    return {
        'quantidade': float(quantidade),
        'preco_medio': float(preco_medio),
        'n_reforcos': 0,
        'preco_ultimo_reforco': None,
        'faixa_venda_executada': 0.0,  # maior faixa de venda já usada no ciclo (0 = nenhuma)
    }


def _cfg(config):
    return CONFIG_REGRA_PADRAO if config is None else {**CONFIG_REGRA_PADRAO, **config}


def aplicar_resets_por_preco(estado, preco, config=None):
    """Zera os contadores quando o preço anda o suficiente:
    - preço >= preço médio × (1 + 5%): zera o contador de reforços;
    - preço <= preço médio: libera de novo as faixas de venda."""
    cfg = _cfg(config)
    pm = estado['preco_medio']
    if estado['quantidade'] <= 0 or pm <= 0:
        return estado
    if preco >= pm * (1 + cfg['limiar_reset_reforcos']):
        estado['n_reforcos'] = 0
        estado['preco_ultimo_reforco'] = None
    if preco <= pm:
        estado['faixa_venda_executada'] = 0.0
    return estado


def decidir_posicao(preco, estado, config=None, compra_permitida=True, motivos_bloqueio=None,
                    aplicar_limite_reforcos=True):
    """Devolve a posição sugerida pela regra para o preço informado.

    Retorno: dict com 'acao' ('comprar', 'vender', 'manter', 'manter_nao_aumente',
    'sem_posicao'), 'fracao' (da quantidade atual), 'variacao' (preço vs preço
    médio), 'faixa' (limiar que disparou) e 'motivo' (texto simples)."""
    cfg = _cfg(config)
    q, pm = estado['quantidade'], estado['preco_medio']
    if q <= 0 or pm <= 0:
        return {'acao': 'sem_posicao', 'fracao': 0.0, 'variacao': None, 'faixa': None,
                'motivo': 'sem posição aberta neste ativo'}

    var = preco / pm - 1

    # ---- Venda: maior faixa atingida ----
    faixas_venda = sorted(cfg['faixas_venda'])
    atingidas = [f for f in faixas_venda if var >= f[0] - _EPS]
    if atingidas:
        limiar, fracao = atingidas[-1]
        if limiar > estado['faixa_venda_executada'] + 1e-12:
            acao = 'vender'
            if fracao >= 1.0:
                motivo = (f"subiu {var*100:.0f}% sobre o seu preço médio — a regra manda vender "
                          f"tudo e embolsar o lucro")
            else:
                motivo = (f"subiu {var*100:.0f}% sobre o seu preço médio — venda {fracao*100:.0f}% "
                          f"da posição para garantir parte do lucro")
            return {'acao': acao, 'fracao': fracao, 'variacao': var, 'faixa': limiar, 'motivo': motivo}
        proximas = [f for f in faixas_venda if f[0] > estado['faixa_venda_executada'] + 1e-12]
        prox_txt = (f" A próxima venda é a partir de +{proximas[0][0]*100:.0f}%." if proximas else "")
        return {'acao': 'manter', 'fracao': 0.0, 'variacao': var, 'faixa': None,
                'motivo': (f"subiu {var*100:.0f}% sobre o seu preço médio, mas você já vendeu "
                           f"nesta faixa.{prox_txt}")}

    # ---- Compra na queda: faixa mais funda atingida ----
    faixas_compra = sorted(cfg['faixas_compra'])  # mais funda primeiro
    atingidas_c = [f for f in faixas_compra if var <= f[0] + _EPS]
    if atingidas_c:
        limiar, fracao = atingidas_c[0]
        if aplicar_limite_reforcos and estado['n_reforcos'] >= cfg['max_reforcos']:
            return {'acao': 'manter_nao_aumente', 'fracao': 0.0, 'variacao': var, 'faixa': limiar,
                    'motivo': (f"caiu {abs(var)*100:.0f}% sobre o seu preço médio, mas você já "
                               f"comprou mais {estado['n_reforcos']} vezes na queda. Espere o preço "
                               f"voltar a subir antes de colocar mais dinheiro.")}
        dist = cfg.get('distancia_minima_nova_compra')
        ult = estado.get('preco_ultimo_reforco')
        if dist is not None and ult is not None and preco > ult * (1 - dist):
            return {'acao': 'manter_nao_aumente', 'fracao': 0.0, 'variacao': var, 'faixa': limiar,
                    'motivo': (f"você já comprou mais a {ult:.2f}; a próxima compra só vale abaixo "
                               f"de {ult * (1 - dist):.2f}.")}
        if not compra_permitida:
            razoes = '; '.join(motivos_bloqueio or []) or 'os fundamentos não confirmam a compra'
            return {'acao': 'manter_nao_aumente', 'fracao': 0.0, 'variacao': var, 'faixa': limiar,
                    'motivo': (f"caiu {abs(var)*100:.0f}% sobre o seu preço médio, mas não é hora de "
                               f"aumentar: {razoes}.")}
        return {'acao': 'comprar', 'fracao': fracao, 'variacao': var, 'faixa': limiar,
                'motivo': (f"caiu {abs(var)*100:.0f}% sobre o seu preço médio — compre mais "
                           f"{fracao*100:.0f}% da posição para baixar o seu preço médio")}

    # ---- Zona de manter ----
    if var >= 0:
        motivo = (f"subiu {var*100:.0f}% sobre o seu preço médio. Ainda não é hora de vender "
                  f"(a regra realiza lucro a partir de +{faixas_venda[0][0]*100:.0f}%).")
    else:
        motivo = (f"caiu {abs(var)*100:.0f}% sobre o seu preço médio. Queda pequena: mantenha "
                  f"(a regra só compra mais a partir de {faixas_compra[-1][0]*100:.0f}%).")
    return {'acao': 'manter', 'fracao': 0.0, 'variacao': var, 'faixa': None, 'motivo': motivo}


def aplicar_compra(estado, quantidade, preco, eh_reforco):
    """Atualiza o estado após uma compra (preço médio ponderado)."""
    q_nova = estado['quantidade'] + quantidade
    estado['preco_medio'] = ((estado['preco_medio'] * estado['quantidade'] + preco * quantidade) / q_nova
                             if q_nova > 0 else 0.0)
    estado['quantidade'] = q_nova
    if eh_reforco:
        estado['n_reforcos'] += 1
        estado['preco_ultimo_reforco'] = preco
    estado['faixa_venda_executada'] = 0.0  # preço médio mudou: faixas de venda recomeçam
    return estado


def aplicar_venda(estado, quantidade, preco, config=None):
    """Atualiza o estado após uma venda. Vender não altera o preço médio
    (mesma regra da Receita Federal). Registra a faixa de venda usada."""
    cfg = _cfg(config)
    pm = estado['preco_medio']
    if pm > 0:
        var = preco / pm - 1
        usadas = [f[0] for f in sorted(cfg['faixas_venda']) if var >= f[0] - _EPS]
        if usadas:
            estado['faixa_venda_executada'] = max(estado['faixa_venda_executada'], usadas[-1])
    estado['quantidade'] = max(estado['quantidade'] - quantidade, 0.0)
    if estado['quantidade'] <= 1e-9:
        estado.update(novo_estado(0.0, 0.0))
    return estado


def estado_a_partir_de_operacoes(operacoes, historico_precos=None, config=None):
    """Reconstrói o estado da regra a partir das operações reais do usuário
    (aba Operações/Resumo da planilha). Uma compra feita ABAIXO do preço médio
    da época conta como reforço na queda. Se o histórico de preços for
    informado, os resets (preço voltou a +5% / voltou ao preço médio) são
    aplicados dia a dia entre as operações e até a data mais recente."""
    ops = sorted(operacoes, key=lambda o: pd.Timestamp(o['data']))
    estado = novo_estado(0.0, 0.0)
    hist = None
    if historico_precos is not None and len(historico_precos) > 0:
        hist = historico_precos.dropna().sort_index()

    def _passar_precos(de, ate):
        if hist is None or estado['quantidade'] <= 0:
            return
        trecho = hist[(hist.index > de) & (hist.index <= ate)] if de is not None else hist[hist.index <= ate]
        for p in trecho.values:
            aplicar_resets_por_preco(estado, float(p), config)

    data_anterior = None
    for op in ops:
        data = pd.Timestamp(op['data'])
        _passar_precos(data_anterior, data)
        if op['tipo'] == 'compra':
            eh_reforco = estado['quantidade'] > 0 and op['preco'] < estado['preco_medio']
            aplicar_compra(estado, float(op['quantidade']), float(op['preco']), eh_reforco)
        elif op['tipo'] == 'venda':
            aplicar_venda(estado, float(op['quantidade']), float(op['preco']), config)
        data_anterior = data
    if hist is not None and len(hist) > 0 and data_anterior is not None:
        _passar_precos(data_anterior, hist.index.max())
    return estado


def quantidade_para_acao(estado, decisao, permitir_fracionario=False):
    """Converte a fração sugerida em quantidade de cotas/ações (inteira,
    a menos que o ativo aceite fração). Devolve 0 se a posição for pequena
    demais para a fração pedida."""
    if decisao['acao'] not in ('comprar', 'vender'):
        return 0
    q = estado['quantidade']
    if decisao['acao'] == 'vender' and decisao['fracao'] >= 1.0:
        return q if permitir_fracionario else int(round(q))
    bruto = q * decisao['fracao']
    return bruto if permitir_fracionario else int(round(bruto))


# ---------------------------------------------------------------------------
# Filtros para compras na queda
# ---------------------------------------------------------------------------

def filtro_renda_compra(dividendos, data_ref):
    """Filtro de renda (testável no passado, porque o histórico de dividendos
    existe): bloqueia a compra na queda se o ativo pagava rendimentos e parou
    de pagar, ou se os rendimentos dos últimos 12 meses caíram mais de 15% em
    relação aos 12 meses anteriores. Ativos que não pagam dividendos não são
    bloqueados. Devolve (permitido: bool, motivos: list[str])."""
    if dividendos is None or len(dividendos) == 0:
        return True, []
    d = dividendos
    if getattr(d.index, 'tz', None) is not None:
        d = d.copy()
        d.index = d.index.tz_localize(None)
    data_ref = pd.Timestamp(data_ref)
    ult12 = float(d[(d.index > data_ref - pd.Timedelta(days=365)) & (d.index <= data_ref)].sum())
    ant12 = float(d[(d.index > data_ref - pd.Timedelta(days=730)) &
                    (d.index <= data_ref - pd.Timedelta(days=365))].sum())
    if ant12 <= 0:
        return True, []
    if ult12 <= 0:
        return False, ["o ativo parou de pagar rendimentos nos últimos 12 meses"]
    queda = 1 - ult12 / ant12
    if queda > 0.15:
        return False, [f"os rendimentos caíram {queda*100:.0f}% em relação ao ano anterior"]
    return True, []


def _eh_fii_de_papel(fund):
    """Mesmos 3 sinais da Seção 6 (segmento, qtd_imoveis, vacância extrema)."""
    segmento = str(fund.get('segmento') or '').lower()
    por_segmento = any(p in segmento for p in ('papel', 'recebív', 'recebiv', 'título', 'titulo', 'renda fixa'))
    por_imoveis = fund.get('qtd_imoveis') in (None, 0)
    vac = fund.get('vacancia_media')
    por_vacancia = vac is not None and vac >= 0.999
    return por_segmento or por_imoveis or por_vacancia


def filtro_fundamentos_compra(tipo, fund, dividendos=None, data_ref=None):
    """Filtro usado AO VIVO antes de sugerir compra na queda. Combina o filtro
    de renda com fundamentos atuais (que não têm série histórica gratuita, por
    isso não entram no backtest). ETFs não passam por filtro (são cestas
    diversificadas). Dado ausente não bloqueia. Devolve (permitido, motivos)."""
    if tipo in ('etf_br', 'etf_us'):
        return True, []
    motivos = []
    fund = fund or {}
    if dividendos is not None and data_ref is not None:
        _, motivos_renda = filtro_renda_compra(dividendos, data_ref)
        motivos.extend(motivos_renda)
    if tipo in ('acao', 'acao_us'):
        roe = fund.get('roe')
        if roe is not None and roe <= 0:
            motivos.append("a empresa está dando prejuízo")
    if tipo == 'acao':
        divida = fund.get('divida_bruta_patrimonio')
        if divida is not None and divida > 1.5:  # 0.0 = setor financeiro/não aplicável
            motivos.append(f"a dívida da empresa está alta ({divida:.1f} vezes o patrimônio)")
        liq = fund.get('liquidez_corrente')
        if liq is not None and 0 < liq < 1.0:
            motivos.append("a empresa tem pouco dinheiro para as contas de curto prazo")
    if tipo == 'fii':
        pvp = fund.get('pvp')
        if pvp is not None and pvp > 1.10:
            motivos.append(f"a cota ainda está cara em relação ao patrimônio do fundo (P/VP {pvp:.2f})")
        vac = fund.get('vacancia_media')
        if not _eh_fii_de_papel(fund) and vac is not None and vac > 0.15:
            motivos.append(f"o fundo tem muitos imóveis vazios (vacância de {vac*100:.0f}%)")
    return (len(motivos) == 0), motivos


# ---------------------------------------------------------------------------
# Simulação histórica (backtest da regra)
# ---------------------------------------------------------------------------

def _dividendos_em_pregoes(fechamento, dividendos):
    """Mapeia cada dividendo para o pregão da data (ou o seguinte)."""
    mapa = {}
    if dividendos is None or len(dividendos) == 0:
        return mapa
    d = dividendos
    if getattr(d.index, 'tz', None) is not None:
        d = d.copy()
        d.index = d.index.tz_localize(None)
    idx = fechamento.index
    for data, valor in d.items():
        pos = idx.searchsorted(pd.Timestamp(data))
        if pos < len(idx):
            mapa[pos] = mapa.get(pos, 0.0) + float(valor)
    return mapa


def simular_estrategia(fechamento, dividendos=None, estrategia='regra', capital_inicial=10000.0,
                       rendimento_caixa_aa=0.0, config=None, usar_limite_reforcos=True,
                       usar_filtro_renda=False, inicio_idx=252):
    """Simula, dia a dia, 'segurar' (compra no início e não mexe) ou 'regra'
    (faixas de compra/venda) sobre o histórico de um ativo.

    - Compras na queda usam dinheiro NOVO (aporte), somado ao total aportado.
    - Vendas vão para o caixa, que rende `rendimento_caixa_aa` ao ano.
    - Dividendos recebidos vão para o caixa nas duas estratégias.
    - Quantidades fracionárias (comparação justa entre estratégias).
    Resultado principal: ganho sobre o total aportado."""
    serie = fechamento.dropna()
    if len(serie) <= inicio_idx + 20:
        return None
    serie = serie.iloc[inicio_idx:]
    precos = serie.values.astype(float)
    divs = _dividendos_em_pregoes(serie, dividendos)
    taxa_diaria = (1 + rendimento_caixa_aa) ** (1 / 252) - 1

    p0 = precos[0]
    estado = novo_estado(capital_inicial / p0, p0)
    aportado = capital_inicial
    caixa = 0.0
    n_compras = n_vendas = n_bloqueios = 0

    for i, preco in enumerate(precos):
        if i > 0:
            caixa *= (1 + taxa_diaria)
        if i in divs and estado['quantidade'] > 0:
            caixa += estado['quantidade'] * divs[i]
        if estrategia != 'regra' or i == 0:
            continue
        aplicar_resets_por_preco(estado, preco, config)
        decisao = decidir_posicao(preco, estado, config, aplicar_limite_reforcos=usar_limite_reforcos)
        if decisao['acao'] == 'comprar' and usar_filtro_renda:
            permitido, _ = filtro_renda_compra(dividendos, serie.index[i])
            if not permitido:
                n_bloqueios += 1
                continue
        if decisao['acao'] == 'comprar':
            q = estado['quantidade'] * decisao['fracao']
            aportado += q * preco
            aplicar_compra(estado, q, preco, eh_reforco=True)
            n_compras += 1
        elif decisao['acao'] == 'vender':
            q = estado['quantidade'] * min(decisao['fracao'], 1.0)
            caixa += q * preco
            aplicar_venda(estado, q, preco, config)
            n_vendas += 1

    valor_final = estado['quantidade'] * precos[-1] + caixa
    ganho = valor_final - aportado
    return {
        'estrategia': estrategia, 'data_inicio': serie.index[0], 'data_fim': serie.index[-1],
        'aportado': aportado, 'aporte_extra': aportado - capital_inicial,
        'valor_final': valor_final, 'ganho': ganho, 'ganho_pct': ganho / aportado,
        'n_compras': n_compras, 'n_vendas': n_vendas, 'n_bloqueios': n_bloqueios,
    }


VARIANTES_BACKTEST_REGRA = {
    'Só segurar': dict(estrategia='segurar'),
    'Regra sem limite de reforços': dict(estrategia='regra', usar_limite_reforcos=False),
    'Regra (padrão: até 2 reforços)': dict(estrategia='regra'),
    'Regra + distância mínima de 10%': dict(estrategia='regra',
                                            config={'distancia_minima_nova_compra': 0.10}),
    'Regra + filtro de renda': dict(estrategia='regra', usar_filtro_renda=True),
}


def comparar_variantes(fechamento, dividendos=None, rendimento_caixa_aa=0.0, capital_inicial=10000.0,
                       variantes=None):
    """Roda todas as variantes para um ativo. Devolve {nome: resultado}."""
    variantes = variantes or VARIANTES_BACKTEST_REGRA
    saida = {}
    for nome, params in variantes.items():
        r = simular_estrategia(fechamento, dividendos, capital_inicial=capital_inicial,
                               rendimento_caixa_aa=rendimento_caixa_aa, **params)
        if r is not None:
            saida[nome] = r
    return saida


def resumir_por_tipo(resultados_por_ticker, tipo_por_ticker, referencia='Só segurar',
                     principal='Regra (padrão: até 2 reforços)'):
    """Agrega por tipo de ativo: ganho médio de cada variante (por R$ 100
    aportados) e em quantos ativos cada variante bateu 'só segurar'."""
    linhas = []
    tipos = sorted(set(tipo_por_ticker.get(t) for t in resultados_por_ticker if tipo_por_ticker.get(t)))
    for tipo in tipos:
        tickers = [t for t in resultados_por_ticker if tipo_por_ticker.get(t) == tipo
                   and referencia in resultados_por_ticker[t]]
        if not tickers:
            continue
        nomes = [n for n in resultados_por_ticker[tickers[0]]]
        for nome in nomes:
            validos = [t for t in tickers if nome in resultados_por_ticker[t]]
            if not validos:
                continue
            ganhos = [resultados_por_ticker[t][nome]['ganho_pct'] for t in validos]
            venceu = sum(1 for t in validos if resultados_por_ticker[t][nome]['ganho_pct']
                         > resultados_por_ticker[t][referencia]['ganho_pct'] + 1e-12)
            linhas.append({
                'tipo': tipo, 'variante': nome, 'n_ativos': len(validos),
                'ganho_medio_pct': float(np.mean(ganhos)),
                'valor_de_100': 100 * (1 + float(np.mean(ganhos))),
                'venceu_segurar': venceu if nome != referencia else None,
                'aporte_extra_medio': float(np.mean([resultados_por_ticker[t][nome]['aporte_extra'] for t in validos])),
            })
    return pd.DataFrame(linhas)


def texto_simples_por_tipo(df_resumo, referencia='Só segurar', principal='Regra (padrão: até 2 reforços)'):
    """Frases em linguagem simples, uma por tipo de ativo."""
    frases = []
    for tipo in df_resumo['tipo'].unique():
        sub = df_resumo[df_resumo['tipo'] == tipo].set_index('variante')
        if referencia not in sub.index or principal not in sub.index:
            continue
        ref, reg = sub.loc[referencia], sub.loc[principal]
        n = int(reg['n_ativos'])
        dif = reg['valor_de_100'] - ref['valor_de_100']
        if dif > 1:
            veredito = "👉 a regra ajudou"
        elif dif < -1:
            veredito = "👉 só segurar foi melhor"
        else:
            veredito = "👉 praticamente empate"
        frases.append(
            f"{tipo} ({n} ativo(s)): seguindo a regra, cada R$ 100 investidos viraram "
            f"R$ {reg['valor_de_100']:.0f} em média; só segurando, R$ {ref['valor_de_100']:.0f}. "
            f"A regra ganhou de 'só segurar' em {int(reg['venceu_segurar'])} de {n}. {veredito}.")
    return frases


# ---------------------------------------------------------------------------
# Card em linguagem simples (Seção 11)
# ---------------------------------------------------------------------------

def formatar_valor(valor, moeda='R$'):
    """R$ no padrão brasileiro (R$ 1.234,56); US$ no americano (US$ 1,234.56)
    — mesma convenção do painel."""
    texto = f"{abs(valor):,.2f}"
    if moeda == 'R$':
        texto = texto.replace(',', '_').replace('.', ',').replace('_', '.')
    return f"{moeda} {'-' if valor < 0 else ''}{texto}"


def montar_card_posicao(ticker, estado, preco_atual, decisao, qtd_sugerida, renda_12m_por_cota=None,
                        moeda='R$'):
    """Monta o conteúdo do card de um ativo, em linguagem simples.
    Devolve dict com 'titulo', 'posicao' (rótulo), 'acao' (código final),
    'linhas' (frases) e 'prioridade' (0 = pede ação hoje, 1 = manter)."""
    q, pm = estado['quantidade'], estado['preco_medio']
    acao = decisao['acao']
    linhas = []

    if acao == 'sem_posicao':
        return {'titulo': ticker, 'posicao': '⚪ Sem posição aberta', 'acao': acao,
                'linhas': ['Você não tem mais este ativo na carteira.'], 'prioridade': 2}

    var = decisao['variacao']
    titulo = (f"{ticker} — {q:.0f} cota(s) · preço médio {formatar_valor(pm, moeda)} · "
              f"hoje {formatar_valor(preco_atual, moeda)} ({var*100:+.0f}%)")

    if acao in ('comprar', 'vender') and qtd_sugerida <= 0:
        rotulo = '⚪ Manter'
        linhas.append(decisao['motivo'][0].upper() + decisao['motivo'][1:] + '.')
        linhas.append(f"Mas sua posição é pequena demais para aplicar {decisao['fracao']*100:.0f}% "
                      f"(daria menos de 1 cota). Mantenha.")
        acao_final, prioridade = 'manter', 1
    elif acao == 'comprar':
        rotulo = f"🟢 Comprar mais {qtd_sugerida:.0f} cota(s)"
        linhas.append(decisao['motivo'][0].upper() + decisao['motivo'][1:] +
                      f" (cerca de {formatar_valor(qtd_sugerida * preco_atual, moeda)}).")
        acao_final, prioridade = acao, 0
    elif acao == 'vender':
        tudo = decisao['fracao'] >= 1.0
        rotulo = (f"💰 Vender tudo ({qtd_sugerida:.0f} cota(s))" if tudo
                  else f"💰 Vender {qtd_sugerida:.0f} cota(s)")
        lucro = (preco_atual - pm) * qtd_sugerida
        linhas.append(decisao['motivo'][0].upper() + decisao['motivo'][1:] +
                      f". Lucro estimado nesta venda: {formatar_valor(lucro, moeda)}.")
        acao_final, prioridade = acao, 0
    elif acao == 'manter_nao_aumente':
        rotulo = '🟡 Manter, sem aumentar'
        linhas.append(decisao['motivo'][0].upper() + decisao['motivo'][1:])
        acao_final, prioridade = acao, 1
    else:
        rotulo = '⚪ Manter'
        linhas.append(decisao['motivo'][0].upper() + decisao['motivo'][1:])
        acao_final, prioridade = 'manter', 1

    if renda_12m_por_cota is not None and renda_12m_por_cota > 0:
        renda_mes = renda_12m_por_cota / 12 * q
        frase = f"Renda estimada: cerca de {formatar_valor(renda_mes, moeda)} por mês com as suas {q:.0f} cota(s)"
        if acao_final == 'vender':
            restante = max(q - qtd_sugerida, 0)
            frase += f"; depois da venda, cerca de {formatar_valor(renda_12m_por_cota / 12 * restante, moeda)}"
        elif acao_final == 'comprar':
            frase += f"; depois da compra, cerca de {formatar_valor(renda_12m_por_cota / 12 * (q + qtd_sugerida), moeda)}"
        linhas.append(frase + '.')

    return {'titulo': titulo, 'posicao': rotulo, 'acao': acao_final, 'linhas': linhas, 'prioridade': prioridade}
