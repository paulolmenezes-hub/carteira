import numpy as np
import pandas as pd
import pytest
from regra_posicao import *


def est(q=100, pm=100.0):
    return novo_estado(q, pm)

# ---------------- decidir_posicao: faixas ----------------
@pytest.mark.parametrize("preco,acao,fracao", [
    (95, 'manter', 0), (86, 'manter', 0), (85, 'comprar', 0.10), (80, 'comprar', 0.10),
    (75, 'comprar', 0.25), (60, 'comprar', 0.25),
    (105, 'manter', 0), (115, 'manter', 0), (124.9, 'manter', 0),
    (125, 'vender', 0.10), (135, 'vender', 0.20), (145, 'vender', 0.30),
    (160, 'vender', 0.40), (200, 'vender', 1.0), (350, 'vender', 1.0),
])
def test_faixas(preco, acao, fracao):
    d = decidir_posicao(preco, est())
    assert d['acao'] == acao
    assert d['fracao'] == pytest.approx(fracao)
    assert d['motivo']

def test_sem_posicao():
    assert decidir_posicao(10, novo_estado(0, 0))['acao'] == 'sem_posicao'

def test_salto_de_faixas_vende_a_maior():
    d = decidir_posicao(150, est())
    assert d['acao'] == 'vender' and d['faixa'] == 0.45

def test_faixa_venda_dispara_uma_vez():
    e = est()
    aplicar_venda(e, 10, 126)
    assert e['faixa_venda_executada'] == 0.25
    d = decidir_posicao(128, e)
    assert d['acao'] == 'manter' and '+35%' in d['motivo']
    assert decidir_posicao(136, e)['acao'] == 'vender'

def test_venda_nao_altera_preco_medio():
    e = est()
    aplicar_venda(e, 30, 130)
    assert e['preco_medio'] == 100 and e['quantidade'] == 70

def test_venda_total_zera_estado():
    e = est()
    aplicar_venda(e, 100, 200)
    assert e['quantidade'] == 0 and e['preco_medio'] == 0
    assert decidir_posicao(200, e)['acao'] == 'sem_posicao'

def test_faixas_venda_liberadas_ao_voltar_ao_pm():
    e = est(); aplicar_venda(e, 10, 126)
    aplicar_resets_por_preco(e, 100)
    assert e['faixa_venda_executada'] == 0
    assert decidir_posicao(126, e)['acao'] == 'vender'

def test_compra_recalcula_pm_e_libera_faixas_venda():
    e = est(); e['faixa_venda_executada'] = 0.35
    aplicar_compra(e, 10, 85, eh_reforco=True)
    assert e['preco_medio'] == pytest.approx((100*100 + 10*85) / 110)
    assert e['faixa_venda_executada'] == 0 and e['n_reforcos'] == 1 and e['preco_ultimo_reforco'] == 85

# ---------------- limite de reforços ----------------
def test_limite_de_2_reforcos():
    e = est()
    for p in (85, 83.5):
        d = decidir_posicao(p, e); assert d['acao'] == 'comprar'
        aplicar_compra(e, e['quantidade'] * d['fracao'], p, True)
    assert e['n_reforcos'] == 2
    d = decidir_posicao(70, e)
    assert d['acao'] == 'manter_nao_aumente' and '2 vezes' in d['motivo']

def test_referencia_eh_pm_atual():
    # após reforço a 85, PM ~98,64: a 85 a variação é ~-13,8% => manter
    e = est(); aplicar_compra(e, 10, 85, True)
    assert decidir_posicao(85, e)['acao'] == 'manter'
    assert decidir_posicao(83.8, e)['acao'] == 'comprar'

def test_reset_reforcos_a_mais_5_pct():
    e = est(); aplicar_compra(e, 10, 85, True); aplicar_compra(e, 11, 83, True)
    pm = e['preco_medio']
    aplicar_resets_por_preco(e, pm * 1.049); assert e['n_reforcos'] == 2
    aplicar_resets_por_preco(e, pm * 1.05); assert e['n_reforcos'] == 0 and e['preco_ultimo_reforco'] is None

def test_sem_limite_permite_cascata():
    e = est(); e['n_reforcos'] = 5
    assert decidir_posicao(80, e, aplicar_limite_reforcos=False)['acao'] == 'comprar'

def test_distancia_minima_opcional():
    cfg = {'distancia_minima_nova_compra': 0.10}
    e = est(); aplicar_compra(e, 10, 85, True)
    d = decidir_posicao(83.8, e, cfg)
    assert d['acao'] == 'manter_nao_aumente' and '76.50' in d['motivo']
    assert decidir_posicao(76, e, cfg)['acao'] == 'comprar'
    # padrão: desligada
    assert CONFIG_REGRA_PADRAO['distancia_minima_nova_compra'] is None

def test_bloqueio_por_filtro():
    d = decidir_posicao(80, est(), compra_permitida=False, motivos_bloqueio=['a dívida está alta'])
    assert d['acao'] == 'manter_nao_aumente' and 'dívida' in d['motivo']

def test_filtro_nao_afeta_venda():
    assert decidir_posicao(130, est(), compra_permitida=False)['acao'] == 'vender'

# ---------------- quantidade ----------------
def test_quantidade_inteira_e_posicao_pequena():
    e = est(q=43)
    assert quantidade_para_acao(e, decidir_posicao(126, e)) == 4
    e2 = est(q=3)
    assert quantidade_para_acao(e2, decidir_posicao(126, e2)) == 0
    assert quantidade_para_acao(e2, decidir_posicao(200, e2)) == 3
    assert quantidade_para_acao(e, decidir_posicao(100, e)) == 0
    assert quantidade_para_acao(est(q=7.5), {'acao': 'comprar', 'fracao': 0.1}, True) == pytest.approx(0.75)

# ---------------- reconstrução a partir de operações ----------------
def op(d, t, q, p): return {'data': pd.Timestamp(d), 'tipo': t, 'quantidade': q, 'preco': p}

def test_estado_de_operacoes_conta_reforcos():
    ops = [op('2024-01-02', 'compra', 100, 100), op('2024-03-01', 'compra', 10, 85),
           op('2024-04-01', 'compra', 11, 83)]
    e = estado_a_partir_de_operacoes(ops)
    assert e['n_reforcos'] == 2 and e['quantidade'] == 121
    assert decidir_posicao(70, e)['acao'] == 'manter_nao_aumente'

def test_compra_acima_do_pm_nao_eh_reforco():
    e = estado_a_partir_de_operacoes([op('2024-01-02', 'compra', 100, 100), op('2024-02-01', 'compra', 50, 110)])
    assert e['n_reforcos'] == 0

def test_ordem_das_operacoes_nao_importa():
    ops = [op('2024-04-01', 'compra', 11, 83), op('2024-01-02', 'compra', 100, 100)]
    assert estado_a_partir_de_operacoes(ops)['n_reforcos'] == 1

def test_historico_zera_reforcos():
    ops = [op('2024-01-02', 'compra', 100, 100), op('2024-03-01', 'compra', 10, 85)]
    datas = pd.bdate_range('2024-01-02', '2024-06-28')
    precos = pd.Series(np.linspace(100, 110, len(datas)), index=datas)
    e = estado_a_partir_de_operacoes(ops, precos)
    assert e['n_reforcos'] == 0

def test_historico_sem_recuperacao_mantem_reforcos():
    ops = [op('2024-01-02', 'compra', 100, 100), op('2024-03-01', 'compra', 10, 85)]
    datas = pd.bdate_range('2024-01-02', '2024-06-28')
    precos = pd.Series(np.linspace(100, 80, len(datas)), index=datas)
    assert estado_a_partir_de_operacoes(ops, precos)['n_reforcos'] == 1

def test_venda_registra_faixa():
    ops = [op('2024-01-02', 'compra', 100, 100), op('2024-05-01', 'venda', 10, 127)]
    e = estado_a_partir_de_operacoes(ops)
    assert e['faixa_venda_executada'] == 0.25 and e['quantidade'] == 90

def test_venda_abaixo_da_faixa_nao_registra():
    e = estado_a_partir_de_operacoes([op('2024-01-02', 'compra', 100, 100), op('2024-05-01', 'venda', 10, 110)])
    assert e['faixa_venda_executada'] == 0

# ---------------- filtros ----------------
def divs_mensais(inicio, fim, valor):
    idx = pd.date_range(inicio, fim, freq='MS')
    return pd.Series(valor, index=idx, dtype=float)

def test_filtro_renda_estavel_permite():
    assert filtro_renda_compra(divs_mensais('2022-01-01', '2024-06-01', 1.0), '2024-06-15') == (True, [])

def test_filtro_renda_queda_bloqueia():
    d = pd.concat([divs_mensais('2022-07-01', '2023-06-01', 1.0), divs_mensais('2023-07-01', '2024-06-01', 0.7)])
    ok, m = filtro_renda_compra(d, '2024-06-15')
    assert not ok and '30%' in m[0]

def test_filtro_renda_parou_de_pagar():
    ok, m = filtro_renda_compra(divs_mensais('2022-07-01', '2023-06-01', 1.0), '2024-07-15')
    assert not ok and 'parou' in m[0]

def test_filtro_renda_nao_pagador_e_vazio():
    assert filtro_renda_compra(pd.Series(dtype=float), '2024-01-01') == (True, [])
    assert filtro_renda_compra(None, '2024-01-01') == (True, [])
    assert filtro_renda_compra(divs_mensais('2024-01-01', '2024-06-01', 1.0), '2024-06-15') == (True, [])

def test_filtro_renda_tz():
    d = divs_mensais('2022-01-01', '2024-06-01', 1.0); d.index = d.index.tz_localize('America/Sao_Paulo')
    assert filtro_renda_compra(d, '2024-06-15')[0]

def test_filtro_fundamentos_acao():
    ok, m = filtro_fundamentos_compra('acao', {'roe': -0.05, 'divida_bruta_patrimonio': 2.0, 'liquidez_corrente': 0.8})
    assert not ok and len(m) == 3
    assert filtro_fundamentos_compra('acao', {'roe': 0.2, 'divida_bruta_patrimonio': 0.0, 'liquidez_corrente': 0.0})[0]
    assert filtro_fundamentos_compra('acao', {})[0]

def test_filtro_fundamentos_acao_us():
    assert not filtro_fundamentos_compra('acao_us', {'roe': -0.1})[0]
    assert filtro_fundamentos_compra('acao_us', {'roe': 0.1, 'divida_bruta_patrimonio': 9})[0]

def test_filtro_fii_tijolo_e_papel():
    tijolo = {'pvp': 0.9, 'vacancia_media': 0.20, 'qtd_imoveis': 10, 'segmento': 'Logística'}
    ok, m = filtro_fundamentos_compra('fii', tijolo)
    assert not ok and 'vazios' in m[0]
    papel = {'pvp': 0.9, 'vacancia_media': 1.0, 'qtd_imoveis': 0, 'segmento': 'Títulos e Val. Mob.'}
    assert filtro_fundamentos_compra('fii', papel)[0]
    ok, m = filtro_fundamentos_compra('fii', {'pvp': 1.2, 'qtd_imoveis': 5, 'segmento': 'Shoppings'})
    assert not ok and 'P/VP' in m[0]

def test_etf_sem_filtro():
    d = divs_mensais('2022-07-01', '2023-06-01', 1.0)
    assert filtro_fundamentos_compra('etf_br', {}, d, '2024-07-15') == (True, [])

def test_filtro_fundamentos_inclui_renda():
    d = divs_mensais('2022-07-01', '2023-06-01', 1.0)
    ok, m = filtro_fundamentos_compra('fii', {'pvp': 0.9, 'qtd_imoveis': 0}, d, '2024-07-15')
    assert not ok and 'parou' in m[0]

# ---------------- simulação ----------------
def serie(valores, inicio='2020-01-01'):
    return pd.Series(valores, index=pd.bdate_range(inicio, periods=len(valores)), dtype=float)

def test_segurar_ganho_simples():
    s = serie([100.0] * 260 + [150.0] * 30)
    r = simular_estrategia(s, estrategia='segurar')
    assert r['ganho_pct'] == pytest.approx(0.5) and r['n_compras'] == 0

def test_historico_curto_devolve_none():
    assert simular_estrategia(serie([100.0] * 100)) is None

def test_regra_vende_na_alta():
    s = serie([100.0] * 253 + list(np.linspace(100, 210, 60)))
    r = simular_estrategia(s, estrategia='regra')
    assert r['n_vendas'] == 5 and r['n_compras'] == 0
    assert r['valor_final'] > r['aportado']

def test_regra_limita_reforcos_e_sem_limite_nao():
    s = serie([100.0] * 253 + list(np.linspace(100, 40, 120)))
    com = simular_estrategia(s, estrategia='regra')
    sem = simular_estrategia(s, estrategia='regra', usar_limite_reforcos=False)
    assert com['n_compras'] == 2
    assert sem['n_compras'] > 2 and sem['aporte_extra'] > com['aporte_extra']

def test_distancia_minima_espaca_compras():
    s = serie([100.0] * 253 + list(np.linspace(100, 80, 60)))
    padrao = simular_estrategia(s, estrategia='regra')
    dist = simular_estrategia(s, estrategia='regra', config={'distancia_minima_nova_compra': 0.10})
    assert padrao['n_compras'] == 2 and dist['n_compras'] == 1

def test_filtro_renda_bloqueia_na_simulacao():
    s = serie([100.0] * 253 + list(np.linspace(100, 70, 60)), inicio='2021-01-01')
    d = divs_mensais('2019-01-01', '2021-06-01', 1.0)  # parou de pagar
    r = simular_estrategia(s, d, estrategia='regra', usar_filtro_renda=True)
    assert r['n_compras'] == 0 and r['n_bloqueios'] > 0

def test_dividendos_e_caixa():
    s = serie([100.0] * 300)
    d = pd.Series([1.0], index=[s.index[260]])
    r = simular_estrategia(s, d, estrategia='segurar')
    assert r['ganho'] == pytest.approx(100.0)
    s2 = serie([100.0] * 253 + [130.0] * 253)
    sem = simular_estrategia(s2, estrategia='regra')
    com = simular_estrategia(s2, estrategia='regra', rendimento_caixa_aa=0.10)
    assert com['valor_final'] > sem['valor_final']

def test_dividendo_tz_e_fora_do_periodo():
    s = serie([100.0] * 300)
    d = pd.Series([1.0, 5.0], index=pd.DatetimeIndex([s.index[260], pd.Timestamp('2030-01-01')]).tz_localize('UTC'))
    r = simular_estrategia(s, d, estrategia='segurar')
    assert r['ganho'] == pytest.approx(100.0)

def test_comparar_e_resumir():
    s1 = serie([100.0] * 253 + list(np.linspace(100, 210, 60)))
    s2 = serie([100.0] * 253 + list(np.linspace(100, 60, 60)))
    res = {'A.SA': comparar_variantes(s1), 'B.SA': comparar_variantes(s2), 'C': {}}
    assert len(res['A.SA']) == 5
    df = resumir_por_tipo(res, {'A.SA': 'FII', 'B.SA': 'FII', 'C': 'ETF EUA'})
    assert set(df['tipo']) == {'FII'}
    linha = df[df['variante'] == 'Só segurar'].iloc[0]
    assert linha['venceu_segurar'] is None or pd.isna(linha['venceu_segurar'])
    frases = texto_simples_por_tipo(df)
    assert len(frases) == 1 and 'R$ 100' in frases[0] and 'de 2' in frases[0]

def test_texto_veredito():
    df = pd.DataFrame([
        {'tipo': 'X', 'variante': 'Só segurar', 'n_ativos': 1, 'valor_de_100': 110, 'venceu_segurar': None},
        {'tipo': 'X', 'variante': 'Regra (padrão: até 2 reforços)', 'n_ativos': 1, 'valor_de_100': 120, 'venceu_segurar': 1},
        {'tipo': 'Y', 'variante': 'Só segurar', 'n_ativos': 1, 'valor_de_100': 120, 'venceu_segurar': None},
        {'tipo': 'Y', 'variante': 'Regra (padrão: até 2 reforços)', 'n_ativos': 1, 'valor_de_100': 110, 'venceu_segurar': 0},
        {'tipo': 'Z', 'variante': 'Só segurar', 'n_ativos': 1, 'valor_de_100': 110, 'venceu_segurar': None},
        {'tipo': 'Z', 'variante': 'Regra (padrão: até 2 reforços)', 'n_ativos': 1, 'valor_de_100': 110.5, 'venceu_segurar': 1},
        {'tipo': 'W', 'variante': 'Só segurar', 'n_ativos': 1, 'valor_de_100': 110, 'venceu_segurar': None},
    ])
    f = texto_simples_por_tipo(df)
    assert 'ajudou' in f[0] and 'segurar foi melhor' in f[1] and 'empate' in f[2] and len(f) == 3

# ---------------- card ----------------
def test_card_venda_com_renda():
    e = est(q=40, pm=100)
    d = decidir_posicao(127, e)
    c = montar_card_posicao('HGLG11.SA', e, 127, d, quantidade_para_acao(e, d), renda_12m_por_cota=12.0)
    assert c['posicao'] == '💰 Vender 4 cota(s)' and c['prioridade'] == 0
    assert 'Lucro estimado nesta venda: R$ 108.00' in c['linhas'][0]
    assert 'R$ 40.00 por mês' in c['linhas'][1] and 'R$ 36.00' in c['linhas'][1]
    assert '+27%' in c['titulo']

def test_card_vender_tudo():
    e = est(q=10); d = decidir_posicao(210, e)
    c = montar_card_posicao('X', e, 210, d, quantidade_para_acao(e, d))
    assert c['posicao'].startswith('💰 Vender tudo (10')

def test_card_compra():
    e = est(q=100); d = decidir_posicao(84, e)
    c = montar_card_posicao('X', e, 84, d, quantidade_para_acao(e, d), 12.0, moeda='US$')
    assert c['posicao'] == '🟢 Comprar mais 10 cota(s)' and 'US$ 840.00' in c['linhas'][0]
    assert 'depois da compra' in c['linhas'][1]

def test_card_posicao_pequena_vira_manter():
    e = est(q=3); d = decidir_posicao(126, e)
    c = montar_card_posicao('X', e, 126, d, quantidade_para_acao(e, d))
    assert c['posicao'] == '⚪ Manter' and c['acao'] == 'manter' and 'pequena demais' in c['linhas'][1]

def test_card_manter_e_nao_aumente_e_sem_posicao():
    e = est()
    c = montar_card_posicao('X', e, 110, decidir_posicao(110, e), 0, 0)
    assert c['posicao'] == '⚪ Manter' and len(c['linhas']) == 1
    d = decidir_posicao(80, e, compra_permitida=False, motivos_bloqueio=['a dívida está alta'])
    c = montar_card_posicao('X', e, 80, d, 0)
    assert c['posicao'] == '🟡 Manter, sem aumentar' and c['prioridade'] == 1
    z = novo_estado(0, 0)
    assert montar_card_posicao('X', z, 10, decidir_posicao(10, z), 0)['prioridade'] == 2
