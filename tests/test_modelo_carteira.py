"""O modelo oferecido para download na aba Carteira precisa ser lido pelo
próprio painel sem erros nem avisos."""
from pathlib import Path

import pandas as pd

from carteira_analise.planilha import (NOMES_ABA_OPERACOES, NOMES_ABA_RESUMO, achar_aba, linhas_resumo_por_ticker,
                                       normalizar_aba_operacoes, normalizar_aba_resumo, normalizar_ticker,
                                       operacoes_do_ticker, tipos_da_planilha)

MODELO = Path(__file__).resolve().parent.parent / "modelos" / "modelo_carteira.xlsx"


def _ler(caminho):
    abas = pd.read_excel(caminho, sheet_name=None)
    return (normalizar_aba_resumo(achar_aba(abas, NOMES_ABA_RESUMO)),
            normalizar_aba_operacoes(achar_aba(abas, NOMES_ABA_OPERACOES)))


def test_modelo_existe_e_tem_as_tres_abas():
    abas = pd.read_excel(MODELO, sheet_name=None)
    assert list(abas) == ["Resumo", "Operações", "Instruções"]


def test_modelo_e_lido_sem_avisos():
    resumo, ops = _ler(MODELO)
    tipos, avisos = tipos_da_planilha(resumo, ops)
    assert not avisos
    tickers = {normalizar_ticker(t)[0] for t in resumo["ticker"]} | {normalizar_ticker(t)[0] for t in ops["ticker"]}
    assert set(tipos) == tickers  # todos com tipo de ativo
    por_ticker = linhas_resumo_por_ticker(resumo)
    ops = ops.assign(_t=[normalizar_ticker(t)[0] for t in ops["ticker"]])
    for ticker in tickers:
        grupo = ops[ops["_t"] == ticker]
        lista, avisos_op, _ = operacoes_do_ticker(por_ticker.get(ticker), grupo if len(grupo) else None)
        assert lista and not avisos_op, (ticker, avisos_op)


def test_gerador_reproduz_o_modelo(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "gerar_modelo", Path(__file__).resolve().parent.parent / "scripts" / "gerar_modelo_carteira.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    novo = mod.gerar(tmp_path / "m.xlsx")
    for a, b in zip(_ler(novo), _ler(MODELO)):
        pd.testing.assert_frame_equal(a, b)
