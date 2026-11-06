"""O notebook importa do pacote (Fase 1, 11/2026): estes testes garantem que
toda importação do notebook continua existindo no pacote — renomear ou
remover uma função do pacote sem ajustar o notebook quebra aqui, e não no
Colab do usuário."""
from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parent.parent / "analise_carteira_acoes_fiis.ipynb"


def _celulas_de_codigo():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for i, celula in enumerate(nb["cells"]):
        if celula["cell_type"] == "code":
            fonte = "".join(celula["source"])
            # comandos de shell do Colab (!pip ...) não são Python
            fonte = "\n".join("#" + l if l.lstrip().startswith(("!", "%")) else l for l in fonte.split("\n"))
            yield i, fonte


def test_celulas_do_notebook_sao_python_valido():
    for i, fonte in _celulas_de_codigo():
        ast.parse(fonte, filename=f"célula {i}")


def test_importacoes_do_pacote_existem():
    faltando = []
    for i, fonte in _celulas_de_codigo():
        for no in ast.walk(ast.parse(fonte)):
            if isinstance(no, ast.ImportFrom) and (no.module or "").startswith("carteira_analise"):
                modulo = importlib.import_module(no.module)
                for nome in no.names:
                    if not hasattr(modulo, nome.name):
                        try:
                            importlib.import_module(f"{no.module}.{nome.name}")
                        except ImportError:
                            faltando.append(f"célula {i}: {no.module}.{nome.name}")
    assert not faltando, faltando


def test_notebook_baixa_o_pacote_na_secao_1():
    fontes = dict(_celulas_de_codigo())
    assert "VERSAO_PACOTE" in fontes[1] and "import carteira_analise" in fontes[1]
