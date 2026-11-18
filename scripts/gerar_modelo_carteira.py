"""Gera modelos/modelo_carteira.xlsx, o arquivo que a aba Carteira oferece
para download. Exemplos tirados da carteira do usuário-piloto (Base
Operações, 18/11/2026), sem as linhas vazias e com as vendas em quantidade
positiva. Rodar: python scripts/gerar_modelo_carteira.py"""
from datetime import date
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

SAIDA = Path(__file__).resolve().parent.parent / "modelos" / "modelo_carteira.xlsx"

RESUMO = [
    ("VALE3.SA", 33, 78.28, date(2021, 12, 20), "Ação"),
    ("BRK-B", 1.25, 318.98, date(2023, 5, 25), "Ação"),
    ("IAU", 5.41004, 36.97, date(2023, 5, 25), "ETF"),
    ("NOBL", 2.14184, 93.38, date(2023, 6, 20), "ETF"),
    ("POSB11.SA", 29, 106.51, date(2026, 9, 10), "ETF"),
    ("SCVB11.SA", 70, 76.43, date(2026, 10, 26), "ETF"),
    ("HGLG11.SA", 20, 142.97, date(2019, 7, 24), "FII"),
    ("HGCR11.SA", 49, 108.54, date(2019, 9, 4), "FII"),
]

OPERACOES = [
    ("HGCR11.SA", "Compra", 17, 110.05, date(2019, 12, 18), "FII"),
    ("HGCR11.SA", "Compra", 28, 110.05, date(2020, 1, 23), "FII"),
    ("HGLG11.SA", "Compra", 9, 155.37, date(2020, 7, 30), "FII"),
    ("HGLG11.SA", "Venda", 29, 172.40, date(2021, 4, 1), "FII"),
    ("HGCR11.SA", "Compra", 7, 107.55, date(2022, 4, 12), "FII"),
    ("VALE3.SA", "Compra", 32, 82.99, date(2022, 5, 4), "Ação"),
    ("VALE3.SA", "Compra", 150, 67.78, date(2022, 7, 18), "Ação"),
    ("HGCR11.SA", "Venda", 41, 102.83, date(2022, 8, 10), "FII"),
    ("NOBL", "Compra", 1.08454, 92.20, date(2023, 6, 22), "ETF"),
    ("VALE3.SA", "Venda", 40, 72.24, date(2026, 8, 13), "Ação"),
    ("POSB11.SA", "Compra", 25, 106.87, date(2026, 9, 17), "ETF"),
]

INSTRUCOES = [
    ("Como preencher", None),
    ("1. Apague os exemplos e lance os seus ativos. Mantenha os nomes das abas e dos cabeçalhos.", None),
    ("2. Resumo: o ponto de partida de cada ativo (quantidade, preço médio e data de entrada).", None),
    ("   Com a aba Operações: lance a primeira compra; a posição de hoje o painel calcula.", None),
    ("   Só com o Resumo: lance a posição de hoje, com o preço médio e a data de entrada.", None),
    ("3. Operações (opcional): compras e vendas depois da data do Resumo. Quantidade sempre positiva.", None),
    ("   Eventos: Bonificação (ações recebidas; preço = custo atribuído ou 0), Desdobramento", None),
    ("   (ações recebidas; preço em branco) e Grupamento (ações que deixaram de existir; preço em branco).", None),
    ("4. Ticker: B3 com ou sem .SA (VALE3 ou VALE3.SA); EUA como no Yahoo Finance (BRK-B, NOBL).", None),
    ("5. Tipo Ativo: Ação, FII ou ETF. Preços em reais (B3) ou dólares (EUA). Datas em dd/mm/aaaa.", None),
    ("A planilha enviada fica só na sua sessão do navegador: não é gravada no servidor.", "nota"),
]


def _aba(ws, cabecalho, linhas, larguras, formatos):
    ws.append(cabecalho)
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="1E3A8A")
        c.alignment = Alignment(horizontal="center")
    for linha in linhas:
        ws.append(linha)
    for i, largura in enumerate(larguras):
        ws.column_dimensions[chr(65 + i)].width = largura
    for col, fmt in formatos.items():
        for cel in ws[col][1:]:
            cel.number_format = fmt
    ws.freeze_panes = "A2"


def gerar(saida: Path = SAIDA) -> Path:
    wb = Workbook()
    resumo = wb.active
    resumo.title = "Resumo"
    _aba(resumo, ["Ticker", "Quantidade", "Preço", "Data", "Tipo Ativo"], RESUMO,
         [14, 12, 12, 12, 12], {"C": "#,##0.00", "D": "DD/MM/YYYY"})
    ops = wb.create_sheet("Operações")
    _aba(ops, ["Ticker", "Tipo", "Quantidade", "Preço", "Data", "Tipo Ativo"], OPERACOES,
         [14, 15, 12, 12, 12, 12], {"D": "#,##0.00", "E": "DD/MM/YYYY"})

    tipo_ativo = DataValidation(type="list", formula1='"Ação,FII,ETF"', allow_blank=True)
    resumo.add_data_validation(tipo_ativo)
    tipo_ativo.add("E2:E1000")
    tipo_ativo_ops = DataValidation(type="list", formula1='"Ação,FII,ETF"', allow_blank=True)
    ops.add_data_validation(tipo_ativo_ops)
    tipo_ativo_ops.add("F2:F2000")
    tipo_op = DataValidation(type="list", formula1='"Compra,Venda,Bonificação,Desdobramento,Grupamento"',
                             allow_blank=True)
    ops.add_data_validation(tipo_op)
    tipo_op.add("B2:B2000")

    instr = wb.create_sheet("Instruções")
    instr.column_dimensions["A"].width = 100
    for texto, estilo in INSTRUCOES:
        instr.append([texto])
        cel = instr.cell(instr.max_row, 1)
        if estilo is None and texto == "Como preencher":
            cel.font = Font(bold=True, size=13, color="1E3A8A")
        elif estilo == "nota":
            cel.font = Font(italic=True, color="4B5563")
    saida.parent.mkdir(parents=True, exist_ok=True)
    wb.save(saida)
    return saida


if __name__ == "__main__":
    print(gerar())
