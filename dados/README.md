# dados/

Bases geradas pelo notebook e lidas pelo painel.

- `cotacoes_b3_mensais.csv.gz` — fechamento do último pregão e média dos fechamentos de cada mês, por papel
  (ações, units, FIIs e ETFs do mercado à vista), a partir dos arquivos oficiais de cotações históricas da B3
  (COTAHIST): anos encerrados e, do ano corrente, os meses já encerrados. Gerada na Seção 15 do notebook; atualizar de
  tempos em tempos (ex.: a cada trimestre). O painel usa esta base
  quando o Yahoo Finance não tem cotações antigas de um papel (ex.: AFHI11, só a partir de 01/2024). Preços sem
  ajuste por proventos; desdobramentos são ajustados no uso, com os eventos registrados pelo Yahoo.
