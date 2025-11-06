import requests
import json
import time
from pathlib import Path
import pandas as pd

# --- Configurações ---
PASTA_RAIZ = Path("dataset")
API_ENDPOINT = "https://brasil.io/api/v1/dataset/gastos-diretos/gastos/data/"
CAMINHO_JSON = PASTA_RAIZ / "raw"
CAMINHO_PARQUET = PASTA_RAIZ / "bronze"
AUTH_KEY = "0a3c26398bb82d107772cf4a05967ebbb7c95bcc"
LIMITE_PAGINAS = 3
DELAY_REQUISICAO_SEG = 1
DELAY_RATE_LIMIT_SEG = 15


def preparar_diretorios():
    for camada in ["raw", "bronze", "silver", "gold"]:
        (PASTA_RAIZ / camada).mkdir(parents=True, exist_ok=True)


def verificar_paginas_baixadas() -> set:
    paginas_existentes = set()
    for arquivo in CAMINHO_JSON.glob("page_*.json"):
        num_pagina = int(arquivo.stem.split('_')[-1])
        paginas_existentes.add(num_pagina)
    return paginas_existentes


def coletar_dados_api():
    print("Iniciando coleta de dados da API...")
    request_headers = {"Authorization": f"Token {AUTH_KEY}"}
    paginas_ja_coletadas = verificar_paginas_baixadas()
    pagina_atual = 1

    while pagina_atual <= LIMITE_PAGINAS:
        if pagina_atual in paginas_ja_coletadas:
            print(f"Página {pagina_atual} já foi baixada.")
            pagina_atual += 1
            continue
        try:
            params = {"page": pagina_atual}
            resposta = requests.get(
                API_ENDPOINT, headers=request_headers, params=params)

            if resposta.status_code == 429:
                print(
                    f"Rate limit atingido. Aguardando {DELAY_RATE_LIMIT_SEG}s...")
                time.sleep(DELAY_RATE_LIMIT_SEG)
                continue

            resposta.raise_for_status()
            dados_json = resposta.json()

            caminho_arquivo = CAMINHO_JSON / f"page_{pagina_atual}.json"
            with open(caminho_arquivo, "w", encoding="utf-8") as f_json:
                json.dump(dados_json, f_json, ensure_ascii=False, indent=2)

            print(f"Página {pagina_atual} salva com sucesso.")

            pagina_atual += 1
            time.sleep(DELAY_REQUISICAO_SEG)

        except requests.exceptions.RequestException as e:
            print(f"Erro ao buscar página {pagina_atual}: {e}")

    print("Coleta de dados finalizada.")


def transformar_json_para_parquet():
    print("\nIniciando processamento JSON para Parquet (Bronze)...")
    lista_de_registros = []

    arquivos_json = sorted(CAMINHO_JSON.glob("page_*.json"))

    if not arquivos_json:
        print("Nenhum arquivo JSON encontrado na pasta 'raw'.")
        return

    for arquivo_path in arquivos_json:
        try:
            with open(arquivo_path, 'r', encoding="utf-8") as f_leitura:
                dados_pagina = json.load(f_leitura)
                if "results" in dados_pagina:
                    lista_de_registros.extend(dados_pagina["results"])
        except json.JSONDecodeError:
            print(f"Erro ao ler o JSON: {arquivo_path}. Arquivo corrompido?")
        except Exception as e:
            print(f"Erro inesperado ao processar {arquivo_path}: {e}")

    if not lista_de_registros:
        print("Nenhum registro encontrado dentro dos arquivos JSON.")
        return

    dataframe_completo = pd.DataFrame(lista_de_registros)
    print(f"Total de registros consolidados: {len(dataframe_completo)}")

    dados_agrupados = dataframe_completo.groupby(["ano", "mes"])

    for (ano, mes), grupo_df in dados_agrupados:
        months = ["janeiro", "fevereiro", "março", "abril", "maio", "junho",
                  "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"]
        mn = months[mes-1]

        diretorio_particao = CAMINHO_PARQUET / f"{ano}" / f"{mes}-{mn}"
        diretorio_particao.mkdir(parents=True, exist_ok=True)
        arquivo_parquet_path = diretorio_particao / "dados.parquet"
        grupo_df.to_parquet(arquivo_parquet_path,
                            index=False, engine="pyarrow")

    print("Processamento para Parquet concluído!")


def executar_pipeline():
    preparar_diretorios()
    coletar_dados_api()
    transformar_json_para_parquet()
    print("\nPipeline finalizado.")


if __name__ == "__main__":
    executar_pipeline()
