import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# --- Configurações ---
PASTA_RAIZ = Path("dataset")
CAMINHO_BRONZE = PASTA_RAIZ / "bronze"
CAMINHO_SILVER = PASTA_RAIZ / "silver"


def analisar_estrutura_dados():
    """Análise exploratória dos dados da camada Bronze"""
    print("="*80)
    print("ANALISE EXPLORATORIA DOS DADOS - CAMADA BRONZE")
    print("="*80)

    arquivos_parquet = list(CAMINHO_BRONZE.rglob("*.parquet"))

    if not arquivos_parquet:
        print("AVISO: Nenhum arquivo Parquet encontrado na camada Bronze!")
        return None

    print(f"\nTotal de arquivos encontrados: {len(arquivos_parquet)}")

    df_amostra = pd.read_parquet(arquivos_parquet[0])

    print("\n" + "="*80)
    print("ESTRUTURA DOS DADOS")
    print("="*80)
    print(f"\nColunas disponíveis ({len(df_amostra.columns)}):")
    print(df_amostra.dtypes)

    print("\n" + "="*80)
    print("PRIMEIRAS LINHAS")
    print("="*80)
    print(df_amostra.head(3))

    print("\n" + "="*80)
    print("ESTATISTICAS DESCRITIVAS")
    print("="*80)
    print(df_amostra.describe())

    print("\n" + "="*80)
    print("ANALISE DE VALORES NULOS")
    print("="*80)
    nulos = df_amostra.isnull().sum()
    total = len(df_amostra)

    for coluna in df_amostra.columns:
        qtd_nulos = nulos[coluna]
        perc = (qtd_nulos / total) * 100
        print(f"{coluna:30s}: {qtd_nulos:6d} nulos ({perc:5.2f}%)")

    print("\n" + "="*80)
    print("VALOR DE NEGOCIO DAS COLUNAS")
    print("="*80)

    analise_colunas = {
        "ano": "CRITICO - Periodo temporal dos gastos",
        "mes": "CRITICO - Detalhamento mensal",
        "codigo_orgao_superior": "ALTO - Identificacao do orgao",
        "nome_orgao_superior": "ALTO - Nome do orgao para analises",
        "codigo_orgao_subordinado": "MEDIO - Identificacao de unidades",
        "nome_orgao_subordinado": "MEDIO - Detalhamento organizacional",
        "codigo_unidade_gestora": "MEDIO - Unidade executora",
        "nome_unidade_gestora": "MEDIO - Identificacao da UG",
        "valor": "CRITICO - Valor do gasto (essencial)",
        "favorecido": "ALTO - Beneficiario do recurso",
        "cpf_cnpj_favorecido": "ALTO - Identificacao fiscal",
        "data_pagamento": "CRITICO - Data de execucao",
        "numero_processo": "MEDIO - Rastreabilidade",
        "categoria": "ALTO - Classificacao do gasto",
        "origem_despesa": "MEDIO - Fonte do recurso"
    }

    for coluna in df_amostra.columns:
        if coluna in analise_colunas:
            print(f"{coluna:30s}: {analise_colunas[coluna]}")
        else:
            print(f"{coluna:30s}: Necessita analise detalhada")

    return df_amostra


def transformar_bronze_para_silver():
    """Transformação e limpeza dos dados Bronze para Silver"""
    print("\n" + "="*80)
    print("INICIANDO TRANSFORMACAO: BRONZE -> SILVER")
    print("="*80)

    arquivos_parquet = list(CAMINHO_BRONZE.rglob("*.parquet"))

    if not arquivos_parquet:
        print("AVISO: Nenhum arquivo encontrado para processar!")
        return

    total_registros_processados = 0
    total_registros_removidos = 0

    for arquivo_path in arquivos_parquet:
        print(f"\nProcessando: {arquivo_path.relative_to(PASTA_RAIZ)}")

        df = pd.read_parquet(arquivo_path)
        registros_originais = len(df)

        # Remover duplicatas
        df = df.drop_duplicates()

        # Remover registros com valores nulos em colunas críticas
        colunas_criticas = ['ano', 'mes', 'valor', 'data_pagamento']
        df = df.dropna(subset=colunas_criticas)

        # Converter tipos de dados
        df['ano'] = pd.to_numeric(df['ano'], errors='coerce').astype('Int64')
        df['mes'] = pd.to_numeric(df['mes'], errors='coerce').astype('Int64')
        df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
        df['data_pagamento'] = pd.to_datetime(
            df['data_pagamento'], errors='coerce')

        # Remover registros com conversões inválidas
        df = df.dropna(subset=['ano', 'mes', 'valor', 'data_pagamento'])

        # Padronização de strings
        colunas_texto = ['nome_orgao_superior', 'nome_orgao_subordinado',
                         'nome_unidade_gestora', 'favorecido', 'categoria']

        for coluna in colunas_texto:
            if coluna in df.columns:
                df[coluna] = df[coluna].astype(str).str.strip().str.upper()
                df[coluna] = df[coluna].replace('NAN', pd.NA)

        # Validações
        df = df[df['valor'] > 0]
        df = df[(df['mes'] >= 1) & (df['mes'] <= 12)]

        # Colunas derivadas
        df['ano_mes'] = df['ano'].astype(
            str) + '-' + df['mes'].astype(str).str.zfill(2)
        df['trimestre'] = ((df['mes'] - 1) // 3 + 1).astype('Int64')

        # Padronizar CPF/CNPJ
        if 'cpf_cnpj_favorecido' in df.columns:
            df['cpf_cnpj_favorecido'] = df['cpf_cnpj_favorecido'].astype(
                str).str.replace(r'\D', '', regex=True)
            df['tipo_pessoa'] = df['cpf_cnpj_favorecido'].apply(
                lambda x: 'PJ' if len(x) == 14 else 'PF' if len(
                    x) == 11 else 'INVALIDO'
            )

        registros_finais = len(df)
        removidos = registros_originais - registros_finais

        print(f"  Registros originais: {registros_originais}")
        print(f"  Registros limpos: {registros_finais}")
        print(
            f"  Registros removidos: {removidos} ({(removidos/registros_originais)*100:.2f}%)")

        # Testes de qualidade
        executar_testes_qualidade(df)

        # Salvar na camada Silver
        estrutura_path = arquivo_path.relative_to(CAMINHO_BRONZE)
        caminho_silver_arquivo = CAMINHO_SILVER / estrutura_path
        caminho_silver_arquivo.parent.mkdir(parents=True, exist_ok=True)

        df.to_parquet(caminho_silver_arquivo, index=False, engine="pyarrow")
        print(f"  Salvo em: {caminho_silver_arquivo.relative_to(PASTA_RAIZ)}")

        total_registros_processados += registros_finais
        total_registros_removidos += removidos

    print("\n" + "="*80)
    print("RESUMO DA TRANSFORMACAO")
    print("="*80)
    print(f"Total de registros processados: {total_registros_processados:,}")
    print(f"Total de registros removidos: {total_registros_removidos:,}")
    print(f"Arquivos salvos em: {CAMINHO_SILVER}")
    print("\nTransformacao concluida com sucesso!")


def executar_testes_qualidade(df):
    """Testes de qualidade dos dados"""
    print("\n  TESTES DE QUALIDADE:")

    # Teste 1: Valores críticos não nulos
    colunas_criticas = ['ano', 'mes', 'valor', 'data_pagamento']
    for coluna in colunas_criticas:
        if df[coluna].isnull().any():
            print(f"    FALHA: {coluna} contem valores nulos")
        else:
            print(f"    OK: {coluna} sem valores nulos")

    # Teste 2: Valores positivos
    if (df['valor'] <= 0).any():
        print("    FALHA: Existem valores <= 0")
    else:
        print("    OK: Todos os valores sao positivos")

    # Teste 3: Mês válido
    if not df['mes'].between(1, 12).all():
        print("    FALHA: Existem meses invalidos")
    else:
        print("    OK: Todos os meses sao validos (1-12)")

    # Teste 4: Consistência ano/mês com data_pagamento
    df_temp = df.copy()
    df_temp['ano_data'] = df_temp['data_pagamento'].dt.year
    df_temp['mes_data'] = df_temp['data_pagamento'].dt.month

    inconsistencias = ((df_temp['ano'] != df_temp['ano_data']) |
                       (df_temp['mes'] != df_temp['mes_data'])).sum()

    if inconsistencias > 0:
        print(
            f"    AVISO: {inconsistencias} inconsistencias entre ano/mes e data_pagamento")
    else:
        print("    OK: Ano/mes consistente com data_pagamento")

    # Teste 5: Estatísticas do valor
    print(f"    Valor minimo: R$ {df['valor'].min():,.2f}")
    print(f"    Valor maximo: R$ {df['valor'].max():,.2f}")
    print(f"    Valor medio: R$ {df['valor'].mean():,.2f}")


def executar_pipeline_silver():
    """Executa o pipeline completo da camada Silver"""
    print("\nPIPELINE DE TRANSFORMACAO SILVER")
    print("="*80)

    analisar_estrutura_dados()
    transformar_bronze_para_silver()

    print("\n" + "="*80)
    print("PIPELINE SILVER FINALIZADO")
    print("="*80)


if __name__ == "__main__":
    executar_pipeline_silver()
