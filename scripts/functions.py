import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk import ngrams
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import re
from google.oauth2.service_account import Credentials
import pandas as pd
from datetime import datetime
import logging
import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from decimal import Decimal
from datetime import date
import math

def preprocess_text(text):
    # Converte para minúsculas, remove caracteres especiais e números
    text = text.lower()
    text = ''.join(e for e in text if e.isalnum() or e.isspace())

    # Remove stopwords
    stop_words = set(stopwords.words('portuguese'))  # Use 'english' se for o caso
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]

    # Reconstrói o texto sem stopwords
    processed_text = ' '.join(filtered_words)

    return processed_text

##### definindo a primeira função de similaridade

def n_grams_const(x,y):
    n = 1
    counts = CountVectorizer(analyzer='word', ngram_range=(n,n))
    n_grams = counts.fit_transform([x,y])
    #vocab = counts.fit([texto_a_ser_comparado,texto_fonte]).vocabulary_
    n_grams_array = n_grams.toarray()
    intersec = np.amin(n_grams.toarray(), axis=0)
    intersec_count = np.sum(intersec)
    index_a = 0
    A_count = np.sum(n_grams.toarray()[index_a])
    similarity_n_grams = intersec_count/A_count

    return similarity_n_grams

print('Primeira função de similaridade carregada ✅')

##### definindo a segunda função de similaridade

def jaccard_similarity(str1, str2, n=1):
    str1 = preprocess_text(str1)
    str2 = preprocess_text(str2)

    # Cria n-grams a partir das strings
    ngrams_set1 = set(ngrams(str1.split(), n))
    ngrams_set2 = set(ngrams(str2.split(), n))

    # Calcula a interseção e a união dos conjuntos de n-grams
    intersection = len(ngrams_set1.intersection(ngrams_set2))
    union = len(ngrams_set1) + len(ngrams_set2) - intersection

    # Calcula a similaridade de Jaccard
    similarity = intersection / union
    return similarity

print('Segunda função de similaridade carregada ✅')

##### definindo a terceira função de similaridade

def levenshtein_similarity(str1, str2):
    str1 = preprocess_text(str1)
    str2 = preprocess_text(str2)

    len_str1 = len(str1)
    len_str2 = len(str2)

    dp = [[0 for _ in range(len_str2 + 1)] for _ in range(len_str1 + 1)]

    for i in range(len_str1 + 1):
        dp[i][0] = i

    for j in range(len_str2 + 1):
        dp[0][j] = j

    for i in range(1, len_str1 + 1):
        for j in range(1, len_str2 + 1):
            cost = 0 if str1[i - 1] == str2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # Deleção
                dp[i][j - 1] + 1,      # Inserção
                dp[i - 1][j - 1] + cost  # Substituição
            )

    # Ajusta o resultado da distância de Levenshtein para o intervalo [0, 1]
    max_len = max(len_str1, len_str2)
    similarity = 1 - (dp[len_str1][len_str2] / max_len)

    return similarity

print('Terceira função de similaridade carregada ✅')

##### definindo a quarta função de similaridade

def calculate_cosine_similarity(str1, str2):
    # Pré-processamento opcional (dependendo dos requisitos específicos)
    str1 = preprocess_text(str1)
    str2 = preprocess_text(str2)

    # Criar o vetor TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([str1, str2])

    # Calcular a similaridade do cosseno
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # A similaridade do cosseno é a entrada [0,1] da matriz
    similarity_score = cosine_sim[0, 1]

    return similarity_score

def extract_quantity_from_product_name(df, product_name_column):
    def extract_quantity(product_name):
        # Padroniza as unidades de quantidade que queremos buscar (ML, G, KG, L)
        quantity_pattern = r"(\d+\.?\d*)\s*(ML|G|KG|L|Unid.|Unid)"

        # Procura o padrão no nome do produto usando regex
        matches = re.findall(quantity_pattern, product_name, re.IGNORECASE)

        # Retorna a quantidade e a unidade de medida, se houver
        if matches:
            return matches[0]
        else:
            return (None, None)

    # Aplica a função extract_quantity na coluna de nomes dos produtos
    df[["SUGGESTED_QUANTITY", "SUGGESTED_UNITYPE"]] = df[product_name_column].apply(extract_quantity).apply(pd.Series)
    return df

print('Função extract quantity carregada com sucesso ✅')

def verifica_variacoes_kg(df, coluna):
    # Criar uma nova coluna chamada "Preco_por_Kg" com valores padrão como "Nao"
    df['PRICE_PER_KG'] = False
    
    # Atualizar os valores para "Possivel preco por kg" onde a coluna "NAME_INT" contém a palavra "KG" ou variações isolada, no final ou junto à última palavra
    df.loc[df[coluna].str.contains(r'(?:\b(?:kg|Kg|kG|KG)\b|\b(?:kg|Kg|kG|KG)\b\W*$)', case=False, regex=True), 'PRICE_PER_KG'] = True

    return df

print('Função para verficar variacoes kg ✅')

def sheets_export(priorization_df, df_rps, country, df_sot, df_complet_rp, snow):
    df_rps.columns = ['RETAILER_ID', 'NAME', 'CAT1_NAME', 'CAT2_NAME', 'CAT3_NAME', 'MAKER',
       'TRADEMARK', 'EAN', 'QUALITY_LABEL', 'PRIORITY', 'COUNTRY', 'SKU',
       'PRODUCT_ID', 'MASTER_PRODUCT_ID', 'MAX_QUANT', 'MIN_QUANT',
       'STEP_QUANT', 'SELL_TYPE', 'PRESENTATION', 'QUANTITY_RP', 'UNITY_P',
       'ASSOCIATED_RETAILERS', 'MP_PHOTO_URL', 'PRODUCT_DR', 'FR',
       'TOTAL_PRODUCTS', 'COUNT_REPLACED', 'GMV_TOTAL', 'NAME_INT',
       'DESCRIPTION', 'TRADEMARK_INT', 'MEDIAN_PRICE', 'IMAGE_URL', 'QUANTITY_INT',
       'UNIT_TYPE', 'MIN_QUANTITY_IN_GRAMS', 'STEP_QUANTITY_IN_GRAMS',
       'IN_SOT', 'N_ASSOCIATED_RETAILERS', 'SIMILARITY', 'SUGGESTED_QUANTITY',
       'SUGGESTED_UNITYPE', 'PRICE_PER_KG']
    
    columns_choice_log = ['COUNTRY', 'RETAILER_ID', 'PRODUCT_ID', 'SKU', 'MASTER_PRODUCT_ID', 'MP_PHOTO_URL', 'IMAGE_URL',
    'NAME', 'NAME_INT', 'TRADEMARK', 'TRADEMARK_INT', 'MEDIAN_PRICE', 'MIN_QUANT', 'STEP_QUANT',
    'SELL_TYPE', 'PRESENTATION', 'QUANTITY_RP', 'UNITY_P', 'UNIT_TYPE', 'MIN_QUANTITY_IN_GRAMS', 'QUANTITY_INT',
    'STEP_QUANTITY_IN_GRAMS', 'FR', 'PRODUCT_DR', 'N_ASSOCIATED_RETAILERS', 'SIMILARITY', 'SUGGESTED_QUANTITY',
    'SUGGESTED_UNITYPE', 'PRICE_PER_KG', 'master_cp', 'master_sot', 'master_product_id', 'master_product_id_sot',
    'price_similarity', 'similarity', 'config_similarity', 'final_weighted_confidence']

    columns_choice_sheets = ['PRODUCT_ID', 'SKU', 'MASTER_PRODUCT_ID',
    'MP_PHOTO_URL', 'IMAGE_URL',
    'NAME', 'NAME_INT', 
    'TRADEMARK', 'TRADEMARK_INT', 
    'MEDIAN_PRICE', 'MEDIAN_PRICE_SOT', 'PRICE_PER_KG',
    'STEP_QUANT', 'SELL_TYPE', 'UNITY_P',
    'QUANTITY_RP', 'STEP_QUANTITY_IN_GRAMS', 'QUANTITY_INT', 'UNIT_TYPE', 'MP_PHOTO_URL_SOT',
    'NAME_SOT', 'LONG_DESCRIPTION_SOT', 'TRADEMARK_SOT', 'SALE_TYPE_SOT', 'STEP_QUANT_SOT', 'QUANTITY_SOT', 'UNIT_TYPE_SOT',
    'MASTER_PRODUCT_ID_SOT', 'SUGGESTED_QUANTITY', 'SUGGESTED_UNITYPE']
   
    columns_df_sot = ['MEDIAN_PRICE', 'MASTER_PRODUCT_ID', 'TRADEMARK']

    lista_mps = tuple(priorization_df['master_product_id_sot'])

    query_infos_mps = f'''

    SELECT DISTINCT

    ID AS MASTER_PRODUCT_ID,
    NAME,
    LONG_DESCRIPTION,
    REPLACE(SELL_TYPE:"type", '"', '') AS SALE_TYPE,
    REPLACE(PRESENTATION:"unitType", '"', '') AS UNIT_TYPE,
    PRESENTATION:"quantity" AS QUANTITY,
    SELL_TYPE:"stepQuantity" AS STEP_QUANT

    FROM {country}_PGLR_MS_CPGS_CLG_PM_PUBLIC.PRODUCT RP

    WHERE ID IN {lista_mps}

    '''

    df_infos_mps = snow.run_query(query_infos_mps)
    columns_df_complet_rp = ['MASTER_PRODUCT_ID','MP_PHOTO_URL']
    df_complet_rp = df_complet_rp[columns_df_complet_rp]
    df_sot = df_sot[columns_df_sot]

    df_infos_mps = df_infos_mps.merge(right=df_complet_rp, on = 'MASTER_PRODUCT_ID', how='inner')
    df_infos_mps = df_infos_mps.merge(right=df_sot, on = 'MASTER_PRODUCT_ID', how='inner')
    df_infos_mps = df_infos_mps.rename(columns=lambda x: f"{x}_SOT")
    df_infos_mps = df_infos_mps.drop_duplicates().reset_index(drop=True)

    df_log = df_rps.merge(right=priorization_df, right_on = 'master_product_id', left_on= 'MASTER_PRODUCT_ID', how='inner')  
    df_log = df_log[columns_choice_log]
    df_log = df_log.merge(right = df_infos_mps, right_on = 'MASTER_PRODUCT_ID_SOT', left_on = 'master_product_id_sot', how='left')
    df_log = df_log.reset_index(drop=True)
    #print(df_log.columns)

    df_sheets = df_log[columns_choice_sheets]
    #display(df_sheets)

    def replace_unit_type(row):
        if pd.isnull(row['UNIT_TYPE']) or row['UNIT_TYPE'] == 0 or row['UNIT_TYPE'] is None:
            return row['SUGGESTED_UNITYPE']
        else:
            return row['UNIT_TYPE']
 
    df_sheets['UNIT_TYPE'] = df_sheets.apply(replace_unit_type, axis=1)
    df_sheets['QUANTITY_INT'] = df_sheets['QUANTITY_INT'].astype(float)
    df_sheets['QUANTITY_RP'] = df_sheets['QUANTITY_RP'].astype(float)
    df_sheets['MASTER_PRODUCT_ID'] = df_sheets['MASTER_PRODUCT_ID'].astype(str)
    df_sheets['MASTER_PRODUCT_ID_SOT'] = df_sheets['MASTER_PRODUCT_ID_SOT'].astype(str)
    df_sheets['TRADEMARK_INT'] = df_sheets['TRADEMARK_INT'].fillna('None')
    df_sheets['TRADEMARK'] = df_sheets['TRADEMARK_INT'].fillna('None')

    def replace_quantity(quantity, suggested_quantity):
        if pd.isnull(quantity) or quantity == 0 or quantity is None:
            return suggested_quantity
        else:
            return quantity

    df_sheets['QUANTITY_INT'] = df_sheets.apply(lambda row: replace_quantity(row['QUANTITY_INT'], row['SUGGESTED_QUANTITY']), axis=1)
    df_sheets['STEP_QUANTITY_IN_GRAMS'] = df_sheets.apply(lambda row: replace_quantity(row['STEP_QUANTITY_IN_GRAMS'], row['SUGGESTED_QUANTITY']), axis=1)

    df_sheets = df_sheets.drop(columns=['SUGGESTED_UNITYPE', 'SUGGESTED_QUANTITY'])

    #df_choice_mps['MPs'] = df_choice_mps['MPs'].astype('int64')
    df_sheets = df_sheets.drop_duplicates()
    return df_sheets, df_log

print('Função para criar template no sheets e log para table ✅')

def cleaning_and_update_sheets(spreadsheet_id, country, data):
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
    CREDENTIALS_FILE = r"C:\Users\edivaldo.alves\OneDrive\Documentos\VS Code\Credenciais_api_google\client_secret_175938711802-371rq9e9gnpb9o1p6vjaihk5s0tnurio.apps.googleusercontent.com.json"
    TOKEN = r'C:\Users\edivaldo.alves\OneDrive\Documentos\VS Code\Credenciais_api_google\token.json'

    creds = None
    if os.path.exists(TOKEN):
        creds = Credentials.from_authorized_user_file(TOKEN, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(TOKEN, 'w') as token_file:
            token_file.write(creds.to_json())

    worksheet_name = f'{country}!A2'
    clear_range = f'{country}!A2:AB'
    service = build('sheets', 'v4', credentials=creds)
    sheet = service.spreadsheets()

    try:
        # Clear the existing values in the worksheet
        clear_result = sheet.values().clear(
            spreadsheetId=spreadsheet_id,
            range=clear_range
        ).execute()

        print(f'Resultado da atualização: {clear_result}')

        if 'clearedRange' in clear_result:
            # Convert DataFrame to list of lists if necessary
            if isinstance(data, pd.DataFrame):
                data = data.values.tolist()

            # Ensure that all decimal values are converted to strings
            data = [[str(item) if isinstance(item, Decimal) else item for item in row] for row in data]

            result = sheet.values().update(
                spreadsheetId=spreadsheet_id,
                range=worksheet_name,
                valueInputOption="RAW",
                body={"values": data}
            ).execute()

            print(f'Resultado da atualização: {result}')
    except HttpError as err:
        print(f'Erro HTTP: {err}')
        print(f'Resposta completa: {err.resp}')
        raise

print('Função para limpar o sheets e alterar os dados realizada ✅')





