import snowflake.connector
import pandas as pd
import numpy as np
#import plotly.express as px
import time
from datetime import datetime
import time
from functions import sheets_export, cleaning_and_update_sheets
from database_extract import run_df_catalog
from funcition_run_similarity import run_script_similarity
from senha import SF_USER, SF_PWD, sf_account, sf_database, sf_warehouse

print('PIP installs and Python imports completed ✅')

thresholds = {
    
'BR':{  'country': 'BR',
        'integration': True,
        'similarity': 0.8,
        'gmv_coverage': 0,
        'language': 'portuguese'
    },

'CO':{  'country': 'CO',
        'integration': True,
        'similarity': 0.8,
        'gmv_coverage': 0,
        'language': 'spanish'
    },

'MX':{  'country': 'MX',
        'integration': True,
        'similarity': 0.8,
        'gmv_coverage': 0,
        'language': 'spanish'
    },

'AR':{  'country': 'AR',
        'integration': True,
        'similarity': 0.8,
        'gmv_coverage': 0,
        'language': 'spanish'
    },

'CL':{  'country': 'CL',
        'integration': True,
        'similarity': 0.8,
        'gmv_coverage': 0,
        'language': 'spanish'
    },

'CR':{  'country': 'CR',
        'integration': True,
        'similarity': 0.8,
        'gmv_coverage': 0,
        'language': 'spanish'
    },

'EC':{  'country': 'EC',
        'integration': True,
        'similarity': 0.8,
        'gmv_coverage': 0,
        'language': 'spanish'
    },

'UY':{  'country': 'UY',
        'integration': True,
        'similarity': 0.8,
        'gmv_coverage': 0,
        'language': 'spanish'
    }  
}

class Snowflake:
    def __init__(self):
        self.con = snowflake.connector.connect(
            host = f'{sf_account}.snowflakecomputing.com',
            user=SF_USER,
            password= SF_PWD,
            account= sf_account,
            database= sf_database,
            warehouse= sf_warehouse,
            #authenticator = 'externalbrowser',
            session_parameters={
                'QUERY_TAG': 'br-shops-partners',
            }
        )

    def __del__(self):
        self.con.close()

    def destroy(self):
        self.con.close()

    def close(self):
        if self.con: self.con.close()

    def download_data(self, query):
        cur = self.con.cursor()
        cur.execute(f'USE WAREHOUSE {sf_warehouse};')
        cur.execute(f'USE DATABASE {sf_database};')
        cur.execute(query)
        self.con.commit()
        results = cur.fetchall()
        column_names = [x[0] for x in cur.description]

        cur.close()
        df = pd.DataFrame(results, columns=column_names)
        return df
    
    def run_query(self, query, max_attempts=15, retry_delay=5):
        for attempt in range(1, max_attempts + 1):
            try:
                cur = self.con.cursor()
                cur.execute('USE WAREHOUSE CPGS;')
                cur.execute('USE DATABASE FIVETRAN;')
                # cur.execute('USE ROLE CPGS_PLANNING_BR_WRITE_ROLE;')
                cur.execute(query)
                cur.close()
                df = self.download_data(query)
                return df
            except snowflake.connector.errors.DatabaseError as e:
                if attempt < max_attempts:
                    print(f"Erro na tentativa {attempt} - Retentando em {retry_delay} segundos.")
                    time.sleep(retry_delay)
                else:
                    raise e
        
    def run_query_from_sql_file(self, query_path, max_attempts=15, retry_delay = 5):
        for attempt in range(1, max_attempts + 1):
            try:
                file_spam = query_path.open(encoding='utf-8')
                query = file_spam.read()
                file_spam.close()
                df = self.download_data(query)
                return df
            except snowflake.connector.errors.DatabaseError as e:
                if attempt < max_attempts:
                    print(f"Erro na tentativa {attempt} - Retentando em {retry_delay} segundos.")
                    time.sleep(retry_delay)
                else:
                    raise e

# snowflake queries
snow = Snowflake()

### definindo o datetime
date_time = datetime.now().strftime("%Y%m%d_%H%M%S")

### variáveis globais

spreadsheet = '1DvUgpu8njoZsLBMCYserHQD-oJc6FwkuTtjr0hBnsf4'
lista_countries = ['BR', 'CO', 'CR', 'PE', 'MX', 'EC', 'CL', 'AR']
countries_out = ['UY']
similarity = 0.5
expected_rps = 500

### loop para atualizar a planilha com todos os países

for i in range(len(lista_countries)):
    country = lista_countries[i]
    print(country)
    range_spreadsheet = f'{country}!A2:AB'

    weightables_base = run_df_catalog(country=country, snow=snow)

    print(weightables_base)

    k = len(weightables_base[0])
    print(f'Número de produtos(MPs) dentro da SOT: {k}')
    g = weightables_base[1]
    g = len(g[g['IN_SOT']==True])
    print(f'Número de produtos(RPs) dentro da SOT: {g}')
    t = weightables_base[1]
    t = len(t[t['IN_SOT']==False])
    print(f'Número de produtos(RPs) fora da SOT: {t}')
    h = weightables_base[2]
    h = len(h[h['IN_SOT']==False])
    print(f'Número de produtos(MPs) fora da SOT: {h}')

    ### validação se a base extraída está "correta"

    l = weightables_base[1]
    l = l[l['IN_SOT']==True]
    num_groups_with_master_only = l.groupby('MASTER_PRODUCT_ID').ngroups
    n_unique_mps = l['MASTER_PRODUCT_ID'].nunique()
    num_groups_with_all_filled = l.groupby(['COUNTRY', 'NAME', 'MASTER_PRODUCT_ID', 'CAT1_NAME', 'CAT2_NAME', 'CAT3_NAME', 'MAKER', 'TRADEMARK', 'IN_SOT']).ngroups

    print(num_groups_with_master_only)
    print(n_unique_mps)
    print(num_groups_with_all_filled)

    if num_groups_with_master_only == n_unique_mps:

        df_opportunities = weightables_base[2].copy()
        df_sot = weightables_base[0].copy()
        output_df = run_script_similarity(df_opportunities, df_sot)

        produtos_de_referencia = output_df['master_product_id'].unique()

        # DataFrame para armazenar os resultados finais
        df_resultado = pd.DataFrame()

        # Loop sobre cada produto de referência
        for produto_x in produtos_de_referencia:
            df_sorted = output_df[output_df['master_product_id'] == produto_x].sort_values(by='similarity', ascending=False)
            df_resultado = pd.concat([df_resultado, df_sorted.head(1)])

        df_resultado = df_resultado.reset_index(drop=True)
        df_resultado = df_resultado[df_resultado['similarity'] >= similarity]
        df_resultado = df_resultado[['master_products', 'integration', 'master_product_id', 'master_product_id_sot', 
                                'price_similarity', 'similarity', 'config_similarity', 'final_weighted_confidence']]
        df_resultado.rename(columns={'integration': 'master_sot', 'master_products': 'master_cp'}, inplace=True)

        gmv_total = weightables_base[4].copy()
        gmv_total = gmv_total['GMV_T'].sum()
        priorization_df = df_resultado.merge(right=weightables_base[4], how='left', left_on = 'master_product_id', right_on = 'MASTER_PRODUCT_ID')
        priorization_df['gmv_coverage'] = (priorization_df['GMV_T']/gmv_total) * 100

        priorization_df = priorization_df[['COUNTRY', 'NAME','master_cp', 'master_sot', 'master_product_id', 'master_product_id_sot',
            'price_similarity', 'similarity', 'config_similarity',
            'final_weighted_confidence', 'CAT1_NAME', 'CAT2_NAME', 'CAT3_NAME', 'MAKER', 'TRADEMARK', 'M_FR',
            'PRODUCT_DR', 'TOTAL_PRODUCTS', 'COUNT_REPLACED', 'GMV_T', 'N_ASSOCIATED_RETAILERS', 'SIMILARITY', 'MEDIAN_PRICE', 'gmv_coverage']]
        priorization_df = priorization_df.sort_values(by=['gmv_coverage'], ascending=False).reset_index(drop=True)
        priorization_df['cumulative_retailers'] = priorization_df['N_ASSOCIATED_RETAILERS'].cumsum()

        result_df = priorization_df[priorization_df['cumulative_retailers'] <= expected_rps]

        print(priorization_df)

        if len(result_df) < len(priorization_df):
            next_row = priorization_df.iloc[len(result_df)]
            result_df = pd.concat([result_df, next_row.to_frame().T], ignore_index=True)
        else:
            print("Todos os produtos já foram priorizados, não há mais linhas para adicionar.")

        result_df = result_df.drop(columns=['cumulative_retailers'])
        result_df = result_df[['master_cp', 'master_sot', 'master_product_id', 'master_product_id_sot',
        'price_similarity', 'similarity', 'config_similarity', 'final_weighted_confidence']]

        ## montando sheets
        
        df_rps = weightables_base[3].copy().reset_index(drop=True)
        df_complet_rp = weightables_base[1].copy().reset_index(drop=True)

        export = sheets_export(priorization_df=result_df, df_rps=df_rps, country=country, df_sot=df_sot, df_complet_rp=df_complet_rp, snow=snow)

        print(export[0])

        cleaning_and_update_sheets(spreadsheet_id=spreadsheet, country=country, data=export[0])

    else:
        print('Uniformidade dos dados não é válida')











