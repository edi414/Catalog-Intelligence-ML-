import snowflake.connector
import jellyfish as j
import pandas as pd
import numpy as np
from functions import preprocess_text, n_grams_const, jaccard_similarity, levenshtein_similarity, calculate_cosine_similarity, extract_quantity_from_product_name, verifica_variacoes_kg

def run_df_catalog(country, snow):

    query_catalog = f'''

        WITH SO_FR AS (
            
                    SELECT
                    SO.COUNTRY,
                    SO.PRODUCT_ID,
                    SUM(IFF(SO.STOCKOUT = 0, 1, 0)) AS COUNT_FOUND,
                    SUM(IFF(SO.SUBSTITUTED = 1, 1, 0)) AS COUNT_REPLACED,
                    SUM(IFF((SO.STOCKOUT = 1 AND SO.SUBSTITUTED = 0), 1, 0)) AS COUNT_REFUNDED,
                    (COUNT_FOUND + COUNT_REPLACED + COUNT_REFUNDED) AS TOTAL_PRODUCTS,
                    SUM(IFF(SO.STOCKOUT = 0, 1, 0))/TOTAL_PRODUCTS AS FR,
                    SUM(IFF(TOTAL_PRICE IS NULL, 0, TOTAL_PRICE)) AS GMV_TOTAL,
                    COUNT(DISTINCT ORDER_ID) AS N_ORDERS
                    
                    FROM OPS_CPGS_NOW.TIPIFICADOR_{country} SO
                    WHERE 1=1
                    AND SO.CREATED_AT_ORDER::DATE >= DATE_TRUNC('WEEK', CURRENT_DATE::DATE) - INTERVAL '12 weeks'
                    AND SO.COUNTRY = '{country}'
                    GROUP BY 1,2
                    
        )

        , DEFECT_PID AS (

                SELECT
                A.COUNTRY
                ,A.RETAILLER_PRODUCT_ID AS PID
                ,COUNT(1) AS QTY_REQUESTED
                ,COUNT(CASE WHEN B.ORDER_ID IS NOT NULL THEN B.ORDER_ID ELSE NULL END) AS QTY_DEFECTIVE
                ,ROUND(QTY_DEFECTIVE/QTY_REQUESTED,4) AS PRODUCT_DR
                FROM FIVETRAN.CPGS_OPS_SHOPPER_GLOBAL.TB_CPGS_OPS_CAP_ORDER_PRODUCTS AS A
                LEFT JOIN FIVETRAN.CPGS_OPS_SHOPPER_GLOBAL.TB_CPGS_OPS_CAP_TICKETS AS B
                ON (A.COUNTRY = B.COUNTRY) AND	A.ORDER_ID = B.ORDER_ID AND	A.KUSTOMER_CONVERSATION_ID = B.KUSTOMER_CONVERSATION_ID
                AND LEVEL_2 IN  ('Different product','Missing product','Disagree with charge','Double charged','Cost of products','Product in poor condition')
                WHERE A.DT_REF >= CURRENT_DATE::DATE - 30
                GROUP BY 1,2
                HAVING PRODUCT_DR > 0
                ORDER BY 4
            
        )

        , CART_TAMS AS (        

            SELECT DISTINCT 
            
            COUNTRY_CODE,
            CP_RETAILER_ID,
            TAM
            
            FROM CPGS_SALESCAPABILITY.TBL_CORE_STORES_LINK SF
            
            WHERE TAM IS NOT NULL
            AND CP_RETAILER_ID IS NOT NULL
            AND TAM NOT IN ('igor.pollo@rappi.com')
            ORDER BY 3

        )

        , FILTER_INTEGRATION AS (

            SELECT DISTINCT
            
            CASE WHEN(RETAIL_ID IS NOT NULL) THEN '{country}' ELSE '{country}' END AS COUNTRY,
            RETAIL_ID,
            MAX(NAME) AS NAME,
            MAX(DESCRIPTION) AS DESCRIPTION,
            MAX(TRADEMARK) AS TRADEMARK,
            MEDIAN(PRICE) AS MEDIAN_PRICE,
            MAX(IMAGE_URL) AS IMAGE_URL,
            MAX(QUANTITY) AS QUANTITY,
            MAX(UNIT_TYPE) AS UNIT_TYPE,
            MAX(MIN_QUANTITY_IN_GRAMS) AS MIN_QUANTITY_IN_GRAMS,
            MAX(STEP_QUANTITY_IN_GRAMS) AS STEP_QUANTITY_IN_GRAMS
            FROM cpgs_datascience.{country}_integrations_new_catalog_last_event i
            WHERE NAME IS NOT NULL
            AND EXECUTION_DATE::DATE > CURRENT_DATE::DATE - 15
            GROUP BY 1,2

        )

        , VALIDATION_WE_ST AS (
                
                SELECT DISTINCT
                
                MP.COUNTRY,
                RP.SKU,
                RP.ID AS PRODUCT_ID,
                RP.PRODUCT_ID AS MASTER_PRODUCT_ID,
                
                COALESCE(RP.SELL_TYPE:maxQuantity,MP.SELL_TYPE:maxQuantity) AS MAX_QUANT,
                COALESCE(RP.SELL_TYPE:minQuantity,MP.SELL_TYPE:minQuantity) AS MIN_QUANT,
                COALESCE(RP.SELL_TYPE:stepQuantity,MP.SELL_TYPE:stepQuantity) AS STEP_QUANT,
                COALESCE(REPLACE(RP.SELL_TYPE:type,'"'),REPLACE(MP.SELL_TYPE:type,'"')) AS SELL_TYPE
                
                FROM {country}_PGLR_MS_CPGS_CLG_PM_PUBLIC.RETAILER_PRODUCT RP
                LEFT JOIN {country}_PGLR_MS_CPGS_CLG_PM_PUBLIC.PRODUCT MP ON MP.ID = RP.PRODUCT_ID
                
                WHERE 1=1
                AND RP.STATUS = 'published'
                AND MP.STATUS = 'published'
                AND RP._FIVETRAN_DELETED = FALSE
                AND MP._FIVETRAN_DELETED = FALSE

        )

        , VALIDATION_WE_PR AS (
            
                SELECT DISTINCT
            
                MP.COUNTRY,
                RP.SKU,
                RP.ID AS PRODUCT_ID,
                RP.PRODUCT_ID AS MASTER_PRODUCT_ID,
            
                COALESCE(REPLACE(RP.PRESENTATION:description,'"'),REPLACE(MP.PRESENTATION:description,'"')) AS PRESENTATION,
                COALESCE(REPLACE(RP.PRESENTATION:quantity,'"'),MP.PRESENTATION:quantity) AS QUANTITY,
                COALESCE(REPLACE(RP.PRESENTATION:unitType,'"'),REPLACE(MP.PRESENTATION:unitType,'"')) AS UNITY_P
            
                FROM {country}_PGLR_MS_CPGS_CLG_PM_PUBLIC.RETAILER_PRODUCT RP
                LEFT JOIN {country}_PGLR_MS_CPGS_CLG_PM_PUBLIC.PRODUCT MP ON MP.ID = RP.PRODUCT_ID
                
                WHERE 1=1
                AND RP.STATUS = 'published'
                AND MP.STATUS = 'published'
                AND RP._FIVETRAN_DELETED = FALSE
                AND MP._FIVETRAN_DELETED = FALSE

        )

        , CATALOG_COUNTRY AS (

            SELECT DISTINCT
            
            CP.RETAILER_ID,
            CP.MP_NAME AS NAME,
            CP.CAT1_NAME,
            CP.CAT2_NAME,
            CP.CAT3_NAME,
            CP.MAKER,
            CP.TRADEMARK,
            CP.EAN,
            MP.QUALITY_LABEL,
            MP.PRIORITY,
            ST.*,
            PR.PRESENTATION,
            PR.QUANTITY,
            PR.UNITY_P,
            MP.ASSOCIATED_RETAILERS,
            CP.MP_PHOTO_URL
            
            FROM VALIDATION_WE_ST ST
            LEFT JOIN VALIDATION_WE_PR PR
                ON ST.COUNTRY = PR.COUNTRY AND ST.PRODUCT_ID = PR.PRODUCT_ID AND ST.MASTER_PRODUCT_ID = PR.MASTER_PRODUCT_ID
            LEFT JOIN CPGS_LOCAL_ANALYTICS.ANALYTICS_RETAILER_PRODUCT CP
                ON ST.PRODUCT_ID = CP.RETAILER_PRODUCT_ID AND ST.COUNTRY = CP.COUNTRY
            LEFT JOIN CPGS_LOCAL_ANALYTICS.ANALYTICS_MASTER_PRODUCT MP
                ON MP.COUNTRY = ST.COUNTRY AND MP.MASTER_PRODUCT_ID = ST.MASTER_PRODUCT_ID
                
            WHERE CP.MP_PHOTO_URL IS NOT NULL
            AND CP.MP_NAME IS NOT NULL

        )
           
        SELECT DISTINCT 

        CP.*,
        DP.PRODUCT_DR,
        FR.FR,
        FR.TOTAL_PRODUCTS,
        FR.COUNT_REPLACED,
        FR.GMV_TOTAL,
        I.NAME AS NAME_INT,
        I.DESCRIPTION,
        I.TRADEMARK AS TRADEMARK_INT,
        I.MEDIAN_PRICE,
        I.IMAGE_URL,
        I.QUANTITY,
        I.UNIT_TYPE,
        I.MIN_QUANTITY_IN_GRAMS,
        I.STEP_QUANTITY_IN_GRAMS,
        IFF(SOT.MASTER_PRODUCT IS NOT NULL, TRUE, FALSE) AS IN_SOT
        
        FROM CATALOG_COUNTRY CP
        LEFT JOIN SO_FR FR ON FR.COUNTRY = CP.COUNTRY AND FR.PRODUCT_ID = CP.PRODUCT_ID
        LEFT JOIN DEFECT_PID DP ON DP.COUNTRY = CP.COUNTRY AND DP.PID = CP.PRODUCT_ID
        LEFT JOIN FILTER_INTEGRATION I
            ON I.RETAIL_ID = CP.SKU AND I.COUNTRY = CP.COUNTRY
        LEFT JOIN BR_WRITABLE.TBL_TEMP_WEIGHTABLES_SOT SOT
            ON SOT.COUNTRY = CP.COUNTRY AND CP.MASTER_PRODUCT_ID = SOT.MASTER_PRODUCT AND _FIVETRAN_DELETED = FALSE
        --INNER JOIN CART_TAMS CT ON CP.RETAILER_ID = CT.CP_RETAILER_ID AND CP.COUNTRY = CT.COUNTRY_CODE
        
        WHERE 1=1
        AND (CP.SELL_TYPE NOT IN ('U') OR SOT.MASTER_PRODUCT IS NOT NULL) -- FILTRO OU É PESÁVEL OU ESTÁ NA SOT
        --AND I.NAME IS NOT NULL

    '''

    #print(query_catalog)

    df_catalog_rp = snow.run_query(query_catalog)
    df_catalog_rp = df_catalog_rp[~df_catalog_rp['NAME_INT'].isna()] ##filtro de integração
    mps_sot_query = f''' SELECT DISTINCT MASTER_PRODUCT AS MPS FROM BR_WRITABLE.TBL_TEMP_WEIGHTABLES_SOT WHERE COUNTRY = '{country}' AND _FIVETRAN_DELETED = FALSE '''
    mps_sot = snow.run_query(mps_sot_query)
    #print(mps_sot)

    df_catalog_rp['N_ASSOCIATED_RETAILERS'] = df_catalog_rp.groupby('MASTER_PRODUCT_ID')['PRODUCT_ID'].transform('nunique')

    #aplicando as funções de similaridade

    df_catalog_rp['JACCARD_SIMILARITY'] = df_catalog_rp.apply(lambda row: jaccard_similarity(row['NAME'], row['NAME_INT']), axis=1)
    df_catalog_rp['LEV_DISTANCE'] = df_catalog_rp.apply(lambda row: levenshtein_similarity(row['NAME'], row['NAME_INT']), axis=1)
    #df_catalog_rp['N_GRAMS_CONST'] = df_catalog_rp.apply(lambda row: n_grams_const(row['NAME'], row['NAME_INT']), axis=1)
    df_catalog_rp['SIMILARITY_CONSINE'] = df_catalog_rp.apply(lambda row: calculate_cosine_similarity(row['NAME'], row['NAME_INT']), axis=1)
    df_catalog_rp['SIMILARITY'] = df_catalog_rp[['JACCARD_SIMILARITY', 'LEV_DISTANCE', 'SIMILARITY_CONSINE']].median(axis=1)
    #df_output['SIMILARITY_CHATGPT'] = df_output.apply(lambda row: calculate_gpt_similarity(row['MP_NAME'], row['NAME_INT'], API_KEY), axis=1)

    df_catalog_rp = df_catalog_rp.drop(columns=['JACCARD_SIMILARITY', 'LEV_DISTANCE', 'SIMILARITY_CONSINE'])

    # aplicando funcoes de extract quantity e kg

    extract_quantity_from_product_name(df_catalog_rp, 'NAME_INT')
    verifica_variacoes_kg(df_catalog_rp, 'NAME_INT')

    df_catalog_rp_w_sot = df_catalog_rp[~df_catalog_rp['MASTER_PRODUCT_ID'].isin(mps_sot['MPS'])]

    print(df_catalog_rp_w_sot)
    groupby_cols = ['COUNTRY', 'NAME', 'MASTER_PRODUCT_ID', 'CAT1_NAME', 'CAT2_NAME', 'CAT3_NAME', 'MAKER', 'TRADEMARK', 'IN_SOT']
    df_catalog_rp[groupby_cols] = df_catalog_rp[groupby_cols].fillna('NULL_VALUE')
    df_catalog = df_catalog_rp.groupby(groupby_cols).agg({
            'FR': 'mean',
            'PRODUCT_DR': 'mean',
            'TOTAL_PRODUCTS': 'sum',
            'COUNT_REPLACED': 'sum',
            'GMV_TOTAL': 'sum',
            'PRODUCT_ID': 'nunique',
            'SIMILARITY': 'median',
            'MEDIAN_PRICE': 'median'
        }).reset_index()

    df_catalog = df_catalog.rename(columns = {'PRODUCT_ID': 'N_ASSOCIATED_RETAILERS'})

    df_catalog.columns = ['COUNTRY', 'NAME', 'MASTER_PRODUCT_ID', 'CAT1_NAME', 'CAT2_NAME', 'CAT3_NAME', 'MAKER', 'TRADEMARK', 'IN_SOT',
                        'M_FR', 'PRODUCT_DR', 'TOTAL_PRODUCTS', 'COUNT_REPLACED', 'GMV_T', 'N_ASSOCIATED_RETAILERS', 'SIMILARITY', 'MEDIAN_PRICE']


    ### filtrando a SOT

    df_catalog_sot = df_catalog[df_catalog['MASTER_PRODUCT_ID'].isin(mps_sot['MPS'])]
    df_catalog_w_sot = df_catalog[~df_catalog['MASTER_PRODUCT_ID'].isin(mps_sot['MPS'])]

    print(f'MPs dentro da SOT: {df_catalog_sot}')
    print(f'MPs fora da SOT: {df_catalog_w_sot}')

    return df_catalog_sot, df_catalog_rp, df_catalog_w_sot, df_catalog_rp_w_sot, df_catalog

