import pandas as pd
import numpy as np
#import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct  # Leading Juice for us
#from sparse_dot_topn import array_wrappers
import time
import jellyfish as j
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import re
from scipy.sparse import csr_matrix
import math
from fuzzywuzzy import fuzz
import time

def run_script_similarity(df_opportunities, df_sot):

    def awesome_cossim_top(A, B, ntop, lower_bound=0):
        A = A.tocsr()
        B = B.tocsr()
        M, _ = A.shape
        _, N = B.shape

        idx_dtype = np.int32

        nnz_max = M * ntop

        indptr = np.zeros(M + 1, dtype=idx_dtype)
        indices = np.zeros(nnz_max, dtype=idx_dtype)
        data = np.zeros(nnz_max, dtype=A.dtype)

        ct.sparse_dot_topn(
            M, N, np.asarray(A.indptr, dtype=idx_dtype),
            np.asarray(A.indices, dtype=idx_dtype),
            A.data,
            np.asarray(B.indptr, dtype=idx_dtype),
            np.asarray(B.indices, dtype=idx_dtype),
            B.data,
            ntop,
            lower_bound,
            indptr, indices, data
        )
        return csr_matrix((data, indices, indptr), shape=(M, N))

    def get_matches_df(sparse_matrix, A, B, top=100):
        non_zeros = sparse_matrix.nonzero()

        sparserows = non_zeros[0]
        sparsecols = non_zeros[1]

        if top:
            nr_matches = top
        else:
            nr_matches = sparsecols.size

        left_side = np.empty([nr_matches], dtype=object)
        right_side = np.empty([nr_matches], dtype=object)
        similarity = np.zeros(nr_matches)

        for index in range(0, nr_matches):
            left_side[index] = A[sparserows[index]]
            right_side[index] = B[sparsecols[index]]
            similarity[index] = sparse_matrix.data[index]

        return pd.DataFrame({'integration': left_side,
                             'master_products': right_side,
                             'similarity': similarity})

    df_opportunities.rename(columns={'NAME': 'NAME'}, inplace=True)
    df_cat = pd.DataFrame(df_opportunities[['NAME', 'MEDIAN_PRICE', 'SIMILARITY', 'MASTER_PRODUCT_ID']])
    df_cat['NAME'] = df_cat['NAME'].str.lower().str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8').str.replace(r'[,"&]', '')

    df_sot.rename(columns={'NAME': 'NAME_MP'}, inplace=True)
    df_cat_m = pd.DataFrame(df_sot[['NAME_MP', 'MEDIAN_PRICE', 'SIMILARITY', 'MASTER_PRODUCT_ID']])
    df_cat_m['NAME_MP'] = df_cat_m['NAME_MP'].str.lower().str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8').str.replace(r'[,"&]', '')
    df_cat_m.rename(columns={'NAME_MP': 'NAME_I'}, inplace=True)

    def ngrams(string, n=3):
        string = re.sub(r'^\$[a-zA-Z0-9_]+\$|[,-./]|\sBD', r'', string)
        ngrams = zip(*[string[i:] for i in range(n)])
        return [''.join(ngram) for ngram in ngrams]

    vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)

    catalog_name = df_cat['NAME']
    tf_idf_matrix_cat = vectorizer.fit_transform(catalog_name)

    integration_name = df_cat_m['NAME_I']
    tf_idf_matrix_int = vectorizer.transform(integration_name)

    print('Executing similarity algorithm ... ', end='\r')
    start_at = time.time()
    matches = awesome_cossim_top(tf_idf_matrix_int, tf_idf_matrix_cat.transpose(), 10, 0)
    matches_time = round(time.time() - start_at, 2)
    print("Executing similarity algorithm ... completed in {} seconds.".format(str(matches_time)))

    print('Obtaining matches ... ', end='\r')
    start_at = time.time()
    integration_name = df_cat_m['NAME_I'].reset_index(drop=True)
    catalog_name = df_cat['NAME'].reset_index(drop=True)
    matches_df = get_matches_df(matches, integration_name, catalog_name, top=0)
    matches_time = round(time.time() - start_at, 2)
    print("Obtaining matches ... completed in {} seconds.".format(str(matches_time)))

    print('Cleaning data frame ... ', end='\r')
    start_at = time.time()

    matches_df.sort_values(['similarity'], ascending=False)
    output_df = pd.DataFrame()
    output_df['similarity'] = matches_df['similarity']
    output_df['master_products'] = matches_df['master_products'].str.replace(r'\~{1,}$', '', regex=True)
    output_df['integration'] = matches_df['integration'].str.replace(r'\~{1,}$', '', regex=True)
    matches_time = round(time.time() - start_at, 2)
    print("Cleaning data frame ... completed in {} seconds.".format(str(matches_time)))

    print('Calculating alternative algorithms ... ', end='\r')
    start_at = time.time()
    output_df['damerau_levenshtein'] = output_df.apply(lambda row: j.damerau_levenshtein_distance(row['integration'], row['master_products']), axis=1)
    output_df['fuzz_token_sort_ratio'] = output_df.apply(lambda row: fuzz.token_sort_ratio(row['integration'], row['master_products']) / 100, axis=1)
    output_df['fuzz_token_set_ratio'] = output_df.apply(lambda row: fuzz.token_set_ratio(row['integration'], row['master_products']) / 100, axis=1)
    output_df['damerau_levenshtein_normalized'] = output_df.apply(lambda row: round(1 - (1 * math.exp(-15 * math.exp(-0.16 * row['damerau_levenshtein']))), 8), axis=1)
    matches_time = round(time.time() - start_at, 2)
    print("Calculating alternative algorithms ... completed in {} seconds.".format(str(matches_time)))

    print('Calculating accuracy ... ', end='\r')
    start_at = time.time()

    w_similarity = 5
    w_token_sort_ratio = 2
    w_token_set_ratio = 0.5
    w_levenshtein = 1
    w_price_similarity = 1
    w_config_similarity = 3
    w_total = w_similarity + w_token_set_ratio + w_token_sort_ratio + w_levenshtein + w_price_similarity + w_config_similarity

    def price_similarity(row):
        price_integration = df_cat_m.loc[df_cat_m['NAME_I'] == row['integration'], 'MEDIAN_PRICE'].values[0]
        price_master = df_cat.loc[df_cat['NAME'] == row['master_products'], 'MEDIAN_PRICE'].values[0]
        return 1 - abs(price_integration - price_master) / max(price_integration, price_master)

    def config_similarity(row):
        return df_cat_m.loc[df_cat_m['NAME_I'] == row['integration'], 'SIMILARITY'].values[0]

    output_df['price_similarity'] = output_df.apply(price_similarity, axis=1)
    output_df['config_similarity'] = output_df.apply(config_similarity, axis=1)
    output_df['master_product_id'] = matches_df.apply(lambda row: df_cat.loc[df_cat['NAME'] == row['master_products'], 'MASTER_PRODUCT_ID'].values[0], axis=1)
    output_df['master_product_id_sot'] = matches_df.apply(lambda row: df_cat_m.loc[df_cat_m['NAME_I'] == row['integration'], 'MASTER_PRODUCT_ID'].values[0], axis=1)

    output_df['final_weighted_confidence'] = output_df.apply(lambda row: ((row['similarity'] * w_similarity)
                                                                         + (row['damerau_levenshtein_normalized'] * w_levenshtein)
                                                                         + (row['fuzz_token_sort_ratio'] * w_token_sort_ratio)
                                                                         + (row['fuzz_token_set_ratio'] * w_token_set_ratio)
                                                                         + (row['price_similarity'] * w_price_similarity)
                                                                         + (row['config_similarity'] * w_config_similarity)) / w_total, axis=1)
    matches_time = round(time.time() - start_at, 2)
    print("Calculating accuracy ... completed in {} seconds.".format(str(matches_time)))

    output_df.sort_values(['final_weighted_confidence'], ascending=False)

    print(output_df)
    return output_df

