import pandas as pd

df = pd.read_csv('top_influencers.csv',sep=",")

'''
print(df.head())
print(df.isnull().sum())
df.dropna(subset=['country'], inplace=True)
print(df.isnull().sum())
print(df.describe())
print(df.duplicated().sum())
print(df.dtypes)
'''
#df['60_day_eng_rate'] = df['60_day_eng_rate'].str.replace('%', '').astype(float) / 100
print(df.isnull().sum())
df.dropna(subset=['country'], inplace=True)

# Função para remover letras e converter os valores
def remove_suffix(value):
    if isinstance(value, str):
        if 'k' in value:
            return float(value.replace('k', '')) * 1000  # Converte 'k' em milhar
        elif 'm' in value:
            return float(value.replace('m', '')) * 1000000  # Converte 'm' em milhão
        elif 'b' in value:
            return float(value.replace('b', '')) * 1000000000  # Converte 'b' em bilhão
        else:
            return float(value)  # Se não tiver sufixo, apenas converte para float
    return value

# Aplicando a função nas colunas que têm sufixos
df['posts'] = df['posts'].apply(remove_suffix)
df['followers'] = df['followers'].apply(remove_suffix)
df['avg_likes'] = df['avg_likes'].apply(remove_suffix)
df['new_post_avg_like'] = df['new_post_avg_like'].apply(remove_suffix)
df['total_likes'] = df['total_likes'].apply(remove_suffix)

def reduzir_decimais(valor):

    if isinstance(valor, (int, float)):
        if valor >= 1000000000:  # Se for maior ou igual a 1 bilhão
            return round(valor / 1000000000, 1)  # Divide por 1 bilhão e arredonda
        elif valor >= 1000000:  # Se for maior ou igual a 1 milhão
            return round(valor / 1000000, 1)  # Divide por 1 milhão e arredonda
        elif valor >= 1000:  # Se for maior ou igual a 1 milhar
            return round(valor / 1000, 1)  # Divide por 1 milhar e arredonda
        else:
            return round(valor, 1)  # Se for menor que milhar, mantém o valor com 1 casa decimal
    return valor

df['posts'] = df['posts'].apply(reduzir_decimais)
df['followers'] = df['followers'].apply(reduzir_decimais)
df['avg_likes'] = df['avg_likes'].apply(reduzir_decimais)
df['new_post_avg_like'] = df['new_post_avg_like'].apply(reduzir_decimais)
df['total_likes'] = df['total_likes'].apply(reduzir_decimais)


df.columns = ['RANK', 'NOME_CANAL', 'PONTO_DE_INFLUENCIA', 'POSTAGEM', 'SEGUDORES', 'MEDIA_CURTIDAS', 'TAXA_ENG_60_DIAS', 
              'NOVA_POST_MEDIA_CURTIDAS', 'TOTAL_CURTIDAS', 'PAIS']

#print(df.head(1))

#Salvar
df.to_csv('dados_limpos.csv', index=False)
print('Arquivo CSV salvo com sucesso!')
