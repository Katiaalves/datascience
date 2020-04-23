#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np


# In[4]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[5]:


black_friday.head()


# In[4]:


print(black_friday.info())


# In[6]:


black_friday.describe()


# Checados os valores das variáveis numéricas, podemos reduzir a memória ocupada convertendo os dtypes.

# In[6]:


from numpy import int32, float32

black_friday = black_friday.astype({
    "User_ID": int32,
    "Occupation": int32,
    "Marital_Status": int32,
    "Product_Category_1": int32,
    "Product_Category_2": float32,
    "Product_Category_3": float32,
    "Purchase": int32
    })


# In[7]:


black_friday.info()


# Houve uma redução de \~15 MB (\~29%) de uso de memória.

# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.
# 

# In[7]:


def q1():
    return black_friday.shape
#visualização do retorno
q1()


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[8]:


# quais são as idades das mulheres disponiveis no dataset?
black_friday[black_friday['Gender'] == 'F']['Age'].unique()


# In[9]:


black_friday_woman = black_friday[black_friday['Gender'] == 'F']
black_friday_woman_anos_26_35 = black_friday_woman[black_friday_woman['Age'] =='26-35']


# In[10]:


def q2():
    return black_friday[(black_friday['Age'] == '26-35')
                        & (black_friday['Gender'] == 'F')].shape[0]

q2()


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.
# #pandas.DataFrame.nunique
# 

# In[11]:


def q3():
    return black_friday['User_ID'].nunique()

q3()


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.
# pandas.DataFrame.dtypes
# pandas.DataFrame.nunique
# 

# In[12]:


print(black_friday.dtypes)


# In[11]:


def q4():
    return black_friday.dtypes.nunique()

q4()


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[12]:


def q5():
    return 1 - black_friday.dropna().shape[0]/black_friday.shape[0]

q5()


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[13]:


print(black_friday.isna().sum().max())


# In[14]:


def q6():
    return int(black_friday.isna().sum().max())

q6()


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[15]:


def q7():
    # mode() exclui null por padrão. Explicitei para clareza.
    return float(black_friday['Product_Category_3'].mode())

q7()


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[16]:


purchases = black_friday['Purchase'].copy()


# In[19]:


def q8():
    norm_max = purchases.max()
    norm_min = purchases.min()
    
    norm_purchase = (purchases - norm_min)                      / (norm_max - norm_min)
    
    return float(norm_purchase.mean())

q8()


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[20]:


def q9():
    std_purchase = (purchases - purchases.mean())                     / purchases.std()
    return int(std_purchase[(-1 <= std_purchase) & (std_purchase <= 1)].shape[0])

q9()


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[21]:


def q10():
      # basta comparar as duas séries de nan
    return black_friday['Product_Category_2'].isna().equals(black_friday['Product_Category_2'].isna())
    
# visualização do retorno
q10()


# In[ ]:




