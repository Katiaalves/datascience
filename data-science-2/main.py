#!/usr/bin/env python
# coding: utf-8

# # Desafio 4
# 
# Neste desafio, vamos praticar um pouco sobre testes de hipóteses. Utilizaremos o _data set_ [2016 Olympics in Rio de Janeiro](https://www.kaggle.com/rio2016/olympic-games/), que contém dados sobre os atletas das Olimpíadas de 2016 no Rio de Janeiro.
# 
# Esse _data set_ conta com informações gerais sobre 11538 atletas como nome, nacionalidade, altura, peso e esporte praticado. Estaremos especialmente interessados nas variáveis numéricas altura (`height`) e peso (`weight`). As análises feitas aqui são parte de uma Análise Exploratória de Dados (EDA).
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[16]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns


# In[17]:


#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[3]:


athletes = pd.read_csv("athletes.csv")


# In[4]:


def get_sample(df, col_name, n=100, seed=42):
    """Get a sample from a column of a dataframe.
    
    It drops any numpy.nan entries before sampling. The sampling
    is performed without replacement.
    
    Example of numpydoc for those who haven't seen yet.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Source dataframe.
    col_name : str
        Name of the column to be sampled.
    n : int
        Sample size. Default is 100.
    seed : int
        Random seed. Default is 42.
    
    Returns
    -------
    pandas.Series
        Sample of size n from dataframe's column.
    """
    np.random.seed(seed)
    
    random_idx = np.random.choice(df[col_name].dropna().index, size=n, replace=False)
    
    return df.loc[random_idx, col_name]


# ## Inicia sua análise a partir daqui

# In[5]:


# Sua análise começa aqui.
athletes.head()


# In[6]:


athletes.isna().sum()


# In[7]:


athletes.shape


# In[8]:


athletes.describe()


# In[7]:


athletes.dtypes


# In[ ]:





# ## Questão 1
# 
# Considerando uma amostra de tamanho 3000 da coluna `height` obtida com a função `get_sample()`, execute o teste de normalidade de Shapiro-Wilk com a função `scipy.stats.shapiro()`. Podemos afirmar que as alturas são normalmente distribuídas com base nesse teste (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[8]:


altura = get_sample(athletes, 'height', n=3000, seed=42)
peso = get_sample(athletes, 'weight', n=3000)


# In[18]:


def q1():
    # Retorne aqui o resultado da questão 1.
    pass
        # Teste de normalidade de Shapiro-Wik
    shapiro = sct.shapiro(altura)
        # Comparando a p-value com a significancia de 5%
    return shapiro[1]>0.05
q1()


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Plote o qq-plot para essa variável e a analise.
# * Existe algum nível de significância razoável que nos dê outro resultado no teste? (Não faça isso na prática. Isso é chamado _p-value hacking_, e não é legal).

# In[13]:


#Plote o histograma dessa variável (com, por exemplo, bins=25). A forma do gráfico e o resultado do teste são condizentes? Por que?
# o histograma e o valor de curtose mostram que o resultado do teste q1 é condizente. Um valor de curtose < 0
# indica que a distribuição é mais achatada que a distribuição normal.
athletes['height'].hist(bins=25)
curtose = athletes['height'].kurtosis()
skew = athletes['height'].skew()
mean = athletes['height'].mean()
std = athletes['height'].std()
print(f'Curtose: {curtose} - Skew: {skew}')
print(f'Media: {mean} - STD: {std}')


# In[14]:


#Plote o qq-plot para essa variavel e analise
#Confirmando as outras analises, o qq-plot mostra uma variancia na cauda esquerda e na direita
import statsmodels.api as sm
sm.qqplot(athletes.height.dropna(), fit=True, line="45");


# ## Questão 2
# 
# Repita o mesmo procedimento acima, mas agora utilizando o teste de normalidade de Jarque-Bera através da função `scipy.stats.jarque_bera()`. Agora podemos afirmar que as alturas são normalmente distribuídas (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[23]:


def q2():
    # Retorne aqui o resultado da questão 2.
    samples = get_sample(athletes, col_name='height', n=3000)
    jb1, jb2 = sct.jarque_bera(samples)
    return bool(jb2 >0.05)
    
q2()
    


# __Para refletir__:
# 
# * Esse resultado faz sentido?

# Sim. O teste Jarque-Bera leva em consideração a curtose e a assimetria para determinar se a amostra é normal.
# #como vimos anteriormente, a curtose é < 0, logo conclui-se que a amostra não é normal, por isso o teste q2 retornou false

# ## Questão 3
# 
# Considerando agora uma amostra de tamanho 3000 da coluna `weight` obtida com a função `get_sample()`. Faça o teste de normalidade de D'Agostino-Pearson utilizando a função `scipy.stats.normaltest()`. Podemos afirmar que os pesos vêm de uma distribuição normal ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[18]:


def q3():
    # Retorne aqui o resultado da questão 3.
    samples = get_sample(athletes, col_name='weight', n=3000)
    normal_t1, normal_t2 = sct.normaltest(samples)
    return bool(normal_t2 > .05)
q3()


# In[20]:


#plote o historigrma dessa variavel 
# o histograma e o valor de curtose mostram que o resultado do teste q3 é condizente. Um valor de curtose > 0
# indica que a distribuição é mais acentuada que a distribuição normal.
athletes['weight'].hist(bins=25)
curtose = athletes['weight'].kurtosis()
skew=athletes['weight'].skew()
print(f'Curtose: {curtose} - skew: {skew}')


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Um _box plot_ também poderia ajudar a entender a resposta.

# ## Questão 4
# 
# Realize uma transformação logarítmica em na amostra de `weight` da questão 3 e repita o mesmo procedimento. Podemos afirmar a normalidade da variável transformada ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[21]:


def q4():
    # Retorne aqui o resultado da questão 4.
    log_weight = pd.DataFrame({'weight':np.log(athletes['weight'])})
    samples = get_sample(log_weight, col_name='weight', n=3000)
    normal_t1, normal_t2 = sct.normaltest(samples)
    
    sm.qqplot(log_weight['weight'].dropna(), fit=True, line='45');
    
    return bool(normal_t2 > .05)
q4()


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Você esperava um resultado diferente agora?

# > __Para as questão 5 6 e 7 a seguir considere todos testes efetuados ao nível de significância de 5%__.

# ## Questão 5
# 
# Obtenha todos atletas brasileiros, norte-americanos e canadenses em `DataFrame`s chamados `bra`, `usa` e `can`,respectivamente. Realize um teste de hipóteses para comparação das médias das alturas (`height`) para amostras independentes e variâncias diferentes com a função `scipy.stats.ttest_ind()` entre `bra` e `usa`. Podemos afirmar que as médias são estatisticamente iguais? Responda com um boolean (`True` ou `False`).

# In[29]:


bra = athletes[athletes['nationality']=='BRA']['height'].dropna()
usa = athletes[athletes['nationality']=='USA']['height'].dropna()
can = athletes[athletes['nationality']=='CAN']['height'].dropna()


# In[31]:


def q5():
    # Retorne aqui o resultado da questão 5.
   
    test_bra_usa = sct.ttest_ind(bra, usa, equal_var=False)
    return bool(test_bra_usa[1]>0.05) 
q5()


# ## Questão 6
# 
# Repita o procedimento da questão 5, mas agora entre as alturas de `bra` e `can`. Podemos afimar agora que as médias são estatisticamente iguais? Reponda com um boolean (`True` ou `False`).

# In[32]:


def q6():
    # Retorne aqui o resultado da questão 6.
    test_bra_can = sct.ttest_ind(bra,can, equal_var=False)
    return bool(test_bra_can[1]>0.05)
q6()


# ## Questão 7
# 
# Repita o procedimento da questão 6, mas agora entre as alturas de `usa` e `can`. Qual o valor do p-valor retornado? Responda como um único escalar arredondado para oito casas decimais.

# In[35]:


def q7():
    # Retorne aqui o resultado da questão 7.
    test_usa_can = sct.ttest_ind(usa,can, equal_var=False)
    return float(round(test_usa_can[1], 8))
q7()


# __Para refletir__:
# 
# * O resultado faz sentido?
# * Você consegue interpretar esse p-valor?
# * Você consegue chegar a esse valor de p-valor a partir da variável de estatística?

# In[ ]:




