import pandas as pd

tabela = pd.read_csv(r"local do arquivo")
display(tabela)
print(tabela.info())

import matplotlib.pyplot as plt
import seaborn as sns

print(tabela.corr())

# Criar um gráfico
sns.heatmap(tabela.corr(), cmap="Blues", annot=True)

# exibe o gráfico
plt.show()

y = tabela["Vendas"]
x = tabela[["TV", "Radio", "Jornal"]]

from sklearn.model_selection import train_test_split

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y)

#Importar I.A.
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

#Criar a inteligencia
modelo_regressaolinear = LinearRegression()
modelo_arvoredecisao = RandomForestRegressor()

#Treinar inteligencia
modelo_regressaolinear.fit(x_treino, y_treino)
modelo_arvoredecisao.fit(x_treino, y_treino)

# Testando inteligência
# Criar pervisão
previsao_regressaolinear = modelo_regressaolinear.predict(x_teste)
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)

from sklearn.metrics import r2_score

print(r2_score(y_teste, previsao_regressaolinear))
print(r2_score(y_teste, previsao_arvoredecisao))

# Visualização gráfica das previsões
tabela_auxiliar = pd.DataFrame()
tabela_auxiliar["y_teste"] = y_teste
tabela_auxiliar["previsao Arvore Decisao"] = previsao_arvoredecisao
tabela_auxiliar["Previsao RegressaoLinear"] = previsao_regressaolinear

sns.lineplot(data=tabela_auxiliar)
plt.show()

# Fazendo nova precisão
nova_tabela = pd.read_csv(r"Local para nova tabela de comparação")
display(nova_tabela)

# Fazendo nova precisão

previsao = modelo_arvoredecisao.predict(nova_tabela)
print(previsao)
