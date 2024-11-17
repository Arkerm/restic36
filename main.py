import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Lasso, Ridge
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar o arquivo CSV
data = pd.read_csv("dados_limpos.csv")

# Selecionar as colunas que vamos utilizar como features
features = ['PONTO_DE_INFLUENCIA', 'POSTAGEM', 'SEGUDORES', 'MEDIA_CURTIDAS', 'NOVA_POST_MEDIA_CURTIDAS', 'TOTAL_CURTIDAS']
target = 'TAXA_ENG_60_DIAS'

# Remover símbolos %
data[target] = data[target].str.replace('%', '').astype(float) / 100

# Remover valores ausentes
data = data.dropna(subset=features + [target])

X = data[features].values  # Features
y = data[target].values    # Target (taxa de engajamento)

# Normalização
scaler = StandardScaler()
X = scaler.fit_transform(X)

#Correlação
correlation_matrix = data[features + [target]].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Matriz de Correlação")
plt.show()

corr_with_target = correlation_matrix[target].drop(target)
selected_features = corr_with_target[abs(corr_with_target) > 0.1].index.tolist()
print(f"Features selecionadas: {selected_features}")

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar os modelos Lasso e Ridge
lasso = Lasso(alpha=0.01)
ridge = Ridge(alpha=1.0)

# Treinar os modelos
lasso.fit(X_train, y_train)
ridge.fit(X_train, y_train)

# Previsões
y_pred_lasso = lasso.predict(X_test)
y_pred_ridge = ridge.predict(X_test)

# Calcular as métricas de desempenho
r2_lasso = r2_score(y_test, y_pred_lasso)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)

r2_ridge = r2_score(y_test, y_pred_ridge)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)

# Exibir as métricas de desempenho
print("Desempenho do Modelo Lasso:")
print(f"R²: {r2_lasso:.4f}")
print(f"MSE: {mse_lasso:.4f}")
print(f"MAE: {mae_lasso:.4f}")

print("\nDesempenho do Modelo Ridge:")
print(f"R²: {r2_ridge:.4f}")
print(f"MSE: {mse_ridge:.4f}")
print(f"MAE: {mae_ridge:.4f}")

# Interpretação dos Coeficientes (para Lasso e Ridge)
lasso_coef = pd.DataFrame({
    'Variável': selected_features,
    'Coeficiente': lasso.coef_
}).sort_values(by='Coeficiente', ascending=False)

ridge_coef = pd.DataFrame({
    'Variável': selected_features,
    'Coeficiente': ridge.coef_
}).sort_values(by='Coeficiente', ascending=False)

# Exibir os coeficientes dos modelos Lasso e Ridge
print("\nCoeficientes do Modelo Lasso:")
print(lasso_coef)

print("\nCoeficientes do Modelo Ridge:")
print(ridge_coef)

# Visualizações
# Modelo Lasso
residuals_lasso = y_test - y_pred_lasso

# Modelo Ridge
residuals_ridge = y_test - y_pred_ridge

# Gráfico de Resíduos para Lasso e Ridge
plt.figure(figsize=(14, 6))

# Resíduos Lasso
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_pred_lasso, y=residuals_lasso)
plt.axhline(0, color='r', linestyle='--')
plt.title("Resíduos do Modelo Lasso")
plt.xlabel("Previsões")
plt.ylabel("Resíduos")

# Resíduos Ridge
plt.subplot(1, 2, 2)
sns.scatterplot(x=y_pred_ridge, y=residuals_ridge)
plt.axhline(0, color='r', linestyle='--')
plt.title("Resíduos do Modelo Ridge")
plt.xlabel("Previsões")
plt.ylabel("Resíduos")

plt.tight_layout()
plt.show()

# Gráfico de Previsões vs Valores Reais para Lasso e Ridge
plt.figure(figsize=(14, 6))

# Previsões vs Reais para Lasso
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test, y=y_pred_lasso)
plt.plot([0, 1], [0, 1], color='r', linestyle='--')  # Linha diagonal
plt.title("Previsões vs Valores Reais (Lasso)")
plt.xlabel("Valores Reais")
plt.ylabel("Previsões")

# Previsões vs Reais para Ridge
plt.subplot(1, 2, 2)
sns.scatterplot(x=y_test, y=y_pred_ridge)
plt.plot([0, 1], [0, 1], color='r', linestyle='--')  # Linha diagonal
plt.title("Previsões vs Valores Reais (Ridge)")
plt.xlabel("Valores Reais")
plt.ylabel("Previsões")

plt.tight_layout()
plt.show()

# Comparação de Métricas
metrics = ['R²', 'MSE', 'MAE']
lasso_metrics = [r2_lasso, mse_lasso, mae_lasso]
ridge_metrics = [r2_ridge, mse_ridge, mae_ridge]

#DataFrame para as métricas
df_metrics = pd.DataFrame({
    'Métrica': metrics,
    'Lasso': lasso_metrics,
    'Ridge': ridge_metrics
})

# Plotar gráfico de barras
df_metrics.set_index('Métrica').plot(kind='bar', figsize=(10, 6), colormap='viridis')
plt.title("Comparação de Métricas entre Modelos Lasso e Ridge")
plt.ylabel("Valor das Métricas")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
