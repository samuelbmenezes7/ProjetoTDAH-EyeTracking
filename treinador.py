import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib # Para salvar o modelo

# --- Configurações ---
NOME_ARQUIVO_DADOS = 'features_data.csv'
NOME_MODELO_SAIDA = 'modelo_tdah_svm.joblib'
NOME_SCALER_SAIDA = 'scaler_tdah_svm.joblib'

# 1. Carregar os Dados
try:
    df = pd.read_csv(NOME_ARQUIVO_DADOS)
except FileNotFoundError:
    print(f"Erro: Arquivo '{NOME_ARQUIVO_DADOS}' não encontrado.")
    print("Por favor, rode o script 'coletor.py' primeiro para gerar os dados.")
    exit()

if len(df) < 10: # Mínimo para treinar
    print(f"Você só tem {len(df)} amostras. É recomendado ter pelo menos 10 (5 de cada classe).")
    print("Por favor, rode o 'coletor.py' para coletar mais dados.")
    exit()

print(f"Dados carregados. {len(df)} amostras encontradas.")

# 2. Preparar os Dados para o ML
X = df.drop('label', axis=1) # Features (as colunas de 'std_')
y = df['label']              # Labels (o que queremos prever: 'focado', 'distraido')

# 3. Dividir em Treino e Teste
# random_state=42 garante que a divisão seja sempre a mesma (reprodutibilidade)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Dados divididos: {len(X_train)} para treino, {len(X_test)} para teste.")

# 4. Normalização (StandardScaler) - Muito importante para SVM!
# Ajusta a escala dos dados para que o modelo funcione melhor
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Treinar o Modelo (usando SVM - Support Vector Machine)
print("Treinando o modelo SVM...")
model = SVC(kernel='linear', C=1.0, probability=True) # Probabilidade é útil para ter confiança
model.fit(X_train_scaled, y_train)
print("Modelo treinado!")

# 6. Avaliar o Modelo
y_pred = model.predict(X_test_scaled)
precisao = accuracy_score(y_test, y_pred)

print("\n--- Avaliação do Modelo ---")
print(f"Acurácia no set de teste: {precisao * 100:.2f}%")

# Se tivermos dados de teste suficientes, mostramos o relatório
if len(y_test) > 0:
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

# 7. Salvar o Modelo e o Scaler para usar em tempo real
joblib.dump(model, NOME_MODELO_SAIDA)
joblib.dump(scaler, NOME_SCALER_SAIDA)

print(f"Modelo salvo como '{NOME_MODELO_SAIDA}'")
print(f"Scaler salvo como '{NOME_SCALER_SAIDA}'")
print("--- Treinamento Concluído ---")