import cv2
import mediapipe as mp
import numpy as np
import time
import joblib
import webbrowser

# --- Configurações da Tarefa 1 ---
TOM_E_JERRY_URL = 'https://www.youtube.com/watch?v=NbmhjSxz-Ak' # Clipe de 3 minutos
DURACAO_TOTAL_TESTE = 194 # Em segundos (3 minutos)
JANELA_ANALISE = 10 # Analisar a cada 10 segundos
TEMPO_PREPARACAO = 5

# --- Configurações do MediaPipe e Modelo ---
LARGURA_CAM = 1280
ALTURA_CAM = 720
NOME_MODELO = 'modelo_tdah_svm.joblib'
NOME_SCALER = 'scaler_tdah_svm.joblib'

# --- Carregar Modelo e Scaler ---
try:
    model = joblib.load(NOME_MODELO)
    scaler = joblib.load(NOME_SCALER)
except FileNotFoundError:
    print("Erro: Arquivos de modelo ou scaler não encontrados.")
    print("Por favor, rode o script 'treinador.py' primeiro.")
    exit()

print("Modelo e Scaler carregados.")

# --- Inicialização do MediaPipe ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# --- Contagem Regressiva ---
print(f"--- TAREFA 1: VISUALIZACAO PASSIVA ---")
print(f"O teste começará em {TEMPO_PREPARACAO} segundos.")
print("Um vídeo do 'Tom e Jerry' será aberto. Assista a ele atentamente.")
for i in range(TEMPO_PREPARACAO, 0, -1):
    print(i)
    time.sleep(1)
print("INICIANDO TESTE!")

# --- Abrir Câmera e Vídeo ---
cap = cv2.VideoCapture(0)
cap.set(3, LARGURA_CAM)
cap.set(4, ALTURA_CAM)

webbrowser.open(TOM_E_JERRY_URL)

# --- Variáveis do Loop de Teste ---
lista_previsoes = []
tempo_inicio_teste = time.time()
tempo_inicio_janela = time.time()

lista_pontos_nariz = []
lista_pontos_iris_esq = []
lista_pontos_iris_dir = []

while cap.isOpened():
    tempo_atual = time.time()
    
    if (tempo_atual - tempo_inicio_teste) > DURACAO_TOTAL_TESTE:
        print("Teste da Tarefa 1 finalizado.")
        break

    success, image = cap.read()
    if not success:
        continue
    
    image = cv2.flip(image, 1) # Espelha a imagem
    altura_img, largura_img, _ = image.shape

    # Processamento MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark
        
        nariz = face_landmarks[1]
        lista_pontos_nariz.append([nariz.x * largura_img, nariz.y * altura_img])
        
        iris_esq_pontos = face_landmarks[474:478]
        centro_iris_esq = [
            (sum([p.x for p in iris_esq_pontos]) / 4) * largura_img,
            (sum([p.y for p in iris_esq_pontos]) / 4) * altura_img
        ]
        lista_pontos_iris_esq.append(centro_iris_esq)

        iris_dir_pontos = face_landmarks[469:473]
        centro_iris_dir = [
            (sum([p.x for p in iris_dir_pontos]) / 4) * largura_img,
            (sum([p.y for p in iris_dir_pontos]) / 4) * altura_img
        ]
        lista_pontos_iris_dir.append(centro_iris_dir)

    # --- Feedback Visual na Câmera ---
    tempo_total_decorrido = tempo_atual - tempo_inicio_teste
    progresso_total = tempo_total_decorrido / DURACAO_TOTAL_TESTE
    cv2.rectangle(image_bgr, (0, 0), (int(largura_img * progresso_total), 10), (0, 0, 255), -1)
    
    tempo_janela_decorrido = tempo_atual - tempo_inicio_janela
    progresso_janela = tempo_janela_decorrido / JANELA_ANALISE
    cv2.rectangle(image_bgr, (0, 15), (int(largura_img * progresso_janela), 25), (0, 255, 0), -1)

    cv2.putText(image_bgr, f"TAREFA 1 (TOM & JERRY): {int(tempo_total_decorrido)}s / {DURACAO_TOTAL_TESTE}s", (20, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    if lista_previsoes:
        cv2.putText(image_bgr, f"Ultima Previsao: {lista_previsoes[-1].upper()}", (20, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow('Camera de Analise - Tarefa 1', image_bgr)

    # --- Lógica de Previsão (A cada JANELA_ANALISE segundos) ---
    if tempo_janela_decorrido > JANELA_ANALISE:
        if not lista_pontos_nariz:
            previsao = "indeterminado"
        else:
            # Calcular Features
            np_nariz = np.array(lista_pontos_nariz)
            np_iris_esq = np.array(lista_pontos_iris_esq)
            np_iris_dir = np.array(lista_pontos_iris_dir)

            std_nariz_x = np.std(np_nariz[:, 0])
            std_nariz_y = np.std(np_nariz[:, 1])
            std_iris_esq_x = np.std(np_iris_esq[:, 0])
            std_iris_esq_y = np.std(np_iris_esq[:, 1])
            std_iris_dir_x = np.std(np_iris_dir[:, 0])
            std_iris_dir_y = np.std(np_iris_dir[:, 1])
            
            rel_iris_esq = np_iris_esq - np_nariz
            rel_iris_dir = np_iris_dir - np_nariz
            std_rel_iris_esq_x = np.std(rel_iris_esq[:, 0])
            std_rel_iris_esq_y = np.std(rel_iris_esq[:, 1])
            std_rel_iris_dir_x = np.std(rel_iris_dir[:, 0])
            std_rel_iris_dir_y = np.std(rel_iris_dir[:, 1])

            features_array = np.array([[
                std_nariz_x, std_nariz_y,
                std_iris_esq_x, std_iris_esq_y,
                std_iris_dir_x, std_iris_dir_y,
                std_rel_iris_esq_x, std_rel_iris_esq_y,
                std_rel_iris_dir_x, std_rel_iris_dir_y
            ]])
            
            features_scaled = scaler.transform(features_array)
            previsao = model.predict(features_scaled)[0]
        
        lista_previsoes.append(previsao)
        print(f"Janela {len(lista_previsoes)} - Previsao: {previsao}")

        # Reiniciar a janela
        tempo_inicio_janela = time.time()
        lista_pontos_nariz = []
        lista_pontos_iris_esq = []
        lista_pontos_iris_dir = []

    if cv2.waitKey(5) & 0xFF == 27: # ESC para cancelar
        break

# --- Finalização e Relatório ---
cap.release()
cv2.destroyAllWindows()

print("\n\n--- RELATORIO FINAL DA TAREFA 1 ---")
if not lista_previsoes:
    print("Nenhuma previsao foi registrada.")
else:
    total_janelas = len(lista_previsoes)
    janelas_focadas = lista_previsoes.count('focado')
    percentual_foco = (janelas_focadas / total_janelas) * 100
    
    print(f"Total de janelas de analise: {total_janelas}")
    print(f" - Janelas 'Focado': {janelas_focadas}")
    print(f" - Janelas 'Distraido': {lista_previsoes.count('distraido')}")
    print(f" - Janelas 'Indeterminado': {lista_previsoes.count('indeterminado')}")
    print(f"\nPERCENTUAL DE FOCO (durante Tom & Jerry): {percentual_foco:.2f}%")