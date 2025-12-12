import cv2
import mediapipe as mp
import numpy as np
import joblib
import os

# --- CONFIGURAÇÃO ---
TAMANHO_BUFFER = 30  # Quantos frames ele "lembra" (30 frames = ~1 segundo de memória)

# --- Carregar IA ---
if not os.path.exists('modelo_tdah_svm.joblib'):
    print("ERRO: 'modelo_tdah_svm.joblib' não encontrado. Rode o treinador.py!")
    exit()

print("Carregando cérebro da IA...")
model = joblib.load('modelo_tdah_svm.joblib')
scaler = joblib.load('scaler_tdah_svm.joblib')

# --- Inicialização do MediaPipe ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# --- Loop Principal (WEBCAM) ---
cap = cv2.VideoCapture(0) # 0 = Webcam padrão
cap.set(3, 1280)
cap.set(4, 720)

buffer_pontos = [] 
estado_atual = "INICIANDO..."
cor_estado = (200, 200, 200)

print("Iniciando Webcam... Pressione ESC para sair.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Erro na webcam.")
        break
    
    # Espelhar a imagem (efeito espelho fica mais natural)
    image = cv2.flip(image, 1)
    altura_img, largura_img, _ = image.shape

    # Processamento MediaPipe
    image.flags.writeable = False
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    
    image.flags.writeable = True
    
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark
        
        # 1. Extrair Dados Brutos
        nariz = face_landmarks[1]
        nx, ny = nariz.x * largura_img, nariz.y * altura_img
        
        iris_esq = face_landmarks[468]
        ix, iy = iris_esq.x * largura_img, iris_esq.y * altura_img
        
        iris_dir = face_landmarks[473]
        dx, dy = iris_dir.x * largura_img, iris_dir.y * altura_img

        # Adiciona ao buffer
        buffer_pontos.append([nx, ny, ix, iy, dx, dy])

        # Se o buffer encheu, remove o frame mais antigo (Janela Deslizante)
        if len(buffer_pontos) > TAMANHO_BUFFER:
            buffer_pontos.pop(0)

        # --- A MÁGICA DA PREVISÃO ---
        # Só prevê se já tivermos dados suficientes (30 frames)
        if len(buffer_pontos) == TAMANHO_BUFFER:
            dados = np.array(buffer_pontos)

            # Calcular as 10 Features Estatísticas (Média do movimento recente)
            std_nariz_x = np.std(dados[:, 0])
            std_nariz_y = np.std(dados[:, 1])

            std_iris_esq_x = np.std(dados[:, 2])
            std_iris_esq_y = np.std(dados[:, 3])
            std_iris_dir_x = np.std(dados[:, 4])
            std_iris_dir_y = np.std(dados[:, 5])

            # Relativo (Olho - Nariz)
            rel_esq_x = dados[:, 2] - dados[:, 0]
            rel_esq_y = dados[:, 3] - dados[:, 1]
            rel_dir_x = dados[:, 4] - dados[:, 0]
            rel_dir_y = dados[:, 5] - dados[:, 1]

            std_rel_iris_esq_x = np.std(rel_esq_x)
            std_rel_iris_esq_y = np.std(rel_esq_y)
            std_rel_iris_dir_x = np.std(rel_dir_x)
            std_rel_iris_dir_y = np.std(rel_dir_y)

            # Vetor Final
            features_vetor = np.array([[
                std_nariz_x, std_nariz_y,
                std_iris_esq_x, std_iris_esq_y,
                std_iris_dir_x, std_iris_dir_y,
                std_rel_iris_esq_x, std_rel_iris_esq_y,
                std_rel_iris_dir_x, std_rel_iris_dir_y
            ]])

            # Normalizar e Prever
            features_scaled = scaler.transform(features_vetor)
            previsao = model.predict(features_scaled)[0]
            
            # Atualizar Texto na Tela
            if previsao == 'focado':
                estado_atual = "FOCADO"
                cor_estado = (0, 255, 0) # Verde
            else:
                estado_atual = "DISTRAIDO"
                cor_estado = (0, 0, 255) # Vermelho

    # --- Desenhar Interface ---
    # Texto
    cv2.putText(image, f"IA: {estado_atual}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, cor_estado, 3)
    
    # Barra de carregamento do buffer (só para visualização)
    progresso = len(buffer_pontos) / TAMANHO_BUFFER
    cv2.rectangle(image, (50, 100), (50 + int(200*progresso), 110), cor_estado, -1)

    cv2.imshow('Teste Webcam - TDAH', image)

    if cv2.waitKey(5) & 0xFF == 27: # ESC para sair
        break

cap.release()
cv2.destroyAllWindows()