import cv2
import mediapipe as mp
import numpy as np
import time
import joblib

# --- Configurações ---
DURACAO_PREVISAO = 10 # Em segundos
LARGURA_CAM = 1280
ALTURA_CAM = 720
NOME_MODELO = 'modelo_tdah_svm.joblib'
NOME_SCALER = 'scaler_tdah_svm.joblib'

# --- Carregar Modelo e Scaler ---
try:
    model = joblib.load(NOME_MODELO)
    scaler = joblib.load(NOME_SCALER)
except FileNotFoundError:
    print("Erro: Arquivos de modelo ('modelo_tdah_svm.joblib') ou scaler ('scaler_tdah_svm.joblib') não encontrados.")
    print("Por favor, rode o script 'treinador.py' primeiro.")
    exit()

print("Modelo e Scaler carregados. Iniciando DEMO.")

# --- Inicialização do MediaPipe ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True, # Habilita íris
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# --- Loop Principal da Demo ---
cap = cv2.VideoCapture(0)
cap.set(3, LARGURA_CAM)
cap.set(4, ALTURA_CAM)

estado_atual = "ANALISANDO..."
cor_estado = (0, 255, 255) # Amarelo

tempo_inicio_ciclo = time.time()
lista_pontos_nariz = []
lista_pontos_iris_esq = []
lista_pontos_iris_dir = []

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue
    image = cv2.flip(image, 1)
    altura_img, largura_img, _ = image.shape
    tempo_atual = time.time()
    
    # Feedback visual do tempo
    tempo_decorrido = tempo_atual - tempo_inicio_ciclo
    progresso = tempo_decorrido / DURACAO_PREVISAO
    cv2.rectangle(image, (0, 0), (int(largura_img * progresso), 10), (0, 255, 0), -1)
    
    # Processamento MediaPipe
    image.flags.writeable = False
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    
    image.flags.writeable = True
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark
        
        # Coletar pontos (igual ao coletor.py)
        nariz = face_landmarks[1]
        lista_pontos_nariz.append([nariz.x * largura_img, nariz.y * altura_img])
        
        iris_esq_pontos = face_landmarks[474:478]
        soma_x_esq = sum([p.x for p in iris_esq_pontos])
        soma_y_esq = sum([p.y for p in iris_esq_pontos])
        centro_iris_esq = [ (soma_x_esq / 4) * largura_img, (soma_y_esq / 4) * altura_img ]
        lista_pontos_iris_esq.append(centro_iris_esq)

        iris_dir_pontos = face_landmarks[469:473]
        soma_x_dir = sum([p.x for p in iris_dir_pontos])
        soma_y_dir = sum([p.y for p in iris_dir_pontos])
        centro_iris_dir = [ (soma_x_dir / 4) * largura_img, (soma_y_dir / 4) * altura_img ]
        lista_pontos_iris_dir.append(centro_iris_dir)

        # Desenha a malha
        mp_drawing.draw_landmarks(
            image=image_bgr,
            landmark_list=results.multi_face_landmarks[0],
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)

    # Mostrar o estado atual
    cv2.putText(image_bgr, f"ESTADO: {estado_atual}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, cor_estado, 3)

    cv2.imshow('Demo TDAH - Projeto', image_bgr)

    # --- Lógica de Previsão ---
    if tempo_decorrido > DURACAO_PREVISAO:
        if not lista_pontos_nariz:
            print("Nenhum rosto detectado no ciclo.")
            estado_atual = "ROSTO NAO DETECTADO"
            cor_estado = (0, 0, 255) # Vermelho
        else:
            # 1. Calcular Features (exatamente como no coletor)
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

            # Criar o vetor de features para o modelo
            features_array = np.array([[
                std_nariz_x, std_nariz_y,
                std_iris_esq_x, std_iris_esq_y,
                std_iris_dir_x, std_iris_dir_y,
                std_rel_iris_esq_x, std_rel_iris_esq_y,
                std_rel_iris_dir_x, std_rel_iris_dir_y
            ]])
            
            # 2. Escalar as Features (IMPORTANTE!)
            features_scaled = scaler.transform(features_array)
            
            # 3. Fazer a Previsão
            previsao = model.predict(features_scaled)[0]
            probabilidade = model.predict_proba(features_scaled)
            
            confianca = np.max(probabilidade) * 100
            
            estado_atual = f"{previsao.upper()} ({confianca:.0f}%)"
            
            if previsao == 'focado':
                cor_estado = (0, 255, 0) # Verde
            else:
                cor_estado = (0, 165, 255) # Laranja

        # Reiniciar o ciclo
        tempo_inicio_ciclo = time.time()
        lista_pontos_nariz = []
        lista_pontos_iris_esq = []
        lista_pontos_iris_dir = []

    if cv2.waitKey(5) & 0xFF == 27: # ESC para sair
        break

cap.release()
cv2.destroyAllWindows()