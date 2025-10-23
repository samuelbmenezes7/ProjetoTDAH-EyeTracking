import cv2
import mediapipe as mp
import numpy as np
import time
import pandas as pd
import os

# --- Configurações ---
DURACAO_COLETA = 10 # Em segundos (comece com 10s para ser rápido, depois aumente)
NOME_ARQUIVO_SAIDA = 'features_data.csv'
LARGURA_CAM = 1280
ALTURA_CAM = 720

# --- Inicialização do MediaPipe ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True, # IMPORTANTE: Habilita os landmarks da íris
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# --- Função Principal de Coleta ---
def coletar_features(label):
    print(f"Iniciando coleta para a classe: '{label}'")
    print("Prepare-se... 3")
    time.sleep(1)
    print("2")
    time.sleep(1)
    print("1")
    time.sleep(1)
    print("GRAVANDO!")

    cap = cv2.VideoCapture(0)
    cap.set(3, LARGURA_CAM)
    cap.set(4, ALTURA_CAM)

    lista_pontos_nariz = []
    lista_pontos_iris_esq = []
    lista_pontos_iris_dir = []
    
    tempo_inicial = time.time()
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignorando frame vazio.")
            continue

        altura_img, largura_img, _ = image.shape
        
        # Para performance
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        
        # Para visualização
        image.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        tempo_atual = time.time()
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark

            # 1. Ponto do Nariz (Landmark 1)
            nariz = face_landmarks[1]
            lista_pontos_nariz.append([nariz.x * largura_img, nariz.y * altura_img])
            
            # 2. Pontos da Íris Esquerda (Landmarks 474-477) e Direita (469-472)
            # Vamos pegar o centro da íris (média dos pontos)
            
            # Índices da Íris Esquerda (do nosso ponto de vista)
            iris_esq_pontos = face_landmarks[474:478] 
            soma_x_esq = sum([p.x for p in iris_esq_pontos])
            soma_y_esq = sum([p.y for p in iris_esq_pontos])
            centro_iris_esq = [ (soma_x_esq / 4) * largura_img, (soma_y_esq / 4) * altura_img ]
            lista_pontos_iris_esq.append(centro_iris_esq)

            # Índices da Íris Direita (do nosso ponto de vista)
            iris_dir_pontos = face_landmarks[469:473]
            soma_x_dir = sum([p.x for p in iris_dir_pontos])
            soma_y_dir = sum([p.y for p in iris_dir_pontos])
            centro_iris_dir = [ (soma_x_dir / 4) * largura_img, (soma_y_dir / 4) * altura_img ]
            lista_pontos_iris_dir.append(centro_iris_dir)

            # Desenha a malha (opcional, mas bom para debug)
            mp_drawing.draw_landmarks(
                image=image_bgr,
                landmark_list=results.multi_face_landmarks[0],
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

        cv2.imshow('Coletor de Dados - MediaPipe', cv2.flip(image_bgr, 1))

        if cv2.waitKey(5) & 0xFF == 27: # ESC para cancelar
            cap.release()
            return None

        if tempo_atual - tempo_inicial > DURACAO_COLETA:
            print("Coleta finalizada!")
            cap.release()
            cv2.destroyAllWindows()
            break
            
    # --- Pós-Processamento: Cálculo das Features ---
    if not lista_pontos_nariz:
        print("Nenhum rosto detectado.")
        return None

    # Converter listas para arrays numpy
    np_nariz = np.array(lista_pontos_nariz)
    np_iris_esq = np.array(lista_pontos_iris_esq)
    np_iris_dir = np.array(lista_pontos_iris_dir)

    # 1. Movimento da Cabeça: Desvio padrão (std) da posição do nariz
    std_nariz_x = np.std(np_nariz[:, 0])
    std_nariz_y = np.std(np_nariz[:, 1])

    # 2. Movimento dos Olhos: Desvio padrão da posição das íris
    std_iris_esq_x = np.std(np_iris_esq[:, 0])
    std_iris_esq_y = np.std(np_iris_esq[:, 1])
    std_iris_dir_x = np.std(np_iris_dir[:, 0])
    std_iris_dir_y = np.std(np_iris_dir[:, 1])

    # 3. Movimento dos Olhos Relativo à Cabeça (Mais avançado, mas melhor)
    # Subtrai a posição do nariz da posição da íris
    rel_iris_esq = np_iris_esq - np_nariz
    rel_iris_dir = np_iris_dir - np_nariz
    
    std_rel_iris_esq_x = np.std(rel_iris_esq[:, 0])
    std_rel_iris_esq_y = np.std(rel_iris_esq[:, 1])
    std_rel_iris_dir_x = np.std(rel_iris_dir[:, 0])
    std_rel_iris_dir_y = np.std(rel_iris_dir[:, 1])

    features = {
        'std_nariz_x': std_nariz_x,
        'std_nariz_y': std_nariz_y,
        'std_iris_esq_x': std_iris_esq_x,
        'std_iris_esq_y': std_iris_esq_y,
        'std_iris_dir_x': std_iris_dir_x,
        'std_iris_dir_y': std_iris_dir_y,
        'std_rel_iris_esq_x': std_rel_iris_esq_x,
        'std_rel_iris_esq_y': std_rel_iris_esq_y,
        'std_rel_iris_dir_x': std_rel_iris_dir_x,
        'std_rel_iris_dir_y': std_rel_iris_dir_y,
        'label': label
    }
    
    return features

# --- Loop Principal para Coletar Amostras ---
if __name__ == "__main__":
    lista_de_todas_features = []
    
    # Carrega dados existentes se o arquivo já existir
    if os.path.exists(NOME_ARQUIVO_SAIDA):
        df_existente = pd.read_csv(NOME_ARQUIVO_SAIDA)
        lista_de_todas_features = df_existente.to_dict('records')
        print(f"Arquivo '{NOME_ARQUIVO_SAIDA}' carregado. {len(lista_de_todas_features)} amostras existentes.")

    while True:
        print("\n--- Coletor de Amostras para Projeto ---")
        print("Digite a classe/label para a próxima amostra (ex: 'focado' ou 'distraido')")
        print("Digite 'sair' para finalizar e salvar.")
        label_input = input("Label: ")

        if label_input.lower() == 'sair':
            break
            
        if label_input.strip() == "":
            print("Label não pode ser vazia.")
            continue

        novas_features = coletar_features(label_input)
        
        if novas_features:
            lista_de_todas_features.append(novas_features)
            print("Amostra adicionada com sucesso!")
            print(f"Total de amostras: {len(lista_de_todas_features)}")

    # Salvar tudo no CSV
    if lista_de_todas_features:
        df_final = pd.DataFrame(lista_de_todas_features)
        df_final.to_csv(NOME_ARQUIVO_SAIDA, index=False)
        print(f"Dados salvos com sucesso em '{NOME_ARQUIVO_SAIDA}'!")
    else:
        print("Nenhuma amostra nova para salvar.")