import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

# --- Configurações ---
PASTA_VIDEOS = 'videos'  # Pasta onde você salvou os mp4
NOME_ARQUIVO_SAIDA = 'features_data.csv' # Salva no mesmo arquivo para juntar com os seus dados
LARGURA_PADRAO = 1280
ALTURA_PADRAO = 720

# --- Inicialização do MediaPipe ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# --- Função de Processamento ---
def processar_video(nome_arquivo, label):
    caminho_completo = os.path.join(PASTA_VIDEOS, nome_arquivo)
    
    # Verifica se o arquivo existe
    if not os.path.exists(caminho_completo):
        print(f"ERRO: Arquivo '{caminho_completo}' não encontrado!")
        return None

    print(f"--> Processando: {nome_arquivo} como '{label}'...")
    
    cap = cv2.VideoCapture(caminho_completo)
    
    lista_pontos_nariz = []
    lista_pontos_iris_esq = []
    lista_pontos_iris_dir = []
    
    frame_count = 0
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break # Fim do vídeo

        frame_count += 1
        
        # Opcional: Redimensionar para manter consistência com a webcam (1280x720)
        image = cv2.resize(image, (LARGURA_PADRAO, ALTURA_PADRAO))

        # Converter para RGB para o MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
            altura_img, largura_img, _ = image.shape

            # 1. Nariz
            nariz = face_landmarks[1]
            lista_pontos_nariz.append([nariz.x * largura_img, nariz.y * altura_img])
            
            # 2. Íris (Centróide)
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

        # Mostra o processamento rápido (1ms de delay)
        cv2.imshow('Processando Video...', image)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # --- Pós-Processamento (Cálculo Estatístico) ---
    if not lista_pontos_nariz:
        print("ALERTA: Nenhum rosto detectado neste vídeo.")
        return None

    # Converter para numpy
    np_nariz = np.array(lista_pontos_nariz)
    np_iris_esq = np.array(lista_pontos_iris_esq)
    np_iris_dir = np.array(lista_pontos_iris_dir)

    # Cálculo das Features (Igual ao coletor original)
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
    
    print(f"Sucesso! Processados {frame_count} frames.")
    return features

# --- Loop Principal ---
if __name__ == "__main__":
    lista_de_todas_features = []
    
    # Carrega dados existentes para não perder o que você já fez
    if os.path.exists(NOME_ARQUIVO_SAIDA):
        try:
            df_existente = pd.read_csv(NOME_ARQUIVO_SAIDA)
            lista_de_todas_features = df_existente.to_dict('records')
            print(f"Carregado: {len(lista_de_todas_features)} amostras anteriores.")
        except:
            print("Arquivo CSV vazio ou inválido. Começando do zero.")

    while True:
        print("\n--- Processador de Vídeos para TDAH ---")
        print(f"Certifique-se que seus vídeos estão na pasta: '/{PASTA_VIDEOS}'")
        nome_arquivo = input("Nome do arquivo (ex: 'menina_tdah.mp4') ou 'sair': ")
        
        if nome_arquivo.lower() == 'sair':
            break
            
        label = input(f"Qual a classificação para '{nome_arquivo}'? (focado/distraido): ")
        
        resultado = processar_video(nome_arquivo, label)
        
        if resultado:
            lista_de_todas_features.append(resultado)
            print("Dados adicionados à lista temporária.")
            
            # Salva a cada vídeo processado para garantir
            df_final = pd.DataFrame(lista_de_todas_features)
            df_final.to_csv(NOME_ARQUIVO_SAIDA, index=False)
            print(f"--> CSV Atualizado! Total de amostras: {len(df_final)}")