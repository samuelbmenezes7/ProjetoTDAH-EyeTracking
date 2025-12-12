import customtkinter as ctk
import cv2
import mediapipe as mp
import numpy as np
import joblib
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import webbrowser
import os
import time

# --- Configuração Visual ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class AppAnaliseTDAH(ctk.CTk):
    def __init__(self):
        super().__init__()

        # 1. Configuração da Janela Principal
        self.title("Sistema de Triagem TDAH - Versão Final (Sem Travamentos)")
        self.geometry("1280x720")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # 2. Carregar IA
        try:
            self.modelo = joblib.load('modelo_tdah_svm.joblib')
            self.scaler = joblib.load('scaler_tdah_svm.joblib')
            print("IA Carregada: OK")
        except:
            print("ERRO: Rode 'python treinador.py' primeiro!")
            self.destroy()
            return

        # 3. MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

        # 4. Variáveis de Controle
        self.cap = None
        self.rodando = False
        self.tempo_inicio = 0
        
        # Variáveis de Janelas (Para evitar travamento)
        self.janela_tarefa = None
        self.janela_relatorio = None 

        # Buffers de Dados
        self.buffer_pontos = []      
        self.historico_heatmap_x = [] 
        self.historico_heatmap_y = [] 
        self.historico_classificacao = [] 
        self.frames_sem_rosto = 0

        # === MENU LATERAL ===
        self.frame_menu = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.frame_menu.grid(row=0, column=0, sticky="nswe")

        self.lbl_titulo = ctk.CTkLabel(self.frame_menu, text="NeuroVision Pro", font=("Roboto", 22, "bold"))
        self.lbl_titulo.grid(row=0, column=0, padx=20, pady=30)

        self.btn_arquivo = ctk.CTkButton(self.frame_menu, text="📂 Analisar Vídeo", command=self.escolher_arquivo)
        self.btn_arquivo.grid(row=1, column=0, padx=20, pady=5)

        self.btn_webcam = ctk.CTkButton(self.frame_menu, text="📷 Webcam Livre", command=lambda: self.iniciar_analise(0))
        self.btn_webcam.grid(row=2, column=0, padx=20, pady=5)

        self.lbl_tarefas = ctk.CTkLabel(self.frame_menu, text="--- TAREFAS ---", font=("Roboto", 12))
        self.lbl_tarefas.grid(row=4, column=0, pady=(20, 5))

        self.btn_tarefa1 = ctk.CTkButton(self.frame_menu, text="1️⃣ Tarefa Vídeo (3 min)", 
                                         command=self.iniciar_tarefa_1, fg_color="#E59113", hover_color="#B8730D")
        self.btn_tarefa1.grid(row=5, column=0, padx=20, pady=5)

        self.btn_tarefa2 = ctk.CTkButton(self.frame_menu, text="2️⃣ Tarefa Busca", 
                                         command=self.iniciar_tarefa_2, fg_color="#8E2CC9", hover_color="#70229F")
        self.btn_tarefa2.grid(row=6, column=0, padx=20, pady=5)

        self.btn_parar = ctk.CTkButton(self.frame_menu, text="⏹ GERAR RELATÓRIO", 
                                       command=self.parar_analise, fg_color="#C92C2C", state="disabled")
        self.btn_parar.grid(row=8, column=0, padx=20, pady=30)

        # Área de Vídeo
        self.frame_video = ctk.CTkFrame(self)
        self.frame_video.grid(row=0, column=1, padx=20, pady=20, sticky="nswe")
        self.label_video = ctk.CTkLabel(self.frame_video, text="Selecione um teste ao lado", corner_radius=10)
        self.label_video.pack(expand=True, fill="both", padx=10, pady=10)

    # --- Funções de Controle ---

    def limpar_janelas_anteriores(self):
        """Fecha janelas de relatórios ou tarefas antigas para não travar"""
        if self.janela_relatorio is not None:
            try:
                self.janela_relatorio.destroy()
            except:
                pass
            self.janela_relatorio = None
            
        if self.janela_tarefa is not None:
            try:
                self.janela_tarefa.destroy()
            except:
                pass
            self.janela_tarefa = None
            
        # Limpa memória do Matplotlib
        plt.close('all') 

    def iniciar_tarefa_1(self):
        """TAREFA 1: Vídeo do Tom & Jerry"""
        if self.rodando: return
        self.limpar_janelas_anteriores() # Limpeza antes de começar
        
        url_video = "https://www.youtube.com/watch?v=NbmhjSxz-Ak" 
        webbrowser.open(url_video)
        
        self.iniciar_analise(0) 
        self.after(180000, self.parar_analise) # 3 minutos

    def iniciar_tarefa_2(self):
        """TAREFA 2: Imagem local"""
        if self.rodando: return
        self.limpar_janelas_anteriores() # Limpeza antes de começar
        
        caminho_imagem = r"C:\Users\samue\OneDrive\Área de Trabalho\ProjetoTDAH\imagem\1cff69577b94991d1083550ab5.jpg"
        
        if not os.path.exists(caminho_imagem):
            self.label_video.configure(text=f"ERRO: Imagem não encontrada:\n{caminho_imagem}")
            return
        
        self.janela_tarefa = ctk.CTkToplevel(self)
        self.janela_tarefa.title("Encontre o Alvo")
        self.janela_tarefa.geometry("1000x800")
        
        try:
            img_pil = Image.open(caminho_imagem)
            img_ctk = ctk.CTkImage(img_pil, size=(1000, 800))
            lbl_img = ctk.CTkLabel(self.janela_tarefa, text="", image=img_ctk)
            lbl_img.pack(fill="both", expand=True)
            
            # Ao fechar a janela da imagem, parar análise automaticamente
            self.janela_tarefa.protocol("WM_DELETE_WINDOW", self.parar_analise)
            
        except Exception as e:
            print(f"Erro imagem: {e}")
            return

        self.iniciar_analise(0)

    def escolher_arquivo(self):
        self.limpar_janelas_anteriores()
        caminho = ctk.filedialog.askopenfilename(filetypes=[("Vídeos MP4", "*.mp4")])
        if caminho: self.iniciar_analise(caminho)

    def iniciar_analise(self, fonte):
        # Fecha câmera anterior se existir
        if self.cap is not None:
            self.cap.release()
            
        self.cap = cv2.VideoCapture(fonte)
        if not self.cap.isOpened(): return
        
        # Resetar Variáveis
        self.rodando = True
        self.tempo_inicio = time.time()
        self.buffer_pontos = []
        self.historico_heatmap_x = []
        self.historico_heatmap_y = []
        self.historico_classificacao = []
        self.frames_sem_rosto = 0
        
        # UI
        self.btn_tarefa1.configure(state="disabled")
        self.btn_tarefa2.configure(state="disabled")
        self.btn_webcam.configure(state="disabled")
        self.btn_arquivo.configure(state="disabled")
        self.btn_parar.configure(state="normal")
        
        self.loop_processamento()

    def parar_analise(self):
        if not self.rodando: return # Evita chamar duas vezes
        
        self.rodando = False
        if self.cap: 
            self.cap.release()
            self.cap = None
            
        # Fecha janela da tarefa 2 se estiver aberta
        if self.janela_tarefa:
            try:
                self.janela_tarefa.destroy()
            except: pass
            self.janela_tarefa = None

        self.btn_tarefa1.configure(state="normal")
        self.btn_tarefa2.configure(state="normal")
        self.btn_webcam.configure(state="normal")
        self.btn_arquivo.configure(state="normal")
        self.btn_parar.configure(state="disabled")
        self.label_video.configure(image=None, text="Gerando Relatório...")
        
        # Pequeno delay para garantir que recursos foram liberados
        self.after(100, self.gerar_relatorio)

    def loop_processamento(self):
        if not self.rodando: return
        
        ret, frame = self.cap.read()
        if not ret: 
            self.parar_analise()
            return

        frame = cv2.resize(frame, (1280, 720))
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        estado = "..."
        cor = (100,100,100)

        # SE DETECTAR ROSTO
        if results.multi_face_landmarks:
            self.frames_sem_rosto = 0
            lm = results.multi_face_landmarks[0].landmark
            
            nx, ny = lm[1].x * w, lm[1].y * h
            ix, iy = lm[468].x * w, lm[468].y * h 
            dx, dy = lm[473].x * w, lm[473].y * h 
            
            self.historico_heatmap_x.append(ix)
            self.historico_heatmap_y.append(iy)
            
            self.buffer_pontos.append([nx, ny, ix, iy, dx, dy])
            if len(self.buffer_pontos) > 30: self.buffer_pontos.pop(0)
            
            if len(self.buffer_pontos) == 30:
                dados = np.array(self.buffer_pontos)
                feats = []
                for i in range(6): feats.append(np.std(dados[:, i])) 
                feats.append(np.std(dados[:,2]-dados[:,0])) 
                feats.append(np.std(dados[:,3]-dados[:,1]))
                feats.append(np.std(dados[:,4]-dados[:,0]))
                feats.append(np.std(dados[:,5]-dados[:,1]))
                
                res = self.modelo.predict(self.scaler.transform([feats]))[0]
                
                if res == 'focado': 
                    estado = "FOCADO"
                    cor = (0, 255, 0)
                    self.historico_classificacao.append(0)
                else: 
                    estado = "DISTRAIDO"
                    cor = (255, 0, 0)
                    self.historico_classificacao.append(1)

        # SE NÃO DETECTAR ROSTO
        else:
            self.frames_sem_rosto += 1
            if self.frames_sem_rosto > 15: 
                estado = "DISTRAIDO (Ausente)"
                cor = (255, 0, 0)
                self.historico_classificacao.append(1) 
        
        cv2.putText(frame_rgb, f"IA: {estado}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 2)
        
        img_pil = Image.fromarray(frame_rgb)
        # Verifica se o widget ainda existe antes de atualizar
        if self.frame_video.winfo_exists():
            if self.frame_video.winfo_width() > 10:
                ctk_img = ctk.CTkImage(img_pil, size=(self.frame_video.winfo_width(), self.frame_video.winfo_height()))
                self.label_video.configure(image=ctk_img, text="")
        
        if self.rodando:
            self.after(10, self.loop_processamento)

    def gerar_relatorio(self):
        if len(self.historico_classificacao) < 10: 
            self.label_video.configure(text="Teste muito curto para gerar relatório.")
            return

        # Fecha relatório anterior se existir
        if self.janela_relatorio is not None:
            try:
                self.janela_relatorio.destroy()
            except: pass

        # --- CÁLCULOS FINAIS ---
        tempo_total_min = (time.time() - self.tempo_inicio) / 60
        if tempo_total_min < 0.1: tempo_total_min = 0.1
        
        total_frames = len(self.historico_classificacao)
        frames_distraido = sum(self.historico_classificacao)
        probabilidade_tdah = (frames_distraido / total_frames) * 100
        
        # --- INTERFACE DO RELATÓRIO ---
        self.janela_relatorio = ctk.CTkToplevel(self)
        self.janela_relatorio.title("Relatório Clínico - TDAH")
        self.janela_relatorio.geometry("900x600")
        
        self.janela_relatorio.grid_columnconfigure(0, weight=1)
        self.janela_relatorio.grid_columnconfigure(1, weight=2)
        self.janela_relatorio.grid_rowconfigure(0, weight=1)
        
        # Frame Texto
        frame_stats = ctk.CTkFrame(self.janela_relatorio)
        frame_stats.grid(row=0, column=0, padx=20, pady=20, sticky="nswe")
        
        ctk.CTkLabel(frame_stats, text="RESULTADOS DA SESSÃO", font=("Arial", 18, "bold")).pack(pady=20)
        
        texto_relatorio = f"""
        ⏱ Duração: {tempo_total_min*60:.1f} seg
        
        📊 Amostras Analisadas: {total_frames} frames
        
        🧠 Índice de Atenção Sustentada:
        {(100-probabilidade_tdah):.1f}%
        (Tempo que manteve o foco estável)
        
        🚨 Probabilidade de TDAH:
        {probabilidade_tdah:.1f}%
        (Tempo com inquietação ou distração)
        """
        
        lbl_dados = ctk.CTkLabel(frame_stats, text=texto_relatorio, justify="left", font=("Arial", 14))
        lbl_dados.pack(pady=10, padx=10)
        
        if probabilidade_tdah > 50:
            ctk.CTkLabel(frame_stats, text="ALTO RISCO DETECTADO", text_color="red", font=("Arial", 16, "bold")).pack(pady=20)
        else:
            ctk.CTkLabel(frame_stats, text="PADRÃO DE FOCO NORMAL", text_color="green", font=("Arial", 16, "bold")).pack(pady=20)

        # Frame Gráfico
        frame_grafico = ctk.CTkFrame(self.janela_relatorio)
        frame_grafico.grid(row=0, column=1, padx=20, pady=20, sticky="nswe")
        
        fig, ax = plt.subplots(figsize=(5,5))
        if len(self.historico_heatmap_x) > 50:
            try:
                sns.kdeplot(x=self.historico_heatmap_x, y=self.historico_heatmap_y, 
                            fill=True, cmap="inferno", alpha=0.8, thresh=0.05, ax=ax)
                ax.set_facecolor('black')
                ax.invert_yaxis()
                ax.set_title("Mapa de Calor da Atenção")
            except:
                ax.text(0.5, 0.5, "Erro ao gerar gráfico", ha='center')
        else:
            ax.text(0.5, 0.5, "Dados insuficientes para mapa", ha='center')

        canvas = FigureCanvasTkAgg(fig, master=frame_grafico)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

if __name__ == "__main__":
    app = AppAnaliseTDAH()
    app.mainloop()