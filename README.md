\# Sistema de Triagem para TDAH Baseado em Rastreamento Ocular



Este é um protótipo desenvolvido para a disciplina de Visão Computacional na UFRPE, que utiliza Visão Computacional (MediaPipe) e Machine Learning (Scikit-learn) para classificar padrões de atenção ("focado" vs. "distraido") a partir de uma webcam.



\## Metodologia



O projeto é inspirado em metodologias acadêmicas (como Jiang et al., 2023) e utiliza um pipeline 100% Python para extrair e analisar características do movimento facial e ocular.



\### Componentes:

\* `coletor.py`: Script para gravar amostras de dados (ex: 'focado', 'distraido') e extrair features.

\* `treinador.py`: Script que usa os dados do coletor (`features\_data.csv`) para treinar um classificador SVM e salvar o modelo (`.joblib`).

\* `demo.py`: Demonstração em tempo real que classifica o estado do usuário a cada 10 segundos.

\* `teste\_tarefa1\_passiva.py`: Simula a "Tarefa de Visualização Passiva" do artigo, abrindo um vídeo do YouTube.

\* `teste\_tarefa2\_local.py`: Simula a "Tarefa de Busca Visual Ativa", abrindo uma imagem local da pasta `/imagem`.



\## Como Rodar o Projeto



\*\*1. Ambiente:\*\*

Este projeto usa Python 3.10.



```bash

\# Clone o repositório (ou baixe o ZIP)

git clone \[COLE A URL DO SEU REPOSITÓRIO AQUI]

cd ProjetoTDAH-EyeTracking



\# Instale as dependências

pip install opencv-python mediapipe numpy pandas scikit-learn

2. Treine seu Próprio Modelo: O modelo (.joblib) não está no repositório. Você precisa criá-lo:

Bash

# Rode o coletor e grave suas amostras
python coletor.py
Siga as instruções: grave 5 amostras para 'focado' e 5 para 'distraido'.

Digite 'sair' para salvar.

Bash

# Agora, treine o modelo
python treinador.py
Isso irá criar os arquivos modelo_tdah_svm.joblib e scaler_tdah_svm.joblib.

3. Execute as Demos:

Bash

# Demo em tempo real
python demo.py

# Teste da Tarefa 1 (Vídeo)
python teste_tarefa1_passiva.py

# Teste da Tarefa 2 (Busca Visual)
# (Lembre-se de colocar sua imagem em /imagem e ajustar o nome no script)
python teste_tarefa2_local.py
---
