import cv2
from deepface import DeepFace
import time
import os
import numpy as np

# Tradução das emoções
emocoes_dict = {
    'happy': 'Feliz',
    'sad': 'Triste',
    'angry': 'Bravo',
    'surprise': 'Surpreso',
    'fear': 'Medo',
    'disgust': 'Nojo',
    'neutral': 'Neutro'
}

# Caminho para emojis
emoji_path = 'emotions'
emojis_imgs = {}

# Carrega os emojis em um dicionário
for emocao in emocoes_dict.values():
    caminho = os.path.join(emoji_path, emocao.lower() + '.png')
    if os.path.exists(caminho):
        emoji_img = cv2.imread(caminho, cv2.IMREAD_UNCHANGED)
        emojis_imgs[emocao] = emoji_img

# Função para sobrepor imagem com canal alfa
def sobrepor_imagem(fundo, sobreposicao, x, y):
    h, w = sobreposicao.shape[:2]

    if y + h > fundo.shape[0] or x + w > fundo.shape[1]:
        return fundo

    if sobreposicao.shape[2] == 4:
        alpha_s = sobreposicao[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(3):
            fundo[y:y+h, x:x+w, c] = (alpha_s * sobreposicao[:, :, c] +
                                      alpha_l * fundo[y:y+h, x:x+w, c])
    else:
        fundo[y:y+h, x:x+w] = sobreposicao

    return fundo

# Inicializa webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

# Controle de emoção exibida
ultima_emocao = 'Detectando...'
emocao_temp = ''
tempo_ultima_analise = 0
tempo_ultima_mudanca = 0
intervalo_analise = 0.3
tempo_mudanca_emocao = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tempo_atual = time.time()

    if tempo_atual - tempo_ultima_analise > intervalo_analise:
        try:
            resultado = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)
            emocao_detectada = resultado[0]['dominant_emotion']
            emocao_traduzida = emocoes_dict.get(emocao_detectada, 'Desconhecida')

            if emocao_traduzida != ultima_emocao:
                if emocao_traduzida == emocao_temp:
                    if tempo_atual - tempo_ultima_mudanca > tempo_mudanca_emocao:
                        ultima_emocao = emocao_traduzida
                        tempo_ultima_mudanca = tempo_atual
                else:
                    emocao_temp = emocao_traduzida
                    tempo_ultima_mudanca = tempo_atual
        except Exception as e:
            print("Erro:", e)
            ultima_emocao = 'Erro'
            emocao_temp = ''

        tempo_ultima_analise = tempo_atual

    # Fundo semitransparente
    overlay = frame.copy()
    texto = ultima_emocao if ultima_emocao else "Detectando..."
    font = cv2.FONT_HERSHEY_SIMPLEX
    tamanho = 2.5
    espessura = 4

    (largura_texto, altura_texto), _ = cv2.getTextSize(texto, font, tamanho, espessura)
    centro_x = int((frame.shape[1] - largura_texto) / 2)
    centro_y = int((frame.shape[0] + altura_texto) / 2)

    # Desenha fundo escuro
    cv2.rectangle(overlay, (centro_x - 120, centro_y - 60), (centro_x + largura_texto + 100, centro_y + 30), (0, 0, 0), -1)
    alpha = 0.4
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Adiciona texto
    cv2.putText(frame, texto, (centro_x, centro_y), font, tamanho, (255, 255, 255), espessura, cv2.LINE_AA)

    # Adiciona emoji se existir
    if texto in emojis_imgs:
        emoji_img = cv2.resize(emojis_imgs[texto], (80, 80))
        frame = sobrepor_imagem(frame, emoji_img, centro_x - 100, centro_y - 60)

    # Exibe vídeo
    cv2.imshow('Detector de Emoções com Emoji', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
