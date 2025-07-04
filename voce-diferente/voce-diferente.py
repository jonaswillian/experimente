import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import mediapipe as mp
import numpy as np
import datetime

frame_filtrado = None  # será atualizado a cada loop
left_eye_indices = [33, 160, 158, 133, 153, 144]
right_eye_indices = [362, 385, 387, 263, 373, 380]

# Carrega imagens PNG com transparência
def load_png(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)

# Função para sobrepor imagem com canal alpha
def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    x, y = pos
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    img_crop = img[y1:y2, x1:x2]
    overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis] / 255.0

    img[y1:y2, x1:x2] = (1.0 - alpha) * img_crop + alpha * overlay_crop

def enlarge_eyes(image, landmarks, scale=1.5):
    h, w, _ = image.shape
    # Processar olho esquerdo
    left_eye_pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in left_eye_indices])
    right_eye_pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in right_eye_indices])

    # Função auxiliar para ampliar uma região de olho
    def process_eye(image, eye_pts, scale):
        x, y, w_eye, h_eye = cv2.boundingRect(eye_pts)
        if w_eye == 0 or h_eye == 0:
            return image

        # Extrair região do olho
        eye_roi = image[y:y+h_eye, x:x+w_eye]
        eye_mask = np.zeros_like(eye_roi, dtype=np.uint8)
        eye_pts_local = eye_pts - np.array([x, y])
        cv2.fillPoly(eye_mask, [eye_pts_local], (255, 255, 255))
        eye_only = cv2.bitwise_and(eye_roi, eye_mask)

        # Ampliar a região do olho
        enlarged_eye = cv2.resize(eye_only, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        ex = x - int((enlarged_eye.shape[1] - w_eye) / 2)
        ey = y - int((enlarged_eye.shape[0] - h_eye) / 2)
        ex, ey = max(0, ex), max(0, ey)
        center = (x + w_eye // 2, y + h_eye // 2)

        # Criar máscara para fusão
        mask = 255 * np.ones(enlarged_eye.shape, enlarged_eye.dtype)
        try:
            image = cv2.seamlessClone(enlarged_eye, image, mask, center, cv2.NORMAL_CLONE)
        except:
            pass  # Ignorar erros de fusão fora dos limites
        return image

    # Aplicar para ambos os olhos
    image = process_eye(image, left_eye_pts, scale)
    image = process_eye(image, right_eye_pts, scale)
    return image

# Função para aplicar máscara colorida com suavização e transparência
def apply_color_mask(img, mask, color, alpha=0.4, blur_ksize=15):
    colored_mask = np.zeros_like(img, dtype=np.uint8)
    colored_mask[:] = color

    mask_blurred = cv2.GaussianBlur(mask.astype(np.float32), (blur_ksize, blur_ksize), 0) / 255.0
    mask_blurred = mask_blurred[..., np.newaxis]

    img[:] = img * (1 - mask_blurred * alpha) + colored_mask * (mask_blurred * alpha)
    img[:] = img.astype(np.uint8)

# Interface com Tkinter
root = tk.Tk()
root.title("App de Filtros com Webcam")
root.geometry("1320x1100")

# Variáveis de estado dos filtros
filtro_oculos = tk.BooleanVar(root, value=False)
filtro_bigode = tk.BooleanVar(root, value=False)
filtro_cabelo = tk.BooleanVar(root, value=False)
filtro_cabelo2 = tk.BooleanVar(root, value=False)
filtro_cabelo_comprido = tk.BooleanVar(root, value=False)
filtro_barba = tk.BooleanVar(root, value=False)
filtro_chifre = tk.BooleanVar(root, value=False)
filtro_chapeu = tk.BooleanVar(root, value=False)
filtro_tatuagem = tk.BooleanVar(root, value=False)
filtro_maquiagem = tk.BooleanVar(root, value=False)
filtro_cor_cabelo = tk.BooleanVar(root, value=False)
filtro_olhos_grandes = tk.BooleanVar(root, value=False)

# Carrega os filtros PNG
oculos_img = load_png("filtros/oculos.png")
bigode_img = load_png("filtros/bigode.png")
cabelo_img = load_png("filtros/cabelo.png")
cabelo_img2 = load_png("filtros/cabelo2.png")
cabelo_img3 = load_png("filtros/cabelo_comprido.png")
barba_img = load_png("filtros/barba.png")
chifre_img = load_png("filtros/chifre.png")
chapeu_img = load_png("filtros/chapeu.png")
tatuagem_img = load_png("filtros/tatuagem.png")
dog_nose_img = load_png("filtros/dog_nose.png")

filtro_nariz_cachorro = tk.BooleanVar(root, value=False)
# Inicializa MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# Webcam em alta resolução
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

# Índices MediaPipe para regiões (com refine_landmarks=True)
# Lábios e bochechas (exemplos para maquiagem)
# Lábios (mais completos e sem duplicações)
lips = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
        375, 321, 405, 314, 17, 84, 181, 91, 146]

# Bochechas (divididas em esquerda e direita para maior controle)
cheek_left = [234, 93, 132, 58, 172, 136, 150, 149]
cheek_right = [454, 323, 361, 288, 397, 365, 379, 378]

# Região aproximada do cabelo (testa + laterais da cabeça)
hair_points = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
    361, 288, 397, 152, 148, 176, 149, 150, 136, 172, 58,
    132, 93, 234
]

def create_mask_from_points(img_shape, points, landmarks):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    h, w = img_shape[:2]
    pts = np.array([(int(landmarks[p].x * w), int(landmarks[p].y * h)) for p in points], np.int32)
    hull = cv2.convexHull(pts)
    cv2.fillPoly(mask, [hull], 255)
    return mask

def update_video():
    global frame_filtrado
    success, frame = cap.read()
    if not success:
        root.after(10, update_video)
        return

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            # Coordenadas principais
            olho_esq = face_landmarks.landmark[33]
            olho_dir = face_landmarks.landmark[263]
            nariz = face_landmarks.landmark[1]
            testa = face_landmarks.landmark[10]

            x1, y1 = int(olho_esq.x * w), int(olho_esq.y * h)
            x2, y2 = int(olho_dir.x * w), int(olho_dir.y * h)
            nx, ny = int(nariz.x * w), int(nariz.y * h)
            tx, ty = int(testa.x * w), int(testa.y * h)

            # Aplicar filtros PNG já existentes
            if filtro_oculos.get():
                largura = x2 - x1 + 100
                altura = int(largura * oculos_img.shape[0] / oculos_img.shape[1])
                resized = cv2.resize(oculos_img, (largura, altura))
                overlay_image_alpha(frame, resized[:, :, :3], (x1 - 50, y1 - 50), resized[:, :, 3])

            if filtro_bigode.get():
                resized = cv2.resize(bigode_img, (130, 45))
                overlay_image_alpha(frame, resized[:, :, :3], (nx - 60, ny + 5), resized[:, :, 3])

            if filtro_barba.get():
                resized = cv2.resize(barba_img, (230, 190))
                overlay_image_alpha(frame, resized[:, :, :3], (nx - 110, ny - 15), resized[:, :, 3])

            if filtro_cabelo.get():
                largura = x2 - x1 + 100
                resized = cv2.resize(cabelo_img, (largura, 120))
                overlay_image_alpha(frame, resized[:, :, :3], (x1 - 50, ty - 80), resized[:, :, 3])

            if filtro_cabelo2.get():
                largura = x2 - x1 + 145
                resized = cv2.resize(cabelo_img2, (largura, 250))
                overlay_image_alpha(frame, resized[:, :, :3], (x1 - 60, ty - 130), resized[:, :, 3])

            if filtro_cabelo_comprido.get():
                largura = x2 - x1 + 300
                resized = cv2.resize(cabelo_img3, (largura, 550))
                overlay_image_alpha(frame, resized[:, :, :3], (x1 - 60, ty - 130), resized[:, :, 3])

            if filtro_chifre.get():
                resized = cv2.resize(chifre_img, (250, 100))
                overlay_image_alpha(frame, resized[:, :, :3], (nx - 110, ty - 170), resized[:, :, 3])

            if filtro_chapeu.get():
                resized = cv2.resize(chapeu_img, (380, 250))
                overlay_image_alpha(frame, resized[:, :, :3], (tx - 180, ty - 190), resized[:, :, 3])

            if filtro_tatuagem.get():
                resized = cv2.resize(tatuagem_img, (80, 80))
                overlay_image_alpha(frame, resized[:, :, :3], (nx - 40, ny + 160), resized[:, :, 3])

            if filtro_olhos_grandes.get():
                frame = enlarge_eyes(frame, face_landmarks.landmark, scale=1.5)

            # --- NOVOS FILTROS ---

            # Maquiagem virtual: lábios e bochechas com cor suave
            if filtro_maquiagem.get():
                landmarks = face_landmarks.landmark

                # Máscara lábios
                mask_lips = create_mask_from_points(frame.shape, lips, landmarks)
                apply_color_mask(frame, mask_lips, color=(0, 0, 255), alpha=0.5, blur_ksize=25)

                # Máscara bochecha esquerda
                mask_cheek_left = create_mask_from_points(frame.shape, cheek_left, landmarks)
                apply_color_mask(frame, mask_cheek_left, color=(147, 20, 255), alpha=0.3, blur_ksize=35)

                # Máscara bochecha direita
                mask_cheek_right = create_mask_from_points(frame.shape, cheek_right, landmarks)
                apply_color_mask(frame, mask_cheek_right, color=(147, 20, 255), alpha=0.3, blur_ksize=35)

            # Cor do cabelo (aplicação de cor com máscara aproximada)
            if filtro_cor_cabelo.get():
                landmarks = face_landmarks.landmark
                mask_hair = create_mask_from_points(frame.shape, hair_points, landmarks)
                apply_color_mask(frame, mask_hair, color=(255, 0, 0), alpha=0.25, blur_ksize=45)  # azul para teste

            if filtro_nariz_cachorro.get():
                # Calcular largura com base na distância entre os olhos
                largura = x2 - x1 - 30  # Ajuste conforme necessário
                altura = int(largura * dog_nose_img.shape[0] / dog_nose_img.shape[1])
                resized = cv2.resize(dog_nose_img, (largura, altura))
                # Posicionar o nariz abaixo do ponto do nariz
                overlay_image_alpha(frame, resized[:, :, :3], (nx - largura // 2, ny - altura // 4), resized[:, :, 3])

    frame_filtrado = frame.copy()

    # Exibir no Tkinter
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img_pil)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    root.after(10, update_video)

video_label = tk.Label(root)
video_label.pack()

# Painel de controles
painel = ttk.Frame(root)
painel.pack(pady=10)

ttk.Checkbutton(painel, text="Óculos", variable=filtro_oculos).pack(side=tk.LEFT, padx=10)
ttk.Checkbutton(painel, text="Bigode", variable=filtro_bigode).pack(side=tk.LEFT, padx=10)
ttk.Checkbutton(painel, text="Cabelo", variable=filtro_cabelo).pack(side=tk.LEFT, padx=10)
ttk.Checkbutton(painel, text="Cabelo 2", variable=filtro_cabelo2).pack(side=tk.LEFT, padx=10)
ttk.Checkbutton(painel, text="Cabelo Comprido", variable=filtro_cabelo_comprido).pack(side=tk.LEFT, padx=10)
ttk.Checkbutton(painel, text="Barba", variable=filtro_barba).pack(side=tk.LEFT, padx=10)
ttk.Checkbutton(painel, text="Chifre", variable=filtro_chifre).pack(side=tk.LEFT, padx=10)
ttk.Checkbutton(painel, text="Chapéu", variable=filtro_chapeu).pack(side=tk.LEFT, padx=10)
ttk.Checkbutton(painel, text="Tatuagem", variable=filtro_tatuagem).pack(side=tk.LEFT, padx=10)
ttk.Checkbutton(painel, text="Olhos Grandes", variable=filtro_olhos_grandes).pack(side=tk.LEFT, padx=10)
ttk.Checkbutton(painel, text="Maquiagem Virtual", variable=filtro_maquiagem).pack(side=tk.LEFT, padx=10)
ttk.Checkbutton(painel, text="Cor do Cabelo", variable=filtro_cor_cabelo).pack(side=tk.LEFT, padx=10)
ttk.Checkbutton(painel, text="Nariz de Cachorro", variable=filtro_nariz_cachorro).pack(side=tk.LEFT, padx=10)

root.after(10, update_video)
root.mainloop()
