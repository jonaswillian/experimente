import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import mediapipe as mp
import numpy as np
import datetime

frame_filtrado = None  # será atualizado a cada loop

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

# Interface com Tkinter
root = tk.Tk()
root.title("App de Filtros com Webcam")
root.geometry("1320x1100")

# Variáveis de estado dos filtros (devem ser criadas depois do root)
filtro_oculos = tk.BooleanVar(root, value=False)
filtro_bigode = tk.BooleanVar(root, value=False)
filtro_cabelo = tk.BooleanVar(root, value=False)
filtro_barba = tk.BooleanVar(root, value=False)
filtro_chifre = tk.BooleanVar(root, value=False)
filtro_chapeu = tk.BooleanVar(root, value=False)
filtro_tatuagem = tk.BooleanVar(root, value=False)

# Carrega os filtros (PNG)
oculos_img = load_png("filtros/oculos.png")
bigode_img = load_png("filtros/bigode.png")
cabelo_img = load_png("filtros/cabelo.png")
barba_img = load_png("filtros/barba.png")
chifre_img = load_png("filtros/chifre.png")
chapeu_img = load_png("filtros/chapeu.png")
tatuagem_img = load_png("filtros/tatuagem.png")

# Inicializa MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False)

# Webcam em alta resolução
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

# Função para atualizar o vídeo
def update_video():
    success, frame = cap.read()
    if not success:
        return

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            olho_esq = face_landmarks.landmark[33]
            olho_dir = face_landmarks.landmark[263]
            nariz = face_landmarks.landmark[1]
            testa = face_landmarks.landmark[10]

            x1, y1 = int(olho_esq.x * w), int(olho_esq.y * h)
            x2, y2 = int(olho_dir.x * w), int(olho_dir.y * h)
            nx, ny = int(nariz.x * w), int(nariz.y * h)
            tx, ty = int(testa.x * w), int(testa.y * h)

            # Filtro: óculos
            if filtro_oculos.get():
                largura = x2 - x1 + 100
                altura = int(largura * oculos_img.shape[0] / oculos_img.shape[1])
                resized = cv2.resize(oculos_img, (largura, altura))
                overlay_image_alpha(frame, resized[:, :, :3], (x1 - 50, y1 - 50), resized[:, :, 3])

            # Filtro: bigode
            if filtro_bigode.get():
                resized = cv2.resize(bigode_img, (130, 45))
                overlay_image_alpha(frame, resized[:, :, :3], (nx - 60, ny + 5), resized[:, :, 3])

            # Filtro: barba
            if filtro_barba.get():
                resized = cv2.resize(barba_img, (230, 190))
                overlay_image_alpha(frame, resized[:, :, :3], (nx - 110, ny-15), resized[:, :, 3])

            # Filtro: cabelo
            if filtro_cabelo.get():
                largura = x2 - x1 + 150
                resized = cv2.resize(cabelo_img, (largura, 120))
                overlay_image_alpha(frame, resized[:, :, :3], (x1 - 75, ty - 140), resized[:, :, 3])

            # Filtro: chifre de unicórnio
            if filtro_chifre.get():
                resized = cv2.resize(chifre_img, (100, 100))
                overlay_image_alpha(frame, resized[:, :, :3], (nx - 50, ty - 170), resized[:, :, 3])

            # Filtro: chapéu
            if filtro_chapeu.get():
                resized = cv2.resize(chapeu_img, (380, 250))
                overlay_image_alpha(frame, resized[:, :, :3], (tx - 180, ty - 190), resized[:, :, 3])

            # Filtro: tatuagem no rosto
            if filtro_tatuagem.get():
                resized = cv2.resize(tatuagem_img, (80, 80))
                overlay_image_alpha(frame, resized[:, :, :3], (nx - 40, ny + 160), resized[:, :, 3])

    global frame_filtrado
    frame_filtrado = frame.copy()  # salva o frame já com os filtros aplicados

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
ttk.Checkbutton(painel, text="Barba", variable=filtro_barba).pack(side=tk.LEFT, padx=10)
ttk.Checkbutton(painel, text="Chifre", variable=filtro_chifre).pack(side=tk.LEFT, padx=10)
ttk.Checkbutton(painel, text="Chapéu", variable=filtro_chapeu).pack(side=tk.LEFT, padx=10)
ttk.Checkbutton(painel, text="Tatuagem", variable=filtro_tatuagem).pack(side=tk.LEFT, padx=10)


# Iniciar loop de vídeo
update_video()

def tirar_foto():
    if frame_filtrado is not None:
        filename = f"print/foto_com_filtros_{datetime.datetime.now().strftime('%H%M%S')}.png"
        cv2.imwrite(filename, frame_filtrado)
        print(f"✅ Foto salva como: {filename}")
    else:
        print("⚠️ Nenhuma imagem disponível para salvar.")

btn_foto = ttk.Button(root, text="Tirar Foto", command=tirar_foto)
btn_foto.pack(pady=10)


root.mainloop()

# Finalizar câmera ao sair
cap.release()
cv2.destroyAllWindows()
