import cv2
from ultralytics import YOLO

# 1. Carregar o modelo YOLO pré-treinado (o 'n' significa nano, que é o mais rápido e leve)
# Na primeira vez que rodar, ele vai baixar um arquivo chamado yolov8n.pt automaticamente
modelo = YOLO('yolov8s.pt')

# 2. Ligar a câmera do computador
camera = cv2.VideoCapture(0)

while True:
    sucesso, frame = camera.read()
    if not sucesso:
        print("Erro ao capturar a imagem.")
        break

    # 3. O YOLO analisa a foto e procura pessoas, cadeiras, mesas, etc.
    resultados = modelo(frame)

    # 4. Desenhar os retângulos em volta do que ele achou
    frame_desenhado = resultados[0].plot()

    # 5. Mostrar o vídeo já com as detecções na tela
    cv2.imshow("Deteccao YOLO", frame_desenhado)

    # Espera a tecla 'q' ser pressionada para fechar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()