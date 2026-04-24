import cv2
import threading
import time
import datetime 
from ultralytics import YOLO

# --- VARIÁVEIS GERAIS E FLAGS ---
COR_AZUL = (217, 89, 63)
COR_VERMELHA = (0, 0, 255)


MODO_PRODUCAO = False 

# --- CLASSES E FUNÇÕES AUTÓNOMAS ---

class LeitorDeVideoAutonomo:
    def __init__(self, fonte_video):
        self.video = cv2.VideoCapture(fonte_video)
        self.sucesso, self.frame = self.video.read()
        self.rodando = True
        self.thread = threading.Thread(target=self.atualizar, args=())
        self.thread.daemon = True 
        self.thread.start()

    def atualizar(self):
        while self.rodando:
            if self.video.isOpened():
                sucesso_temp, frame_temp = self.video.read()
                if sucesso_temp:
                    self.sucesso = sucesso_temp
                    self.frame = frame_temp
                else:
                    self.sucesso = False
                    self.rodando = False
            time.sleep(0.01)

    def ler(self):
        return self.sucesso, self.frame

    def parar(self):
        self.rodando = False
        self.thread.join()
        self.video.release()

def rastrear_objetos(modelo_ia, imagem):
    return modelo_ia.track(imagem, persist=True, verbose=False, classes=0)

def contar_pessoas_atuais(resultados):
    dados = resultados[0].boxes
    if dados is not None and dados.id is not None:
        return len(dados.id)
    return 0

def atualizar_memoria_metricas(resultados, memoria, tempo_atual_segundos, altura_tela):
    dados = resultados[0].boxes
    linha_y = int(altura_tela / 2)
    
    if dados is not None and dados.id is not None:
        coordenadas = dados.xyxy.cpu().numpy()
        ids = dados.id.int().cpu().tolist()
        
        for caixa, id_pessoa in zip(coordenadas, ids):
            y_centro = int((caixa[1] + caixa[3]) / 2)
            
            if y_centro > linha_y:
                if id_pessoa not in memoria:
                    memoria[id_pessoa] = {'entrada': tempo_atual_segundos, 'saida': tempo_atual_segundos}
                else:
                    memoria[id_pessoa]['saida'] = tempo_atual_segundos

def gerar_relatorio_txt(memoria, max_p, min_p):
    tempos_espera = []
    
    for id_pessoa, dados in memoria.items():
        tempo_total = dados['saida'] - dados['entrada']
        if tempo_total > 0: 
            tempos_espera.append(tempo_total)
            
    if len(tempos_espera) > 0:
        tempo_medio = sum(tempos_espera) / len(tempos_espera)
        tempo_maximo = max(tempos_espera)
        tempo_minimo = min(tempos_espera)
    else:
        tempo_medio = tempo_maximo = tempo_minimo = 0

    agora = datetime.datetime.now()
    carimbo_tempo = agora.strftime("%Y-%m-%d_%H-%M")
    nome_dinamico = f"Relatorio_Agencia_{carimbo_tempo}.txt"

    with open(nome_dinamico, "w", encoding="utf-8") as arquivo:
        arquivo.write("="*40 + "\n")
        arquivo.write(f"RELATÓRIO DE FLUXO - GERADO EM: {agora.strftime('%d/%m/%Y %H:%M')}\n")
        arquivo.write("="*40 + "\n\n")
        arquivo.write(f"Total de Clientes Atendidos: {len(tempos_espera)} pessoas\n")
        arquivo.write(f"Pico Máximo de Lotação: {max_p} pessoas ao mesmo tempo\n")
        arquivo.write(f"Mínimo de Lotação: {min_p} pessoas ao mesmo tempo\n\n")
        arquivo.write("--- MÉTRICAS DE TEMPO (Simuladas) ---\n")
        arquivo.write(f"Tempo Médio de Espera: {tempo_medio:.2f} segundos\n")
        arquivo.write(f"Tempo Máximo de Espera: {tempo_maximo:.2f} segundos\n")
        arquivo.write(f"Tempo Mínimo de Espera: {tempo_minimo:.2f} segundos\n\n")
        arquivo.write("="*40 + "\n")
        
    print(f"\n[SUCESSO] Relatório '{nome_dinamico}' gerado na sua pasta!")

def desenhar_contador(imagem, quantidade, maxima):
    frame_desenhado = imagem.copy()
    texto_atual = f"Pessoas na Tela: {quantidade}"
    cv2.putText(frame_desenhado, texto_atual, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, COR_AZUL, 3)
    
    texto_max = f"Lotacao Maxima: {maxima}"
    cv2.putText(frame_desenhado, texto_max, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame_desenhado

def desenhar_ids_e_centro(imagem, resultados):
    frame_desenhado = imagem.copy()
    dados = resultados[0].boxes
    altura, largura = frame_desenhado.shape[:2]
    linha_y = int(altura / 2)
    
    if dados is not None and dados.id is not None:
        coordenadas = dados.xyxy.cpu().numpy()
        ids = dados.id.int().cpu().tolist()
        
        for caixa, id_pessoa in zip(coordenadas, ids):
            x1, y1, x2, y2 = map(int, caixa)
            x_centro = int((x1 + x2) / 2)
            y_centro = int((y1 + y2) / 2)
            
            if y_centro > linha_y:
                cor_atual = COR_VERMELHA
            else:
                cor_atual = COR_AZUL
            
            cv2.rectangle(frame_desenhado, (x1, y1), (x2, y2), cor_atual, 2)
            cv2.putText(frame_desenhado, f"ID: {id_pessoa}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_atual, 2)
            cv2.circle(frame_desenhado, (x_centro, y_centro), 4, cor_atual, -1)
            
    return frame_desenhado

def desenhar_linha_virtual(imagem):
    frame_desenhado = imagem.copy()
    altura, largura = frame_desenhado.shape[:2]
    posicao_y = int(altura / 2)
    cv2.line(frame_desenhado, (0, posicao_y), (largura, posicao_y), (255, 255, 255), 2)
    cv2.putText(frame_desenhado, "LINHA DA PORTA", (10, posicao_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return frame_desenhado

# --- CÓDIGO PRINCIPAL ---

modelo = YOLO('yolov8s.pt')
caminho_video = 'videoplayback.mp4'

memoria_pessoas = {}
max_pessoas_agencia = 0
min_pessoas_agencia = 99999
contador_frames = 0

if MODO_PRODUCAO:
    leitor_thread = LeitorDeVideoAutonomo(caminho_video)
    fps_real = leitor_thread.video.get(cv2.CAP_PROP_FPS) 
    time.sleep(1.0)
else:
    video = cv2.VideoCapture(caminho_video)
    if not video.isOpened():
        print("Erro ao abrir o vídeo.")
        exit()
    fps_real = video.get(cv2.CAP_PROP_FPS) 

print(f"Iniciando... O FPS real do seu vídeo é: {fps_real}") 

while True:
    if MODO_PRODUCAO:
        sucesso, frame = leitor_thread.ler()
    else:
        sucesso, frame = video.read()
    
    if not sucesso:
        print("Fim do vídeo ou perda de conexão!")
        break

    contador_frames += 1

    segundos_simulados = contador_frames / fps_real 

    resultados = rastrear_objetos(modelo, frame)
    qtd_pessoas = contar_pessoas_atuais(resultados)
    
    if qtd_pessoas > max_pessoas_agencia:
        max_pessoas_agencia = qtd_pessoas
    if qtd_pessoas < min_pessoas_agencia and qtd_pessoas > 0: 
        min_pessoas_agencia = qtd_pessoas

    altura_video = frame.shape[0]
    atualizar_memoria_metricas(resultados, memoria_pessoas, segundos_simulados, altura_video)
    
    frame_desenhado = desenhar_linha_virtual(frame)
    frame_desenhado = desenhar_ids_e_centro(frame_desenhado, resultados)
    frame_desenhado = desenhar_contador(frame_desenhado, qtd_pessoas, max_pessoas_agencia)

    cv2.imshow("Monitoramento e Coleta de Dados", frame_desenhado)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("\nProcessando dados e gerando relatório...")
gerar_relatorio_txt(memoria_pessoas, max_pessoas_agencia, min_pessoas_agencia)

if MODO_PRODUCAO:
    leitor_thread.parar()
else:
    video.release()
cv2.destroyAllWindows()
