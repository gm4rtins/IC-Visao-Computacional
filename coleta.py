import numpy as np
import cv2
import pandas as pd


# classe que encapsula todos os métodos de captura, processamento e armazenamento das imagens das amostras
class Analyser:
    # captura de imagem da amostra na cubeta
    def capturar(self, name):
        # iniciando captura de video de uma webcam
        webcam = cv2.VideoCapture(0)

        # dimensionando largura e altura da captura da imagem da câmera 1 e da câmera 2
        webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # le o frame das câmeras
        working, frame = webcam.read()

        # mostrando imagem da camera
        cv2.imshow("Video da Webcam", frame)  # exibir preview da imagem

        # tempo para visualizar a imagem adquirida de 5000 ms
        key = cv2.waitKey(5000)

        # salvando arquivos de imagem (etapa temporária para avaliar os arquivos)
        cv2.imwrite(str(name) + '.jpg', frame)

        # recortando imagens
        framecut = frame[370:570, 300:400]

        # salvando arquivos de imagem (etapa temporária para avaliar os arquivos)
        cv2.imwrite(str(name) + 'cut.jpg', framecut)

        # fechando as câmeras e as janelas
        webcam.release()
        cv2.destroyAllWindows()

        return framecut

    # processa as imagens e as fatora em valores de RGB e HSV
    def processar(self, framecut):

        # calculando a média dos valores de RGB
        meanRGB = np.mean(framecut, axis=(0, 1))

        # convertendo RGB em HSV
        frameHSV = cv2.cvtColor(framecut, cv2.COLOR_BGR2HSV)

        # calculando a média dos valores de HSV
        meanHSV = np.mean(frameHSV, axis=(0, 1))

        values = [meanRGB, meanHSV]

        return values

    # metodo que lê e salva os arquivos localmente
    def salvar(self, name, values):
        # lendo os dados antigos
        old_samples = pd.read_excel("DataCV.xlsx")

        # transformando dados de RGB e HSV da camera 1 e 2 em vetores 1 x n
        RGBcam1 = np.transpose(pd.DataFrame(values[0]))
        HSVcam1 = np.transpose(pd.DataFrame(values[1]))

        # criando um dataframe com os dados da amostra
        finaldata = pd.DataFrame(
            {'Name': name, 'valueR_cam1': RGBcam1[0], 'valueG_cam1': RGBcam1[1], 'valueB_cam1': RGBcam1[2],
             'valueH_cam1': HSVcam1[0], 'valueS_cam1': HSVcam1[1], 'valueV_cam1': HSVcam1[2]})

        # concatenando a planilha antiga com a planilha nova
        gatherdata = pd.concat([old_samples, finaldata])

        # salvando em cima do arquivo original os novos dados concatenados
        gatherdata.to_excel('DataCV.xlsx', index=False)

    def backup(self):
        return

cap = Analyser()
cap.capturar('teste')
