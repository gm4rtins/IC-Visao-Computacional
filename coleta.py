import numpy as np
import cv2
import pandas as pd
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


# classe que encapsula todos os métodos de captura, processamento e armazenamento das imagens das amostras
class Analyser:
    # captura de imagem da amostra na cubeta
    def capturar(self):

        # solicitando ao usuario o nome da amostra
        name = input('Insira o nome da amostra')

        # iniciando captura de video de uma webcam
        webcam1 = cv2.VideoCapture(0)
        webcam2 = cv2.VideoCapture(1)

        # dimensionando largura e altura da captura da imagem da câmera 1 e da câmera 2
        webcam1.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        webcam1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        webcam2.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        webcam2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # le o frame das câmeras
        # read camera frame
        working, frame1 = webcam1.read()
        working, frame2 = webcam2.read()

        # mostrando imagem da camera
        cv2.imshow("Video da Webcam", frame1)  # exibir preview da imagem
        cv2.imshow("Video da Webcam", frame2)  # exibir preview da imagem

        # tempo para visualizar a imagem adquirida de 5000 ms
        key = cv2.waitKey(5000)

        # salvando arquivos de imagem (etapa temporária para avaliar os arquivos)
        cv2.imwrite(str(name) + '.jpg', frame1)
        cv2.imwrite(str(name) + '.jpg', frame2)

        # recortando imagens
        framecut1 = frame1[0:140, 0:639]
        framecut2 = frame2[0:140, 0:639]

        # salvando arquivos de imagem (etapa temporária para avaliar os arquivos)
        cv2.imwrite(str(name) + 'cut.jpg', framecut1)
        cv2.imwrite(str(name) + 'cut.jpg', framecut2)

        # fechando as câmeras e as janelas
        webcam1.release()
        webcam2.release()
        cv2.destroyAllWindows()

        return framecut1, framecut2, name


    # processa as imagens e as fatora em valores de RPG e HSV
    def processar(self, framecut1, framecut2):
        # calculando a média dos valores de RGB
        meanRGB_webcam1 = np.mean(framecut1, axis=(0, 1))
        meanRGB_webcam2 = np.mean(framecut2, axis=(0, 1))

        # convertendo RGB em HSV
        frameHSV_webcam1 = cv2.cvtColor(framecut1, cv2.COLOR_BGR2HSV)
        frameHSV_webcam2 = cv2.cvtColor(framecut2, cv2.COLOR_BGR2HSV)

        # Calcular a média dos valores de HSV
        # Calculate the average of the HSV values
        meanHSV_webcam1 = np.mean(frameHSV_webcam1, axis=(0, 1))
        meanHSV_webcam2 = np.mean(frameHSV_webcam2, axis=(0, 1))

        values = [meanRGB_webcam1, meanRGB_webcam2, meanHSV_webcam1, meanHSV_webcam2]

        return values

    # metodo que lê e salva os arquivos localmente, utilizar quando não houver rede
    def salvar(self, name, values):
        # lendo os dados antigos
        old_samples = pd.read_excel("DataCV.xlsx")


        # transformando dados de RGB e HSV da camera 1 e 2 em vetores 1 x n
        RGBcam1 = np.transpose(pd.DataFrame(values[0]))
        RGBcam2 = np.transpose(pd.DataFrame(values[1]))
        HSVcam1 = np.transpose(pd.DataFrame(values[2]))
        HSVcam2 = np.transpose(pd.DataFrame(values[3]))

        # criando um dataframe com os dados da amostra
        finaldata = pd.DataFrame(
            {'Name': name, 'valueR_cam1': RGBcam1[0], 'valueG_cam1': RGBcam1[1], 'valueB_cam1': RGBcam1[2],
             'valueH_cam1': HSVcam1[0], 'valueS_cam1': HSVcam1[1], 'valueV_cam1': HSVcam1[2], 'valueR_cam2': RGBcam2[0],
             'valueG_cam2': RGBcam2[1], 'valueB_cam2': RGBcam2[2], 'valueH_cam2': HSVcam2[0], 'valueS_cam2': HSVcam2[1],
             'valueV_cam2': HSVcam2[2]})

        # concatenando a planilha antiga com a planilha nova
        gatherdata = pd.concat([old_samples, finaldata])

        # salvando em cima do arquivo original os novos dados concatenados
        gatherdata.to_excel('DataCV.xlsx', index=False)

    def backup(self):
        return


