import board
import busio
import numpy as np
import pandas as pd
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import os


class NarizEletronico:
    def __init__(self):
        self.arrayforvoltage = np.zeros(50000)
        self.arrayfortime = np.zeros(50000)

    def read_sample(self, name):
        # lendo as portas SCL e SDA no formato I2C
        i2c = busio.I2C(board.SCL, board.SDA)
        ads = ADS.ADS1115(i2c)
        channelA0 = AnalogIn(ads, ADS.P0)
        print(channelA0.value, channelA0.voltage)

        starttime = time.time()  # inicializando a contagem de tempo

        # preenchendo os valores nos vetores de voltagem e tempo
        for i in range(0, 50000):
            self.arrayforvoltage[i] = float(channelA0.voltage)
            self.arrayfortime[i] = float(time.time() - starttime)

    def process_data(self):
        # calculando a voltagem media
        meanvoltage = np.mean(self.arrayforvoltage)

        ENmean = np.mean(self.arrayforvoltage)

        # Transformar valor de média em dataframe
        # ENmean = pd.DataFrame(ENmean)

        return ENmean, meanvoltage

    def plot(self):
        # transpondo vetores
        arrayforvoltagetransposed = np.transpose(self.arrayforvoltage)
        arrayfortimetransposed = np.transpose(self.arrayfortime)

        figureofchart = plt.figure()

        # criar subgráfico na figura
        chart = figureofchart.add_subplot(111)

        # plotando gráfico de tempo corrido por voltagem adquirida em função do mesmo
        plt.plot(arrayfortimetransposed, arrayforvoltagetransposed)

    def update_data(self, name, ENmean):
        # criando um df para valores de voltagem
        # df = pd.DataFrame(self.arrayforvoltage)

        # abrindo planilha de amostras antigas
        old_samples = pd.read_excel('Datanose.xlx')

        # criando um dataframe com dados novos
        finaldata = pd.DataFrame({'Name': name, 'ENmean': ENmean})

        # concatenando planilha de dados antigos com dado novo
        gatherdata = pd.concat([old_samples, finaldata])

        # salvando a planilha nova em cima do arquivo original
        gatherdata.to_excel('Datanose.xls', index=False)









