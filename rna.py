from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class RNA:
    def __init__(self):
        # criando o modelo
        self.model = keras.Sequential([
            layers.Input(shape=(6,)),  # camada de entrada com 6 neurônios (RGB e HSV)
            layers.Dense(100, activation='relu'),  # camada oculta com 100 neurônios
            layers.Dense(1, activation='sigmoid')  # camada de saída com 1 neurônio e ativação sigmoidal
        ])

        # compilando o modelo
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    # função para separar df em treino teste sem separar triplicatas
    def split_df(self, df):
        df['grupo'] = (df.index // 3).astype(str)

        grupos_unicos = df['grupo'].unique()

        # dividindo grupos em treino teste
        grupos_treino, grupos_teste = train_test_split(grupos_unicos, test_size=0.2)

        # filtrando df com base na partição
        df_treino = df[df['grupo'].isin(grupos_treino)]
        df_teste = df[df['grupo'].isin(grupos_teste)]

        # dropando a coluna de grupo
        df_treino.drop(columns=['grupo'], inplace=True)
        df_teste.drop(columns=['grupo'], inplace=True)

        y_train = df_treino['Name']
        y_test = df_teste['Name']
        x_train = df_treino.drop(columns=['Name'])
        x_test = df_teste.drop(columns=['Name'])

        return x_train, y_train, x_test, y_test

    def treinar(self, x_train, y_train, x_test, y_test, epochs):
        # obtendo numero de linhas dos dados de treino
        nrows = x_train.shape[0]

        # calculando o batch_size
        batch_size = nrows / epochs

        # treinando o modelo
        history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test)
        )

        return pd.DataFrame(history.history)

    def avaliar(self):

        return

    def plot(self, y_test, y_pred):
        # plotando valores esperados vs previstos
        plt.scatter(y_test, y_pred)
        plt.xlabel('Valores Esperados')
        plt.ylabel('Valores Previstos')
        plt.title('Valores Esperados vs Valores Previstos')
        plt.show()


rna = RNA()
df = pd.read_excel('DataCV.xlsx')

# particionando dados de treino e teste
x_train, y_train, x_test, y_test = rna.split_df(df)


# avaliando o modelo
test_loss = rna.model.evaluate(x_test, y_test)
metrics_df = pd.DataFrame({'Loss': [test_loss]})

# obtendo valores previstos
y_pred = rna.model.predict(x_test)






