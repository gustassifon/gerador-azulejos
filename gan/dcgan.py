import math
import tensorflow as tf
import keras
from keras import Sequential
from keras import layers



class DCGAN:
    def __init__(
            self,
            dimensao_ruido=100, # Representa o tamanho do array/ruído de entrada para o gerador.
            tamanho_imagem=100, # O tamanho, em píxeis, da imagem que será processada. Deve ser uma imagem quadrada.
            canais_imagem=3 # Quantidade de canais que tem a imagem, o padrão será 3 canais (RGB).
    ):
        # O maior tamanho e inicial de filtro usado como parâmetro para as funções Conv2D e Conv2DTranspose
        self.tamanho_filtro_maximo = 1024
        self.tamanho_filtro_minimo = 16
        self.tamanho_kernel = 3
        self.tamanho_strides = 2
        self.dimensao_ruido = dimensao_ruido
        self.tamanho_imagem = tamanho_imagem
        self.canais_imagem = canais_imagem

        # O gerador e discriminador serão "constuidos" quando a função construir_dcgan for chamada.
        self.discriminador = None
        self.gerador = None

        # Função do Keras utilizada para calcular a perda do discriminador e gerador. Usa-se para calcular o gradiente
        # e ir "aproximando" o erro para chegar a um "padrão comum" das imagens
        self.funcao_cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)

    def __calcular_perda_discriminador(self, imagens_reais, imagens_falsas):
        perda_dados_reais = self.funcao_cross_entropy(tf.ones_like(imagens_reais), imagens_reais)
        perda_dados_falsos = self.funcao_cross_entropy(tf.zeros_like(imagens_falsas), imagens_falsas)
        return perda_dados_reais + perda_dados_falsos

    def __calcular_perda_gerador(self, images_falsas):
        # Para o gerador as imagens falsas são as reais, pois foram gerados por ele. Assim, é usada o tf.ones_like por
        # essa razão.
        return self.funcao_cross_entropy(tf.ones_like(images_falsas), images_falsas)

    def __construir_gerador(self):
        # Como a transformação da imagem acontece sempre dobrando seu tamanho, ou seja, de 4x4 depois temos 8x8, 16x16
        # e assim por diante, temos uma repetição baseada em uma potência de 2. Dessa forma, usando o tamanho da imagem
        # em uma função logarítimica na base 2 é possível encontrar o total de repetições necessárias para fazer esta
        # transformação. A imagem é aumentada até ficar um tamanho maior que o tamanho final da imagem. Ao final o
        # "resizing" é feito de uma imagem maior para uma menor. Por isso, considerando que já se inicia o processo a
        # partir de uma imagem 4x4 ou 2 ao quadrado, é preciso diminuir o número de repetições em 1.
        repeticoes = int(math.log2(self.tamanho_imagem)) - 1

        # Cria o modelo
        modelo = Sequential()

        # Inclui o "input" do modelo que, neste caso, será um array de 100 posições com informações aleatórias. Projeta
        # o vetor de ruído para um tensor 4x4 com 1024 filtros.
        modelo.add(layers.Input(shape=(self.dimensao_ruido,)))
        modelo.add(layers.Dense(4 * 4 * self.tamanho_filtro_maximo, use_bias=False))
        modelo.add(layers.BatchNormalization(momentum=0.8))
        modelo.add(layers.LeakyReLU())

        # Remodela para tensor 4x4x1024.
        modelo.add(layers.Reshape((4, 4, self.tamanho_filtro_maximo)))

        # O número de camadas do modelo será determinado pelo tamanho da imagem. A cada iteração o tamanho da imagem,
        # que se inicia em 4x4, é duplicado e os filtros caem pela metade. Ao final é feito um "resizing" "retornando" o
        # tamanho para o valor informado como parâmetro da imagem.
        for i in range(repeticoes):
            i += 1

            # "Bit shift": equivale a uma divisão por 2 em potências de 2 a cada iteração.
            filtros = self.tamanho_filtro_maximo >> i

            # Garante que não fique menor que 16
            filtros = max(filtros, self.tamanho_filtro_minimo)

            modelo.add(
                layers.Conv2DTranspose(
                    filters=filtros,
                    kernel_size=self.tamanho_kernel,
                    padding='same',
                    use_bias=False,
                    strides=self.tamanho_strides
                )
            )
            modelo.add(layers.BatchNormalization(momentum=0.8))
            modelo.add(layers.LeakyReLU())

        # Camada para ajustar exatamente para o tamanho da imagem.
        modelo.add(layers.Resizing(self.tamanho_imagem, self.tamanho_imagem))

        # Camada final: Gera a imagem colorida com 3 canais. A ativação tanh normaliza a saída para [-1,1].
        modelo.add(
            layers.Conv2DTranspose(
                filters=self.canais_imagem,
                kernel_size=self.tamanho_kernel,
                padding='same',
                use_bias=False,
                activation='tanh'
            )
        )

        return modelo

    def __construir_discriminador(self):
        modelo = Sequential()

        # Inclui o "input" do modelo. A entrada para o discriminador será uma imagem. Terá o tamanho definido por parâ-
        # metro quando na criação do objeto da classe DCGAN. Possui A quantidade de canais, conforme o informado na
        # criação do objeto. Por padrão usa 3 canais (RGB).
        modelo.add(layers.Input(shape=(self.tamanho_imagem, self.tamanho_imagem, self.canais_imagem)))

        # Mesma ideia do comentário feito no __construir_gerador, contudo aqui funciona ao contrário os filtros aumentam
        # a medida que a imagem diminui.
        repeticoes = int(math.log2(self.tamanho_imagem)) - 1

        for i in range(repeticoes):
            i = repeticoes - i

            # "Bit shift": equivale a uma divisão por 2 em potências de 2 a cada iteração.
            filtros = self.tamanho_filtro_maximo >> i

            # Garante que não fique menor que 16
            filtros = max(filtros, self.tamanho_filtro_minimo)

            modelo.add(
                layers.Conv2D(
                    filters=filtros,
                    kernel_size=self.tamanho_kernel,
                    padding='same',
                    use_bias=False,
                    strides=self.tamanho_strides
                )
            )
            modelo.add(layers.LeakyReLU())
            modelo.add(layers.Dropout(0.3))

        # O resultado precisa ser um valor único que é avalido probabilisticamente conforme o padrão das imagens usadas
        # para o treinamento.
        modelo.add(layers.Flatten())
        modelo.add(layers.Dense(1, activation='sigmoid'))

        return modelo

    def __realizar_treinamento(self, dataset, epocas=100):
        pass

    def construir_dcgan(self, treinamento=True):

        if treinamento: # Treinar o modelo gerador e discriminador, salva o gerador para uso posterior
            # Constrói a estrutuda do modelo gerador.
            self.gerador = self.__construir_gerador()

            # Constrói a estrutuda do modelo dicriminador.
            self.discriminador = self.__construir_discriminador()

            self.__realizar_treinamento()
        else: # Carregar modelo já treinado.
            self.gerador = keras.saving.load_model('models/gerador.keras')