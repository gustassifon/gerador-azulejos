import datetime
import math
import os
import time

import numpy as np
import cv2

import tensorflow as tf
import keras
from keras import Sequential
from keras import layers
from tqdm import tqdm


class DCGAN:
    def __init__(
            self,
            # Representa o tamanho do array/ruído de entrada para o gerador.
            dimensao_ruido=100,
            # O tamanho, em píxeis, da imagem que será processada. Deve ser uma imagem quadrada.
            tamanho_imagem=128,
            # Quantidade de canais que tem a imagem, o padrão será 3 canais (RGB).
            canais_imagem=3,
            # Tamanho do lote/batch usado para o treinamento do modelo.
            tamanho_lote=32,


    ):
        # O maior tamanho e inicial de filtro usado como parâmetro para as funções Conv2D e Conv2DTranspose
        self.tamanho_filtro_maximo = 1024
        self.tamanho_filtro_minimo = 16
        self.tamanho_kernel = 3
        self.tamanho_strides = 2
        self.dimensao_ruido = dimensao_ruido
        self.tamanho_imagem = tamanho_imagem
        self.canais_imagem = canais_imagem
        self.tamanho_lote = tamanho_lote

        # O gerador e discriminador serão "constuidos" quando a função construir_dcgan for chamada. A decisao de treinar
        # ou não o modelo também será estabelecido na função construir_dcgan.
        self.discriminador = None
        self.gerador = None
        self.epocas = None
        # Parâmetro opcional caso queira treinar o modelo.
        self.caminho_imagens_dataset = None,
        # Parâmetro opcional caso queira salvar o histórico da execução do programa.
        self.caminho_resultado = None,
        self.caminho_historico_execucao = None
        self.caminho_imagens_treinamento = None
        self.caminho_modelo_treinado = None

        # Função do Keras utilizada para calcular a perda do discriminador e gerador. Usa-se para calcular o gradiente
        # e ir "aproximando" o erro para chegar a um "padrão comum" das imagens
        self.funcao_cross_entropy = keras.losses.BinaryCrossentropy()

        # Otimizadores usados para o treinamento
        self.otimizador_discriminador = keras.optimizers.Adam(1e-3)
        self.otimizador_gerador = keras.optimizers.Adam(1e-3)

        self.dataset = None


    def __logger(self, mensagem):
        if self.caminho_historico_execucao is not None:
            momento = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open(self.caminho_historico_execucao, 'a') as arquivo:
                arquivo.write(f'{momento} -> {mensagem}{os.linesep}')
        else:
            print(mensagem)


    def __calcular_perda_discriminador(self, resultado_real, resultado_falso):
        perda_dados_reais = self.funcao_cross_entropy(tf.ones_like(resultado_real), resultado_real)
        perda_dados_falsos = self.funcao_cross_entropy(tf.zeros_like(resultado_falso), resultado_falso)
        return perda_dados_reais + perda_dados_falsos


    def __calcular_perda_gerador(self, resultado_falso):
        # Para o gerador as imagens falsas são as reais, pois foram gerados por ele. Assim, é usada o tf.ones_like por
        # essa razão.
        return self.funcao_cross_entropy(tf.ones_like(resultado_falso), resultado_falso)


    def __construir_gerador(self):
        # Como a transformação da imagem acontece sempre dobrando seu tamanho, ou seja, de 4x4 depois temos 8x8, 16x16
        # e assim por diante, temos uma repetição baseada em uma potência de 2. Dessa forma, usando o tamanho da imagem
        # em uma função logarítimica na base 2 é possível encontrar o total de repetições necessárias para fazer esta
        # transformação. A imagem é aumentada até ficar um tamanho maior que o tamanho final da imagem. Ao final o
        # "resizing" é feito de uma imagem maior para uma menor. Por isso, considerando que já se inicia o processo a
        # partir de uma imagem 4x4 ou 2 ao quadrado, é preciso diminuir o número de repetições em 2.
        repeticoes = int(math.log2(self.tamanho_imagem)) - 2

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
        repeticoes = int(math.log2(self.tamanho_imagem)) - 2

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
            modelo.add(layers.Dropout(0.3))
            modelo.add(layers.LeakyReLU(negative_slope=0.2))

        # O resultado precisa ser um valor único que é avalido probabilisticamente conforme o padrão das imagens usadas
        # para o treinamento.
        modelo.add(layers.Flatten())
        modelo.add(layers.Dense(1, activation='sigmoid'))

        return modelo


    def __preparar_dataset(self):
        self.__logger('Preparando o dataset...')

        if self.caminho_imagens_dataset is not None and os.path.exists(self.caminho_imagens_dataset):
            dataset = []
            lista_arquivos = os.listdir(self.caminho_imagens_dataset)
            qtde_imagens = len(lista_arquivos)

            # Se não existir imagens dentro do diretório informado, não executa o processamento do dataset
            if qtde_imagens > 0:
                self.__logger('Carregando as imagens e convertendo em ndarray...')
                for arquivo in tqdm(lista_arquivos):
                    imagem = cv2.imread(os.path.join(self.caminho_imagens_dataset, arquivo))
                    imagem = np.asarray(imagem)
                    dataset.append(imagem)

                dataset = np.asarray(dataset).astype(np.float32)

                # Normaliza os dados para um conjunto de valores entre −1 e 1 para o uso da função "tanh". Por isso a
                # subtração do valor por 127,5 (metade de 255, valor máximo de uma cor RGB) e na sequência a divisão por
                # este mesmo valor
                self.__logger('Normaliza o dataset (-1, 1)')
                dataset = (dataset - 127.5) / 127.5

                self.__logger('Cria o dataset no formato do TensorFlow e atribui a propriedade do objeto')
                self.dataset = (
                    tf.data.Dataset.from_tensor_slices(dataset).shuffle(qtde_imagens).batch(self.tamanho_lote)
                )
                return True

        self.__logger("Não foi indicado um dataset para treinamento.")
        return False


    def __salvar_imagem(self, ruidos, epoca):
        for i, ruido in enumerate(ruidos):
            imagem_gerada = self.gerador(ruido, training=False)
            imagem_gerada = imagem_gerada.numpy().astype("float32")
            imagem_gerada = imagem_gerada + 127.5
            imagem_gerada = imagem_gerada * 127.5
            imagem_gerada = imagem_gerada.astype("uint8")
            caminho_imagem_gerada = (
                os.path.join(self.caminho_imagens_treinamento, f'epoca_{epoca + 1}_{str(i + 1).zfill(2)}.png')
            )
            cv2.imwrite(caminho_imagem_gerada, imagem_gerada[0])


    def __realizar_treinamento(self):
        @tf.function
        def treinar_etapa(imagens_dataset):
            ruido = tf.random.normal([self.tamanho_lote, self.dimensao_ruido])

            # O tf.GradienteTape() é um recurso do TensorFlow para gravar as operações executadas para posterior cálculo
            # dos gradientes de cada uma das redes neurais sendo treinadas, gerador e discriminador.
            with tf.GradientTape() as gerador_tape, tf.GradientTape() as discriminador_tape:
                # Gera imagens a partir do gerador em "modo" de treino
                imagens_geradas = self.gerador(ruido, training=True)

                # Recupera do discriminador o resultado para o treinamento com as imagens reais e a falsas geradas pelo
                # gerador.
                resultado_real_discriminador = self.discriminador(imagens_dataset, training=True)
                resultado_falso_discriminador = self.discriminador(imagens_geradas, training=True)

                # Calcula, usando a "loss function" do Keras, a perda do gerador e do discriminador
                perda_gerador = self.__calcular_perda_gerador(resultado_falso_discriminador)
                perda_discriminador = (
                    self.__calcular_perda_discriminador(resultado_real_discriminador, resultado_falso_discriminador)
                )

            # A partir do cálculo da função de perda é possivel calcular o gradiente
            gradiente_gerador = gerador_tape.gradient(perda_gerador, self.gerador.trainable_variables)
            gradiente_discriminador = (
                discriminador_tape.gradient(perda_discriminador, self.discriminador.trainable_variables)
            )

            # Aplica os gradientes no otimizador
            self.otimizador_gerador.apply_gradients(zip(gradiente_gerador, self.gerador.trainable_variables))
            self.otimizador_discriminador.apply_gradients(
                zip(gradiente_discriminador, self.discriminador.trainable_variables)
            )

            return perda_gerador, perda_discriminador


        if self.dataset is not None:
            self.__logger('Treinando o modelo...')
            inicio_treinamento = time.time()
            ruidos_fixos = []

            for i in range(10):
                ruidos_fixos.append(tf.random.normal([1, self.dimensao_ruido]))

            lista_perda_gerador = []
            lista_perda_discriminador = []

            for epoca in range(self.epocas):
                inicio_epoca = time.time()

                for lote in self.dataset:
                    resultado = treinar_etapa(lote)
                    lista_perda_gerador.append(resultado[0])
                    lista_perda_discriminador.append(resultado[1])

                perda_total_gerador = sum(lista_perda_gerador) / len(lista_perda_gerador)
                perda_total_discriminador = sum(lista_perda_discriminador) / len(lista_perda_discriminador)
                fim_epoca = time.time()
                duracao_epoca = fim_epoca - inicio_epoca

                self.__logger(
                    f'Epoca {epoca + 1} finalizada em {duracao_epoca:.2f} segundos. PERDA GERADOR: '
                    f'{perda_total_gerador} >> PERDA DISCRIMINADOR: {perda_total_discriminador}'
                )

                # Salva imagens de exemplo, com a progressão do trabalho, apenas nas épocas múltiplas de 10
                if epoca % 5 == 0:
                    self.__salvar_imagem(ruidos=ruidos_fixos, epoca=epoca)

            fim_treinamento = time.time()
            duracao_treinamento = fim_treinamento - inicio_treinamento
            self.__logger(f'Treinamento finalizado em {duracao_treinamento:.2f} segundos.')


    def construir_dcgan(self, treinar=True, epocas=100, caminho_imagens_dataset=None, caminho_resultado=None):
        # Quantidade de etapas para o treinamento do modelo.
        self.epocas = epocas
        self.caminho_imagens_dataset = caminho_imagens_dataset
        self.caminho_resultado = caminho_resultado

        # Testa se foi definido um local para salvar o progresso da execução do programa, ou o caminho para carregar o
        # modelo treinado. Se não houver esse parâmetro a aplicação não pode ser executada
        if self.caminho_resultado is not None and os.path.exists(self.caminho_resultado):
            # Treinar o modelo gerador e discriminador, salva o gerador para uso posterior
            if treinar:
                self.caminho_resultado = (
                    os.path.join(self.caminho_resultado, datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
                )
                os.mkdir(self.caminho_resultado)
                self.caminho_historico_execucao = os.path.join(self.caminho_resultado, 'historico_execucao.txt')
                self.caminho_imagens_treinamento = os.path.join(self.caminho_resultado, 'imgs')
                self.caminho_modelo_treinado = os.path.join(self.caminho_resultado, 'model')
                os.mkdir(self.caminho_imagens_treinamento)
                os.mkdir(self.caminho_modelo_treinado)
                self.caminho_modelo_treinado = os.path.join(self.caminho_modelo_treinado, 'gerador_azulejos.h5')

                if self.__preparar_dataset():
                    # Constrói a estrutuda do modelo gerador.
                    self.gerador = self.__construir_gerador()

                    # Constrói a estrutuda do modelo dicriminador.
                    self.discriminador = self.__construir_discriminador()

                    self.__realizar_treinamento()
                    self.gerador.save(self.caminho_modelo_treinado)
            else: # Carregar modelo já treinado.
                self.gerador = keras.saving.load_model(self.caminho_resultado)
        else:
            self.__logger("Não foi indicado um local para salvar/carregar o modelo.")

