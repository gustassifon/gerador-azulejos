import datetime
import os
import sys

import keras
import tensorflow as tf
from keras import layers
from keras import ops

# Necessário para conseguir realizar o import dos scripts do módulo "config"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.parametros_dcgan_keras3 import ParametrosDcganKeras3



caminho_historico_execucao = 'historico_execucao.txt'
caminho_resultado = 'resultado'
caminho_imagens_treinamento = 'imgs'
caminho_modelo_treinado = 'model'



class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim=128):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.seed_generator = keras.random.SeedGenerator(1337)

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real_images):
        # Sample random points in the latent space
        batch_size = ops.shape(real_images)[0]
        random_latent_vectors = keras.random.normal(
            shape=(batch_size, self.latent_dim), seed=self.seed_generator
        )

        # Decode them to fake images
        generated_images = self.generator(random_latent_vectors)

        # Combine them with real images
        combined_images = ops.concatenate([generated_images, real_images], axis=0)

        # Assemble labels discriminating real from fake images
        labels = ops.concatenate(
            [ops.ones((batch_size, 1)), ops.zeros((batch_size, 1))], axis=0
        )
        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space
        random_latent_vectors = keras.random.normal(
            shape=(batch_size, self.latent_dim), seed=self.seed_generator
        )

        # Assemble labels that say "all real images"
        misleading_labels = ops.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }



class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=3, latent_dim=128, save_img_path=None, logger_path=None):
        super().__init__()
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.seed_generator = keras.random.SeedGenerator(42)
        self.save_img_path = save_img_path
        self.logger_path = logger_path


    def __logger(self, mensagem):
        momento = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mensagem = f'{momento} -> {mensagem}{os.linesep}'

        if self.logger_path is not None:
            with open(self.logger_path, 'a') as arquivo:
                arquivo.write(mensagem)
        else:
            print(mensagem)

    # def on_train_batch_end(self, batch, logs=None):
    #     self.__logger("Batch {batch} - Logs: {logs}")


    def on_epoch_end(self, epoch, logs=None):
        # Salva imagens de exemplo, com a progressão do trabalho, apenas nas épocas múltiplas de 10
        if epoch % 5 == 0:
            random_latent_vectors = keras.random.normal(
                shape=(self.num_img, self.latent_dim), seed=self.seed_generator
            )
            generated_images = self.model.generator(random_latent_vectors)
            generated_images *= 255
            generated_images.numpy()

            for i in range(self.num_img):
                img = keras.utils.array_to_img(generated_images[i])
                save_img_path = "generated_img_%03d_%d.png" % (epoch, i)

                if self.save_img_path is not None:
                    save_img_path = os.path.join(self.save_img_path, save_img_path)

                img.save(save_img_path)

        self.__logger(f'Terminando o processamento da época: {epoch} - Logs: {logs}')



def logger(mensagem):
    momento = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mensagem = f'{momento} -> {mensagem}{os.linesep}'

    if caminho_historico_execucao is not None:
        with open(caminho_historico_execucao, 'a') as arquivo:
            arquivo.write(mensagem)
    else:
        print(mensagem)



def funcao_principal(epocas:int=50, diretorio_dataset:str=None, diretorio_resultado:str=None):
    # O trecho de código abaixo é importante quando tento trabalhar com imagens de 128 pixels. O meu equipeamento não
    # consegue processar floats32 (padrão) para imagens "grandes". Trabalhando com imagens de 64 pixels é possível usar
    # a configuração padrão de 32. Interessante que ao usar o float16 a aplicação demora mais e se comportar pior que
    # ao usar o float32 padrão.
    # mixed_precision.set_global_policy('mixed_float16')

    global caminho_historico_execucao
    global caminho_resultado
    global caminho_imagens_treinamento
    global caminho_modelo_treinado

    if diretorio_dataset is None or (not os.path.exists(diretorio_dataset)):
        print("Não foi indicado um dataset para treinamento.")
        return # Finaliza a execução

    if diretorio_resultado is None:
        print("Não foi indicado um local para salvar/carregar o modelo.")
        return # Finaliza a execução

    caminho_resultado = diretorio_resultado

    # Prepara os diretorios para gravar o resultado
    caminho_resultado = (os.path.join(caminho_resultado, datetime.datetime.now().strftime('%Y%m%d%H%M%S')))
    os.mkdir(caminho_resultado)
    caminho_historico_execucao = os.path.join(caminho_resultado, caminho_historico_execucao)
    caminho_imagens_treinamento = os.path.join(caminho_resultado, caminho_imagens_treinamento)
    caminho_modelo_treinado = os.path.join(caminho_resultado, caminho_modelo_treinado)
    os.mkdir(caminho_imagens_treinamento)
    os.mkdir(caminho_modelo_treinado)
    caminho_modelo_treinado = os.path.join(caminho_modelo_treinado, 'gerador_azulejos.h5')

    logger('-----> Início treinamento DCGANKera3 <-----')
    logger('Criando o dataset...')
    dataset = keras.utils.image_dataset_from_directory(
        diretorio_dataset, label_mode=None, image_size=(64, 64), batch_size=32
    )
    dataset = dataset.map(lambda x: x / 255.0)

    logger('Criando o discriminador...')
    discriminator = keras.Sequential(
        [
            keras.Input(shape=(64, 64, 3)), # Fiz alguns testes com 128
            layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="discriminator",
    )
    discriminator.summary(print_fn=logger)

    # 09/08/2025 - Para ficar igual ao modelo dcgan.py, alterei o latent_dim para 100.
    latent_dim = 100

    logger(f'Criando o gerador com ruido de tamanho: {latent_dim}')
    generator = keras.Sequential(
        [
            keras.Input(shape=(latent_dim,)),
            layers.Dense(8 * 8 * 128), # Quando trabalhar com 128px a entrada precisa ser 16 e 64px deve ser 8
            layers.Reshape((8, 8, 128)),
            layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Conv2D(3, kernel_size=5, padding="same", activation="sigmoid"),
        ],
        name="generator",
    )
    generator.summary(print_fn=logger)

    epochs = epocas  # In practice, use ~100 epochs
    logger(f'Quantidade de épocas: {epochs}')

    gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
    gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss_fn=keras.losses.BinaryCrossentropy(),
    )

    gan.fit(
        dataset,
        epochs=epochs,
        callbacks=[
            GANMonitor(
                num_img=5,
                latent_dim=latent_dim,
                save_img_path=caminho_imagens_treinamento,
                logger_path=caminho_historico_execucao
            )
        ]
    )

    # 09/08/2025 - Antes eu estava salvando o modelo todo e não apenas o gerador. Quando tentava usar o arquivo na API
    # resultava em um erro, pois tentava carregar todo o modelo
    gan.generator.save(caminho_modelo_treinado)



if __name__ == '__main__':
    parametros_aplicacao = ParametrosDcganKeras3().recuperar_parametros()
    diretorio_dataset = parametros_aplicacao.diretorio_dataset
    diretorio_resultado = parametros_aplicacao.diretorio_resultado
    epocas = parametros_aplicacao.epocas
    funcao_principal(diretorio_dataset=diretorio_dataset, diretorio_resultado=diretorio_resultado, epocas=epocas)