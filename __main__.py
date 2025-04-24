import math
import numpy as np
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
import keras

from gan.dcgan import DCGAN

if __name__ == "__main__":
    arquivo_dataset = 'datasets/20250108114727678_100x100_dataset.npy'
    imagens = np.load(file=arquivo_dataset)
    print(imagens.shape)
    print(imagens.shape[0])
    ruido = tf.random.normal([imagens.shape[0], 100])
    print(ruido.shape)
    print(ruido[0])
    print(len(ruido))
    # print(imagens.min())
    # print(imagens.max())
    # plt.imshow(imagens[0])
    # plt.axis('off')
    # plt.show()
    # cv2.imshow("Imagem", imagens[0])
    # cv2.waitKey(0)

    # dcgan = DCGAN()
    # dcgan.construir_dcgan()
    # dcgan.discriminador.summary()
    # dcgan.gerador.summary()

