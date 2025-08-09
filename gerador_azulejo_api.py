import io
import os
import random
from pathlib import Path

import cv2
import keras
import tensorflow as tf
from flask import Flask, send_file

from config.parametros_api import ParametrosApi


def levantar_api(caminho_arquivo_modelo: str=None):
    app = Flask(__name__)
    caminho_modelo = 'modelos'
    caminho_modelo = caminho_arquivo_modelo if caminho_arquivo_modelo is not None else caminho_modelo

    def recuperar_imagem_modelo(caminho_arquivo: str):
        caminho_arquivo = Path(caminho_arquivo)

        if caminho_arquivo.is_dir(): # Significa que é uma pasta com vários modelos
            lista_arquivos = os.listdir(caminho_arquivo)
            nome_arquivo = random.choice(lista_arquivos)
            caminho_arquivo = caminho_arquivo.joinpath(nome_arquivo)

        print(caminho_arquivo)

        gerador = keras.models.load_model(str(caminho_arquivo))
        dimensao_ruido = gerador.input_shape[1]
        ruido = tf.random.normal([1, dimensao_ruido])
        imagem_gerada = gerador(ruido)

        imagem_gerada = imagem_gerada.numpy().astype("float32")
        imagem_gerada = imagem_gerada + 127.5
        imagem_gerada = imagem_gerada * 127.5
        imagem_gerada = imagem_gerada.astype("uint8")

        # Aumenta a imagem para 256x256 para melhorar a resposta da API
        imagem = cv2.resize(imagem_gerada[0], (256, 256), interpolation=cv2.INTER_CUBIC)

        # Detectar tamanho da imagem e ajustar fonte e posição
        altura, largura = imagem.shape[:2]

        # Adicionar texto na imagem com a data do modelo para identificar qual está sendo o modelo usado
        cv2.putText(
            imagem,
            caminho_arquivo.name.split('_')[0],
            (0, altura - 2),
            cv2.FONT_HERSHEY_PLAIN,
            0.8,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )
        resultado, buffer = cv2.imencode('.png', imagem)

        # Exclusão das variáveis para evitar estouro de memória
        del gerador
        del ruido
        del imagem_gerada

        return io.BytesIO(buffer)

    @app.route('/', methods=['GET'])
    def gerar_azulejo():
        return send_file(recuperar_imagem_modelo(caminho_modelo), mimetype='image/png')

    return app


if __name__ == '__main__':
    parametros_aplicacao = ParametrosApi().recuperar_parametros()
    levantar_api(parametros_aplicacao.caminho_arquivo_modelo).run(debug=True)