import keras
import tensorflow as tf
import cv2
from flask import Flask, send_file
import io

from config.parametros_api import ParametrosApi

def levantar_api(caminho_arquivo_modelo: str=None):
    app = Flask(__name__)
    caminho_modelo = 'resultado/20250609113406/model/gerador_azulejos.h5'
    tamanho_ruido = 100
    caminho_modelo = caminho_arquivo_modelo if caminho_arquivo_modelo is not None else caminho_modelo

    def recuperar_imagem_modelo(caminho_arquivo_modelo: str, dimensao_ruido: int):
        gerador = keras.saving.load_model(caminho_arquivo_modelo)
        ruido = tf.random.normal([1, dimensao_ruido])
        imagem_gerada = gerador(ruido)
        imagem_gerada = imagem_gerada.numpy().astype("float32")
        imagem_gerada = imagem_gerada + 127.5
        imagem_gerada = imagem_gerada * 127.5
        imagem_gerada = imagem_gerada.astype("uint8")
        resultado, buffer = cv2.imencode('.png', imagem_gerada[0])
        return io.BytesIO(buffer)

    @app.route('/', methods=['GET'])
    def gerar_azulejo():
        return send_file(recuperar_imagem_modelo(caminho_modelo, tamanho_ruido), mimetype='image/png')

    return app


if __name__ == '__main__':
    parametros_aplicacao = ParametrosApi().recuperar_parametros()
    levantar_api(parametros_aplicacao.caminho_arquivo_modelo).run(debug=True)