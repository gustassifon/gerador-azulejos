import keras
import tensorflow as tf
import cv2
from flask import Flask, send_file
import io

from config.parametros_api import ParametrosApi

app = Flask(__name__)
caminho_modelo = ''
tamanho_ruido = 100


def recuperar_imagem_modelo(caminho_arquivo_modelo: str, dimensao_ruido: int):
    print('No recuperar_imagem: ' + caminho_arquivo_modelo)
    gerador = keras.saving.load_model(caminho_arquivo_modelo)
    ruido = tf.random.normal([1, dimensao_ruido])
    imagem_gerada = gerador(ruido)
    imagem_gerada = imagem_gerada.numpy().astype("float32")
    imagem_gerada = imagem_gerada + 127.5
    imagem_gerada = imagem_gerada * 127.5
    imagem_gerada = imagem_gerada.astype("uint8")
    resultado, buffer = cv2.imencode('.png', imagem_gerada[0])
    return io.BytesIO(buffer)


@app.route('/gerar-azulejo', methods=['GET'])
def gerar_azulejo():
    return send_file(recuperar_imagem_modelo(caminho_modelo, tamanho_ruido), mimetype='image/png')


def levantar_api(caminho_arquivo_modelo: str):
    global caminho_modelo
    caminho_modelo = caminho_arquivo_modelo
    app.run(debug=True)


if __name__ == '__main__':
    parametros_aplicacao = ParametrosApi().recuperar_parametros()
    print('No main: ' + parametros_aplicacao.caminho_arquivo_modelo)
    levantar_api(parametros_aplicacao.caminho_arquivo_modelo)