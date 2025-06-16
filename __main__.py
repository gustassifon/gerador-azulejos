import io
import os
import time

import cv2
import tensorflow as tf
from flask import Flask, send_file
from tqdm import tqdm

from config.parametros_main import ParametrosMain
from formatador.formatador_imagem import FormatadorImagem
from gan.dcgan import DCGAN




# Usa a classe FormatadorImagem para ler um diretório e transformar todas as imagens conforme os parâmetros passados na
# construção do objeto. Os valores defauts do FormatadorImagem já foram configurados para os dados utilizados neste
# projeto
def preparar_imagens(diretorio_origem: str, diretorio_destino: str, tamanho_imagem: int):
    inicio = time.time()
    print('Validando o diretório de origem...')

    if not os.path.exists(diretorio_origem) or len(os.listdir(diretorio_origem)) == 0:
        print('Não existe diretório de origem ou arquivos na pasta, não é possível preparar as imagens.')
        return

    print('Validando o diretório de destino...')

    if not os.path.exists(diretorio_destino):
        os.mkdir(diretorio_destino)
        print('O diretório de destino não existia e foi criado.')
    else:
        print('O diretório de destino já existe.')

    print('Inciando a preparação das imagens...')
    print('Diretório de origem: ', diretorio_origem)
    print('Diretório de destino: ', diretorio_destino)

    lista_arquivos = os.listdir(diretorio_origem)
    lista_arquivos.sort()

    for arquivo in tqdm(lista_arquivos):
        formatador_imagem = FormatadorImagem(
            caminho_imagem=os.path.join(diretorio_origem, arquivo),
            nova_altura=tamanho_imagem,
            nova_largura=tamanho_imagem,
        )
        formatador_imagem.salvar(diretorio_destino)

    fim = time.time()
    duracao = fim - inicio
    print(f'Imagens preparadas em {duracao:.2f} segundos.')


if __name__ == "__main__":
    parametros_aplicacao = ParametrosMain().recuperar_parametros()

    if parametros_aplicacao.acao == 'redimensionar':
        diretorio_origem = parametros_aplicacao.diretorio_origem
        diretorio_destino = parametros_aplicacao.diretorio_destino
        tamanho_imagem = parametros_aplicacao.tam
        preparar_imagens(diretorio_origem, diretorio_destino, tamanho_imagem)
    elif parametros_aplicacao.acao == 'treinar':
        diretorio_dataset = parametros_aplicacao.diretorio_dataset
        diretorio_resultado = parametros_aplicacao.diretorio_resultado
        epocas = parametros_aplicacao.epocas
        dcgan = DCGAN()
        dcgan.construir_dcgan(
            treinar=True,
            caminho_imagens_dataset=diretorio_dataset,
            caminho_resultado=diretorio_resultado,
            epocas=epocas,
        )
    else: # api
        caminho_arquivo_modelo = parametros_aplicacao.caminho_arquivo_modelo
        app.run(debug=True)