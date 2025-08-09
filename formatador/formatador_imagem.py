import os

import cv2


# Recebe o caminho de uma imagem para formatá-la conforme o esperado pela rede neural. Guarda em memória a imagem origi-
# nal e também as várias transformações feitas na imagem. O construtor já converte a imagem automaticamente, então não
# há a necesidade de chamar os métodos. Contudo, é possível utilizar os métodos com novas dimensões que não aquelas que
# representam as propriedades do objeto.
class FormatadorImagem:
    def __init__(self, caminho_imagem: str, nova_altura: int = 128, nova_largura: int = 128):
        self.imagem_original = cv2.imread(caminho_imagem)
        self.nome_imagem = os.path.basename(caminho_imagem)
        self.altura_original = self.imagem_original.shape[0]
        self.largura_original = self.imagem_original.shape[1]
        self.nova_altura = nova_altura
        self.nova_largura = nova_largura
        self.imagem_redimensionada = self.redimensionar()
        self.imagem_rotacionada_90 = self.rotacionar_90()
        self.imagem_rotacionada_180 = self.rotacionar_180()
        self.imagem_rotacionada_270 = self.rotacionar_270()
        self.imagem_espelhada_horizontal = self.espelhar_horizontal()
        self.imagem_espelhada_vertical = self.espelhar_vertical()

    # Ao utilizar a função print() para o objeto, imprime os dados e shapes das imagens que compões aquele objeto.
    def __str__(self):
        return (
            "Nome: {0}\n"
            "Altura original: {1}\n"
            "Largura original: {2}\n"
            "Nova altura: {3}\n"
            "Nova largura: {4}\n"
            "--- Formatos (altura, largura, canais):\n"
            "Formato redimensionada: {5}\n"
            "Formato rotacionada 90: {6}\n"
            "Formato rotacionada 180: {7}\n"
            "Formato rotacionada 270: {8}\n"
            "Formato espelhada vertical: {9}\n"
            "Formato espelhada horizontal: {10}\n"
            .format(
                self.nome_imagem,
                self.altura_original,
                self.largura_original,
                self.nova_altura,
                self.nova_largura,
                self.imagem_redimensionada.shape,
                self.imagem_rotacionada_90.shape,
                self.imagem_rotacionada_180.shape,
                self.imagem_rotacionada_270.shape,
                self.imagem_espelhada_vertical.shape,
                self.imagem_espelhada_horizontal.shape,
            )
        )

    def redimensionar(self, imagem: cv2=None,  nova_altura: int=None, nova_largura: int=None):
        if imagem is None:
            imagem = self.imagem_original

        if nova_altura is None:
            nova_altura = self.nova_altura

        if nova_largura is None:
            nova_largura = self.nova_largura

        return cv2.resize(imagem, (nova_largura, nova_altura), interpolation=cv2.INTER_AREA)

    def rotacionar_90(self, imagem: cv2=None):
        if imagem is None:
            imagem = self.imagem_redimensionada

        return cv2.rotate(imagem, cv2.ROTATE_90_CLOCKWISE)

    def rotacionar_180(self, imagem: cv2 = None):
        if imagem is None:
            imagem = self.imagem_redimensionada

        return cv2.rotate(imagem, cv2.ROTATE_180)

    def rotacionar_270(self, imagem: cv2 = None):
        if imagem is None:
            imagem = self.imagem_redimensionada

        return cv2.rotate(imagem, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def espelhar_horizontal(self, imagem: cv2 = None):
        if imagem is None:
            imagem = self.imagem_redimensionada

        return cv2.flip(imagem, flipCode=1)

    def espelhar_vertical(self, imagem: cv2 = None):
        if imagem is None:
            imagem = self.imagem_redimensionada

        return cv2.flip(imagem, flipCode=0)

    # Salva as imagens que fazem parte do objeto. Criadas no construtor no momento de instanciar a classe
    def salvar(self, diretorio: str):
        nome_arquivo, extensao = os.path.splitext(self.nome_imagem)

        caminho_nova_imagem = os.path.join(
            diretorio, nome_arquivo + '_{0}x{1}'.format(self.nova_altura, self.nova_largura) + extensao
        )
        cv2.imwrite(caminho_nova_imagem, self.imagem_redimensionada)

        caminho_nova_imagem = os.path.join(
            diretorio, nome_arquivo + '_{0}x{1}_90'.format(self.nova_altura, self.nova_largura) + extensao
        )
        cv2.imwrite(caminho_nova_imagem, self.imagem_rotacionada_90)

        caminho_nova_imagem = os.path.join(
            diretorio, nome_arquivo + '_{0}x{1}_180'.format(self.nova_altura, self.nova_largura) + extensao
        )
        cv2.imwrite(caminho_nova_imagem, self.imagem_rotacionada_180)

        caminho_nova_imagem = os.path.join(
            diretorio, nome_arquivo + '_{0}x{1}_270'.format(self.nova_altura, self.nova_largura) + extensao
        )
        cv2.imwrite(caminho_nova_imagem, self.imagem_rotacionada_270)

        caminho_nova_imagem = os.path.join(
            diretorio, nome_arquivo + '_{0}x{1}_hor'.format(self.nova_altura, self.nova_largura) + extensao
        )
        cv2.imwrite(caminho_nova_imagem, self.imagem_espelhada_horizontal)

        caminho_nova_imagem = os.path.join(
            diretorio, nome_arquivo + '_{0}x{1}_ver'.format(self.nova_altura, self.nova_largura) + extensao
        )
        cv2.imwrite(caminho_nova_imagem, self.imagem_espelhada_vertical)