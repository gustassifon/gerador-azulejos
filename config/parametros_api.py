import argparse

class ParametrosApi():
    def __init__(self):
        self.parametros = argparse.ArgumentParser(
            prog="Gerador de Azulejos",
            description = "Aplicação criada como parte do projeto de conclusão de curso do Mestrado em Media Digitais "
                "Interativos",
        )

        self.parametros.add_argument(
            "caminho_arquivo_modelo",
            type=str,
            help="Indicar o caminho para o arquivo que contém o modelo da DCGAN/IA treinado.",
            nargs="?",
        )

    def recuperar_parametros(self):
        return self.parametros.parse_args()