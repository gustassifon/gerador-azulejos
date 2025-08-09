import argparse

class ParametrosDcganKeras3():
    def __init__(self):
        self.parametros = argparse.ArgumentParser(
            prog="Gerador de Azulejos",
            description = "Aplicação criada como parte do projeto de conclusão de curso do Mestrado em Media Digitais "
                "Interativos",
        )

        # A chamada isolada do arquivo dcgan_keras3.py faz unica e exclusivamente o treinamento
        self.parametros.add_argument(
            "diretorio_dataset",
            type=str,
            help="Indicar o diretório das imagens já redimensionadas e tratadas para o dataset.",
        )
        self.parametros.add_argument(
            "diretorio_resultado",
            type=str,
            help="Indicar o diretório onde serão salvas as imagens geradas pelas épocas e o log do treinamento.",
        )
        self.parametros.add_argument(
            "-epocas",
            required=False,
            type=int,
            default=100,
            help="Indicar a quantidade de épocas de treinameto, parâmetro opcional e por default utliza 100 épocas.",
        )


    def recuperar_parametros(self):
        return self.parametros.parse_args()