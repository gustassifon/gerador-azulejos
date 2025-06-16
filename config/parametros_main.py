import argparse

class ParametrosMain():
    def __init__(self):
        self.parametros = argparse.ArgumentParser(
            prog="Gerador de Azulejos",
            description = "Aplicação criada como parte do projeto de conclusão de curso do Mestrado em Media Digitais "
                "Interativos",
        )

        acao = self.parametros.add_subparsers(
            dest="acao",
            help="Indica qual a ação a aplicação deve realizar: REDIMENSIONAR as imagens que compõe o dataset para o "
                "treinamento; TREINAR o modelo propriamente dito e 'subir' uma API REST para usar o modelo treinado a "
                "partir de requisição HTTP."
        )
        acao.required = True

        # Tratamento para a função REDIMENSIONAR
        redimensionar = acao.add_parser(
            name="redimensionar",
            help="Redimensiona um conjunto de imagens a partir de um diretório indicado. Cria imagens a partir da "
                "rotação das imagens em diversos posicionamentos e o espelhamento das mesmas. Indicar também o "
                "diretório de destino."
        )
        redimensionar.set_defaults(acao="redimensionar")
        redimensionar.add_argument(
            "diretorio_origem",
            type=str,
            help="Indicar o diretório das imagens que serão usadas para compor o dataset.",
        )
        redimensionar.add_argument(
            "diretorio_destino",
            type=str,
            help="Indicar o diretório onde as novas imagens criadas para serem usadas pelo modelo serão salvas.",
        )
        redimensionar.add_argument(
            "-tam",
            required=False,
            type=int,
            default=128,
            help="Indicar o tamanho da imagem, por padrão usa 128x128. As imagens são sempre quadradas. Parâmetro "
                "opcional",

        )

        # Tratamento para a função TREINAR
        treinar = acao.add_parser(
            name="treinar",
            help="Treina um modelo do tipo DCGAN a partir de um diretório de imagens, já redimensionadas."
        )
        treinar.set_defaults(acao="treinar")

        treinar.add_argument(
            "diretorio_dataset",
            type=str,
            help="Indicar o diretório das imagens já redimensionadas e tratadas para o dataset.",
        )
        treinar.add_argument(
            "diretorio_resultado",
            type=str,
            help="Indicar o diretório onde serão salvas as imagens geradas pelas épocas e o log do treinamento.",
        )
        treinar.add_argument(
            "-epocas",
            required=False,
            type=int,
            default=100,
            help="Indicar a quantidade de épocas de treinameto, parâmetro opcional e por default utliza 100 épocas.",

        )

        # Tratamento para a função "levantar" API
        api = acao.add_parser(
            name="api",
            help="'Levanta' API REST que usa modelo treinado para gerar uma imagem aleatória."
        )
        api.set_defaults(acao="api")
        api.add_argument(
            "caminho_arquivo_modelo",
            type=str,
            help="Indicar o caminho para o arquivo que contém o modelo da DCGAN/IA treinado.",
        )


    def recuperar_parametros(self):
        return self.parametros.parse_args()