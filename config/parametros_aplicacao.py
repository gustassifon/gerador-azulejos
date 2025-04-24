import argparse

class ParametrosAplicacao(argparse.ArgumentParser):
    def __init__(self):
        super(ParametrosAplicacao, self).__init__(
            prog="Gerador Azulejos: aplicação criada como parte do projeto de conclusão de curso do Mestrado "
        )