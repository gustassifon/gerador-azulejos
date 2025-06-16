Preparando as imagens para o Dataset
=====

Como o GitHub em usa versão gratuita não aceita arquivos grandes, as imagens que
compõe o dataset foram colocadas em servidor separado. Antes de utilizar a aplicação,
realizar o download das imagens através do link abaixo. Utilize o diretório "/imagens"
do projeto para adicionar o arquivo proceda conforme descrito.

````link
https://ln5.sync.com/4.0/dl/e023c7610#mdfpm6y3-72ixrgpw-puze3ren-xhs2igiv
````

Após baixado o arquivo de imagens, descompactar o conteúdo do arquivo em um 
diretório.

````bash
tar -xJf archive_name.tar.xz

# Exemplo
# tar -xJf ims_originais_preparadas.taz.xz
````

Utilizar a pasta descompactada para gerar o dataset através da aplicação:

````bash
python __main__.py redimensionar caminho/para/imagens_originais caminho/para/diretorio/destino_imagens 

# Exemplo:
# python __main__.py redimensionar imagens/originais_preparadas imagens/redimensionadas
````

Pronto, temos a pasta de imagens para treinamento pronta. Volte a raiz do projeto
para instruções para a próxima etapa.