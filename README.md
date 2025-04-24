# Gerador de Azulejos
## Inteligência Artificial para a Reprodução da Estética Patrimonial Portuguesa

### Índice

- [ABSTRACT](#abstract)
- [INTRODUÇÃO](#introdução)
- [DESIGN DE SUPERFÍCIE](#design-de-superfície)
- [AZULEJO, PORTUGAL, ARTE E CULTURA](#azulejo-portugal-arte-e-cultura)
- [INTELIGÊNCIA ARTIFICIAL](#inteligência-artificial)
- [GENERATIVE ADVERSARIAL NETWORKS (GANs)](#generative-adversarial-networks-gans)
- [METODOLOGIA](#metodologia)
- [CONCLUSÃO](#conclusão)
- [REFERÊNCIAS](#referências)

---

### ABSTRACT

Este trabalho tem por objetivo a criação de um software gerador de imagens. O output esperado por esta aplicação será um módulo inspirado nos padrões da azulejaria portuguesa. Para isso, será criado um conjunto de dados a partir de imagens das fachadas dos edifícios das cidades de Portugal. Esta informação será utilizada para treinar um algoritmo de inteligência artificial. Para chegar ao resultado este projeto passa por uma explicação sobre o design de superfície, conceituando-o. Descreve a azulejaria portuguesa como fator de identidade do povo português e representante cultural de identidade, bem como seus padrões de estrutura e forma de arte. Por fim, faz um breve estado da arte sobre inteligência artificial e a arquitetura de processamento de imagem que será utilizada para o desenvolvimento.

---

### INTRODUÇÃO

Desde a disponibilização do ChatGPT ao público em geral, em novembro de 2022 (Teixeira, 2023), o tema Inteligência Artificial (IA) está em voga, principalmente quando se considera o temor relativo ao seu potencial negativo (Arntz et al., 2017). Porém, é fato que as IA's vieram para ficar (Howley, 2023) e, assim como a prensa de Gutenberg, esta tecnologia tem a capacidade de revolucionar o desenvolvimento da humanidade.

Apesar do exposto, a utilização de inteligência artificial tem crescido em diversas áreas e as possibilidades são infinitas, inclusive nas áreas artísticas. Desponta como recurso de grande impacto para artistas, designers e produtores de mídias digitais.

Neste contexto, este trabalho se baseia e pretende experimentar. Visa-se construir uma aplicação, treinada com arquiteturas de IA existentes, para criar módulos de azulejos com inspiração na estética patrimonial e cultural portuguesa.

---

### DESIGN DE SUPERFÍCIE

Para delimitação do âmbito deste projeto, é necessário definir o conceito de “módulo” utilizado em Design de Superfície (DS). Esta concepção demarcará o comportamento da aplicação a ser desenvolvida.

No DS, o módulo é a parte mais básica de um todo. “É a unidade básica de medida para a coordenação dimensional dos componentes e das partes da construção” (Andrade, 2005). É a unidade do desenho que, replicado e organizado, gera um padrão, e este, por sua vez, gera uma nova textura (Schwartz, 2008).

---

### AZULEJO, PORTUGAL, ARTE E CULTURA

Segundo Mateus Miguel de Souza (2019), o azulejo refere-se ao ladrilho cerâmico com uma das faces decorada com esmaltes, destinado à ornamentação de superfícies.

Susana Nunes (2014) destaca que, em Portugal, o azulejo ultrapassou a função utilitária e assumiu o estatuto de arte, sendo componente essencial da arquitetura desde o século XV.

A criação de azulejos em Portugal divide-se em dois tipos: pintura sobre cerâmica e azulejo de padrão. Este projeto considera apenas o segundo tipo — os que permitem composição em padrões.

---

### INTELIGÊNCIA ARTIFICIAL

O conceito moderno de IA remonta à década de 1950, com Alan Turing e seu artigo “Computing Machinery and Intelligence” (Turing, 1950). O primeiro grande trabalho prático foi de McCulloch e Pitts (1943), que propuseram o modelo de neurônio artificial baseado em estímulos biológicos.

As redes neurais artificiais (RNAs) possuem arquitetura e algoritmo de aprendizagem, permitindo adaptações dinâmicas no processo decisório (Rauber, 2005).

---

### GENERATIVE ADVERSARIAL NETWORKS (GANs)

Este trabalho utiliza GANs, compostas por duas redes: a geradora (G) e a discriminadora (D). A G tenta criar imagens realistas, enquanto a D tenta distinguir imagens reais das geradas. Essa competição resulta em redes cada vez mais eficientes (Machado, 2018).

Há diversas arquiteturas de GANs (DCGAN, WGAN, BigGAN). O projeto buscará a mais adequada para gerar módulos de azulejos (Iglesias et al., 2023).

---

### METODOLOGIA

A figura abaixo representa as etapas do projeto:

![Figura 1 - Etapas do trabalho](caminho/para/figura1.jpg)

1. Coleta de fotografias de fachadas, principalmente na cidade do Porto.
2. Formatação das imagens para extrair apenas os módulos das superfícies.
3. Preparação do dataset.
4. Treinamento das redes neurais (geradora e discriminadora) utilizando Python e TensorFlow.
5. Disponibilização do gerador de azulejos via API REST.

---

### CONCLUSÃO

Este trabalho entrega à comunidade acadêmica um software gerador e um dataset de azulejos. Explora a arte generativa com redes neurais, propondo futuras aplicações que permitam a criação automatizada de superfícies completas por IA.

---

### REFERÊNCIAS

- Abadi et al. (2015). *TensorFlow: Large-Scale Machine Learning...*
- Andrade, M. (2005). *A representação gráfica de projetos modulares...*
- Arntz et al. (2017). *Revisiting the risk of automation...*
- Gomes, D. dos S. (2010). *Inteligência Artificial: Conceitos e Aplicações...*
- Howley, D. (2023). *There’s no going back on A.I...*
- Iglesias et al. (2023). *A survey on GANs...*
- Machado, D. L. (2018). *Construção de modelos neurais...*
- Nicolelis, M. (2023). *Inteligência artificial: tudo que você precisa saber...*
- Nunes, S. (2014). *Azulejos de Padrão e Relevo...*
- Rauber, T. W. (2005). *Redes Neurais Artificiais...*
- Schwartz, A. R. D. (2008). *Design de superfície...*
- Souza, M. M. (2019). *Azulejaria portuguesa...*
- Teixeira, P. M. (2023). *O ChatGPT e os desafios às universidades...*
- Turing, A. M. (1950). *Computing Machinery and Intelligence...*

---

Instituto Politécnico do Porto  
Escola Superior de Artes Media e Design  
Mestrado em Sistemas e Media Interativos  
Seminários em Ambientes Multimédia  
**Autor:** Gustavo Ventorim Glória Leal  
**Orientador:** Professor Jorge Lima
