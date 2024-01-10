# Projeto de Iniciação Científica
# Descritor baseado em autômato celular para classificação depixels


## Visão Geral
Este repositório abriga o código-fonte e materiais relacionados ao projeto de pesquisa dedicado à análise e classificação de pixels por meio de autômatos celulares. 
O objetivo central é desenvolver uma abordagem eficaz para a análise e descrição de pixels, utilizando autômatos celulares para modelar relações discretas locais e propagar esses processos para discriminar eficientemente imagens como um todo.

## Estrutura do Projeto
A pasta src abriga os códigos do projeto, os quais são responsáveis por gerar e carregar os histogramas de cada imagem, como também testar diferentes combinaçõs dos histogramas no treinamento, com o objetivo de encontrar a melhor opção, e gerar a matriz de confusão para analisar em qual classe o algoritmo está com maior dificuldade.
A pasta data abriga o dataset utilizado (Corel-1k), os logs e as matrizes de confusão gerados em cada combinação de histograma testada 
O código main.py, por onde os outros códigos são chamados.
O arquivo requirements.txt onde as depêndencias do projeto estão listadas.
