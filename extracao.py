#!/usr/bin/env python3
'''Este Script abre arquivos pgm com o opencv, os 
   exporta para png em pastas ordenadas pelo nome do arq'''
import cv2
import glob, os, sys
import numpy as np
import mahotas
from skimage import feature
from matplotlib import pyplot as plt
from scipy import stats

from algoritmos.haralick import haralick

from algoritmos.fractal import fractal_dimension
from algoritmos.fractal import fractal_lacunaridade
from algoritmos.fractal import calcsucolaridade

from algoritmos.estruturais import area
from algoritmos.estruturais import perimetro

from algoritmos.estatisticos import desvio_padrao
from algoritmos.estatisticos import curtose
from algoritmos.estatisticos import mediana
from algoritmos.estatisticos import variancia


def main():
  directory = "/Users/divensi/Dataset/MAIUSCULAS"
  if len(sys.argv) > 1: # se foi passado por parametro
    directory = sys.argv[1]

  header = [
    'haralick01',
    'haralick02',
    'haralick03',
    'haralick04',
    'haralick05',
    'haralick06',
    'haralick07',
    'haralick08',
    'haralick09',
    'haralick10',
    'haralick11',
    'haralick12',
    'haralick13',
    'area',
    'perimetro',
    'desvio padrao',
    'mediana',
    'curtose',
    'variância',
    'lacunaridade',
    'dimensao fractal',
    'sucolaridade de cima para baixo',
    'sucolaridade da direita para a esquerda',
    'sucolaridade de baixo para cima',
    'sucolaridade da esquerda para a direita',
    'desvio padrão histograma',
    'mediana histograma',
    'curtose histograma',
    'variancia histograma',
  ]

  dataset = []

  dataset.append(header)

  for file in glob.glob("{}/*.pgm".format(directory)):
    image = cv2.imread(file, -1)
    descritores = list(haralick(image))

    #métodos estruturais
    descritores.append(area(image))
    descritores.append(perimetro(image))

    # métodos estatísticos
    descritores.append(desvio_padrao(image)) # desvio padrão
    descritores.append(mediana(image)) # mediana
    descritores.append(curtose(image)) # curtose
    descritores.append(variancia(image)) # variancia

    descritores.append(fractal_lacunaridade(10, image))
    descritores.append(fractal_dimension(image))

    sucolaridades = calcsucolaridade(image)

    descritores.append(sucolaridades[0])
    descritores.append(sucolaridades[1])
    descritores.append(sucolaridades[2])
    descritores.append(sucolaridades[3])

    # # cv2.imshow("Imagem", image)
    lbp = feature.local_binary_pattern(image, 8, 2)


    # cv2.imshow("LBP", lbp)
    # cv2.waitKey()
    # descritores.append(lbp)
    # hist = cv2.calcHist(lbp,[0],None,[256],[0,256])
    # histograma_lbp = lbp.ravel()
    histograma, bins = np.histogram(lbp.ravel(), bins=np.range(0, 60), range=(0, 59))

    descritores.append(histograma)
   
    descritores.append(desvio_padrao(histograma)) # desvio padrão histograma
    descritores.append(mediana(histograma)) # mediana histograma
    descritores.append(curtose(histograma)) # curtose histograma
    descritores.append(variancia(histograma)) # variancia histograma

    # # print(descritores)
    dataset.append(descritores)

  np.savetxt('dataset.csv', dataset,delimiter=",", fmt='%s')

if __name__ == '__main__':
  main()
