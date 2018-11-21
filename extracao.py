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
  ]

  dataset = []

  dataset.append(header)

  for file in glob.glob("{}/A*.pgm".format(directory)):
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

    # # cv2.imshow("Imagem", image)
    # lbp = feature.local_binary_pattern(image, 8, 2)
    # # cv2.imshow("LBP", lbp)
    # descritores.append(lbp)


    # # print(descritores)
    dataset.append(descritores)

    # ## calcular histograma sobre o lbp.ravel()
    # # cv2.waitKey()

  np.savetxt('dataset.csv', dataset,delimiter=",", fmt='%s')

if __name__ == '__main__':
  main()