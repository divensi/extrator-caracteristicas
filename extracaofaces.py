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

from get_caracteristicas import get_caracteristicas

def main():
  directory = "/Users/divensi/Dataset/lfwcrop_grey/faces"
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
    'lbp1',
    'lbp2',
    'lbp3',
    'lbp4',
    'lbp5',
    'lbp6',
    'lbp7',
    'lbp8',
    'lbp9',
    'lbp10',
    'lbp11',
    'lbp12',
    'lbp13',
    'lbp14',
    'lbp15',
    'lbp16',
    'lbp17',
    'lbp18',
    'lbp19',
    'lbp20',
    'lbp21',
    'lbp22',
    'lbp23',
    'lbp24',
    'lbp25',
    'lbp26',
    'lbp27',
    'lbp28',
    'lbp29',
    'lbp30',
    'lbp31',
    'lbp32',
    'lbp33',
    'lbp34',
    'lbp35',
    'lbp36',
    'lbp37',
    'lbp38',
    'lbp39',
    'lbp40',
    'lbp41',
    'lbp42',
    'lbp43',
    'lbp44',
    'lbp45',
    'lbp46',
    'lbp47',
    'lbp48',
    'lbp49',
    'lbp50',
    'lbp51',
    'lbp52',
    'lbp53',
    'lbp54',
    'lbp55',
    'lbp56',
    'lbp57',
    'lbp58',
    'lbp59',
    'classe'
  ]

  dataset = []

  dataset.append(header)
  os.chdir(directory)
  x = glob.glob("*.pgm")

  # selecionar apenas classes que possuem mais de 10 instâncias
  classes = {}
  for instancia in x:
    try:
      classes[instancia[:-9]] += 1
    except:
      classes[instancia[:-9]] = 1
  
  classes_selecionadas = []

  for classe, valor in classes.items():
    if (valor >= 60):
      classes_selecionadas.append(classe)

  # gerar as medidas
  for key, filename in enumerate(glob.glob("*.pgm")):
    if filename[:-9] in classes_selecionadas:
      image = cv2.imread(filename, -1)

      descritores = get_caracteristicas(image)
      #adiciona a clase aos descritores
      descritores.append(filename[:-9])

      dataset.append(descritores)

  np.savetxt('/Users/divensi/Projects/coisas-de-imagens/extracao-caracteristicas/dataset-gray.60.csv', dataset, delimiter=",", fmt='%s')

if __name__ == '__main__':
  main()
