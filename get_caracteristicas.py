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

def get_caracteristicas(image):
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
  dim = fractal_dimension(image)
  descritores.append(fractal_dimension(image))

  sucolaridades = calcsucolaridade(image)

  descritores.append(sucolaridades[0])
  descritores.append(sucolaridades[1])
  descritores.append(sucolaridades[2])
  descritores.append(sucolaridades[3])

  lbp = feature.local_binary_pattern(image, 8, 2)

  (histograma, _) = np.histogram(lbp.ravel(),
    bins=np.arange(0, 60), range=(0, 59))


  # normalize the histogram
  histograma = histograma.astype("float")
  histograma /= (histograma.sum() + 1e-7)


  descritores.extend(histograma) # desvio padrão histograma

  return descritores
