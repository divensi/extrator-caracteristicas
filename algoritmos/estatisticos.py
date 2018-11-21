import numpy as np
from scipy import stats

def desvio_padrao(image):
  return np.array(image).std()

def mediana(image):
  return np.median(np.array(image))

def curtose(image):
  return stats.kurtosis(np.array(image.reshape(-1)))

def variancia(image):
  return stats.variation(np.array(image.reshape(-1)))

def moda(image):
  return stats.mode(np.array(image.reshape(-1)))