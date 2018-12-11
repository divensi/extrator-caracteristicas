import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

image = cv2.imread('color_test.png')


def distanciaEuclididana3D(p1, p2):
  return math.sqrt(
      (p1[0] - p2[0])**2 + 
      (p1[1] - p2[1])**2 + 
      (p1[2] - p2[2])**2)

def calcula_proximidade(centroides, image):
  ''' calcula imagens para cada centroide '''
  
  #reseta imagem pra evitar que um pixel seja atribuido pra duas centroides
  for i in centroides:
    centroides[i]["image"] = np.zeros(image.shape)
  
  
  altura = image.shape[0]
  largura = image.shape[1]

  # calcula imagens para cada centroide
  for x in range(altura):
    for y in range(largura):
      mindist = 255**3
      mais_proximo = None

      for index, centroide in centroides.items():
        # calcula a distancia euclidiana entre a cor da centroide 
        # e a imagem na posicao x, y
        dist = distanciaEuclididana3D(centroide["cor"], image[x, y])
        
        if dist < mindist:
          mindist = dist
          mais_proximo = index
        
      centroides[mais_proximo]["posicoes"].append([x, y])
      centroides[mais_proximo]["image"][x, y] = image[x, y]
  
  return centroides

def recalcula_centroides(centroides):
  '''recalcula a cor da centroide'''
  
  for x in centroides:
    
    soma = [0, 0, 0]
    cont = len(centroides[x]["posicoes"])
    
    for ponto in centroides[x]["posicoes"]:
      soma += centroides[x]["image"][ponto[0], ponto[1]]

    centroides[x]["cor"] = np.array(soma/cont, np.uint8)
  
  return centroides

def kmeans(image, k, iteracoes):
  altura = image.shape[0]
  largura = image.shape[1]
  
  # gera 3 números inteiros randômicos, um número 'k' de vezes 
  centroides = {}
  
  for x in range(k):
    centroides[x] = {
      # cor do centroide
      "cor": np.random.randint(255, size=(3)),
      # posicoes que pertencem a centroide
      "posicoes": [],
      # imagem do K
      "image": np.zeros(image.shape)
    }
  
  for i in range(iteracoes):
    centroides = calcula_proximidade(centroides, image)
    centroides = recalcula_centroides(centroides)
  
  for x in centroides:
    plt.imshow(centroides[x]["image"])
    plt.show()
    
  
#     print(centroides[x]["image"])

kmeans(image, 6, 10)
 