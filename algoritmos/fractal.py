
import cv2
import glob, os, sys
import numpy as np
import mahotas
from skimage import feature
from matplotlib import pyplot as plt


def fractal_dimension(Z, threshold=0.9):

  # Only for 2d image
  assert(len(Z.shape) == 2)

  # From https://github.com/rougier/numpy-100 (#87)
  def boxcount(Z, k):
    S = np.add.reduceat(
      np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
      np.arange(0, Z.shape[1], k), axis=1)

    # We count non-empty (0) and non-full boxes (k*k)
    return len(np.where((S > 0) & (S < k*k))[0])


  # Transform Z into a binary array
  Z = (Z < threshold)

  # Minimal dimension of image
  p = min(Z.shape)

  # Greatest power of 2 less than or equal to p
  n = 2**np.floor(np.log(p)/np.log(2))

  # Extract the exponent
  n = int(np.log(n)/np.log(2))

  # Build successive box sizes (from 2**n down to 2**1)
  sizes = 2**np.arange(n, 1, -1)

  # Actual box counting with decreasing size
  counts = []
  for size in sizes:
    counts.append(boxcount(Z, size))

  # Fit the successive log(sizes) with log (counts)
  coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
  return -coeffs[0]


def fractal_lacunaridade(theta, image):
  imagewidth = image.shape[1]
  imageheight = image.shape[0]
  roiwidth = int(imagewidth/theta)
  roiheight = int(imageheight/theta)

  # print ("lacunalidade iniciada para a imagem {}x{} e roi de tamanho: {}x{}"
  #   .format(imagewidth, imageheight, roiwidth, roiheight))

  eqlacunaup = 0
  eqlacunabottom = 0

  line = 0
  while (line * roiheight < imageheight):

    col = 0
    while (col * roiwidth < imagewidth):
      #faz o roi da imagem
      roi = image[int(line*roiheight):int((line+1)*roiheight), int(col*roiwidth):int((col+1)*roiwidth)]

      minimo = maximo = -1

      x = 0
      while (x < roiwidth):
      
        y = 0
        while (y < roiheight):
          try:
            if (maximo == -1):
              maximo = minimo = roi[x][y]
            else:
              value = roi[x][y]
              
              if maximo < value:
                maximo = value

              if minimo > value:
                minimo = value
          except IndexError:
            pass

          y += 1
        x += 1

      altura = (maximo/roiheight) - (minimo/roiheight) + 1

      eqlacunaup += (altura * (altura/theta))
      eqlacunabottom += (altura/theta)

      col += 1
    line += 1

  eqlacunabottom *= eqlacunabottom
  result = eqlacunaup/eqlacunabottom
  
  # print("lacunalirade terminada com resultado: {}".format(result))
  
  return result

def calcsucolaridade(imagemOg):
  '''
  recebe uma imagem binaria e retorna os 4 valores:
  1 - sucolaridade de cima para baixo;
  2 - sucolaridade da direita para a esquerda;
  3 - sucolaridade de baixo para cima;
  4 - sucolaridade da esquerda para a direita;
  '''

  image = cv2.cvtColor(imagemOg, cv2.COLOR_GRAY2BGR)# convert to 24 bits 

  # copia imagens
  # cima para baixo
  i1 = image.copy()
  # direita para esquerda
  i2 = np.rot90(i1.copy())
  # baixo para cima
  i3 = np.rot90(i2.copy())
  # esquerda para direita
  i4 = np.rot90(i3.copy())

  imagens = [i1, i2, i3, i4]

  rows = image.shape[0]-1
  vermelho = [255, 0, 0]
  fator = [255, 255, 255]

  pressoes = []

  for imagem in imagens:
    # plt.imshow(imagem)
    # plt.show()

    for col in range(imagem.shape[1]):


      ## verifica se os dois sets sao iguais
      if set(imagem[0][col]) == set(fator):
        # por algum motivo, se nao for feita uma copia da uma execao
        imagem = imagem.copy()
        cv2.floodFill(imagem, None, (col, 0), vermelho)

    # substitui a cor branca por cor preta (deixando apenas o flood fill)
    imagem[np.where((imagem==[255, 255, 255]).all(axis=2))] = [0, 0, 0]

    pressoes.append(calcular_pressao(imagem))

  return pressoes

def calcular_pressao(image, pressao=3):
  """ Calcula a pressão da imagem
  """
  roi = [int(image.shape[0]/pressao), int(image.shape[1]/pressao)]
  resolucao_roi = roi[0] * roi[1]
  delta_ponto_central = roi[0]/2

  ocupacao_total = 0
  ocupacao_maxima = 0

  #basicamente dividir a imagem em quadros 3X3 ai calcular a ocupacao deles e divida-la peal ocupacao maxima 
  #soma a porcentagem de ocupacao do quadrante 3*3 ex: for do roi da imagem é ir em cada caixinha dois for linha e coluna
  for linroi in range(roi[0]):
    for colroi in range(roi[1]):
      ocupacao = 0

      #calcula ocupacao do roi
      for lin in range(linroi * pressao, (linroi + 1) * pressao):
        for col in range(colroi * pressao, (colroi + 1) * pressao):
          if set(image[lin][col]) == set([255, 0, 0]):
            #acumula ocupacao do roi
            ocupacao += 1
      # rows numero de colunas
      pontocentral = (linroi * roi[0]) + delta_ponto_central

      porc_ocupacao = ocupacao/resolucao_roi

      #ocupacao total+=ocupacao do roi*pontocentral 
      ocupacao_total += (porc_ocupacao * pontocentral)
      ocupacao_maxima += 1 * pontocentral
 
  resultado = ocupacao_total/ocupacao_maxima

  return resultado

