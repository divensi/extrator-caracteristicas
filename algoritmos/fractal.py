
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
