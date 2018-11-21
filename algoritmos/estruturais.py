import cv2
import glob, os, sys
import numpy as np
import mahotas
from skimage import feature
from matplotlib import pyplot as plt

def area(image):
  (_, contornos, _) = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  area = cv2.contourArea(contornos[0])
  
  return area

def perimetro(image):
  (_, contornos, _) = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  perimetro = cv2.arcLength(contornos[0], True)
  
  return perimetro

