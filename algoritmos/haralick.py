import cv2
import glob, os, sys
import numpy as np
import mahotas
from skimage import feature
from matplotlib import pyplot as plt

def haralick(image):
  # convert the image to grayscale
  # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # compute the haralick texture feature vector
  haralick = mahotas.features.haralick(image).mean(axis=0)
  # return the result
  return haralick
