import os
import re
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from PIL import Image

def rename_file(path, label):
  i = 0

  for filename in os.listdir(path):
    try:
      f, extension = os.path.splitext(filename)
      src = path + filename
      dst = path + label + str(i) + extension

      if not os.path.exists(dst):
        os.rename(src, dst)
      else:
        print(f'File {dst} already exists, skipping.')

      i += 1

    except Exception as e:
      print(e)
      i += 1
path = "D:\FYP\dataset\dirty\\"
label = "dusty"
# print(os.path.splitext(path+ "20210917_151202.jpg"))
rename_file(path,label)
