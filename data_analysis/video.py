from cv2 import cv2
import numpy as np

pathin = '/media/kun/Seagate Expansion Drive/data/data/'
pathout = '/media/kun/Seagate Expansion Drive/data/data/bz.avi'
fps = 10

frame_array = []
for count in range(0, 40001, 500):
    print(count)
    filename = pathin + 'bz' + str(count) + '.png'
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    frame_array.append(img)

out = cv2.VideoWriter(pathout, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

for i in range(len(frame_array)):
    out.write(frame_array[i])
out.release()
