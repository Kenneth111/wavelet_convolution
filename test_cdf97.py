import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import color
from cdf97 import cdf97_2d, inverse_cdf97_2d

def plt_gray(img):
    plt.imshow(img, cmap="gray")
    plt.show()    

img = Image.open("d:\\desktop.png")
img_arr = np.array(img)
# only the R channel is used
r = img_arr[:, :, 0]
img0 = cdf97_2d(r)
img1 = inverse_cdf97_2d(img0)
# compare the difference between the original graph and the reconstructed graph
d = (img1[3:-3, 3:-3] - r)
print(d.min(), d.max())