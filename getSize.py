
from PIL import Image

path = 'original_data_after_augmentation\\original\\happy\\img_0_afraid_asian_103_0_2303.jpeg'

img = Image.open(path)
imgSize = img.size

imgSize = img.size  #大小/尺寸
w = img.width       #图片的宽
h = img.height      #图片的高
f = img.format      #图像格式
 
print(imgSize)
print(w, h, f)