import sys
sys.path.append('../')
import imUtils 
from pathlib import Path
from PIL import Image

img = imUtils.imread(Path('1.CR3'))
img = img[:,:,::-1]
print(img.shape)
img = Image.fromarray(img)
print(img.size)
img.save('test.jpg')