from PIL import Image, ImageTk
import os

def max_pool(img_file):
    if not os.path.isfile(img_file):
        print(f'Could not find file \"{img_file}\"')
        return
    img = Image.open(img_file)
    w = img.width
    h = img.height
    img = img.load()
    pixels = [0 for i in range(h*w//4)]
    for i in range(0, w, 2):
        for j in range(0, h, 2):
            m = max(img[i, j], img[i+1, j], img[i, j+1], img[i+1, j+1])
            pixels[(j//2)*(w//2) + (i//2)] = m
    img2 = Image.new('L', (w//2, h//2))
    img2.putdata(pixels)
    img2.save('mp' + img_file)

os.chdir('src/custom_images')

max_pool('5.jpg')
max_pool('mp5.jpg')
max_pool('6.jpg')
max_pool('mp6.jpg')
max_pool('2.jpg')
max_pool('mp2.jpg')
max_pool('3.jpg')
max_pool('mp3.jpg')
