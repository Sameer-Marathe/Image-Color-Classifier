from PIL import Image
import glob

diff=0
images=[Image.open(file).convert('RGB') for file in glob.glob('file/path/images/*.jpg')]
for im in images:
    
    w,h=im.size
    for i in range(w):
        for j in range(h):
            r,g,b=im.getpixel((i,j))
            rg = abs(r-g)
            rb=abs(r-b)
            gb=abs(g-b)
            diff=diff+rg+rb+gb
    fact=((diff/(h*w)))
    print(fact)
    if fact<1:
        print('image is grayscale')
    else:
        print('image is color')
            
    
'''
    the current criterion for the image segregation is
    hardcoded as the gpu was unavailable for procesing ml on number of images.
    This works for normal images as well as challenging color images 
'''
    
