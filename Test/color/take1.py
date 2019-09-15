from PIL import Image,ImageStat
import math
##def is_greyscal(img_path='clr_test_1.jpg'):
im=Image.open('10025WHJ6D6_20_20180418T194951778Z.jpg').convert('RGB')
w,h=im.size
diff=0
a=0
for i in range(w):
    for j in range(h):
        r,g,b=im.getpixel((i,j))
        rg = abs(r-g)
        rb=abs(r-b)
        gb=abs(g-b)
        diff=diff+rg+rb+gb
        if r!=g!=b:
            a=a+1
            #print(a)
if a==0:
    print('image is grey')
else:
    print('image is color')

##is_grayscal()
stat=ImageStat.Stat(im)
if sum(stat.sum)/3==stat.sum[0]:
    print('grey')
else:
    print('color')

print((diff/(h*w)))
