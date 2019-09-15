from PIL import Image,ImageStat
import glob
from keras.models import Sequential
from keras.layers import Dense,LSTM
images=[Image.open(file).convert('RGB') for file in glob.glob('F:\ML stuff\pyscripts\Dataset\Train\grey\*.jpg')]
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.layers import Dropout
from keras import optimizers
from keras import backend as K
x=[]
y=[]
'''
for im in images:
    
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
            
    '''
'''
    the current criterion for the image segregation is
    hardcoded as the gpu was unavailable for procesing ml on number of images.
    This works for normal images as well as challenging color images 
    '''
    #fact=((diff/(h*w)))
'''
    print(fact)
    if fact<1:
        print('image is grayscale')
    else:
        print('image is color')'''
'''
    x.append(fact)
    y.append(0)
##---------------------------------------------------------------------------------------------------------------------------------------------------------------##
images=[Image.open(file).convert('RGB') for file in glob.glob('F:\ML stuff\pyscripts\Dataset\Train\color\*.jpg')]
for im in images:
    
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
            
'''
    
   # the current criterion for the image segregation is
    #hardcoded as the gpu was unavailable for procesing ml on number of images.
    #This works for normal images as well as challenging color images 
'''
    
    fact=((diff/(h*w)))
    #print(fact)
    #if fact<1:
     #   print('image is grayscale')
    #else:
        #print('image is color')'''
    #x.append(fact)
    #y.append(1)
#print(x)
#print(y)

#np.savetxt("springer.csv", [x,y], delimiter=",")


names=['par','label']
df = pd.read_csv('springer2.csv',names=names)
array = df.values
#array=array.reshape(-1, 1)
#print(array)
X=array[:,0]
X=X.reshape(-1,1)
Y= array[:,1]
#Y=Y.reshape(-1,1)
#print(Y)
X_train,X_test,y_train,y_test=model_selection.train_test_split(X,Y,test_size=0.2,random_state=7)
#clf=LogisticRegression()
#clf.fit(X_train,y_train)
#accuracy=clf.score(X_test,y_test)
#print(accuracy)
labelencoder_y_1 = LabelEncoder()
y = labelencoder_y_1.fit_transform(Y)
#print(y)
model= Sequential()
model.add(Dense(8,input_dim=1,activation='relu'))
#model.add(Dense(9,activation='relu'))#optional
model.add(Dense(10,activation='relu'))
model.add(Dense(7,activation='relu'))
model.add(Dense(6,activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(20,activation='relu'))#16,tanh
#model.add(Dense(13,activation='relu'))#optional
model.add(Dense(10,activation='relu'))#optional
model.add(Dense(4,activation='relu'))
#model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='hard_sigmoid'))
#Adam=optimizers.Adam(lr=0.0001, beta_1=0.99, beta_2=0.9999, epsilon=K.epsilon(), decay=0.0, amsgrad=False)
model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])
model.fit(X,y,epochs=500,batch_size=10)
score= model.evaluate(X,y)
##print(model.metrics_name[1])
print(score[1]*100)
'''
p=[]
images=[Image.open(file).convert('RGB') for file in glob.glob('F:\ML stuff\pyscripts\Dataset\Test\color\*.jpg')]

for im in images:
    
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
    
    fact=((diff/(h*w)))
    p.append(fact)
pred=model.predict(np.array(p))
print(pred)






    

'''
