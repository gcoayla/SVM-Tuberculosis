import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skimage import exposure

Categories=['normal','tuber']

#Cargamos los datos de entrenamiento
flat_data_arr=[] #input array
target_arr=[] #output array

datadir='/content/drive/MyDrive/radio/small'

def carga():
  for i in Categories:
      print(f'loading... category : {i}')

      path=os.path.join(datadir,i)

      for img in os.listdir(path):
          print(f'loading:{img}')
          #cargamos la imagen y convertimos a escala de grises
          img_array=imread(os.path.join(path,img),as_gray=True)
          #aplicamos resize a la imagen
          img_array=resize(img_array,(512,512,3))
          #aplicamos constrast stretching a la imagen
          p2, p98 = np.percentile(img_array, (2, 98))
          img_rescale = exposure.rescale_intensity(img_array, in_range=(p2, p98))
          flat_data_arr.append(img_rescale.flatten())
          target_arr.append(Categories.index(i))
      print(f'loaded category:{i} successfully')
  flat_data=np.array(flat_data_arr)
  target=np.array(target_arr)
  df=pd.DataFrame(flat_data)
  df['Target']=target
  x=df.iloc[:,:-1] #input data 
  y=df.iloc[:,-1] #output data
  x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=77,stratify=y)
  return x_train,x_test,y_train,y_test

x_train,x_test,y_train,y_test = carga()
print('Splitted Successfully')


model = svm.SVC()

#Se realiza el entrenamiento del SVM
model.fit(x_train,y_train)

print('The Model is trained well with the given images')


#Prediccion con una cantidad pequeña de tests
y_pred=model.predict(x_test)
print("The predicted Data is :")
print(y_pred)
print("The actual data is:")
print(np.array(y_test))
print(f"The model is {accuracy_score(y_pred,y_test)*100}% accurate")

flat_data_arr_res=[] #input array
target_arr_res=[] #output array

#cargamos la data de testing

datadir2='/content/drive/MyDrive/radio/small/test'

def carga2():
  for i in Categories:
      print(f'loading... category : {i}')

      path=os.path.join(datadir2,i)

      for img in os.listdir(path):
          print(f'loading:{img}')
          img_array=imread(os.path.join(path,img),as_gray=True)
          img_array=resize(img_array,(512,512,3))
          p2, p98 = np.percentile(img_array, (2, 98))
          img_rescale = exposure.rescale_intensity(img_array, in_range=(p2, p98))
          flat_data_arr_res.append(img_rescale.flatten())
          target_arr_res.append(Categories.index(i))
      print(f'loaded category:{i} successfully')
  flat_data=np.array(flat_data_arr_res)
  target=np.array(target_arr_res)
  df=pd.DataFrame(flat_data)
  df['Target']=target
  x=df.iloc[:,:-1] #input data 
  y=df.iloc[:,-1] #output data
  x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.01,random_state=77,stratify=y)
  return x_train,x_test,y_train,y_test

x_train2,x_test2,y_train2,y_test2 = carga2()
print("Predicting :")
y_pred2=model.predict(x_train2)
print("The predicted Data is :")
print(y_pred2)
print("The actual data is:")
print(np.array(y_train2))
print(f"The model is {accuracy_score(y_pred2,y_train2)*100}% accurate")


from sklearn import metrics
#imprimimos los resultados de la ultima clasificacion
print(f"Classification report for classifier {model}:\n"
      f"{metrics.classification_report(y_train2, y_pred2)}\n")


disp = metrics.plot_confusion_matrix(model, x_train2, y_train2)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()