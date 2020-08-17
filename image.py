import keras
from keras.datasets import cifar10
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout, GlobalAveragePooling2D
from keras.models import Sequential
from keras import regularizers
from keras.optimizers import Adam

(x_trains,y_trains),(x_test,y_test)=cifar10.load_data() 

y_train_hot=to_categorical(y_trains)
t_test_hot=to_categorical(y_test)

img=plt.imshow(x_trains[0])

x_trains=x_trains/255
x_test=x_test/255 
#Normalize values to be between 0 and 1

model=Sequential()

model.add(Conv2D(32,(3,3),activation='elu',input_shape=(32,32,3),padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),activation='elu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#model.add(Conv2D(64,(3,3),activation='elu',padding='same'))
#model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),activation='elu',padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),activation='elu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(128,(3,3),activation='elu',padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),activation='elu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

#model.add(Conv2D(256,(3,3),activation='elu',padding='same'))
#model.add(BatchNormalization())
#model.add(Conv2D(256,(3,3),activation='elu',padding='same'))
#model.add(BatchNormalization())
#model.add(Conv2D(256,(3,3),activation='elu',padding='same'))
#model.add(BatchNormalization())
#model.add(MaxPooling2D(pool_size=(2,2)))

#model.add(GlobalAveragePooling2D())
#model.add(Dense(512,activation='elu'))
#model.add(BatchNormalization())
#model.add(Dropout(0.25))
#model.add(Dense(512,activation='elu'))
#model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=3e-4), metrics=['accuracy'])

# Validation = 30 % , inorder to compare the resulting inputs with the 30% validation set inorder to improve the validation on subsequent epochs
hist = model.fit(x_trains, y_train_hot, epochs=15, validation_split=0.3, batch_size=32) 
#print('\nTest result: %.3f loss: %.3f' % (hist[1]*100,hist[0]))
#accuracy

#plt.plot(hist.history['acc'])
#plt.plot(hist.history['val_acc'])
#plt.title('Model Accuracy')
#plt.ylabel('Accuracy')
#plt.xlabel('Epochs')
#plt.legend(['Train','Validation'],loc='upper left')
#plt.show()

#plt.plot(hist.history['loss'])
#plt.plot(hist.history['val_loss'])
#plt.title('Model Loss')
#plt.ylabel('Loss')
#plt.xlabel('Epochs')
#plt.legend(['Train','Validation'],loc='upper left')
#plt.show()
model.save('cifar_model.h5')
model.save_weights('cifar_model_weights.h5')

