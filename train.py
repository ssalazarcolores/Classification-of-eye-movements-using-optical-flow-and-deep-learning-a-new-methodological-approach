
#LIBRARIES
from utils import get_data_from_fold, edit_trainable_layers
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.applications import efficientnet
from tensorflow.compat.v1 import ConfigProto
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import Dense
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
from tensorflow.keras import layers
from tqdm import tqdm
import pandas as pd


warnings.filterwarnings('ignore')

#PARAMETERS
WEIGHTS='imagenet'
IMAGE_HEIGHT = IMAGE_WIDTH = 224
NUM_CHANNELS = 3
BATCH_SIZE = 192
EPOCHS = 45
SPLITS_KFOLD = 5
LEARNING_RATE= 5e-3
OPTIMIZER = tf.keras.optimizers.Adam(LEARNING_RATE)
LOSS = tf.keras.losses.CategoricalCrossentropy()

METRICS=['accuracy', 'AUC', 'Precision', 'Recall']
ds=  tf.data.experimental.load('/home/rtx3090/Desktop/attention_dataset_ds')
ds_lund= tf.data.experimental.load('/home/rtx3090/Desktop/Alea/lund2013_fo/test_ds')
#PARAMETERS TRAINNING CONFIG
with_fp16 = False
with_numpy = False
enable_cache = False
enable_XLA = True

#CALLBACKS
valores=[]
stop_early = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=15)
reduce_lr= tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',factor=0.1,patience=15)



def get_model():
  i = tf.keras.layers.Input([IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS], dtype = tf.uint8)
  x = tf.cast(i, tf.float32)
  x = tf.keras.applications.efficientnet.preprocess_input(x)  
  modelo = efficientnet.EfficientNetB0(weights='imagenet', input_shape=(IMAGE_HEIGHT,IMAGE_WIDTH,NUM_CHANNELS), classifier_activation="softmax", include_top = True)

  y=tf.keras.layers.Dense(2,activation='softmax',name='prediccion')(modelo.layers[-2].output)
  model=tf.keras.Model(inputs=modelo.input, outputs=[y])
  model = edit_trainable_layers(model, layers  = 'conv_stage')
  x= model(x)
  model= tf.keras.Model(inputs=[i], outputs=[x])


  data_augmentation= tf.keras.Sequential([
      tf.keras.layers.RandomFlip('horizontal'),
      tf.keras.layers.RandomRotation(0.2),])

  inputs = tf.keras.Input(shape=(224,224,3))
  x = data_augmentation(inputs)
  x = model(x, training=True)
  model = tf.keras.Model(inputs, x)


  return model


def train():
    
    testing_gazecom=[]
    times_history=[]
    best_score_history=[]

    
    for i in tqdm(range(SPLITS_KFOLD)):
        
   
        LEARNING_RATE= 5e-3
        OPTIMIZER = tf.keras.optimizers.Adam(LEARNING_RATE)
        LOSS = tf.keras.losses.CategoricalCrossentropy()
        checkpoint= tf.keras.callbacks.ModelCheckpoint('/home/rtx3090/Desktop/Alea/models/'
                                                       + 'efficient_trial__fold_'+str(i) +'.hdf5')
        #Training
        print("\r**********             FOLD", i +1,"             **********\n")
        ds_train, ds_val = get_data_from_fold(SPLITS_KFOLD, i, ds)
        path = os.path.join('/home/rtx3090/Desktop/Alea/ds_val_trial_1', "ds_val_"+str(i+1))
        tf.data.experimental.save(ds_val, path)
        print('ds_val saved correctly!')
        
        print('\ntrain:',ds_train.cardinality().numpy(), 'val:', ds_val.cardinality().numpy())
        model = get_model()
        
        if enable_cache == True:
            ds_train = ds_train.shuffle(100).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE).cache()
            ds_val = ds_val.shuffle(100).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE).cache()
        else:
            ds_train = ds_train.shuffle(100).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
            ds_val = ds_val.shuffle(100).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE) 
            
        print("\r**********             DS content ready               **********\n")


        if with_fp16 == True:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            print("gpus",gpus)
            tf.keras.backend.clear_session()
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            print("\n----------           Running with mixed precision             ----------\n")
        tf.config.optimizer.set_jit(enable_XLA)

        tic_complete = time.time()
        
        
        # Load model plot grphviz library
        os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
        
        
       
        if with_fp16 ==True:
            model.compile(optimizer = tf.keras.mixed_precision.LossScaleOptimizer(OPTIMIZER), loss = LOSS, metrics = METRICS)
        else:
            model.compile(optimizer=OPTIMIZER, loss = LOSS, metrics = METRICS)

        print("\n----------                 Training                   ----------\n")



        tic_train =  time.time()
        csv_logger=tf.keras.callbacks.CSVLogger('log_efficientnet_trial_1.csv', separator=',', append=True)
        historia = model.fit(ds_train, validation_data = ds_val, epochs = EPOCHS, batch_size = BATCH_SIZE, callbacks = [stop_early, reduce_lr,csv_logger, checkpoint])
        toc_train = time.time()
        time_training = toc_train - tic_train 
        
        print("\n**********             Training complete               **********")
        scores_gazecom = model.evaluate(ds_val, batch_size=BATCH_SIZE)
        testing_gazecom.append(scores_gazecom)
        
        
        print('Testing GazeCom:',scores_gazecom)
      

        experiment_path='/home/rtx3090/Desktop/Alea'
        
        
        
        toc_complete = time.time()
        time_complete = toc_complete - tic_complete
        times_history.append(time_complete)
        time_training, time_complete

                
        best_score = np.max(historia.history['val_accuracy'])
        best_score_i = np.argmax(historia.history['val_accuracy'])

        print('The best accuracy score in validation was ', best_score )
        best_score_history.append(best_score)
        try:    
            tf.keras.clear_session()
            del model, X_test, Y_test, X_train, y_train, X_val, y_val,  
        except:
            try:
                del ds_train, ds_val
            except: pass
        
        print('Testing GazeCom:')
        print(testing_gazecom)
        



def get_X_and_Y(ds):
    Xl, Yl = list(), list()
    for image, label in tqdm(ds):
      Yl.append(label.numpy()), Xl.append(image.numpy())
    Y, X = np.array(Yl), np.array(Xl)    
    print('ready')
    return X, Y

if __name__ == '__main__':
    print("\n**********             EfficientNetB0                **********")

    train()
