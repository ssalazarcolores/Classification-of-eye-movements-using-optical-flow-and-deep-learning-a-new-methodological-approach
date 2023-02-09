from tqdm import tqdm
import pandas as pd 
import os
import numpy as np
def get_data_from_fold(K, iteration, ds):
    ds_val = ds.shard(num_shards = K, index = iteration)
    ds_train = None
    for j in tqdm(range(K)):
        if iteration != j:
            if ds_train == None:
                ds_train = ds.shard(num_shards = K, index = j)
            else:
                ds_train = ds_train.concatenate(ds.shard(num_shards = K, index = j))
                print(ds_train.cardinality().numpy())
    return ds_train, ds_val

def edit_trainable_layers(model, layers  = 'conv_stage'):
    total_layers, pattern_extraction_layers = count_layers_of_net_stages(model)
    if layers == None:
        print('None layers will be trained...')
        for layer in model.layers:
            layer.trainable = False 
    elif layers == 'conv_stage':
        print('Only top layers will be trained...')
        for layer in model.layers[:pattern_extraction_layers]:
            layer.trainable = False
    elif layers == 'all':
      print('All layers will be trained...')      
    else:
          if (len(model.layers) < layers):
              print('The number of trainable layers defined is bigger than model size, all the layers will be trained...')
          else:
              print('The last ', layers, ' layers will be trained...')
              for layer in model.layers[:(len(model.layers) - layers)]:
                  layer.trainable = False
    # for i, layer in enumerate(model.layers):
    #     print(i, l
    return model
    
def count_layers_of_net_stages(modelo):  
    total_layers = len(modelo.layers)   
    for i, layer in enumerate(modelo.layers):
        if (layer.name ==  'flatten'):# or (layer.name ==  'prediction'):
            pattern_extraction_layers = i
        else:
            pattern_extraction_layers = 3
    return total_layers, pattern_extraction_layers

def plot_training_history(experiment_path, training_history_file,fold, dpi = 1200):
    """Plot training"""
    
    print("\n\n----------         Plotting training metrics         ----------", end = "")
    
    df = pd.read_csv(os.path.join(experiment_path, training_history_file))
    
    plt.plot(df["accuracy"])
    plt.plot(df['val_accuracy'])
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Training","Validation"])
    plt.grid(color='r', linestyle='dotted', linewidth=1)
    plt.savefig(os.path.join(experiment_path, 'fold_'+ fold +"training_acc_plot_english_" + str(df["val_accuracy"].max()) + ".png"), dpi = dpi)
    plt.show()
     
    plt.plot(df['loss'])
    plt.plot(df['val_loss'])
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Training","Validation"])
    plt.grid(color='r', linestyle='dotted', linewidth=1)
    plt.savefig(os.path.join(experiment_path, 'fold_'+ fold + "training_loss_plot_english.png"), dpi = dpi)
    plt.show()   
    
    plt.plot(df["accuracy"])
    plt.plot(df['val_accuracy'])
    plt.ylabel("Exactitud")
    plt.xlabel("Época")
    plt.legend(["Entrenamiento","Validación"])
    plt.grid(color='r', linestyle='dotted', linewidth=1)
    plt.savefig(os.path.join(experiment_path, 'fold_'+ fold+"training_acc_plot_español.png"), dpi = dpi)
    plt.show() 
    
    plt.plot(df['loss'])
    plt.plot(df['val_loss'])
    plt.ylabel("Pérdida")
    plt.xlabel("Época")
    plt.legend(["Entrenamiento","Validación"])
    plt.grid(color='r', linestyle='dotted', linewidth=1)
    plt.savefig(os.path.join(experiment_path, 'fold_'+ fold+ "training_loss_plot_español.png"), dpi = dpi)
    plt.show()
    
    print("\r**********           Training plots ready             **********\n\n")
    
    def create_report(BATCH_SIZE, LEARNING_RATE, OPTIMIZER, LOSS, METRICS, 
                      historia, dataset_path, log_path, experiment_path,
                      testing_gazecom,testing_lund2013,
                      classification_report_1, classification_report_2,
                      time_training, time_complete, 
                      file_name = 'Report_EfficientB0_Trial_.json'):
        

   
        print("\n----------           Creating test report ...          ----------", end = "")
        
        Hiperparametros = historia.params
        Hiperparametros['Verbose'] = Hiperparametros.pop('verbose')
        Hiperparametros['Epochs'] = Hiperparametros.pop('epochs')
        Hiperparametros['Steps'] = Hiperparametros.pop('steps')
        Hiperparametros.update([('Batch size', BATCH_SIZE), ('Learning rate', LEARNING_RATE), ('Classes', CLASSES), ('Optimizer', str(OPTIMIZER)), ('Loss_function', str(LOSS)), ('Metrics', str(METRICS))])
        
    #    linea_parametros= '-----------------------------------------Training data-----------------------------------------'
        
        line_as_bytes = subprocess.check_output("nvidia-smi -L", shell=True)
        line = line_as_bytes.decode("ascii")
        _, line = line.split(":", 1)
        line, _ = line.split("(")
        gpu_info = line.strip()
        
        setup = {
            
            'Date and time': time.strftime("%c"),
            
            'Hardware setup': 
            [{
                    'Platform': platform.platform(),
                    'CPU processor': get_cpu_info()['brand_raw'],
                    'RAM': str(round(psutil.virtual_memory().total/(2**32), 2)) + ' RAM',
                    'GPU': gpu_info}],
    
            'Software setup':
            [{
                    'Python software version': sys.version,
                    'Tensorflow software version': tf.__version__,
                    'Keras software version': keras.__version__,
                    'Scikit-learn software version': sklearn.__version__, }],
        
        
        }
    
        execution_time = {
            'Execution_Time':
                [{
                    'Elapsed training time (sec)': time_training,
                    'Approximated elapsed time per epoch (sec)': time_training/Hiperparametros['Epochs'],
                    'Elapsed total time (sec)': time_complete,
                    
                    }],
      
        }         
                 
        with open(os.path.join(experiment_path, file_name), 'w') as f:
              
              json.dump('---------------         General        ---------------', f, indent=4)
              json.dump(setup, f, indent=4)
              json.dump('--------------- Evaluate Process Metrics ---------------', f, indent=4)
              json.dump(testing_gazecom, f, indent=4)
              json.dump(testing_lund2013, f, indent=4)
              json.dump('--------------- Classification Metrics ---------------', f, indent=4)
              json.dump('----------- GazeCom -----------', f, indent=4)
              json.dump(classification_report_1, f, indent=4)
              json.dump('----------- Lund2013 -----------', f, indent=4)
              json.dump(classification_report_2, f, indent=4)
              #json.dump(eval(str(historia.history)), f, indent=4)                                                            '*50, f, indent=4)
              json.dump('---------------      Execution Time    ---------------', f, indent=4)
              json.dump(execution_time, f, indent=4)
    
    
        print("\r**********             Test report ready              **********\n")
def from_ds_get_X_and_Y(ds):
    """From a tf.data.dataset are created X and Y"""
   
    print("\n----------     Extracting X & Y from DS    ----------\n")
   
    Y_testl = list()
    X_testl = list()
#    cnct = 0
    for image, label in tqdm(ds.unbatch()):
      Y_testl.append(label.numpy())      
      X_testl.append(image.numpy())

      #print(type(image.numpy()), type(label.numpy()))
      #print(image.shape, label.shape)
#    print(Y_testl, type(X_testl))
    Y_test = np.array(Y_testl)

    X_test = np.array(X_testl)
    
    print("X & Y shapes: ", X_test.shape, Y_test.shape)
    print("\n**********            X & Y ready           **********\n")
    
    return X_test, Y_test