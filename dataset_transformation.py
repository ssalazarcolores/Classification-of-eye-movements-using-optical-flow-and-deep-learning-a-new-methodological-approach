import pandas as pd
import numpy as np
import cv2 as cv2
import os


def interpolation(x,cantidad):
    temporal=[]
    for g in range(len(x)-1):
        b=np.linspace(x[g],x[g+1],cantidad)
        for j in b:
            temporal.append(round(j,4))    
    return temporal


def generar_video(t_coordenadas,test,name,fondo_gaze, path_save):

    point_color_l = (199,179,102)
   
    # Reading an image in default mode
    imagen = cv2.imread(fondo_gaze)
    height, width  = imagen.shape[:2]
    video = cv2.VideoWriter(path_save+name +'_'+test+'.avi',cv2.VideoWriter_fourcc(*'DIVX'),10,(width,height))

    #Frames creation                     
    for point in t_coordenadas:
        imagen = cv2.imread(fondo_gaze)    
        start_point = (point[0], point[1])
        end_point = (point[0]+30, point[1]+30)
        frame_f=cv2.circle(imagen,(start_point),30,point_color_l,-1)
        video.write(frame_f)


    video.release() 
    cv2.destroyAllWindows()


def extraccion_video_sc(dataframe,subject,fondo,fondo_gaze, path_save):
    posiciones=[]
    extraccion_x=[]
    extraccion_y=[]
    video=1
    frame_temporal= pd.DataFrame()

    for n in range(len(dataframe)-1):
        if dataframe.index[n+1]-dataframe.index[n] ==1:
            posiciones.append(dataframe.index[n])

        else:
            posiciones.append(dataframe.index[n])
            for i in posiciones:
                extraccion_x.append(dataframe.x[i])
                extraccion_y.append(dataframe.y[i])
          

            if len(extraccion_x)<20:
                frame_temporal['X']=interpolation(extraccion_x,60)
                frame_temporal['Y']=interpolation(extraccion_y,60)
            else:
                frame_temporal['X']=interpolation(extraccion_x,7)
                frame_temporal['Y']=interpolation(extraccion_y,7)

            t = frame_temporal.to_numpy()
            t=t.astype(int)
            t = tuple(map(tuple, t))
            extraccion_x=[]
            extraccion_y=[]
        
            generar_video(t,'sc_'+str(video),subject,fondo_gaze, path_save)
            video=video+1
            posiciones=[]
            frame_temporal= pd.DataFrame()


    posiciones.append(dataframe.index[-1])

    for i in posiciones:
        extraccion_x.append(dataframe.x[i])
        extraccion_y.append(dataframe.y[i])
            
    if len(extraccion_x)<15:
        frame_temporal['X']=interpolation(extraccion_x,60)
        frame_temporal['Y']=interpolation(extraccion_y,60)
    else:
        frame_temporal['X']=interpolation(extraccion_x,4)
        frame_temporal['Y']=interpolation(extraccion_y,4)

    t = frame_temporal.to_numpy()
    t=t.astype(int)
    t = tuple(map(tuple, t))
    extraccion_x=[]
    extraccion_y=[]
        
    generar_video(t,'sc_'+str(video),subject,fondo_gaze, path_save)


def extracion_fix(dataframe,subject,fondo_gaze, path_save):
    posiciones=[]
    extraccion_x=[]
    extraccion_y=[]
    video=1
    frame_temporal= pd.DataFrame()

    for n in range(len(dataframe)-1):
        if dataframe.index[n+1]-dataframe.index[n] ==1:
            posiciones.append(dataframe.index[n])
            #print(n)
        else:
            posiciones.append(dataframe.index[n])
            
            for i in posiciones:
                extraccion_x.append(dataframe.x[i])
                extraccion_y.append(dataframe.y[i])
                
            if len(extraccion_x)<20:
                frame_temporal['X']=interpolation(extraccion_x,60)
                frame_temporal['Y']=interpolation(extraccion_y,60)
            else:
                frame_temporal['X']=interpolation(extraccion_x,7)
                frame_temporal['Y']=interpolation(extraccion_y,7)

            t = frame_temporal.to_numpy()
            t=t.astype(int)
            t = tuple(map(tuple, t))
            extraccion_x=[]
            extraccion_y=[]
        
            generar_video(t,'fix_'+str(video),subject,fondo_gaze, path_save)
            video=video+1
            posiciones=[]
            frame_temporal= pd.DataFrame()

    for i in posiciones:
        extraccion_x.append(dataframe.x[i])
        extraccion_y.append(dataframe.y[i])
            
    if len(extraccion_x)<20:
        frame_temporal['X']=interpolation(extraccion_x,60)
        frame_temporal['Y']=interpolation(extraccion_y,60)
    else:
        frame_temporal['X']=interpolation(extraccion_x,7)
        frame_temporal['Y']=interpolation(extraccion_y,7)

    t = frame_temporal.to_numpy()
    t=t.astype(int)
    t = tuple(map(tuple, t))
    extraccion_x=[]
    extraccion_y=[]
        
    generar_video(t,'fix_'+str(video),subject,fondo_gaze, path_save)
    video=video+1
    posiciones=[]
    frame_temporal= pd.DataFrame()


def dataset_transformation(ent_input_dir,path_save,fondo_gaze):
    input_img_paths = sorted(
    [
        os.path.join(ent_input_dir, fname)
        for fname in os.listdir(ent_input_dir) 
        if fname.endswith(".csv")
    ]
            )

    for n in range(len(input_img_paths)-1):  
        data_copy = pd.read_csv(input_img_paths[n])
        #data.columns = ['hopudi', 'vepudi', 'extra','x', 'y', 'label'] 
        #data_copy=data.drop(columns=['hopudi', 'vepudi', 'extra']) #Delete all other columns, only x,y,Label
        
        subject= input_img_paths[n][36:51]

        print('Subject:',subject)
        print('---------------')
        data_c_fix=data_copy[data_copy['label'] == 1]
        data_c_fix=data_c_fix.drop(data_c_fix[data_c_fix['x']==0.0].index)
        data_c_fix=data_c_fix.drop(data_c_fix[data_c_fix['y']==0.0].index)
        data_c_fix=data_c_fix.drop(data_c_fix[data_c_fix['x']<0.0].index)
        ata_c_fix=data_c_fix.drop(data_c_fix[data_c_fix['y']<0.0].index)

        data_c_sc=data_copy[data_copy['label'] == 2]
        data_c_sc=data_c_sc.drop(data_c_sc[data_c_sc['x']==0.0].index)
        data_c_sc=data_c_sc.drop(data_c_sc[data_c_sc['y']==0.0].index)
        data_c_sc=data_c_sc.drop(data_c_sc[data_c_sc['x']<0.0].index)
        data_c_sc=data_c_sc.drop(data_c_sc[data_c_sc['y']<0.0].index)

        print('---------------')

        if len(data_c_sc) !=0:
            print('SACCADES')
            extraccion_video_sc(data_c_sc,subject,fondo_gaze, path_save)
            print('-------------------------------')
        else:
            print('NO INFORMATION')
        
        print('FIXATIONS')
        extracion_fix(data_c_fix,subject,fondo_gaze,path_save)
        print('-------------------------------')

dataset_transformation(ent_input_dir='/home/rtx3090a/Desktop/Alea/csv/img/',
                path_save='/home/rtx390a/Desktop/Alea/dataset/',
                fondo_gaze='/home/rtx3090a/Desktop/Alea/fondo_claro.png')
print('Data transformation done!')