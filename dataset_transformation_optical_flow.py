import numpy as np
import os
import time
import cv2
print(cv2.__version__) #'4.6.0-dev'


def compute(folder, path_video):
    max_flow=compute_all(folder, path_video)
    return max_flow

def compute_all(folder, path_video):

    maxi=90
    path_optic_flow_color_hsv_global = folder + 'optical_flow_color_hsv_global.avi'
    max_flow= estimate_optical_flow(path_video, path_optic_flow_color=path_optic_flow_color_hsv_global)
    return max_flow

def estimate_optical_flow(path_input, path_optic_flow_color, maxi = 194):
    cap = cv2.VideoCapture(path_input)
    ret, frame1 = cap.read()

    prvs = np.uint8(cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY) * (255.0 / (maxi + 1.0)))
    gpu_prvs=cv2.cuda_GpuMat()
    gpu_prvs.upload(prvs)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    cnt = 0
    alto, ancho, z = frame1.shape

   #Generate video
    
    max_mag = estimate_max_magnitude_gradient(path_input, maxi = maxi)
    print(max_mag)

    cnt = 0
    max_map = np.zeros((frame1.shape[0], frame1.shape[1]), np.uint8)
    max_map_hsv = np.zeros_like(frame1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 180)
    max_flow = np.zeros((max_map.shape[0], max_map.shape[1], 2))
    cap = cv2.VideoCapture(path_input)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    
    while(True):
        ret, frame2 = cap.read()
        if ret == False:
             break

        temp1= np.uint8(cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY) * (255.0 / (maxi * 1.0)))
        gpu_next= cv2.cuda_GpuMat()
        gpu_next.upload(temp1)

        gpu_flow = cv2.cuda_FarnebackOpticalFlow.create(numLevels = 4, pyrScale = 0.5, fastPyramids = False, winSize = 5, numIters = 3, polyN = 5, polySigma = 1.1, flags = 0)
        gpu_flow = cv2.cuda_FarnebackOpticalFlow.calc(gpu_flow, gpu_prvs, gpu_next, None,)

        gpu_flow_x = cv2.cuda_GpuMat(gpu_flow.size(), cv2.CV_32FC1)
        gpu_flow_y = cv2.cuda_GpuMat(gpu_flow.size(), cv2.CV_32FC1)
        
        cv2.cuda.split(gpu_flow, [gpu_flow_x, gpu_flow_y])
        flow=gpu_flow.download()
        flow_x=gpu_flow_x.download()
        flow_y= gpu_flow_y.download()
        rgb, hsv, mag, ang = draw_hsv(prvs, flow, max_mag)
        next_temporal = gpu_next.download()

        prvs =next_temporal

        gpu_prvs.upload(prvs)
        max_map_hsv[max_map < mag] = rgb[max_map < mag]
        max_map[max_map < mag] =  mag[max_map < mag]        


        cnt += 1

    return  max_map_hsv

def estimate_max_magnitude_gradient(path_input, maxi = 194):
    cap = cv2.VideoCapture(path_input)
    ret, frame1 = cap.read()
    prvs = np.uint8(cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY) * (255.0 / (maxi * 1.0)))
    gpu_prvs=cv2.cuda_GpuMat()
    gpu_prvs.upload(prvs)
    cnt=0
    max_mag=0

    cap = cv2.VideoCapture(path_input)
    while True:
        
        ret, frame2 = cap.read()
        if cnt%25 == 0:
                # upload frame to GPU

                if not ret:
                    break

            # upload resized frame to GPU
                temp=np.uint8(cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY) * (255.0 / (maxi * 1.0)))
                gpu_current = cv2.cuda_GpuMat()
                gpu_current.upload(temp)

                gpu_flow = cv2.cuda_FarnebackOpticalFlow.create(numLevels = 4, pyrScale = 0.5, fastPyramids = False, winSize = 5, numIters = 3, polyN = 5, polySigma = 1.1, flags = 0)
                gpu_flow = cv2.cuda_FarnebackOpticalFlow.calc(gpu_flow, gpu_prvs, gpu_current, None,)
                
                #Separate the flow in x and y
                #First generate the matrices where to place it (x,y)
                gpu_flow_x = cv2.cuda_GpuMat(gpu_flow.size(), cv2.CV_32FC1)
                gpu_flow_y = cv2.cuda_GpuMat(gpu_flow.size(), cv2.CV_32FC1)


                cv2.cuda.split(gpu_flow, [gpu_flow_x, gpu_flow_y])
                flow_x=gpu_flow_x.download()
                flow_y= gpu_flow_y.download()
         
                mag = np.sqrt(flow_x**2 +flow_y**2).max()
           

                if  max_mag < mag:
                    max_mag = mag
        cnt += 1
    return max_mag


def draw_hsv(prvs, flow, max_mag):
#maximum value of the magnitude of all the frames = 255
    hsv = np.zeros((prvs.shape[0], prvs.shape[1], 3), np.uint8)
    mag,ang= cv2.cartToPolar(flow[...,0], flow[...,1])
    mag= mag * (255.0/ (max_mag * 1.0))
    hsv[..., 0] = ang*180/(np.pi*2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.uint8(mag) * 60

    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return rgb, hsv, mag, ang


def main_optical_flow(ent_input_dir,folder_1):

    input_img_paths = sorted(
    [
        os.path.join(ent_input_dir, fname)
        for fname in os.listdir(ent_input_dir) 
        if fname.endswith(".avi")
    ]
        )
    tiempos=[]

    for n in range(len(input_img_paths)-1):
        cap = cv2.VideoCapture(input_img_paths[n])

        # get default video FPS
        fps = cap.get(cv2.CAP_PROP_FPS)

        # get total number of video frames
        num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        if num_frames != 0.0:
            path_video = os.path.normpath(input_img_paths[n])
            folder = folder_1 + os.path.basename(path_video).split('.')[0]
             
            folder = folder + '/'



            ancho, alto = 1280, 720
      
            start = time.time()
            max_map_hsv = compute(folder,path_video)

            cv2.imwrite(folder_1+os.path.basename(input_img_paths[n]).split('.')[0] +'.png', max_map_hsv)
            end = time.time()
            result= end - start
            tiempos.append(result)
        else:
            continue


'''
Indidicate the folders:

ent_input_dir -- folder with the videos generated by the previuos transformation
folder_1 -- folder where the result of the optical flow transformation to the dataset
'''
main_optical_flow(ent_input_dir='/home/rtx390a/Desktop/Alea/dataset/',folder_1='res/')
print('Optical Flow transformation Done!')