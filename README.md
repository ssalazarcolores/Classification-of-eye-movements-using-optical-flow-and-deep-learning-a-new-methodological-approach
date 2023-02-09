# Classification-of-eye-movements-using-optical-flow-and-deep-learning-a-new-methodological-approach

# Abstract
As is well known, the patterns generated in eye movements contain helpful information used in scientific fields such as neuroscience and psychology. Eye-tracking is the most widely used non-invasive method for recording and monitoring eye behavior. Currently, the identification of eye movements, such as fixations, saccades, and smooth pursuit in raw datasets, is carried out by specialists. However, this task is sometimes impractical due to the large amount of data to be handled. Therefore, there is significant interest in the discipline to find automatic methodologies that can solve this problem with a quality comparable to the specialists. While the problem has been addressed before, there are still considerable areas of opportunity in terms of performance. This contribution presents a novel method to identify the main eye movements. This method introduces the idea of analyzing the information in a two-dimensional optical flow domain through dense optical flow techniques and convolutional neural networks. The most common metrics in the research field were accuracy, precision, recall, f1-score, Cohen's kappa coefficient, and IoU, for which values of 96.59\%, 96.61\%, 96.58\%, 96.60\%, 0.9319, and 93.42\%, respectively, were obtained. The results show that the present contribution outperforms the state-of-the-art by at least 2\% in each metric. Above mentioned demonstrates the feasibility and robustness of the proposed method, which opens up multiple possibilities for new applications.

# Sample Result
Specifically, this proposal is based on domain change. Instead of processing the information as a one-dimensional time sequence, it is processed in a two-dimensional optical flow signal. Here is an example of how the transformation of eye movement information works:

![UL47_img_konijn_sc_6](https://user-images.githubusercontent.com/42470952/217583747-0179dbd3-738c-428f-904f-b0b6a1d85143.png)
![UL47_img_konijn_fix_11](https://user-images.githubusercontent.com/42470952/217583775-654f6641-a121-4129-9dff-b4ece465daef.png)


# Method
The proposed approach: (a) one-dimensional simple time series containing $x$ and $y$ coordinates of the gaze, (b) result of using dense optical flow techniques, obtaining images of the movement performed by the gaze, (c) the resulting two-dimensional images are used to train a CNN model for classifying the main eye movements: fixations and saccades.
![1](https://user-images.githubusercontent.com/42470952/217582881-77d4549a-9bb7-454a-b976-222c91d0f28c.png)

## Dataset 
The main dataset of this contribution is Gazecom, obtained from: https://gin.g-node.org/ioannis.agtzidis/gazecom_annotations. The proposed method was additionally tested on the lund2013 database, obtained from: https://github.com/richardandersson/EyeMovementDetectorEvaluation/tree/master/annotated_data/data%20used%20in%20the%20article

# Results
The results obtained from the experimentation carried out through 10 trials and a five-fold cross-validation in GazeCom.

| -  | Accuracy | Precision | Recall | F1-score | Kappa | IoU |
| ------- | -------- | -------- | -------- | -------- | -------- | -------- |
| Fixations | 96.61%  | 95.93%  | 97.31%  | 96.63%  | 0.9319 | 93.45% |
| Saccades | 96.53%  | 97.31% |  95.85%  | 96.59%   | 0.9320 |  93.38% |

The results obtained from the experimentation carried out through 10 trials and a five-fold cross-validation in Lund2013.
| -  | Accuracy | Precision | Recall | F1-score | Kappa | IoU |
| ------- | -------- | -------- | -------- | -------- | -------- | -------- |
| Fixations | 99.29%   | 99.17% |  99.17%  |  99.26%  |  0.9850 | 98.53% |
| Saccades | 99.28% | 99.29% |    99.26%  | 99.26% |  0.9851 |  98.56%|


# Requeriments
- python 3.10
- tensorflow 2.8.1 
- sklearn 1.2.1
- opencv-gpu 4.6.0-dev


# Transformation to the new domain
To transform the information of a dataset to the new domain for training or replicate the experiments, a csv file with the following characteristics is required: *x* and *y* correspond to the coordinates of the gaze position, and *L* corresponds to the label of the executed movement. This procedure corresponds to Stage I in our method. The csv file should look like:
| x  | y | L |
| ------- | -------- | -------- |
| 450.9	  | 278.1 | 1  | 
| 450.4	  | 278.4 | 1  | 
| 449.3	  | 278.7 | 1  |
| 449.1	  | 278.9	| 1  |

## Stage II
Stage II corresponds to a first transformation. It consists of taking the information from one-dimensional domain to two-dimensional domain through the generation of frame sequences where the coordinates corresponding to an eye movement event are plotted. To perform this process, it is required to run the file *dataset_transformation.py*, you will obtain a sequence of frames joined in an mp4 video.
![4](https://user-images.githubusercontent.com/42470952/217714955-2713653d-ffe8-480e-a7c2-258952c56d42.png)

## Stage III 
Stage III consists of the transformation to optical flow, to perform this procedure it is required to execute the file *dataset_transformation_optical_flow.py* indicating the folder where the information generated by Stage II is located and the destination folder.
![5 (1)](https://user-images.githubusercontent.com/42470952/217714958-fa13ee19-e2b3-4500-abe2-03facdfae256.png)

# How to train
## Preprocessing
First it is required to perform a preprocessing to the images generated by the Stage III transformation, through the file *dataset_preprocessing.py* it is only required to indicate the folder where the information of the previous stage is located and the destination folder.

## Training
To train the model it is required to generate a tf.data type structure, this is done through the script *create_ds.py*. The training script requires the path to the file generated by *create_ds.py* and the hyperparameters of the model:

```
IMAGE_HEIGHT = IMAGE_WIDTH = 224
NUM_CHANNELS = 3
BATCH_SIZE = 192
EPOCHS = 45
SPLITS_KFOLD = 5
LEARNING_RATE= 5e-3
```

# Reference
The manuscript is currently under revision.

# Contact us
For more questions, please feel free to contact us:
- abello15@alumnos.uaq.mx
- sebastian.salazar@cio.mx
- marco.aceves@uaq.mx
