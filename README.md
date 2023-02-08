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
The main dataset of this contribution is Gazecom, available in: 

# Requeriments
- python 3.10
- tensorflow 2.8.1 
- opencv-gpu 4.6.0-dev

# Reference
The manuscript is currently under revision.

# Contact us
For more questions, please feel free to contact us:
- abello15@alumnos.uaq.mx
- sebastian.salazar@cio.mx
- marco.aceves@uaq.mx
