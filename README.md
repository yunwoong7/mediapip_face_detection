<h2 align="center">
MediaPipe Face Detection
</h2>

<div align="center">
  <img src="https://img.shields.io/badge/python-v3.10-blue.svg"/>
  <img src="https://img.shields.io/badge/mediapipe-v0.8.9.1-blue.svg"/>
</div>

[미디어파이프(MediaPipe)](https://google.github.io/mediapipe/)는 구글에서 인체를 대상으로 하는 인식에 대해 다양한 형태로 기능과 모델까지 제공하는 서비스입니다. Python 이외에도 다양한 프로그램언어와 환경에서에서 사용하기 편리한 라이브러리 형태로 제공되며 설치 후 즉시 간편하게 사용이 가능합니다.

이전 글에서 소개한 dlib을 이용한 얼굴인식을 처음 사용해본건 2018년이였는데 처음 사용 했을 때 신기하기도 했고 상당히 빠르기때문에 영상에 적용하여 다양한 응용도 했었습니다. 하지만 dlib을 사용하면서 이런 부분은 조금 문제가 있어서 어려움이 있겠구나 하는 부분도 많았죠. (물론 dlib를 모두 이해하고 사용해본것은 아니지만..)

그런데 최근 MediaPipe라는 라이브러리를 사용해보고 다시 한번 놀랬습니다. 

<div align="center">
  <img src="/asset/images/img.gif" width="70%">
</div>

MediaPipe의 얼굴인식(Face Detection)은 6개의 얼굴 랜드마크와 다중 얼굴 인식 기능을 지원합니다. 이 모듈은 가볍고 성능이 뛰어난 얼굴 검출기인 [BlazeFace](https://arxiv.org/abs/1907.05047)에 기반을 두었습니다. GPU 없이 CPU만으로도 작업이 가능합니다.

------

#### **1. Install**

Python 환경에서는 간단하게 mediapipe만 설치하면 사용이 가능합니다.

```python
pip install mediapipe
```

#### **2. Import Packages**

```python
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
 
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
```

#### **3. Function**

Colab 또는 Jupyter Notebook에서 이미지를 확인하기 위한 Function입니다.

```python
def plt_imshow(title='image', img=None, figsize=(8 ,5)):
    plt.figure(figsize=figsize)
 
    if type(img) == list:
        if type(title) == list:
            titles = title
        else:
            titles = []
 
            for i in range(len(img)):
                titles.append(title)
 
        for i in range(len(img)):
            if len(img[i].shape) <= 2:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_GRAY2RGB)
            else:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)
 
            plt.subplot(1, len(img), i + 1), plt.imshow(rgbImg)
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
 
        plt.show()
    else:
        if len(img.shape) < 3:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
        plt.imshow(rgbImg)
        plt.title(title)
        plt.xticks([]), plt.yticks([])
        plt.show()
```

#### **4. Load Image**

```python
image_path = 'asset/images/2021_g7_1.jpg'
image = cv2.imread(image_path)
```

#### **5. Face Detection**

```python
with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Draw face detections of each face.
    if not results.detections:
        print("Face not found in image")
    else:
        print('Found {} faces.'.format(len(results.detections)))
        
        annotated_image = image.copy()
        
        for detection in results.detections:
            # print(mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
            mp_drawing.draw_detection(annotated_image, detection, bbox_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=7))
            
        plt_imshow(["Original", "Find Faces"], [image, annotated_image], figsize=(16,10))
```

![img](https://blog.kakaocdn.net/dn/be2n9w/btrr1MVzc1A/oQfGS8tQZe1fANx4V0e130/img.png)

***더 나아가서..***

MediaPipe는 얼굴 검출(Face Detection) 외에도 별도의 센서 없이 얼굴에서 486개의 랜드마크를 추정하여 3D로 유추하는 기능(Face Mesh)도 제공합니다.

#### **6. Face Mesh**

```python
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=10, circle_radius=1)
static_image_mode = True
max_num_faces = 10
refine_landmarks = True
min_detection_confidence = 0.5
image_path = 'asset/images/kim.jpg'
image = cv2.imread(image_path)
with mp_face_mesh.FaceMesh(static_image_mode=static_image_mode, 
                           max_num_faces=max_num_faces, 
                           refine_landmarks=refine_landmarks, 
                           min_detection_confidence=min_detection_confidence) as face_mesh:
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if not results.multi_face_landmarks:
        print("Face not found in image")
    else:
        print('Found {} faces.'.format(len(results.multi_face_landmarks)))
        
        annotated_image = image.copy()
        for face_landmarks in results.multi_face_landmarks:
            # print('face_landmarks:', face_landmarks)
            mp_drawing.draw_landmarks(image=annotated_image, 
                                      landmark_list=face_landmarks, 
                                      connections=mp_face_mesh.FACEMESH_TESSELATION,
                                      landmark_drawing_spec=drawing_spec, 
                                      connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(image=annotated_image,
                                      landmark_list=face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_CONTOURS,
                                      landmark_drawing_spec=drawing_spec,
                                      connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(image=annotated_image,
                                      landmark_list=face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_IRISES,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
            
        plt_imshow(["Original", "Find Faces"], [image, annotated_image], figsize=(16,10))
```

![img](https://blog.kakaocdn.net/dn/cKddde/btrr1iNYe8P/OtcbIEt7kKcIwKfxqJGZn0/img.png)

<div align="center">
  <img src="/asset/images/img2.gif" width="70%">
</div>

<hr/>

MediaPipe는 인공지능을 이용한 얼굴, 포즈인식 모듈입니다. Python, Android, iOS, C++, JS, Coral의 언어들에서 사용이 가능하며 다양한 기능을 제공하고 있습니다.
<div align="center">
  <img src="https://blog.kakaocdn.net/dn/beF0RJ/btrrUtBB3Jx/eWvg82dwiK2a5yKCFBRdqK/img.png"/> <br/>&nbsp;

|                         | **Android** | **iOS** | **C++** | **Python** | **JS** | **Coral** |
| ----------------------- | ----------- | ------- | ------- | ---------- | ------ | --------- |
| Face Detection          | ✅           | ✅       | ✅       | ✅          | ✅      | ✅         |
| Face Mesh               | ✅           | ✅       | ✅       | ✅          | ✅      |           |
| Iris                    | ✅           | ✅       | ✅       |            |        |           |
| Hands                   | ✅           | ✅       | ✅       | ✅          | ✅      |           |
| Pose                    | ✅           | ✅       | ✅       | ✅          | ✅      |           |
| Holistic                | ✅           | ✅       | ✅       | ✅          | ✅      |           |
| Selfie Segmentation     | ✅           | ✅       | ✅       | ✅          | ✅      |           |
| Hair Segmentation       | ✅           |         | ✅       |            |        |           |
| Object Detection        | ✅           | ✅       | ✅       |            |        | ✅         |
| Box Tracking            | ✅           | ✅       | ✅       |            |        |           |
| Instant Motion Tracking | ✅           |         |         |            |        |           |
| Objectron               | ✅           |         | ✅       | ✅          | ✅      |           |
| KNIFT                   | ✅           |         |         |            |        |           |
| AutoFlip                |             |         | ✅       |            |        |           |
| MediaSequence           |             |         | ✅       |            |        |           |
| YouTube 8M              |             |         | ✅       |            |        |           |
  </div>
