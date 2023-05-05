import cv2
import numpy as np

L = 256

def Erosion(imgin, imgout):
    w = cv2.getStructuringElement(cv2.MORPH_RECT,(45,45))
    cv2.erode(imgin,w,imgout)
    return imgout

def Dilation(imgin, imgout):
    w = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    cv2.dilate(imgin,w,imgout)
    return imgout
def OpeningClosing(imgin, imgout):
    w = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    temp = cv2.morphologyEx(imgin, cv2.MORPH_OPEN, w)
    cv2.morphologyEx(temp, cv2.MORPH_CLOSE, w, imgout)
    return imgout
def Boundary(imgin):
    w = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    temp = cv2.erode(imgin,w)
    imgout = imgin - temp
    return imgout



import cv2
import numpy as np
#import easyocr as ocr  
from PIL import Image

L = 256
#-----Function Chapter 3-----#
def Negative(imgin,imageout):
    M, N, chanel = imgin.shape
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            s = L - 1 - r
            imageout[x, y] = s
    return imageout
def Logarit(imgin,imageout):
    print(np.max(imgin))  
    c = (255)/np.log(1+np.max(imgin))
    s = c*(np.log(1 + imgin))
    imageout = np.array(s, dtype=np.uint8)
    return imageout

def Power(imgin,imageout):
    imageout
    M, N, chanel= imgin.shape
    gamma = 5.0
    c = np.power(256 - 1, 1 - gamma)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            s = c*np.power(r, gamma)
            imageout[x, y] = s.astype(np.uint8)
    return imageout

def HistogramEqualization(imgin, imgout):
    M, N,ncl = imgin.shape
    h = np.zeros(L, np.int32)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            h[r] = h[r] + 1

    p = np.zeros(L, np.float)
    for r in range(0, L):
        p[r] = h[r]/(M*N)

    s = np.zeros(L, np.float)
    for k in range(0, L):
        for j in range(0, k + 1):
            s[k] = s[k] + p[j]
        s[k] = s[k]*(L-1)

    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            imgout[x, y] = s[r].astype(np.uint8)
    return imgout



def Smoothing(imgin):
    M, N, h = imgin.shape
    m = 21
    n = 21
    a = m // 2
    b = m // 2
    w = np.ones((m,n),np.float)/(m*n)
    imgout = cv2.filter2D(imgin,cv2.CV_8UC1, w)
    # imgout = cv2.blur(imgin, (m,n))
    return imgout

def SmoothingGauss(imgin):
    # M, N, h = imgin.shape
    # m = 51
    # n = 51
    # a = m // 2
    # b = m // 2
    # sigma = 7.0
    # w = np.zeros((m,n), np.float)
    # for s in range(-a, a+1):
    #     for t in range(-b, b+1):
    #         w[s+a, t+b] = np.exp(-(s*s + t*t)/(2*sigma*sigma))
    # sum = np.sum(w)
    # w = w/sum
    # imgout = cv2.filter2D(imgin,cv2.CV_8UC1, w)
    imgout = cv2.GaussianBlur(imgin, (5,5),cv2.BORDER_DEFAULT)
    return imgout

def MeanFilter(imgin):
    kernel = np.ones((10,10),np.float32)/25
    imgout = cv2.filter2D(imgin,-1,kernel)
    return imgout
def MedianFilter(imgin):
    imgout = cv2.medianBlur(imgin,5)
    return imgout

def Sharpen(imgin):
    w = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], np.int32)
    temp = cv2.filter2D(imgin, cv2.CV_32FC1, w)
    result = imgin - temp
    result = np.clip(result, 0, L-1)
    imgout = result.astype(np.uint8)
    return imgout
def Bileteral(imgin):
    imgout = cv2.bilateralFilter(imgin,60,60,60)
    return imgout
def UnSharpMasking(imgin):
    blur = cv2.GaussianBlur(imgin, (3, 3), 1.0).astype(np.float)
    mask = imgin - blur
    k = 10.0
    imgout = imgin + k*mask
    imgout = np.clip(imgout, 0, L-1).astype(np.uint8)
    return imgout
def LowPass(imgin):
    kernel = np.ones((10,10),np.float32)/25
    lp = cv2.filter2D(imgin,-1,kernel)
    lp = imgin - lp
    return lp

def Gradient(imgin):
    wx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.int32)
    wy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.int32)
    gx = cv2.filter2D(imgin, cv2.CV_32FC1, wx);
    gy = cv2.filter2D(imgin, cv2.CV_32FC1, wy);
    g = abs(gx) + abs(gy)
    imgout = np.clip(g, 0, L-1).astype(np.uint8)
    return imgout


import cv2
import streamlit as st
import numpy as np

from PIL import Image
from streamlit_option_menu import option_menu
import os
import sys
import joblib
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.utils import label_map_util   
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import datetime
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
background: linear-gradient(#FFBB98, #FBE0C3);
background-size: 100%;
background-position: center;
background-repeat: no-repeat;
background-attachment: fixed;
color: #344648;
font-size:24px;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

config = tf.compat.v1.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


CONFIG_PATH = 'Tensorflow/workspace/models_TTD/my_ssd_mobnet/pipeline.config'

CHECKPOINT_PATH = 'Tensorflow/workspace/models_TTD/my_ssd_mobnet/'

ANNOTATION_PATH = 'Tensorflow/workspace/annotations'

configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-6')).expect_partial()




def brighten_image(image, amount):
    img_bright = cv2.convertScaleAbs(image, beta=amount)
    return img_bright


def blur_image(image, amount):
    blur_img = cv2.GaussianBlur(image, (11, 11), amount)
    return blur_img


def enhance_details(img):
    hdr = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    return hdr
def Balance_Color(img_bgr, R,G,B):
    # Chuyển từ hệ màu BGR - RGB
    # Thực tế, chúng ta thường sử dụng hệ màu RGB nhưng mặc định
    # opencv sử dụng hệ màu BGR. Do đó, để làm việc chúng ta cần chuyển đổi
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # Hàm Convert sang hệ màu COLOR_BGR2RGB  
    # nhân màu với hệ số trên thanh trượt
    img_rgb[:,:,0] = img_rgb[:,:,0] + R# Lấy tất cả những điểm ảnh [hệ số R] * R.get()
    img_rgb[:,:,1] = img_rgb[:,:,1] + G
    img_rgb[:,:,2] = img_rgb[:,:,2] + B
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    return img_rgb

def EditImage_loop():
    
    st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background: linear-gradient(#FFBB98,#FBE0C3);
        color: #7D8E95;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    chek = st.radio("Get Image From: ", ('File', 'Camera'), horizontal=True)
    st.sidebar.title("Edit Area")
    #st.markdown(f'<p style="color:#FCB0B4;font-size:24px;">{"Edit Area"}</p>', unsafe_allow_html=True)

    R_rate = st.sidebar.slider("R", min_value=0, max_value=255)
    G_rate = st.sidebar.slider("G", min_value=0, max_value=255)
    B_rate = st.sidebar.slider("B", min_value=0, max_value=255)

    blur_rate = st.sidebar.slider("Blurring", min_value=0.5, max_value=3.5)
    brightness_amount = st.sidebar.slider("Brightness", min_value=-50, max_value=50, value=0)
    
    apply_enhancement_filter = st.sidebar.checkbox('Enhance Details')
      
    st.sidebar.text("Smoothing Image")
    apply_Smoothing_filter= st.sidebar.checkbox("Smoothing")
    apply_Gauss_filter= st.sidebar.checkbox("Smoothing Gauss")
    apply_Mean_filter= st.sidebar.checkbox("Mean Filter")
    apply_Median_filter= st.sidebar.checkbox("Median Filter")
  

    st.sidebar.text("Sharpening")
    apply_Sharpen_filter= st.sidebar.checkbox("Sharpen")
    apply_UnSharpMasking_filter= st.sidebar.checkbox("UnSharpMasking")
    apply_Bileteral_filter= st.sidebar.checkbox("Bileteral")

    st.sidebar.text("Others")
    apply_Erosion_filter= st.sidebar.checkbox("Erosion")
    apply_Dilation_filter= st.sidebar.checkbox("Dilation")
    apply_Boundary_filter= st.sidebar.checkbox("Boundary")
    apply_LowPass_filter= st.sidebar.checkbox("Low Pass")
    apply_Gradient_filter= st.sidebar.checkbox("Gradient")
    apply_OpeningClosing_filter= st.sidebar.checkbox("OpeningClosing")
    apply_Negative_filter= st.sidebar.checkbox("Negative")
    apply_Power_filter= st.sidebar.checkbox("Power")
    apply_HistogramEqualization_filter= st.sidebar.checkbox("HistogramEqualization")

    if chek == 'File':
        image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
        if not image_file:
            return None
    elif chek == 'Camera':
        image_file = st.camera_input("Take a picture")
        if image_file:
            st.image(image_file)
        else:
            return None

    original_image = Image.open(image_file)
    original_image = np.array(original_image)
    
    processed_image = blur_image(original_image, blur_rate)
    processed_image = brighten_image(processed_image, brightness_amount)
    processed_image = Balance_Color(processed_image, R_rate, G_rate, B_rate)
    
    if apply_enhancement_filter:
        processed_image = enhance_details(processed_image)
    # C3 ----------------------------------------------------------------- 
    if apply_Negative_filter: 
        processed_image = Negative(original_image,processed_image)
    
    if apply_Power_filter:
        processed_image = Power(original_image,processed_image)
    
    if apply_HistogramEqualization_filter:
        processed_image = HistogramEqualization(original_image,processed_image)
    
    if apply_Smoothing_filter:
        processed_image = Smoothing(original_image) 
    if apply_Gauss_filter:
        processed_image = SmoothingGauss(original_image)
    
    if apply_Sharpen_filter:
        processed_image = Sharpen(original_image)
    if apply_UnSharpMasking_filter:
        processed_image = UnSharpMasking(original_image)
    
    if apply_Gradient_filter: #O
        processed_image = Gradient(original_image)

    if apply_Erosion_filter: #O
        processed_image = Erosion(original_image,processed_image)
    if apply_Dilation_filter: #O
        processed_image = Dilation(original_image,processed_image)
    if apply_OpeningClosing_filter: #O
        processed_image = OpeningClosing(original_image,processed_image)
    if apply_Boundary_filter: #O
        processed_image = Boundary(original_image)
    if apply_Mean_filter:
        processed_image = MeanFilter(original_image)

    if apply_Median_filter: 
        processed_image = MedianFilter(original_image)

    if apply_Bileteral_filter:
        processed_image = Bileteral(original_image)

    if apply_LowPass_filter:
        processed_image = LowPass(original_image)
    if chek == 'File':
        original_image = cv2.resize(original_image,(512,512))
        processed_image = cv2.resize(processed_image,(512,512))
    st.markdown(f'<p style="color:#344648;font-size:24px;">Original Image vs Processed Image</p>', unsafe_allow_html=True)

    # Hiển thị ảnh lên màn hình
    st.image([original_image, processed_image])

    # Lưu ảnh
    img = Image.fromarray(processed_image)
    dt_now = datetime.datetime.now()
    ten = f'D:/{dt_now.microsecond}.png'
    img.save(ten)
    title = st.text_input('Picture name:', placeholder = 'Name')
    tenanh = title+'.png'
    with open(ten, "rb") as file:
        st.download_button(
                    label="Download image",
                    data=file,
                    file_name=tenanh,
                    mime="image/png"
                )
    if os.path.isfile(ten):
        os.remove(ten)
@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections
category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'/label_map.pbtxt')

def XoaTrung(a, L):
    index = []
    flag = np.zeros(L, np.bool)
    for i in range(0, L):
        if flag[i] == False:
            flag[i] = True
            x1 = (a[i,0] + a[i,2])/2
            y1 = (a[i,1] + a[i,3])/2
            for j in range(i+1, L):
                x2 = (a[j,0] + a[j,2])/2
                y2 = (a[j,1] + a[j,3])/2
                d = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                if d < 0.2:
                    flag[j] = True
            index.append(i)
    for i in range(0, L):
        if i not in index:
            flag[i] = False
    return flag    
def DetectFruit(): 
    image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
    if not image_file:
        return None

    imgin = np.array(Image.open(image_file))
    #r, g, b = cv2.split(imgin)
    #imgin=cv2.merge([b, g, r])
    image_np = np.array(imgin)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
              for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    my_box = detections['detection_boxes']
    my_class = detections['detection_classes']+label_id_offset
    my_score = detections['detection_scores']

    my_score = my_score[my_score >= 0.7]
    L = len(my_score)
    my_box = my_box[0:L]
    my_class = my_class[0:L]
        
    flagTrung = XoaTrung(my_box, L)
    my_box = my_box[flagTrung]
    my_class = my_class[flagTrung]
    my_score = my_score[flagTrung]

    # viz_utils.visualize_boxes_and_labels_on_image_array(
    #         image_np_with_detections,
    #         detections['detection_boxes'],
    #         detections['detection_classes']+label_id_offset,
    #         detections['detection_scores'],
    #         category_index,
    #         use_normalized_coordinates=True,
    #         max_boxes_to_draw=5,
    #         min_score_thresh=.5,
    #         agnostic_mode=False)

    viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            my_box,
            my_class,
            my_score,
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=.7,
            agnostic_mode=False)
    st.image(image_np_with_detections)

def DetectFace_loop():
    
    detector = cv2.FaceDetectorYN.create(
    "./face_detection_yunet_2022mar.onnx",
    "",
    (320, 320),
    0.9,
    0.3,
    5000
    )
    detector.setInputSize((320, 320))

    recognizer = cv2.FaceRecognizerSF.create(
            "./face_recognition_sface_2021dec.onnx","")

    svc = joblib.load('svc.pkl')
    mydict =   ['DucAnh', 'Huy','Ngoc','Phat','Vi'
            ]
    image_file=st.file_uploader("Upload Your Image", accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None,  disabled=False, label_visibility="visible")
    if not image_file:
        return None
    
    imgin = np.array(Image.open(image_file))
        #r, g, b = cv2.split(imgin)
        #imgin=cv2.merge([b, g, r])
        
    cv2.namedWindow("ImageIn", cv2.WINDOW_AUTOSIZE)
    imgin = cv2.resize(imgin,(320,320),interpolation =cv2.INTER_AREA)
    faces = detector.detect(imgin) 
    try:
        face_align = recognizer.alignCrop(imgin, faces[1][0])
        face_feature = recognizer.feature(face_align)
        test_prediction = svc.predict(face_feature)

        result = mydict[test_prediction[0]]
        
        st.image(image_file)
        st.text("Bạn này là: "+ result)
    except:
        st.markdown(f'<p style="color:#344648;font-size:24px;">Can\'\t detect this face</p>', unsafe_allow_html=True)
        st.image(image_file)



#color = st.color_picker('Pick A Color', '#fff1ac')
#st.write('The current color is', color )


    
st.markdown(f'<h1 style="color:#344648;font-size:24px;">Web for Image Processing</h1>', unsafe_allow_html=True)
st.markdown(f'<p style="color:#344648;font-size:24px;">Made by Phan Văn Đức Anh - 20110609 & Phạm Quang Huy - 20110653</p>', unsafe_allow_html=True)


selected = option_menu(
        menu_title= "OPTION",
        options= ["Image Editing", 'Facial Recognition', "Fruit Identification"], 
        icons=["image", "people", "flower1"], menu_icon="cast",
        default_index=0, 
        orientation="horizontal",
        styles={
                "container": {"padding": "0!important", "background-color": "rgb(240, 242, 246)"},
                "icon": {"color":"#7D8E95", "font-size": "25px"},
                "nav-link": {
                    "font-size": "25px",
                    "text-align": "center",
                    "margin": "0px",
                    "--hover-color": "#eee",
                    "color":"#344648"
                    
                },
                "nav-link-selected": {"background-color": "#FFBB98","color":"#7D8E95"},
                # "nav-item-selected": {"background-color": "rgb(255, 187, 152)"}
            },
            )
selected

if selected == "Image Editing":
    EditImage_loop()
if selected == "Facial Recognition":
    DetectFace_loop()  
if selected == "Fruit Identification":
    DetectFruit()
