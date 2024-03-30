#-*- coding:utf-8 -*-
#import model as shinemuscat_chk 
import sys, os
from PIL import Image
import numpy as np
#import cv2 
import streamlit as st
import cv2
from ultralytics import YOLO
#import matplotlib.pyplot as plt
#import datetime

image_size = 50


st.set_option("deprecation.showfileUploaderEncoding", False)

st.sidebar.title("シャインマスカット認識認識アプリ")
st.sidebar.write("オリジナルの画像認識モデルを使ってシャインマスカット物体認識をします。")

st.sidebar.write("")

img_source = st.sidebar.radio("画像のソースを選択してください。",
                              ("画像をアップロード", "カメラで撮影"))
if img_source == "画像をアップロード":
    img_file = st.sidebar.file_uploader("画像を選択してください。", type=["png", "jpg"])
elif img_source == "カメラで撮影":
    img_file = st.camera_input("カメラで撮影")

if img_file is not None:
    with st.spinner("推定中..."):
        img = Image.open(img_file)
        st.image(img, caption="対象の画像", width=480)
        st.write("")

        img = img.convert("RGB")
       # img = img.resize((image_size,image_size))
        
        model = YOLO('last.pt')
        ret = model(img,save=True, conf=0.6, iou=0.1)
        annotated_frame = ret[0].plot(labels=True,conf=True)
        annotated_frame = cv2.cvtColor(annotated_frame , cv2.COLOR_BGR2RGB)
        
        #in_data = np.asarray(img)
        #X = []
        #X.append(in_data)
        #X = np.array(X)
        # CNNのモデルを構築 --- (※3)
        #model = shinemuscat_chk.build_model(X.shape[1:])
        #model.load_weights("shinemuscat-color4-model_30_300_yellow-b_green-1-2_grenn-3-4b.hdf5")
     　 #####model.load_weights("shinemuscat-color4-model_30_60_bk_yellowBK_.hdf5")
        ##### 20220903 model.load_weights("tomato-color2-model2.hdf5")
# データを予測 --- (※4)
      
        #pre = model.predict(X)
        #y=0
        #y = pre.argmax()
        #st.image(img, caption="対象の画像", width=480)
        #st.write("")

        # 予測
        #results = predict(img)

        # 結果の表示
        st.subheader("判定結果")
        st.image(annotated_frame, caption='出力画像') 
        #st.write(camerapos[y] + "です。")
        #st.write(categories[y] + "です。")
