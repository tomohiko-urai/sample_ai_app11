#-*- coding:utf-8 -*-

import sys, os
from PIL import Image
import numpy as np

import streamlit as st
import cv2
from ultralytics import YOLO


image_size = 50

  # max_y =max(scores)
st.set_option("deprecation.showfileUploaderEncoding", False)

st.sidebar.title("シャインマスカット収穫時期判定アプリ-ai-app11")
st.sidebar.write("画像認識モデルを使ってシャインマスカットの収穫時期の判定をします。")

st.sidebar.write("")
col1,col2 = st.columns(2)

#with col1:    
img_source = st.sidebar.radio("画像のソースを選択してください。",
                              ("画像をアップロード", "カメラで撮影"))
if img_source == "画像をアップロード":
    #with col1:
         img_file = st.sidebar.file_uploader("画像を選択してください。", type=["png", "jpg", "JPG"])
elif img_source == "カメラで撮影":
    #with col1:    
        img_file = st.camera_input("カメラで撮影")

#with col2:
if img_file is not None:  # max_y =max(scores)
    with st.spinner("推定中..."):
           #if  img_source == "カメラで撮影":
        img = Image.open(img_file)
        if  img_source != "カメラで撮影":
           #st.image(img, caption="対象の画像", width=280)
           st.image(img, caption="対象の画像", width=480)
        st.write("")

        img = img.convert("RGB")
       # img = img.resize((image_size,image_size))ca
        
        model = YOLO('last.pt')
        ret = model(img,save=True, conf=0.2, iou=0.1)
        annotated_frame = ret[0].plot(labels=True,conf=True)
        annotated_frame = cv2.cvtColor(annotated_frame , cv2.COLOR_BGR2RGB)
        #categories = ret[0].boxes.cls
        boxes = ret[0].boxes.xyxy
        scores = ret[0].boxes.conf
        categories = ret[0].boxes.cls
        max_y = max(scores)
     #   max_index = categories.index(max_y)  # max_y =max(scores)
        # 結果の表示
    #with col2:       
        #st.subheader("判定結果")
        st.subheader("判定結果")
      #  st.image(annotated_frame, caption='出力画像', width=280) 
        st.image(annotated_frame, caption='出力画像', width=480)
        #st.write("判定:Shinemuscat", max_y)
        st.write(scores)
        st.write(categories)
        #st.write(categories[0])
