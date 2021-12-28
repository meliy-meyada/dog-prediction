#Library imports
import numpy as np
import streamlit as st
import cv2
from tensorflow.keras.models import load_model


#กำลังโหลดโมเดล
model =  load_model('dog_breed.h5')

#ชื่อคลาส
CLASS_NAMES = ['Scottish Deerhound','Maltese Dog','Bernese Mountain Dog']

#Setting Title ของแอพ
st.title("การทำนายสายพันธุ์สุนัข")
st.markdown("อัพโหลดรูปน้องหมา")

#อัพโหลดภาพน้องหมา
dog_image = st.file_uploader("เลือกภาพ ...", type="png")
submit = st.button('Predict')
#ในปุ่มทำนาย คลิก
if submit:


    if dog_image is not None:

        # แปลงไฟล์เป็นอิมเมจ opencv
        file_bytes = np.asarray(bytearray(dog_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)



        # กำลังแสดงภาพ
        st.image(opencv_image, channels="BGR")
        #ปรับขนาดภาพ
        opencv_image = cv2.resize(opencv_image, (224,224))
        #แปลงภาพเป็น4มิติ
        opencv_image.shape = (1,224,224,3)
        #Make Prediction
        Y_pred = model.predict(opencv_image)

        st.title(str("สายพันธุ์สุนัขคือ "+CLASS_NAMES[np.argmax(Y_pred)]))
