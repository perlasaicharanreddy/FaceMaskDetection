import streamlit as st
from keras.models import load_model
import cv2
import numpy as np
from PIL import Image

model = load_model('model-008.model')
face_clsfr = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
labels_dict = {1: 'MASK', 0: 'NO MASK'}
color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}

def detect_facemask(our_image):
    labels_dict={1:'MASK',0:'NO MASK'}
    img = np.array(our_image.convert('RGB'))
    if (type(img) is np.ndarray):
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        face=face_clsfr.detectMultiScale(gray,1.03,5)
        for (x,y,w,h) in face:
            face_img=gray[y:y+w,x:x+w]
            resized=cv2.resize(face_img,(100,100))
            normalized=resized/255.0
            reshaped=np.reshape(normalized,(1,100,100,1))
            result=model.predict(reshaped)
            label=np.argmax(result,axis=1)[0]
            #st.text(label)
            return (labels_dict[label])


def main():
    """Face Detection App"""
    st.title("Face Detection App")
    st.text("Build with Streamlit and OpenCV")
    activities = ["Photo Detection","Live Detection","About"]
    choice = st.sidebar.selectbox("Select Activty", activities)
    if choice == 'Photo Detection':
        st.subheader("Face Mask Detection")
        image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

        if image_file is not None:
            our_image = Image.open(image_file)
            st.text("Original Image")
            st.image(our_image)

        if st.button('Classify'):
            result=detect_facemask(our_image)
            st.success(result)

    elif choice == 'Live Detection':
        st.subheader("Live  Detection")
        st.title("Webcam Live Feed")
        run = st.checkbox('Run')
        frameST = st.empty()
        #FRAME_WINDOW = st.image([])
        camera = cv2.VideoCapture(1)

        while run:
            _, img = camera.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_clsfr.detectMultiScale(gray, 1.03, 5)

            for (x, y, w, h) in faces:
                face_img = gray[y:y + w, x:x + w]
                resized = cv2.resize(face_img, (100, 100))
                normalized = resized / 255.0
                reshaped = np.reshape(normalized, (1, 100, 100, 1))
                result = model.predict(reshaped)

                label = np.argmax(result, axis=1)[0]

                cv2.rectangle(img, (x, y), (x + w, y + h), color_dict[label], 2)
                cv2.rectangle(img, (x, y - 40), (x + w, y), color_dict[label], -1)
                cv2.putText(img, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            #FRAME_WINDOW.image(img)
            frameST.image(img, channels="BGR")

        else:
            st.write('Stopped')


    elif choice == 'About':
        st.subheader("About Face Detection App")
        st.markdown("Built with Streamlit by ")
        st.text("Sai Charan\nAravind\nAlekhya\nKoushik")
        st.success("")


if __name__ == '__main__':
		main()
