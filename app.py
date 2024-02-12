import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import cv2, numpy as np

st.title('Digit Recognizer')

st.write("""This app predicts the **Digit** from the drawing""")
st.write('---') 


def drawing_grid_canvas():
    #add grid to canvas
    bg_image = Image.fromarray(draw_grid())
    
    canvas_result = st_canvas(
        fill_color='#000000',
        stroke_width=40,
        stroke_color='#FFFFFF',
        background_color='#000000',
        width=560,
        height=560,
        drawing_mode="freedraw",
        key="canvas",
        background_image=bg_image if bg_image else None, 
        
    )

    return canvas_result

def draw_grid():
    img = np.zeros((560,560,3), np.uint8)
    img.fill(0)
    #28x28 grid
    for i in range(28):
        for j in range(28):
            cv2.rectangle(img, (i*20, j*20), (i*20+20, j*20+20), (255,255,255), 1)
    return img

def select_model():
    models={"Logistic Regression": "lr.pkl", "Decision Tree": "dt.pkl", "Support Vector Machine": "svc.pkl", "K-Nearest Neighbors": "knn.pkl", "Random Forest": "random_forest.pkl", "MLP": "mlp.pkl"}
    model_selection = st.sidebar.selectbox("Select Model", list(models.keys()))
    
    return models[model_selection]


canvas_result=drawing_grid_canvas()
model = 'model/'+select_model()

if st.button('Predict'):
    if canvas_result.image_data is not None:
        img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.reshape(1, 28, 28, 1)
        img = img.reshape(784)
        img = img//255.0
        # print(img)

        if os.path.isfile(model): 
            loaded_model = pickle.load(open(model, 'rb'))
            prediction = loaded_model.predict([img])
            st.write("Prediction: ", prediction)
        else:
            st.write("Model not found")
    else:
        st.write("Please draw a digit")
    

