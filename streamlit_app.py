import streamlit as st
import model
import utils
import torch
from PIL import Image

LABELS = ['English', 'Irish', 'Chinese', 'French', 'Portuguese', 'Greek', 'German', 'Dutch', 'Russian', 'Vietnamese', 'Arabic', 'Spanish', 'Scottish', 'Japanese', 'Italian', 'Korean', 'Polish', 'Czech']

model = model.RNN(utils.N_LETTERS, 128, 18)
model.load_state_dict(torch.load("data/model_state"))

image = Image.open('data/nationalities.jpg')
image = image.resize((600,300))
st.image(image)
st.title("Name Classifier")
st.subheader("Harnessing the Power of Machine Learning to Predict Nationalities from Names")
input = st.text_input("Enter a Name")

if input:
  st.write(f"**Your Prediction is -** {model.predict(input, labels = LABELS)}")   

