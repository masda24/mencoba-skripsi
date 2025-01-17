import streamlit as st
import cv2
import numpy as np
import base64
import io
from inference import inference as inferenceYOLO
from chat import chatbot, class_info_dict

labels = []
classes = dict()

def detect(image):
    inferencedImage, classesInDataset, classesInImage = inferenceYOLO(image)
    imageClassesList = list(set(classesInImage))
    label = ""

    for x in range(len(imageClassesList)):
        if x >= len(imageClassesList) - 1:
            label = label + str(classesInDataset[imageClassesList[x]])
        else:    
            label = label + str(classesInDataset[imageClassesList[x]]) + ", "

    global labels 
    labels = imageClassesList
    global classes 
    classes = classesInDataset
    
    return inferencedImage, label

def chatfront(history, message):
    info = ""

    for x in range(len(labels)):
        name = str(classes[labels[x]])
        infoCurrent = str(class_info_dict[name])

        if x >= len(labels) - 1:
            info = info + name + ":" + infoCurrent
        else:
            info = info + name + ":" + infoCurrent + ", "

    response = chatbot(info, history, message)

    return response

def main():
    st.title('Image Detection and Chatbot')

    camera_input = st.camera_input("Capture an image")

    if camera_input is not None:
        img_bytes = camera_input.getvalue()
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)

        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        detected_image, label = detect(img)

        _, img_encoded = cv2.imencode('.jpg', detected_image)
        image_as_text = base64.b64encode(img_encoded).decode('utf-8')

        st.image(detected_image, channels="BGR", caption="Inferred Image")
        st.write(f"Detected Label: {label}")

    st.subheader("Chat with the bot")

    if 'history' not in st.session_state:
        st.session_state.history = []

    user_message = st.text_input("Your message")

    if user_message:
        bot_response = chatfront(st.session_state.history, user_message)
        st.session_state.history.append(f"User: {user_message}")
        st.session_state.history.append(f"Bot: {bot_response}")

        st.write(f"**Bot Response:** {bot_response}")

if __name__ == "__main__":
    main()
