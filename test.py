import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyttsx3
from googletrans import Translator
from PIL import Image, ImageDraw, ImageFont

# Initialize the video capture, hand detector, classifier, and text-to-speech engine
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
classifier = Classifier("C:\\Users\\Lakshya\\Desktop\\converted_keras\\keras_model.h5", "C:\\Users\\Lakshya\\Desktop\\converted_keras\\labels.txt")
offset = 20
imgSize = 300

# Text-to-speech engine initialization
engine = pyttsx3.init()

# Optionally, adjust the rate and volume of the speech
engine.setProperty('rate', 150)  # Adjust the speed of speech
engine.setProperty('volume', 1.0)  # Volume level (1.0 is max)

# Labels for the recognized gestures
labels = ["Hello", "I love you", "Name", "No", "Thank You", "Yes"]

# Initialize the translator for Hindi and Tamil translations
translator = Translator()

# Load fonts for Hindi and Tamil (Make sure you have the appropriate font files)
hindi_font = ImageFont.truetype("D:\\Indian Sign Language\\Tamil\\NotoSansDevanagari-VariableFont_wdth,wght.ttf", 32)  # Path to Hindi font
tamil_font = ImageFont.truetype("D:\\Indian Sign Language\\Hindi\\NotoSansTamil-VariableFont_wdth,wght.ttf", 32)  # Path to Tamil font

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)  # Detect hands
    if hands:
        for hand in hands:  # Loop over each detected hand
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]
            imgCropShape = imgCrop.shape

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap: wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap: hCal + hGap, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            recognized_text = labels[index]

            # Translate the recognized text to Hindi and Tamil
            translation_hindi = translator.translate(recognized_text, dest='hi').text
            translation_tamil = translator.translate(recognized_text, dest='ta').text

            # Convert the OpenCV image (imgOutput) to a PIL image for adding non-ASCII text
            img_pil = Image.fromarray(cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)

            # Draw the English text using OpenCV
            cv2.rectangle(imgOutput, (x - offset, y - offset - 90), (x - offset + 450, y - offset + 60 - 50), (0, 255, 0), cv2.FILLED)
            cv2.putText(imgOutput, recognized_text, (x, y - 90), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)

            # Draw Hindi and Tamil text using Pillow
            draw.text((x, y - 50), f"Hindi: {translation_hindi}", font=hindi_font, fill=(0, 0, 0))
            draw.text((x, y - 20), f"Tamil: {translation_tamil}", font=tamil_font, fill=(0, 0, 0))

            # Convert the PIL image back to an OpenCV image
            imgOutput = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            # Speak the recognized gesture aloud in English
            engine.say(recognized_text)
            engine.runAndWait()

            # Show cropped hand images
            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)

    # Show final image with hand detections
    cv2.imshow('Image', imgOutput)
    cv2.waitKey(1)
