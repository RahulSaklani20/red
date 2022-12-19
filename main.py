import streamlit as st
import cv2
import numpy as np
from PIL import Image



html_temp="""
<body style = "background-color:red;">
<div style ="background-color:tomato;padding:20px">
<h2 style="color:white;text-align:center;">Red Eye Detection And Corretion</h2>
</div>
</body>
"""
st.markdown(html_temp,unsafe_allow_html=True)
image_file=st.file_uploader("upload Image",type=['jpg','png','jpeg','jfif'])




def fillHoles(mask):
  maskFloodfill = mask.copy()
  h, w = maskFloodfill.shape[:2]
  maskTemp = np.zeros((h+2, w+2), np.uint8)
  cv2.floodFill(maskFloodfill, maskTemp, (0, 0), 255)
  mask2 = cv2.bitwise_not(maskFloodfill)
  return mask2 | mask

def detect_eyes():
    ima = Image.open(image_file);
    filebyte = np.array(ima);
    img = cv2.cvtColor(filebyte, cv2.COLOR_RGB2BGR)

    eyesCascade = cv2.CascadeClassifier(r'eye.xml')
    eyeRects = eyesCascade.detectMultiScale(img, 1.2, 20);

    imgOut = img.copy()
    for (x, y, w, h) in eyeRects:
        eyeImage = img[y:y + h, x: x + w]
        blue, green, red = cv2.split(eyeImage)

        bg = cv2.add(blue, green)
        # mask = (red > 150) & (red > bg)
        mask = (red > 50) & (red > bg)
        mask = mask.astype(np.uint8) * 255

        # Clean up mask by filling holes and dilating
        mask = fillHoles(mask)
        mask = cv2.dilate(mask, None, anchor=(-1, -1), iterations=1, borderType=1, borderValue=1)

        mean = bg / 2
        mask = mask.astype(bool)[:, :, np.newaxis]
        mean = mean[:, :, np.newaxis]

        # Copy the eye from the original image.
        eyeOut = eyeImage.copy()
        eyeOut = eyeOut.astype(np.ndarray)

        # Copy the mean image to the output image.
        np.copyto(eyeOut, mean, where=mask)

        # Copy the fixed eye to the output image.
        imgOut[y:y + h, x:x + w, :] = eyeOut
    return imgOut


if image_file is not None:
    our_image= Image.open(image_file)
    st.text('Orignal Image')
    st.image(our_image)
if image_file is not None and st.button('Result'):
    our_image= Image.open(image_file)
    result_image = detect_eyes();
    st.write("Output Image")
    st.image(result_image, channels="BGR")

