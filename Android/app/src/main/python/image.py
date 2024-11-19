from PIL import Image
import numpy as np
import cv2
import base64
import io


def main(imagestring):
    decoded_data= base64.b64decode(imagestring)
    np_data = np.fromstring(decoded_data,np.uint8)
    img=cv2.imdecode(np_data,cv2.IMREAD_UNCHANGED)
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    adaptive_threshold= cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,91,11)
    pil_im=Image.fromarray(adaptive_threshold)
    buff = io.BytesIO()
    pil_im.save(buff,format="PNG")
    img_str = base64.b64encode(buff.getvalue())
    return ""+str(img_str,'utf-8')