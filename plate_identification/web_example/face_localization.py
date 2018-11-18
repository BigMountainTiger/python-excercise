import cv2

import matplotlib.pyplot as plt

file_location = "../config/haarcascade_frontalface_alt.xml"
face_cascade = cv2.CascadeClassifier(file_location)

def get_faces(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    faces = face_cascade.detectMultiScale(gray)
    
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
    
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img)
    plt.show()

img = cv2.imread('../images/other-1.jpeg')
get_faces(img)

img = cv2.imread('../images/lp-1.jpeg')
get_faces(img)

img = cv2.imread('../images/lp-2.jpeg')
get_faces(img)