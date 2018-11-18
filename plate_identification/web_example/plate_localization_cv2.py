import cv2

file_location = "../config/haarcascade_russian_plate_number.xml"
plate_cascade = cv2.CascadeClassifier(file_location)

img = cv2.imread('../images/car-r.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = plate_cascade.detectMultiScale(gray)

print(faces)

for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

cv2.imshow('img', img)
cv2.waitKey(0)


cv2.destroyAllWindows()