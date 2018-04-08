import cv2
import matplotlib.pyplot as plt

#load training file
haar_face_cascade = cv2.CascadeClassifier('/home/bunny/Desktop/AI/ipai/lib/python3.5/site-packages/cv2/data/haarcascade_frontalface_alt.xml')
haar_face_cascade = cv2.CascadeClassifier('/home/bunny/Desktop/AI/ipai/lib/python3.5/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
haar_face_cascade = cv2.CascadeClassifier('/home/bunny/Desktop/AI/ipai/lib/python3.5/site-packages/cv2/data/haarcascade_frontalface_alt_tree.xml')
haar_face_cascade = cv2.CascadeClassifier('/home/bunny/Desktop/AI/ipai/lib/python3.5/site-packages/cv2/data/haarcascade_frontalface_default.xml')
haar_face_cascade = cv2.CascadeClassifier('/home/bunny/Desktop/AI/ipai/lib/python3.5/site-packages/cv2/data/haarcascade_profileface.xml')



def detect_faces(f_cascade, colored_img, scaleFactor = 1.1):
 img_copy = colored_img.copy()          
 gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)          
 faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5);          
 print(faces)
 print("faces found:", len(faces))
 for (x, y, w, h) in faces:
      cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)              
 
 return img_copy

test = cv2.imread('1.jpg')  
  
faces_detected_img = detect_faces(haar_face_cascade, test, scaleFactor = 1.1)  
plt.imshow(faces_detected_img)
plt.show()
