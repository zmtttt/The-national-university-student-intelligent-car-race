import cv2
cap=cv2.VideoCapture(0)
_,image=cap.read()
i=0
while True:
  _,image=cap.read()
  i+=1
  if i>10:
    break
s=330
image[:160,:]=0
cv2.imwrite('test.jpg',image)
cv2.imwrite('test1.jpg',image)