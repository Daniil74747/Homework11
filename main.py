import cv2
from PIL import Image

image_path = 'cats.jpg'
image_cat = cv2.imread(image_path)

cats_face_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface_extended (1).xml')
cats_face = cats_face_cascade.detectMultiScale(image_cat)

for (x, y, w, h) in cats_face:
    print(x, y, w, h)
    cv2.rectangle(image_cat, (x, y), (x + w, y + h), (0, 255, 0), 3)

cv2.imshow('Cats', image_cat)
cv2.waitKey()

cats = Image.open('cats.jpg')
glasses = Image.open('glasses.jpg')

cats = cats.convert('RGBA')
glasses = glasses.convert('RGBA')

for (x, y, w, h) in cats_face:
    print(x, y, w, h)
    glasses = glasses.resize((w, int(h/3)))
    cats.paste(glasses, (x, int(y + h/4)), glasses)

cats.save('cats_in_glasses.png')

cats_in_glasses = cv2.imread('cats_in_glasses.png')

cv2.imshow('Cats in glasses', cats_in_glasses)
cv2.waitKey()
