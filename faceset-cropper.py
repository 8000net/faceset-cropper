import os
import sys
import cv2

image_path = sys.argv[1]
output_path = image_path + "/cropped"

def save_faces(cascade, imgname):
    img = cv2.imread(os.path.join(image_path, imgname))
    for i, face in enumerate(cascade.detectMultiScale(img)):
        x, y, w, h = face
        sub_face = img[y:y + h, x:x + w]
        cv2.imwrite(os.path.join(output_path, "{}_{}.jpg".format(imgname, i)), sub_face)

if __name__ == '__main__':
    face_cascade = "model.xml"
    cascade = cv2.CascadeClassifier(face_cascade)
    # Iterate through files
    for f in [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]:
        print(f)
        save_faces(cascade, f)
