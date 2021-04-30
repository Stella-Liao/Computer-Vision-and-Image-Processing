import os
import sys
import glob
import json

import cv2


FACE_MODEL_DIR = "models/haarcascade_frontalface_alt.xml"


class FaceDetector(object):

    def __init__(self):
        self.face_detector_ = cv2.CascadeClassifier(FACE_MODEL_DIR)


    def detect(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.face_detector_.detectMultiScale(gray_img, 1.1, 3)

        return faces


class DetectionResult(object):

    def __init__(self):
        self.results_ = []

    
    def add(self, file_dir, faces):
        filename = file_dir.split('/')[-1]
        # for windows
        filename = filename.split('\\')[-1]
        
        # {"iname": "3.jpg", "bbox": [12, 38, 10, 12]}
        for (x, y, w, h) in faces:
            self.results_.append(
                {
                    "iname": filename,
                    "bbox": [int(x), int(y), int(w), int(h)]
                }
            )


    def dump(self, output_dir):
        with open(output_dir, "w") as fout:
            json.dump(self.results_, fout)
        
        print("[INFO] Detection result was saved at: {}".format(output_dir))


def get_filenames(root_dir):
    if os.path.isdir(root_dir) == False:
        print("[ERROR] Directory '{}' does NOT exist".format(root_dir))
        exit(1)

    # match number only
    files = glob.glob(root_dir + "/*.*")

    # sort files according to index number
    # filename format "<prefix-dir>/img_idx.jpg"
    cmp = lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1])
    files.sort(key=cmp)

    return files


def face_detection(dataset_dir):
    img_files = get_filenames(dataset_dir)
    face_detector = FaceDetector()
    detection_result = DetectionResult()
    
    for file in img_files:
        img = cv2.imread(file)
        faces = face_detector.detect(img)

        detection_result.add(file, faces)

    output_dir = "/".join(dataset_dir.split('/')[:-1]) + "/results.json"
    detection_result.dump(output_dir)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please specify the directory of the dataset")
        print("Example: python FaceDetector.py <dataset-dir>")
        exit(1)
    
    dataset_dir = sys.argv[1]
    face_detection(dataset_dir)