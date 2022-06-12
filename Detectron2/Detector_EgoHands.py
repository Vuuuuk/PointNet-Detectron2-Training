import cv2
from PIL.Image import BOX
from detectron2.data.catalog import DatasetCatalog
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo

import open3d as o3d
import pyrealsense2 as rs
import cv2 as cv
import numpy as np
import keyboard
import os
import glob

class Detectron:
    def __init__(self):

        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
        self.cfg.MODEL.WEIGHTS = os.path.join("./EgoHands.pth")
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
        self.cfg.MODEL.DEVICE = "cpu"
        self.predictor = DefaultPredictor(self.cfg)

    def detectOnImage(self, image):
        outputs = self.predictor(image)
        instances = outputs["instances"].to("cpu")
        confident_output09 = instances[instances.scores >= 0.6]
        v = Visualizer(image[:, :, ::-1], scale=0.8,)
        out = v.draw_instance_predictions(confident_output09)
        cv.imshow("Detectron2-ErgoHands Prediction", out.get_image()[:, :, ::-1])

detectron = Detectron()

cap = cv.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    cv.imshow('Main Camera Output', frame)
    if cv.waitKey(1) == ord("d"):
       detectron.detectOnImage(frame)
    if cv.waitKey(1) == ord("q"):
       cap.release()
       cv.destroyAllWindows()
       break;


