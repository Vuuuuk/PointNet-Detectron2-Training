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
    def __init__(self, model_type = "OD"):
        self.cfg = get_cfg()
        
        if(model_type == "OD"):
            #OBJECT DETECTION
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        elif(model_type == "IS"):
            #INSTANCE SEGMENTATION
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        elif(model_type == "LVIS"):
            #LVIS INSTANCE SEGMENTATION
            self.cfg.merge_from_file(model_zoo.get_config_file("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml")
        elif(model_type == "KP"):
            #KEYPOINT DETECTION
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        elif(model_type == "customDS"):
            #CUSTOM DETECTION
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
            self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
            self.cfg.MODEL.WEIGHTS = os.path.join("./customDS.pth")
           
            
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
        self.cfg.MODEL.DEVICE = "cpu"
       
        
        self.predictor = DefaultPredictor(self.cfg)
        
    def onImage(self, image):
        predictions = self.predictor(image)
        
        viz = Visualizer(image[:,:,::-1], metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                                                       instance_mode = ColorMode.SEGMENTATION)
                                                        
        output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))
        
        cv.imshow("Prediction", output.get_image()[:,:,::-1])
       
        
    def onImageCustom(self, image):
        outputs = self.predictor(image)
        instances = outputs["instances"].to("cpu")
        confident_output09 = instances[instances.scores >= 0.97]    
        
        for i in confident_output09.pred_boxes: 
            print(i)
            x = i[0]
            y = i[1]
            w = i[2]
            h = i[3]
            pixel_cm_ratio_w = 638.2198 / 11.5
            pixel_cm_ratio_h = 432.7625 / 11.5
            object_width = w / pixel_cm_ratio_w
            object_height = h / pixel_cm_ratio_h

            try:
                distance = depth_frame.get_distance(int(x+w)//2, int((y+h)//2))
                print("The mid position depth is (cm):", distance)
            except:
                print("Couldnt calculate distance!")
                
            cv.circle(image, (int(x+w)//2, int(y+h)//2), 5, (0, 0, 255), 3)
            cv.putText(image, "Width {} cm".format(np.round(object_width)), (int((x+w)//2), int((y+h)//2)-100),cv.FONT_HERSHEY_PLAIN,1,(0, 0, 255),2)          
            cv.putText(image, "Height {} cm".format(np.round(object_height)), (int((x+w)//2), int((y+h)//2)-80), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        v = Visualizer(image[:, :, ::-1],
                        scale=0.8,
                        )
        
        out = v.draw_instance_predictions(confident_output09)
        cv.imshow("Prediction", out.get_image()[:, :, ::-1])

pipe = rs.pipeline()
config = rs.config()   

def startPipeline():
  pipe.start(config)
  align = rs.align(rs.stream.color)
  print("Pipeline started!", end="\n")

def stopPipeline():
  pipe.stop()
  pipe = None
  config = None
  print("Pipeline stopped", end="\n")

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)

startPipeline()

detectron = Detectron(model_type = "customDS")
#image = cv.imread("./testKutije/233652255_1880784022091643_7880584755160553246_n.jpg")
# detectron.onImageCustom(image)
# cv.waitKey(0)

while True:
    frameset = pipe.wait_for_frames()
    color_frame = frameset.get_color_frame()
    depth_frame = frameset.get_depth_frame()
    color_image = np.asanyarray(color_frame.get_data())
    detpth_image = np.asanyarray(depth_frame.get_data())
    cv.imshow("L515", color_image)
    #cv.imshow("L515_depth", detpth_image)
    
    depth_data = depth_frame.as_frame().get_data()
    np_image = np.asanyarray(depth_data)
    depth_colormap = cv.applyColorMap(cv.convertScaleAbs(np_image, alpha=0.05), cv.COLORMAP_RAINBOW)
    cv.imshow('depth_colormap', depth_colormap)
    if cv.waitKey(1) == ord("n"):
        detectron.onImageCustom(color_image)
    if cv.waitKey(1) == ord("q"):
        cv.destroyAllWindows()
        stopPipeline()
        break        
    
stopPipeline()

def midpoint(ptA, ptB):
    	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
 

