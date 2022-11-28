import os

import numpy as np
import cv2


RES_WIDTH = 640
RES_HEIGHT = 480

TRINA_AREA_WIDTH = 625-415
TRINA_AREA_HEIGHT = 680-480
TRINA_AREA_X = 415
TRINA_AREA_Y = 480
TRINA_AREA_END_X = TRINA_AREA_X + TRINA_AREA_WIDTH
TRINA_AREA_END_Y = TRINA_AREA_Y + TRINA_AREA_HEIGHT

CROP_WIDTH = 825-225
CROP_HEIGHT = 875-270
CROP_X = 225
CROP_Y = 270
CROP_END_X = CROP_X + CROP_WIDTH
CROP_END_Y = CROP_Y + CROP_HEIGHT


path = os.path.dirname(__file__)
pathToIntrinsicsData = os.path.join(path, "..", "Data", "IntrinsicsData")
pathToExtrinsicsData = os.path.join(path, "..", "Data", "ExtrinsicsData")
pathToImages = os.path.join(path, "..", "Data", "ImagesStitchingAndTopDown")

#loading all intrinsic data
frontMtx = np.load(os.path.join(pathToIntrinsicsData, "frontMtx.npy"))
frontDist = np.load(os.path.join(pathToIntrinsicsData, "frontDist.npy"))
frontNewCameraMatrix = np.load(os.path.join(pathToIntrinsicsData, "frontNewCameraMatrix.npy"))
frontROI= np.load(os.path.join(pathToIntrinsicsData, "frontROI.npy"))

leftMtx = np.load(os.path.join(pathToIntrinsicsData, "leftMtx.npy"))
leftDist = np.load(os.path.join(pathToIntrinsicsData, "leftDist.npy"))
leftNewCameraMatrix = np.load(os.path.join(pathToIntrinsicsData, "leftNewCameraMatrix.npy"))
leftROI= np.load(os.path.join(pathToIntrinsicsData, "leftROI.npy"))

backMtx = np.load(os.path.join(pathToIntrinsicsData, "backMtx.npy"))
backDist = np.load(os.path.join(pathToIntrinsicsData, "backDist.npy"))
backNewCameraMatrix = np.load(os.path.join(pathToIntrinsicsData, "backNewCameraMatrix.npy"))
backROI = np.load(os.path.join(pathToIntrinsicsData, "backROI.npy"))

rightMtx = np.load(os.path.join(pathToIntrinsicsData, "rightMtx.npy"))
rightDist = np.load(os.path.join(pathToIntrinsicsData, "rightDist.npy"))
rightNewCameraMatrix = np.load(os.path.join(pathToIntrinsicsData, "rightNewCameraMatrix.npy"))
rightROI= np.load(os.path.join(pathToIntrinsicsData, "rightROI.npy"))

trinaFromAbove= cv2.imread(os.path.join(pathToImages, "TrinaTopView.png"))


front_map_1 = np.load(os.path.join(pathToExtrinsicsData, "FrontMap1.npy"))
front_map_2 = np.load(os.path.join(pathToExtrinsicsData, "FrontMap2.npy"))
left_map_1 = np.load(os.path.join(pathToExtrinsicsData, "LeftMap1.npy"))
left_map_2 = np.load(os.path.join(pathToExtrinsicsData, "LeftMap2.npy"))
back_map_1 = np.load(os.path.join(pathToExtrinsicsData, "BackMap1.npy"))
back_map_2 = np.load(os.path.join(pathToExtrinsicsData, "BackMap2.npy"))
right_map_1 = np.load(os.path.join(pathToExtrinsicsData, "RightMap1.npy"))
right_map_2 = np.load(os.path.join(pathToExtrinsicsData, "RightMap2.npy"))
# print(front_map_1.dtype)

#images paths to calculate top down homography from
pathFrontCamActual = os.path.join(pathToImages, "frontCamActual.png")
pathFrontCamDesired = os.path.join(pathToImages, "frontCamDesired.png")
pathRightCamActual = os.path.join(pathToImages, "rightCamActual.png")
pathRightCamDesired = os.path.join(pathToImages, "rightCamDesired.png")
pathLeftCamActual = os.path.join(pathToImages, "leftCamActual.png")
pathLeftCamDesired = os.path.join(pathToImages, "leftCamDesired.png")
pathBackCamActual = os.path.join(pathToImages, "backCamActual.png")
pathBackCamDesired = os.path.join(pathToImages, "backCamDesired.png")

frontActual = cv2.imread(pathFrontCamActual)
frontDesired = cv2.imread(pathFrontCamDesired)
rightActual = cv2.imread(pathRightCamActual)
rightDesired = cv2.imread(pathRightCamDesired)
leftActual = cv2.imread(pathLeftCamActual)
leftDesired = cv2.imread(pathLeftCamDesired)
backActual = cv2.imread(pathBackCamActual)
backDesired = cv2.imread(pathBackCamDesired)


#images to calculate stitching matricies from
frontStitchImage = cv2.imread(os.path.join(pathToImages, "frontStitchingImage.png"))
leftStitchImage = cv2.imread(os.path.join(pathToImages, "leftStitchingImage.png"))
backStitchImage = cv2.imread(os.path.join(pathToImages, "backStitchingImage.png"))
rightStitchImage = cv2.imread(os.path.join(pathToImages, "rightStitchingImage.png"))


#top down matricies and stitching matricies computed from calculateBirdsEye.py
Hfront = np.load(os.path.join(pathToExtrinsicsData, "Hfront.npy"))
Hright = np.load(os.path.join(pathToExtrinsicsData, "Hright.npy"))
Hleft = np.load(os.path.join(pathToExtrinsicsData, "Hleft.npy"))
Hback = np.load(os.path.join(pathToExtrinsicsData, "Hback.npy"))


HFL = np.load(os.path.join(pathToExtrinsicsData, "HFL.npy"))
HFLB = np.load(os.path.join(pathToExtrinsicsData, "HFLB.npy"))
HFLBR = np.load(os.path.join(pathToExtrinsicsData, "HFLBR.npy"))