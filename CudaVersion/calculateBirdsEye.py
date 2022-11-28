import numpy as np
import cv2
import os
import sys
try:
    import trina
except ImportError:
    sys.path.append(os.path.expanduser("~/TRINA"))
    import trina
import trina.utils.BirdsEye.CudaVersion.dataImports as dataImports
from dataImports import path, RES_HEIGHT, RES_WIDTH
import trina.utils.BirdsEye.CudaVersion.stitchingFunctions as stitchingFunctions

path_to_extrinsics_data = os.path.join(path, "..", "Data", "ExtrinsicsData")


front_map_1, front_map_2 = stitchingFunctions.computeUndistortMaps(dataImports.frontMtx, dataImports.frontDist, dataImports.frontNewCameraMatrix)

right_map_1, right_map_2 = stitchingFunctions.computeUndistortMaps(dataImports.rightMtx, dataImports.rightDist, dataImports.rightNewCameraMatrix)

left_map_1, left_map_2 = stitchingFunctions.computeUndistortMaps(dataImports.leftMtx, dataImports.leftDist, dataImports.leftNewCameraMatrix)

back_map_1, back_map_2 = stitchingFunctions.computeUndistortMaps(dataImports.backMtx, dataImports.backDist, dataImports.backNewCameraMatrix)

np.save(os.path.join(path_to_extrinsics_data, "FrontMap1"), front_map_1)
np.save(os.path.join(path_to_extrinsics_data, "FrontMap2"), front_map_2)
np.save(os.path.join(path_to_extrinsics_data, "RightMap1"), right_map_1)
np.save(os.path.join(path_to_extrinsics_data, "RightMap2"), right_map_2)
np.save(os.path.join(path_to_extrinsics_data, "LeftMap1"), left_map_1)
np.save(os.path.join(path_to_extrinsics_data, "LeftMap2"), left_map_2)
np.save(os.path.join(path_to_extrinsics_data, "BackMap1"), back_map_1)
np.save(os.path.join(path_to_extrinsics_data, "BackMap2"), back_map_2)

frontActual = stitchingFunctions.undistortImage(dataImports.frontActual, dataImports.frontMtx, dataImports.frontDist, dataImports.frontNewCameraMatrix, dataImports.frontROI)
frontDesired = stitchingFunctions.undistortImage(dataImports.frontDesired, dataImports.frontMtx, dataImports.frontDist, dataImports.frontNewCameraMatrix, dataImports.frontROI)
rightActual = stitchingFunctions.undistortImage(dataImports.rightActual, dataImports.rightMtx, dataImports.rightDist, dataImports.rightNewCameraMatrix, dataImports.rightROI)
rightDesired = stitchingFunctions.undistortImage(dataImports.rightDesired, dataImports.rightMtx, dataImports.rightDist, dataImports.rightNewCameraMatrix, dataImports.rightROI)
leftActual = stitchingFunctions.undistortImage(dataImports.leftActual, dataImports.leftMtx, dataImports.leftDist, dataImports.leftNewCameraMatrix, dataImports.leftROI)
leftDesired = stitchingFunctions.undistortImage(dataImports.leftDesired, dataImports.leftMtx, dataImports.leftDist, dataImports.leftNewCameraMatrix, dataImports.leftROI)
backActual = stitchingFunctions.undistortImage(dataImports.backActual, dataImports.backMtx, dataImports.backDist, dataImports.backNewCameraMatrix, dataImports.backROI)
backDesired = stitchingFunctions.undistortImage(dataImports.backDesired, dataImports.backMtx, dataImports.backDist, dataImports.backNewCameraMatrix, dataImports.backROI)

#patternSize stores the size of the chessboard you are looking for
patternSize = (8,6)

retFA, cornersFrontActual = cv2.findChessboardCorners(frontActual, patternSize, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
retFD, cornersFrontDesired = cv2.findChessboardCorners(frontDesired, patternSize)
retRA, cornersRightActual = cv2.findChessboardCorners(rightActual, patternSize)
retRD, cornersRightDesired = cv2.findChessboardCorners(rightDesired, patternSize)
retLA, cornersLeftActual = cv2.findChessboardCorners(leftActual, patternSize)
retLD, cornersLeftDesired = cv2.findChessboardCorners(leftDesired, patternSize)
retBA, cornersBackActual = cv2.findChessboardCorners(backActual, patternSize)
retBD, cornersBackDesired = cv2.findChessboardCorners(backDesired, patternSize)


Hfront, _1 = cv2.findHomography(cornersFrontActual, cornersFrontDesired)
Hright, _2 = cv2.findHomography(cornersRightActual, cornersRightDesired)
Hleft, _3 = cv2.findHomography(cornersLeftActual, cornersLeftDesired)
Hback, _4 = cv2.findHomography(cornersBackActual, cornersBackDesired)



np.save(os.path.join(path_to_extrinsics_data, "Hfront"), Hfront)
np.save(os.path.join(path_to_extrinsics_data, "Hright"), Hright)
np.save(os.path.join(path_to_extrinsics_data, "Hleft"), Hleft)
np.save(os.path.join(path_to_extrinsics_data, "Hback"), Hback)

# undistorts the images
frontStitchImage = stitchingFunctions.undistortImage(dataImports.frontStitchImage, dataImports.frontMtx, dataImports.frontDist, dataImports.frontNewCameraMatrix, dataImports.frontROI)
leftStitchImage = stitchingFunctions.undistortImage(dataImports.leftStitchImage, dataImports.leftMtx, dataImports.leftDist, dataImports.leftNewCameraMatrix, dataImports.leftROI)
backStitchImage = stitchingFunctions.undistortImage(dataImports.backStitchImage, dataImports.backMtx, dataImports.backDist, dataImports.backNewCameraMatrix, dataImports.backROI)
rightStitchImage = stitchingFunctions.undistortImage(dataImports.rightStitchImage, dataImports.rightMtx, dataImports.rightDist, dataImports.rightNewCameraMatrix, dataImports.rightROI)



#applies top down homography Uncomment next block of 4 lines if you want this to run
frontStitchImage_warp = cv2.warpPerspective(frontStitchImage, Hfront, (RES_WIDTH, RES_HEIGHT))
leftStitchImage_warp = cv2.warpPerspective(leftStitchImage, Hleft, (RES_WIDTH, RES_HEIGHT))
backStitchImage_warp = cv2.warpPerspective(backStitchImage, Hback, (RES_WIDTH, RES_HEIGHT))
rightStitchImage_warp = cv2.warpPerspective(rightStitchImage, Hright, (RES_WIDTH, RES_HEIGHT))

frontStitchImage_warp_gray = cv2.cvtColor(frontStitchImage_warp, cv2.COLOR_RGB2GRAY)
leftStitchImage_warp_gray = cv2.cvtColor(leftStitchImage_warp, cv2.COLOR_RGB2GRAY)
backStitchImage_warp_gray = cv2.cvtColor(backStitchImage_warp, cv2.COLOR_RGB2GRAY)
rightStitchImage_warp_gray = cv2.cvtColor(rightStitchImage_warp, cv2.COLOR_RGB2GRAY)

HFL = stitchingFunctions.calculateStitchingMatrix(frontStitchImage_warp, frontStitchImage_warp_gray, leftStitchImage_warp, leftStitchImage_warp_gray)
subStitchFL = stitchingFunctions.warpTwoImages(leftStitchImage_warp, frontStitchImage_warp, HFL)
np.save(os.path.join(path_to_extrinsics_data, "HFL"), HFL)

subStitchFL_gray = cv2.cvtColor(subStitchFL, cv2.COLOR_RGB2GRAY)
HFLB = stitchingFunctions.calculateStitchingMatrix(subStitchFL, subStitchFL_gray, backStitchImage_warp, backStitchImage_warp_gray)
np.save(os.path.join(path_to_extrinsics_data, "HFLB"), HFLB)

subStitchFLB = stitchingFunctions.warpTwoImages(backStitchImage_warp, subStitchFL, HFLB)

subStitchFLB_gray = cv2.cvtColor(subStitchFLB, cv2.COLOR_RGB2GRAY)
HFLBR = stitchingFunctions.calculateStitchingMatrix(subStitchFLB, subStitchFLB_gray, rightStitchImage_warp, rightStitchImage_warp_gray)
subStitchFLBR = stitchingFunctions.warpTwoImages(rightStitchImage_warp, subStitchFLB, HFLBR)

np.save(os.path.join(path_to_extrinsics_data, "HFLBR"), HFLBR)