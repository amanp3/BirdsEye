# Aman Penmetcha
# Written for AVATRINA,  Feb 2022
# Code to generate 360 Birds eye of TRINA surrounding using 4 usb cameras
# part of this code are adapted from code already created for 2 images panoramic stitching

import time
from dataclasses import dataclass

import numpy as np
import cv2
cv2.ocl.setUseOpenCL(False)

import sys, os
try:
    import trina
except ImportError:
    sys.path.append(os.path.expanduser("~/TRINA"))
    import trina
import trina.utils.BirdsEye.CudaVersion.dataImports as dataImports
from trina.utils.BirdsEye.CudaVersion.dataImports import RES_WIDTH, RES_HEIGHT
import trina.utils.BirdsEye.CudaVersion.stitchingFunctions as stitchingFunctions
# import pdb

# IMPORTANT**** Cameras are lettered, A must be right camera, B must be front camera, C must be left Camera, and D must be back camera
# This ensures that the proper camera recieves the correct intrinsics for its lens^

# where is each camera
# NOTE: these are being set with their actual (hardware) ports with udev rules. DO NOT CHANGE OR REPLUG CAMERAS!
# @see /etc/udev/rules.d/40-birdseye.rules
FRONT_INDEX = 1 # CAM B
LEFT_INDEX = 90 # CAM C
BACK_INDEX = 92  # CAM D
RIGHT_INDEX = 91 # CAM A
#LEFT_INDEX = 8 # CAM C
#BACK_INDEX = 12  # CAM D
#RIGHT_INDEX = 10 # CAM A
SPARE_CAM = 0

FRONT_PROP_INDEX = 0
LEFT_PROP_INDEX = 1
BACK_PROP_INDEX = 2
RIGHT_PROP_INDEX = 3

# cv2.imwrite('finalOutput.png', subStitchFLBR)
# subStitchFLBR = cv2.resize(subStitchFLBR, (1280, 1280))

# _,thresh = cv2.threshold(cv2.cvtColor(subStitchFLBR, cv2.COLOR_BGR2GRAY),1,255,cv2.THRESH_BINARY)
# x,y,w,h = cv2.boundingRect(thresh)
# subStitchFLBR = subStitchFLBR[y:y+h, x:x+w]

#manual crop
# subStitchFLBR = subStitchFLBR[1620:2450,4240:5200]

#find out where to crop
# plt.imshow(subStitchFLBR)
# plt.show()
# subStitchFLBR = subStitchFLBR[400:1200,200:1200]

#find out where trina goes
# plt.imshow(subStitchFLBR)
# plt.show()
# trinaFromAbove= cv2.resize(trinaFromAbove, (200,200))
# trinaFromAbove = cv2.rotate(trinaFromAbove, cv2.ROTATE_90_COUNTERCLOCKWISE)
# print(trinaFromAbove.shape)
# print("DONE COMPUTING")


# def call_undistort(args):
#     frame,H,resolution,name = args
#     return [name,cv2.cuda.warpPerspective(frame,H, resolution)]


@dataclass
class CameraProp:
    width: int
    height: int
    capture: cv2.VideoCapture
    gpuFrame: cv2.cuda_GpuMat
    # Intrinsics
    mtx: np.ndarray
    dist: np.ndarray
    newCameraMtx: np.ndarray
    roi: np.ndarray
    H: np.ndarray


class BirdsEyeCuda:

    def __init__(self):
        self.cameras = [None] * 4

        start = time.time()
        self._setup_cameras()
        # print(f"setting properties took {time.time() - start}")

    def _setup_cameras(self):
        self.cameras[FRONT_PROP_INDEX] = self._create_camera_prop(
            FRONT_INDEX, RES_WIDTH, RES_HEIGHT, dataImports.frontMtx,
            dataImports.frontDist, dataImports.frontNewCameraMatrix,
            dataImports.frontROI, dataImports.Hfront
        )
        self.cameras[LEFT_PROP_INDEX] = self._create_camera_prop(
            LEFT_INDEX, RES_WIDTH, RES_HEIGHT, dataImports.leftMtx,
            dataImports.leftDist, dataImports.leftNewCameraMatrix,
            dataImports.leftROI, dataImports.Hleft
        )
        self.cameras[BACK_PROP_INDEX] = self._create_camera_prop(
            BACK_INDEX, RES_WIDTH, RES_HEIGHT, dataImports.backMtx,
            dataImports.backDist, dataImports.backNewCameraMatrix,
            dataImports.backROI, dataImports.Hback
        )
        self.cameras[RIGHT_PROP_INDEX] = self._create_camera_prop(
            RIGHT_INDEX, RES_WIDTH, RES_HEIGHT, dataImports.rightMtx,
            dataImports.rightDist, dataImports.rightNewCameraMatrix,
            dataImports.rightROI, dataImports.Hright
        )

    def _create_camera_prop(self, index, width, height, mtx, dist, newCameraMtx, roi, H):
        capture = cv2.VideoCapture(index)
        # set resolution of video feed
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        gpuFrame = cv2.cuda_GpuMat()
        return CameraProp(
            width=width, height=height, capture=capture, gpuFrame=gpuFrame,
            mtx=mtx, dist=dist, newCameraMtx=newCameraMtx, roi=roi, H=H
        )

    def run(self, rtmp_url: str, framerate: int):
        # pub = RTMPVideoPublisher(rtmp_url, framerate)
        prevTime = time.time()
        warpedFrames = [None] * 4

        while True:
            currentTime = time.time()
            fps = 1 / (currentTime - prevTime)
            prevTime = currentTime

            # TODO: Parallelize these operations
            for propIndex, camera in enumerate(self.cameras):
                # Read camera frame
                ret, frame = camera.capture.read()
                # timeAfterCapture = time.time()

                # Undistort frame
                undistortedFrame = stitchingFunctions.undistortImage(
                    frame, camera.mtx, camera.dist, camera.newCameraMtx, camera.roi
                )

                # timeAfterUndistort = time.time()
                # print("Time to undistort: ", timeAfterUndistort - timeAfterCapture)

                camera.gpuFrame.upload(undistortedFrame)

                # Warp frame
                warpedFrames[propIndex] = cv2.cuda.warpPerspective(camera.gpuFrame, camera.H, (camera.width, camera.height))

            timeBeforeStitch = time.time()
            subStitchFL = stitchingFunctions.warpCudaTwoImages(warpedFrames[LEFT_PROP_INDEX], warpedFrames[FRONT_PROP_INDEX], dataImports.HFL)
            # subStitchFL_gray = cv2.cvtColor(subStitchFL, cv2.COLOR_RGB2GRAY)
            subStitchFLB = stitchingFunctions.warpCudaTwoImages(warpedFrames[BACK_PROP_INDEX], subStitchFL, dataImports.HFLB)
            # subStitchFLB_gray = cv2.cvtColor(subStitchFLB, cv2.COLOR_RGB2GRAY)
            subStitchFLBR = stitchingFunctions.warpCudaTwoImages(warpedFrames[RIGHT_PROP_INDEX], subStitchFLB, dataImports.HFLBR)
            timeAfterStitch = time.time()
            # print(f"Time to stitch: {timeAfterStitch - timeBeforeStitch}")

            # result of top down homography and stitching
            result = subStitchFLBR.download()

            # result[874:1000, 1100:1275,:] = trinaFromAbove
            # result = result[400:1200,200:1200]
            # result = cv2.resize(result, (720, 720))

            cv2.putText(result, '{:.2f}'.format(fps), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

            # pub.publish(result)

            cv2.imshow('', result)

            print(fps)
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                break

class BirdsEyeNonCuda:

    def __init__(self):
        self.cameras = [None] * 4

        self.trina_image = cv2.resize(dataImports.trinaFromAbove,
            (dataImports.TRINA_AREA_WIDTH, dataImports.TRINA_AREA_HEIGHT))

        start = time.time()
        self._setup_cameras()
        print(f"setting properties took {time.time() - start}")

    def _setup_cameras(self):
        self.cameras[FRONT_PROP_INDEX] = self._create_camera_prop(
            FRONT_INDEX, RES_WIDTH, RES_HEIGHT, dataImports.frontMtx,
            dataImports.frontDist, dataImports.frontNewCameraMatrix,
            dataImports.frontROI, dataImports.Hfront
        )
        self.cameras[LEFT_PROP_INDEX] = self._create_camera_prop(
            LEFT_INDEX, RES_WIDTH, RES_HEIGHT, dataImports.leftMtx,
            dataImports.leftDist, dataImports.leftNewCameraMatrix,
            dataImports.leftROI, dataImports.Hleft
        )
        self.cameras[BACK_PROP_INDEX] = self._create_camera_prop(
            BACK_INDEX, RES_WIDTH, RES_HEIGHT, dataImports.backMtx,
            dataImports.backDist, dataImports.backNewCameraMatrix,
            dataImports.backROI, dataImports.Hback
        )
        self.cameras[RIGHT_PROP_INDEX] = self._create_camera_prop(
            RIGHT_INDEX, RES_WIDTH, RES_HEIGHT, dataImports.rightMtx,
            dataImports.rightDist, dataImports.rightNewCameraMatrix,
            dataImports.rightROI, dataImports.Hright
        )

    def _create_camera_prop(self, index, width, height, mtx, dist, newCameraMtx, roi, H):
        capture = cv2.VideoCapture(index)
        # set resolution of video feed
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        return CameraProp(
            width=width, height=height, capture=capture, gpuFrame = None,
            mtx=mtx, dist=dist, newCameraMtx=newCameraMtx, roi=roi, H=H
        )

    def run(self, rtmp_url: str, framerate: int):
        # pub = RTMPVideoPublisher(rtmp_url, framerate)
        prevTime = time.time()
        while True:
            currentTime = time.time()
            fps = 1 / max(currentTime - prevTime, 1e-9)
            prevTime = currentTime
            result = self.get_frame()
            # result = cv2.resize(result, (720, 720))
            cv2.putText(result, '{:.2f}'.format(fps), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            # pub.publish(result)
            cv2.imshow('', result)
            # print(fps)
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                break

    def get_frame(self) -> np.ndarray:
        warpedFrames = [None] * 4
        # TODO: Parallelize these operations
        for propIndex, camera in enumerate(self.cameras):
            # Read camera frame
            ret, frame = camera.capture.read()
            # timeAfterCapture = time.time()
            # Undistort frame
            if not ret:
                # print("WARNING: BEV couldn't get frame for camera index {}".format(propIndex))
                warpedFrames[propIndex] = np.zeros((RES_HEIGHT, RES_WIDTH, 3), dtype="uint8")
                continue
            undistortedFrame = stitchingFunctions.undistortImage(
                frame, camera.mtx, camera.dist, camera.newCameraMtx, camera.roi
            )
            # timeAfterUndistort = time.time()
            # print("Time to undistort: ", timeAfterUndistort - timeAfterCapture)
            # Warp frame
            warpedFrames[propIndex] = cv2.warpPerspective(undistortedFrame, camera.H, (camera.width, camera.height))
        timeBeforeStitch = time.time()
        subStitchFL = stitchingFunctions.warpTwoImages(warpedFrames[LEFT_PROP_INDEX], warpedFrames[FRONT_PROP_INDEX], dataImports.HFL)
        # subStitchFL_gray = cv2.cvtColor(subStitchFL, cv2.COLOR_RGB2GRAY)
        subStitchFLB = stitchingFunctions.warpTwoImages(warpedFrames[BACK_PROP_INDEX], subStitchFL, dataImports.HFLB)
        # subStitchFLB_gray = cv2.cvtColor(subStitchFLB, cv2.COLOR_RGB2GRAY)
        subStitchFLBR = stitchingFunctions.warpTwoImages(warpedFrames[RIGHT_PROP_INDEX], subStitchFLB, dataImports.HFLBR)
        timeAfterStitch = time.time()
        #print(f"Time to stitch: {timeAfterStitch - timeBeforeStitch}")
        # result of top down homography and stitching
        result = subStitchFLBR
        # rotate to match viewing orientation
        result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)
        # import matplotlib.pyplot as plt
        # plt.imshow(result)
        # plt.show()
        # pdb.set_trace()
        # Place topdown view of TRINA in the center dead-zone
        #result[dataImports.TRINA_AREA_Y:dataImports.TRINA_AREA_END_Y,
        #        dataImports.TRINA_AREA_X:dataImports.TRINA_AREA_END_X, :] = self.trina_image
        # Crop final image
        result = result[dataImports.CROP_Y:dataImports.CROP_END_Y, dataImports.CROP_X:dataImports.CROP_END_X]
        # Blur image for basic AA
        result = cv2.GaussianBlur(result, (5, 5), 0)
        result = cv2.resize(result, (500,500))
        return result


if __name__ == "__main__":
    # RTMP
    # rtmp_settings = settings.settings()['RTMPStreamer']
    # rtmp_url = str(rtmp_settings['url'])

    # framerate = int(rtmp_settings['framerate'])
    framerate = 15

    # server = BirdsEyeCuda()
    server = BirdsEyeNonCuda()
    server.run("", framerate)
