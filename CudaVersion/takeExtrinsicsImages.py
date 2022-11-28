import cv2

from trina.utils.BirdsEye.CudaVersion.birdsEyeCuda import FRONT_INDEX, LEFT_INDEX, BACK_INDEX, RIGHT_INDEX, SPARE_CAM

from trina.utils.BirdsEye.CudaVersion.dataImports import RES_HEIGHT, RES_WIDTH


cv2.ocl.setUseOpenCL(False)



#this function takes the file name as a string eg: rightCamDesired.png and which camera: 1, 3, 5, or 7 and
#stores a image to later calculate homography with
def takePicture(fileName, camNumber):
    cap = cv2.VideoCapture(camNumber) #choose which camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RES_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RES_HEIGHT)
    width1 = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height1 = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(width1, height1)
    while(True):
        ret, frame = cap.read() # return a single frame in variable `frame`
        cv2.imshow('Take Picture for: ' + fileName + '||| y for take picture q for dont take picture', frame) #display the captured image
        if cv2.waitKey(1) & 0xFF == ord('y'): #save on pressing 'y' 
            cv2.imwrite(fileName, frame)
            cv2.destroyAllWindows()
            break
        elif cv2.waitKey(1) == ord('q'): #dont save the picture on pressing 'q'
            print('No picture taken')
            cv2.destroyAllWindows()
            break

    cap.release()


def calibrate():
    #taking pics for the gram 

    takePicture('backCamActual.png', BACK_INDEX)
    takePicture('frontCamActual.png', FRONT_INDEX)
    takePicture('rightCamActual.png', RIGHT_INDEX)  
    takePicture('leftCamActual.png', LEFT_INDEX)
    takePicture('backCamDesired.png', SPARE_CAM)
    takePicture('frontCamDesired.png', SPARE_CAM) 
    takePicture('rightCamDesired.png',  SPARE_CAM) 
    takePicture('leftCamDesired.png', SPARE_CAM)

    takePicture('frontStitchingImage.png', FRONT_INDEX)
    takePicture('leftStitchingImage.png', LEFT_INDEX)
    takePicture('backStitchingImage.png', BACK_INDEX)
    takePicture('rightStitchingImage.png', RIGHT_INDEX)

#comment out calibrate line below if you want to skip calibration 
#if not commented you can still skip taking pictures but you will have to press q 8 times
calibrate()

