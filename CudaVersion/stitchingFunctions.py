import numpy as np
import cv2
import io

import sys, os
try:
    import trina
except ImportError:
    sys.path.append(os.path.expanduser("~/TRINA"))
    import trina
from trina.utils.BirdsEye.CudaVersion.dataImports import RES_HEIGHT, RES_WIDTH


def detectAndDescribe(image, method='orb'):
    """
    Compute key points and feature descriptors using an specific method
    """
    
    assert method is not None, "You need to define a feature detection method. Values are: 'sift', 'surf'"
    
    # detect and extract features from the image
    if method == 'sift':
        descriptor = cv2.xfeatures2d.SIFT_create()
    elif method == 'surf':
        descriptor = cv2.xfeatures2d.SURF_create()
    elif method == 'brisk':
        descriptor = cv2.BRISK_create()
    elif method == 'orb':
        descriptor = cv2.ORB_create(nfeatures = 50000)
        
    # get keypoints and descriptors
    (kps, features) = descriptor.detectAndCompute(image, None)
    
    return (kps, features)

#creates matcher object
def createMatcher(method,crossCheck):
    "Create and return a Matcher Object"
    
    if method == 'sift' or method == 'surf':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    elif method == 'orb' or method == 'brisk':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
    return bf

#matches the features
def matchKeyPointsBF(featuresA, featuresB, method):
    bf = createMatcher(method, crossCheck=True)
        
    # Match descriptors.
    best_matches = bf.match(featuresA,featuresB)
    
    # Sort the features in order of distance.
    # The points with small distance (more similarity) are ordered first in the vector
    rawMatches = sorted(best_matches, key = lambda x:x.distance)
    print("Raw matches (Brute force):", len(rawMatches))
    return rawMatches


def matchKeyPointsKNN(featuresA, featuresB, ratio, method):
    bf = createMatcher(method, crossCheck=False)
    # compute the raw matches and initialize the list of actual matches
    rawMatches = bf.knnMatch(featuresA, featuresB, 2)
    print("Raw matches (knn):", len(rawMatches))
    matches = []

    # loop over the raw matches
    for m,n in rawMatches:
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe's ratio test)
        if m.distance < n.distance * ratio:
            matches.append(m)
    return matches


def getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh):
    # convert the keypoints to numpy arrays
    kpsA = np.float32([kp.pt for kp in kpsA])
    kpsB = np.float32([kp.pt for kp in kpsB])
    
    if len(matches) > 4:
        # construct the two sets of points
        ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx] for m in matches])
        
        # estimate the homography between the sets of points
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
            reprojThresh)

        return (matches, H, status)
    else:
        return None


def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img



#stitching starts here
DEFAULT_FEATURE_EXTRACTOR = 'orb' # one of 'sift', 'surf', 'brisk', 'orb'
DEFAULT_FEATURE_MATCHER = 'bf'

#pass in the grayscale of images that need to be stitched (these will be the top down warped images)
#needs both grayscale and normal to compute the stitching matrix properly
def calculateStitchingMatrix(img1, img1Gray, img2, img2Gray, feature_extractor=DEFAULT_FEATURE_EXTRACTOR, feature_matching=DEFAULT_FEATURE_MATCHER):
    kpsA, featuresA = detectAndDescribe(img1Gray, method=feature_extractor)
    kpsB, featuresB = detectAndDescribe(img2Gray, method=feature_extractor)
    
    # display the keypoints and features detected on both images
    # fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,8), constrained_layout=False)
    # ax1.imshow(cv2.drawKeypoints(img1,kpsA,None,color=(0,255,0)))
    # ax1.set_xlabel("(a)", fontsize=14)
    # ax2.imshow(cv2.drawKeypoints(img2,kpsB,None,color=(0,255,0)))
    # ax2.set_xlabel("(b)", fontsize=14)
    # plt.show()

    if feature_matching == 'bf':
        matches = matchKeyPointsBF(featuresA, featuresB, method=feature_extractor)
        img3 = cv2.drawMatches(img1, kpsA, img2, kpsB, matches[:100],
                            None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    elif feature_matching == 'knn':
        matches = matchKeyPointsKNN(featuresA, featuresB, ratio=0.75, method=feature_extractor)
        img3 = cv2.drawMatches(img1, kpsA, img2, kpsB, np.random.choice(matches,100),
                            None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    M = getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh=4)
    if M is None:
        print("Error! Stitching Homography Matrix couldnt be calculated for kpsA and kpsB")
    (matches, Hstitch, status) = M
    return Hstitch


def warpTwoImages(img1, img2, H):
    '''warp img2 to img1 with homograph H'''
    black_mask = np.all(img1==np.array([0,0,0]).reshape(1,1,3), axis = 2)
    nonBlackPixels = np.where(black_mask == False)
    
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    # print('these are the points = {}'.format(pts))
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    # print('These are the calculated extrema ',xmin,xmax,ymin,ymax)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate

    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin)).astype(np.int32)
    result_nonblack_mask = (~np.all(result==np.array([0,0,0]).reshape(1,1,3), axis = 2)).astype(np.uint8)
    result_nonblack_mask[t[1]+nonBlackPixels[0],t[0]+nonBlackPixels[1]] += 1
    result_nonblack_mask[result_nonblack_mask == 0] = 1

    result[t[1]+nonBlackPixels[0],t[0]+nonBlackPixels[1],:] += img1[nonBlackPixels[0],nonBlackPixels[1],:]
    result[:, :, 0] //= result_nonblack_mask
    result[:, :, 1] //= result_nonblack_mask
    result[:, :, 2] //= result_nonblack_mask
    return result.astype(np.uint8)

def warpCudaTwoImages(img1, img2, H):
    '''warp img2 to img1 with homograph H'''
    #img1 and img2 are of type gpuMat while img1np and img2np are images stored as np arrays
    img1np = img1.download()
    img2np = img2.download()

    mask = np.all(img1np==np.array([0,0,0]).reshape(1,1,3), axis = 2)

    blackPixels = np.where(mask == False)
    
    h1, w1 = img1.size()[:2]    # size() bc cpu mat

    h2, w2 = img2.size()[:2]

    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    print('these are the points = {}'.format(pts))
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    print('These are the calculated extrema ',xmin,xmax,ymin,ymax)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate
    
    resultMat = cv2.cuda.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))

    result = resultMat.download()
    result[t[1]+blackPixels[0],t[0]+blackPixels[1],:] = img1np[blackPixels[0],blackPixels[1],:]

    gpuFrame = cv2.cuda_GpuMat()
    gpuFrame.upload(result)
    return gpuFrame



#Other Functions
def undistortImage(img, mtx, dist, newcameramtx, roi):
    h,  w = img.shape[:2]
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst

def computeUndistortMaps(mtx, dist, new_cam_mtx):
    map1, map2 = cv2.initUndistortRectifyMap(cameraMatrix=mtx, distCoeffs=dist, R=np.identity(3), m1type = cv2.CV_32FC1, newCameraMatrix=new_cam_mtx, size=(RES_HEIGHT, RES_WIDTH))
    return (map1, map2)

#not working at the moment, under review
def undistortCudaImage(imgMat, map1, map2):
    dst = cv2.cuda.remap(imgMat, np.asnumpy(map1), np.asnumpy(map2), borderMode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_NEAREST)
    return dst
