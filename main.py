from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
import os
#import matplotlib.pyplot as mplp
#import seaborn as sb
import scipy as sp
import pandas as pd
#import h5py

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

## Deal with Tensorflow version issues (only necessary with certain versions of Tensorflow; uncomment if useful):
#f = h5py.File("/Users/dave/Documents/Work/Ellis et al 2024b/fibrillae/fibrillae/fibrillae_erection_rpi/teachable_machine/model_v6-5/converted_keras/keras_Model.h5", mode="r+")
#model_config_string = f.attrs.get("model_config")
#if model_config_string.find('"groups": 1,') != -1:
#    model_config_string = model_config_string.replace('"groups": 1,', '')
#f.attrs.modify('model_config', model_config_string)
#f.flush()
#model_config_string = f.attrs.get("model_config")
#assert model_config_string.find('"groups": 1,') == -1

# Load the model
model = load_model('/keras_model/keras_Model.h5', compile=False)

# Load the labels
class_names = open('/keras_model/labels.txt', 'r').readlines()

def getMaxima(data, n_samples=100, bw_method=0.01):
    kde = sp.stats.gaussian_kde(data, bw_method=bw_method)
    x_grid = np.linspace(min(data) - 1, max(data) + 1, n_samples)
    density = kde.evaluate(x_grid)
    peaks_indices, _ = sp.signal.find_peaks(density)
    maxima = x_grid[peaks_indices]
    
    return maxima
    
def showImage(img):
    '''
    Hack for macOS Monterey to get OpenCV to show an image (e.g. "close"
    returned by findSquares). Displays a window with the image that is closed
    by pushing any key.

    Parameters
    ----------
    img : a cv2 image

    '''
    cv2.namedWindow('image')
    cv2.imshow('image', img)
    cv2.waitKey(0) # 5 sec delay before image window closes
    cv2.destroyWindow('image')
    cv2.waitKey(1)

def findSquares(image_path, previous_xywh=None, threshold=90, kernel_reduction_factor=3, min_area=50000, max_area=110000, buffer=40, flip=False):
    '''
    Finds wells (ROIs) in an imaged plate, then create a dictionary containing
    cropped images of each well.

    Parameters
    ----------
    image_path              : String.
    previous_xywh           : Dictionary. Only required if you wish to predict wells for frames where all 24 were not found
    threshold               : Int (max 255). The threshold for binary conversion of image.
    kernel_reduction_factor : Int. Effects the kernal used to sharpening image.
    min_area                : Int. Minimus size of a well.
    max_area                : Int. Maximum size of a well.
    buffer                  : Int. Amount to pad around each identified well.
    flip                    : Bool. Rotate the image 180 degrees (useful if plate imaged upside down)

    Returns
    -------
    Dictionary of ROIs (cropped well images); Image post-processing/threshold;
    Image showing all the ROIs as boxes; Dictionary of x, y, width, height for
    each ROI.

    '''
    
    # Load image, grayscale, median blur, sharpen image
    img = cv2.imread(image_path)
    
    if flip==True:
        img = cv2.rotate(img, cv2.ROTATE_180)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    sharpen_kernel = np.array([[ 0,-1, 0], 
                               [-1, 6,-1], 
                               [ 0,-1, 0]])
    sharpen = cv2.filter2D(blurred, -1, sharpen_kernel/kernel_reduction_factor)
    
    # Threshold and morph close
    thresh = cv2.threshold(sharpen, threshold, 255, cv2.THRESH_BINARY_INV)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    close = 255 - cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2) # <- this is both closed (i.e. definite boundaries of black or white) and inverted (255 - ...)
    
    # Find contours and filter using threshold area
    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    img_copy = img.copy()
    roi = {} # <- dictonary of images of each ROI, where the [x,y] coords are the keys (these can then be converted to positions e.g. A1)
    xywh = {}
    for c in cnts:
        area = cv2.contourArea(c)
        if area > min_area and area < max_area:
            x,y,w,h = cv2.boundingRect(c)
            x -= int(buffer/2)
            y -= int(buffer/2)
            w += buffer
            h += buffer
            ROI = img[y:y+h, x:x+w]
            if len(ROI[0])==0:
                continue
            roi[str(x) + ',' + str(y)] = cv2.cvtColor(ROI, cv2.COLOR_BGR2RGB)
            xywh[str(x) + ',' + str(y)] = [x, y, w, h]
    
    if len(set(roi.keys())) != 24:
        if previous_xywh!=None:
            roi = {}
            xywh = {}
            img_copy = img.copy()
            for k in previous_xywh.keys():
                x = previous_xywh[k][0]
                y = previous_xywh[k][1]
                w = previous_xywh[k][2]
                h = previous_xywh[k][3]
                ROI = img[y:y+h, x:x+w]
                if len(ROI[0])==0:
                    continue
                roi[str(x) + ',' + str(y)] = cv2.cvtColor(ROI, cv2.COLOR_BGR2RGB)
                xywh[str(x) + ',' + str(y)] = [x, y, w, h]

        else:
            raise Exception('found ' + str(len(roi.keys())) + ' wells rather than 24.')
    
    for k in xywh.keys():
        x = xywh[k][0]
        y = xywh[k][1]
        w = xywh[k][2]
        h = xywh[k][3]
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (36,255,12), 2)
        
    x = []
    y = []
    for k in roi.keys():
        i, j = int(k.split(',')[0]), int(k.split(',')[1])
        x.append(i)
        y.append(j)
    
    lookup1 = pd.DataFrame([['A','A','A','A','B','B','B','B','C','C','C','C','D','D','D','D','E','E','E','E','F','F','F','F'], sorted(y)], dtype=str).T
    lookup2 = pd.DataFrame([[ 1 , 1 , 1 , 1 , 1 , 1 , 2 , 2 , 2 , 2 , 2 , 2 , 3 , 3 , 3 , 3 , 3 , 3 , 4 , 4 , 4 , 4 , 4 , 4 ], sorted(x)], dtype=str).T
    
    coordinates = {}
    for k in roi.keys():
        coord = lookup1.loc[lookup1[1]==k.split(',')[1]].values[0,0]
        coord = coord + lookup2.loc[lookup2[1]==k.split(',')[0]].values[0,0]
        coordinates[k] = coord
        cv2.putText(img_copy, coord, [int(k.split(',')[0]), int(k.split(',')[1])], 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (36,255,12), thickness=2, 
                    lineType=cv2.LINE_AA)
    roi = {coordinates[k]:roi[k] for k in roi}
    xywh = {coordinates[k]:xywh[k] for k in xywh}
    
    if len(set(roi.keys())) != 24:
        if previous_xywh!=None:
            roi = {}
            xywh = {}
            img_copy = img.copy()
            for k in previous_xywh.keys():
                x = previous_xywh[k][0]
                y = previous_xywh[k][1]
                w = previous_xywh[k][2]
                h = previous_xywh[k][3]
                ROI = img[y:y+h, x:x+w]
                if len(ROI[0])==0:
                    continue
                roi[str(x) + ',' + str(y)] = cv2.cvtColor(ROI, cv2.COLOR_BGR2RGB)
                xywh[str(x) + ',' + str(y)] = [x, y, w, h]

            coordinates = {}
            for k in roi.keys():
                coord = lookup1.loc[lookup1[1]==k.split(',')[1]].values[0,0]
                coord = coord + lookup2.loc[lookup2[1]==k.split(',')[0]].values[0,0]
                coordinates[k] = coord
                cv2.putText(img_copy, coord, [int(k.split(',')[0]), int(k.split(',')[1])], 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (36,255,12), thickness=2, 
                            lineType=cv2.LINE_AA)
            roi = {coordinates[k]:roi[k] for k in roi}
            xywh = {coordinates[k]:xywh[k] for k in xywh}    
            
        else:
            raise Exception('found ' + str(len(roi.keys())) + ' wells rather than 24.')
    
    return roi, close, img_copy, xywh

#==============================================================================
# Run the below when current working directory is set to folder containing images:
#==============================================================================

os.mkdir('./tmp/')
data = pd.DataFrame(columns=['file', 'well', 'class', 'confidence'])

threshold = 90 # <- for findSquares()
flip = False # <- for findSquares() Set to True if plate is imaged upside down.

files = [i for i in os.listdir() if i[-4:]=='.jpg']

for fdx, f in enumerate(sorted(files)):
    
    try:
        roi, _, img_copy, _ = findSquares(f, threshold=threshold, flip=flip)
    except Exception:
        continue
    
    cv2.imwrite('./tmp/{}_segments.jpg'.format(f[:-4]), img_copy)
    
    for r in roi.keys():
        
        # Resize the raw image into (224-height,224-width) pixels
        img = cv2.resize(roi[r], (224, 224), interpolation=cv2.INTER_AREA)

        # Make the image a numpy array and reshape it to the models input shape.
        img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)

        # Normalize the image array
        img = (img / 127.5) - 1

        # Predicts the model
        prediction = model.predict(img)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        _ = pd.DataFrame({'file':[f], 'well':[r], 'class':[class_name], 'confidence':[confidence_score]})
        data = pd.concat([data, _]).reset_index(drop=True)

        # Print prediction and confidence score
        print('Class:', class_name[2:], end='')
        print('Confidence Score:', str(np.round(confidence_score * 100))[:-2], '%')
        
    data.to_csv('predictions.txt', sep='\t', header=True, index=False)

#==============================================================================
# To correct for a slowly shifting camera, uncomment and run the below instead:
#==============================================================================

# xywh = None # <- initialise
# shift = 1 # <- how much to shift the y position each iteration (although see the relevant section in the loop...)

# os.mkdir('./tmp/')
# data = pd.DataFrame(columns=['file', 'well', 'class', 'confidence'])

# threshold = 90 # <- for findSquares()
# flip = False # <- for findSquares() Set to True if plate is imaged upside down.

# files = [i for i in os.listdir() if i[-4:]=='.jpg']

# for fdx, f in enumerate(sorted(files)):
        
#     try:
#         roi, _, img_copy, xywh = findSquares(f, previous_xywh=xywh, threshold=threshold, flip=flip)
#     except Exception:
#         continue
    
#     if flip==True:
#         #shift_ = shift + fdx % 2 # <- shift down every other image
#         shift_ = shift + fdx % 3 % 2 # <- shift down every 2/3 images
#     else:
#         #shift_ = shift - fdx % 2 # <- shift up every other image
#         shift_ = shift - fdx % 3 % 2 # <- shift up every 2/3 images
    
#     for i in xywh.keys():
#         xywh[i] = [xywh[i][0], xywh[i][1]-shift_, xywh[i][2], xywh[i][3]]
    
#     cv2.imwrite('./tmp/{}_segments.jpg'.format(f[:-4]), img_copy)
    
#     for r in roi.keys():
        
#         # Resize the raw image into (224-height,224-width) pixels
#         img = cv2.resize(roi[r], (224, 224), interpolation=cv2.INTER_AREA)

#         # Make the image a numpy array and reshape it to the models input shape.
#         img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)

#         # Normalize the image array
#         img = (img / 127.5) - 1

#         # Predicts the model
#         prediction = model.predict(img)
#         index = np.argmax(prediction)
#         class_name = class_names[index]
#         confidence_score = prediction[0][index]

#         _ = pd.DataFrame({'file':[f], 'well':[r], 'class':[class_name], 'confidence':[confidence_score]})
#         data = pd.concat([data, _]).reset_index(drop=True)

#         # Print prediction and confidence score
#         print('Class:', class_name[2:], end='')
#         print('Confidence Score:', str(np.round(confidence_score * 100))[:-2], '%')
        
#     data.to_csv('predictions.txt', sep='\t', header=True, index=False)