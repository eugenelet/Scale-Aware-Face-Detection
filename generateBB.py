import os
import sys
import cv2
import dlib
from imutils import face_utils
import argparse
import numpy as np
import imutils
import scipy.io as sio
import pickle

def drawLandmark(img, shape):
	_x = []
	_y = []
	bbox_info = np.empty(3)
	for idx, (x, y) in enumerate(shape):
		if idx == 36 or idx == 45 or idx == 33 or idx == 48 or idx == 54: 
			cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
			_x.append(x)
			_y.append(y)
	bbox_info[0] = np.mean(_x)
	bbox_info[1] = np.mean(_y)
	bbox_info[2] = np.std(_y)
	# print(bbox_info, _x, _y)
	x1 = int(bbox_info[0] - bbox_info[2]*2.5)
	y1 = int(bbox_info[1] - bbox_info[2]*2.5)
	x2 = int(bbox_info[0] + bbox_info[2]*2.5)
	y2 = int(bbox_info[1] + bbox_info[2]*2.5)
	# cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)

	return x1, y1, x2, y2


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])


# labeled bbox by imdb dataset
bbox = sio.loadmat('imdb.mat')['imdb']
bbox_path = bbox['full_path'][0][0][0]


# Directory of imdb dataset
data_dir = 'imdb/'

# File to store BBox
f = open('bbox.p', 'wb')

for dirname in os.listdir(data_dir):
	for filename in os.listdir(data_dir + dirname):
		# For tracking purpose
		print(dirname + '/' + filename)
		
		bbox_detected = []
		accept_box = True
		img = cv2.imread(data_dir + dirname + '/' + filename)
		ori_img = np.copy(img)

		idx = np.where(bbox_path == dirname + '/' + filename)
		bbox_loc = bbox['face_location'][0][0][0][idx][0][0]
		
		# Only accept box with area < 2/5 of whole image
		width = img.shape[1]
		height = img.shape[0]
		bbox_area = (int(bbox_loc[2]) - int(bbox_loc[0])) * (int(bbox_loc[3]) - int(bbox_loc[1]))
		img_area = float(width) * float(height)
		if bbox_area / img_area > 0.4:
			accept_box = False


		# Projected bbox location
		scale_ratio = 500.0 / float(width)
		bbox_loc = bbox_loc * scale_ratio

		# Transform image to fit predictor (speed and accuracy)
		img = imutils.resize(img, width=500)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# Show bounding box provided by IMDB (w/o filter)
		# cv2.rectangle(img, (int(bbox_loc[0]), int(bbox_loc[1])), (int(bbox_loc[2]), int(bbox_loc[3])), (255, 255, 255), 2)

		# detect faces in the grayscale image
		rects = detector(gray, 1)

		# loop over the face detections
		for (i, rect) in enumerate(rects):
			# convert dlib's rectangle to a OpenCV-style bounding box
			# [i.e., (x, y, w, h)], then draw the face bounding box
			(x, y, w, h) = face_utils.rect_to_bb(rect)
			# Show bounding box by HoG detector
			# cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

			# Eliminate bounding box based on overlap
			xx1 = max(bbox_loc[0], x)
			yy1 = max(bbox_loc[1], y)
			xx2 = min(bbox_loc[2], x + w)
			yy2 = min(bbox_loc[3], y + h)
			w_intersect = max(0, xx2 - xx1 + 1)
			h_intersect = max(0, yy2 - yy1 + 1)
			overlap = float(w_intersect * h_intersect) / (w * h)
			# print(overlap)
			if overlap > 0.2:
				accept_box = False

			# determine the facial landmarks for the face region, then
			# convert the facial landmark (x, y)-coordinates to a NumPy
			# array
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)

			# loop over the (x, y)-coordinates for the facial landmarks
			# and draw them on the image
			# for (x, y) in shape:
				# cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
			bbox_detected.append(drawLandmark(img, shape))


		width = img.shape[1]
		height = img.shape[0]
		# print('img_width: %d img_height: %d bbox_width: %d bbox_height: %d' % (width, height, (int(bbox_loc[2]) - int(bbox_loc[0])),
		 # (int(bbox_loc[3]) - int(bbox_loc[1]))))

		# Only accept box if it doesn't occupy > 2/5 of an image AND has overlap < 1/5 of BBox found using HoG detector
		if accept_box:
			# Show bounding box provided by IMDB (after filter)
			# cv2.rectangle(img, (int(bbox_loc[0]), int(bbox_loc[1])), (int(bbox_loc[2]), int(bbox_loc[3])), (0, 0, 255), 2)
			rect_d = dlib.rectangle(int(bbox_loc[0]), int(bbox_loc[1]), int(bbox_loc[2]), int(bbox_loc[3]))
			shape_d = predictor(gray, rect_d)
			shape_d = face_utils.shape_to_np(shape_d)
			bbox_detected.append(drawLandmark(img, shape_d))

			# for (x, y) in shape_d:
				# cv2.circle(img, (x, y), 1, (255, 0, 0), -1)
		for idx, (x1, y1, x2, y2) in enumerate(bbox_detected):
			x1, y1, x2, y2 = int(x1 / scale_ratio), int(y1 / scale_ratio), int(x2 / scale_ratio), int(y2 / scale_ratio)
			bbox_detected[idx] = x1, y1, x2, y2
			cv2.rectangle(ori_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

		if(len(bbox_detected) != 0):
			pickle.dump((dirname + '/' + filename, bbox_detected), f)
		# cv2.imshow('image', ori_img)
		# cv2.waitKey(0)

f.close()

