import pickle
import cv2

data_dir = 'imdb/'

f = open('bbox.p', 'rb')
while True:
	try:
		file_loc, bbox = pickle.load(f)
		print(file_loc, bbox)
		img = cv2.imread(data_dir + file_loc)
		for x1, y1, x2, y2 in bbox:
			cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
		cv2.imshow('image', img)
		cv2.waitKey(0)
	except EOFError:
		break