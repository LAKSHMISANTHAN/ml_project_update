from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import cv2

class CentroidTracker():
	def __init__(self, maxDisappeared=30):
		self.pics = []
		self.nextObjectID = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()
		self.maxDisappeared = maxDisappeared

	def add(self, centroid):
		self.objects[self.nextObjectID] = centroid
		self.disappeared[self.nextObjectID] = 0
		self.nextObjectID += 1

	def delete(self, objectID):
		del self.objects[objectID]
		del self.disappeared[objectID]

	def update(self, rects, frame):
		if len(rects) == 0:
			for objectID in self.disappeared.keys():
				self.disappeared[objectID] += 1
				if self.disappeared[objectID] > self.maxDisappeared:
					self.delete(objectID)
			return self.objects

		inputCentroids = np.zeros((len(rects), 2), dtype="int")

		for (i, (startX, startY, endX, endY)) in enumerate(rects):
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			inputCentroids[i] = (cX, cY)

		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.add(inputCentroids[i])
			for (i, (startX, startY, endX, endY)) in enumerate(rects):
				margin = 1
				w = endX-startX
				h = endY-startY
				img_h, img_w, _ = np.shape(frame)
				startX = max(int(startX - margin * w), 0)
				startY = max(int(startY - margin * h), 0)
				endX = min(int(endX + margin * w), img_w - 1)
				endY = min(int(endY + margin * h), img_h - 1)
				self.pics.append([i,frame[startY:endY, startX:endX]])

		else:
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())
			D = dist.cdist(np.array(objectCentroids), inputCentroids)
			rows = D.min(axis=1).argsort()

			cols = D.argmin(axis=1)[rows]

			usedRows = set()
			usedCols = set()

			for (row, col) in zip(rows, cols):
				if row in usedRows or col in usedCols:
					continue

				objectID = objectIDs[row]
				self.objects[objectID] = inputCentroids[col]
				self.disappeared[objectID] = 0

				usedRows.add(row)
				usedCols.add(col)

			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)
			if D.shape[0] >= D.shape[1]:
				for row in unusedRows:

					objectID = objectIDs[row]
					self.disappeared[objectID] += 1
					if self.disappeared[objectID] > self.maxDisappeared:
						self.delete(objectID)

			else:
				for col in unusedCols:
					margin = 1
					self.add(inputCentroids[col])
					(startX, startY, endX, endY) = rects[col]
					w = endX-startX
					h = endY-startY
					img_h, img_w, _ = np.shape(frame)
					startX = max(int(startX - margin * w), 0)
					startY = max(int(startY - margin * h), 0)
					endX = min(int(endX + margin * w), img_w - 1)
					endY = min(int(endY + margin * h), img_h - 1)
					ind = self.nextObjectID-1
					self.pics.append([ind,frame[startY:endY, startX:endX]])

		return self.objects
