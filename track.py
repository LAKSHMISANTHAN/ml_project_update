import numpy as np
import imutils
import cv2
from demo import predict_gen
from demo import predict_cv
from centroidtracker import CentroidTracker
from imutils.video import FPS

def track(video_name):
    ct = CentroidTracker()
    (H, W) = (None, None)
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
    print("[INFO] starting video stream...")
    stream = cv2.VideoCapture(video_name)
    fps = FPS().start()

    while True:
        (grabbed,frame) = stream.read()
        if not grabbed:
            break
        # frame = imutils.resize(frame, width=400)

        if W is None or H is None:
            (H, W) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),
            (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        print(detections.shape[2])
        rects = []

        for i in range(0, detections.shape[2]):
            if detections[0, 0, i, 2] > 0.85:
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])

                rects.append(box.astype("int"))
                (startX, startY, endX, endY) = box.astype("int")
                det = frame[startY:endY, startX:endX]
                # cv2.rectangle(frame, (startX, startY), (endX, endY),
                #     (0, 255, 0), 2)

        # update our centroid tracker using the computed set of bounding
        # box rectangles
                objects = ct.update(rects,frame)

        # loop over the tracked objects
            if len(rects)!=0:
                for (objectID, centroid) in objects.items():

                    text = "ID {}".format(objectID)

                    cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF
                fps.update()

                if key == ord("q"):
                    break
    cv2.destroyAllWindows()
    fps.stop()
    return ct.pics,ct.objects

def predict(video_name):
    a,b = track(video_name)
    m = 0
    f = 0
    age = []
    s = set()
    for i in b:
        s.add(i)
    print(s)
    for i in a:
        if i[0] in s:
            gen,age_res = predict_gen(i[1])
            if gen[0][0]<0.5:
                m = m+1
            else:
                f = f+1
            age.append(int(age_res[0]))
    return [m,f,age]
# res = predict("video.mp4")
# print(res)
