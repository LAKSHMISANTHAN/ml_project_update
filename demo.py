import cv2
import numpy as np
from tensorflow.keras import applications
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense

def get_model():
    base_model = getattr(applications, "EfficientNetB3")(
        include_top=False,
        input_shape=(224, 224, 3),
        pooling="avg"
    )
    features = base_model.output
    pred_gender = Dense(units=2, activation="softmax", name="pred_gender")(features)
    pred_age = Dense(units=101, activation="softmax", name="pred_age")(features)
    model = Model(inputs=base_model.input, outputs=[pred_gender, pred_age])
    return model

def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.8, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

def predict_cv(image):
    ageProto="age_deploy.prototxt"
    ageModel="age_net.caffemodel"
    genderProto="gender_deploy.prototxt"
    genderModel="gender_net.caffemodel"
    MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
    ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList=['Male','Female']

    ageNet=cv2.dnn.readNet(ageModel,ageProto)
    genderNet=cv2.dnn.readNet(genderModel,genderProto)

    blob=cv2.dnn.blobFromImage(image, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
    genderNet.setInput(blob)
    genderPreds=genderNet.forward()
    gender=genderList[genderPreds[0].argmax()]


    ageNet.setInput(blob)
    agePreds=ageNet.forward()
    age=ageList[agePreds[0].argmax()]
    return [gender,age]




def predict_gen(image):
    margin = 0.4
    weight_file = "EfficientNetB3_224_weights.11-3.44.hdf5"
    img_size = 224
    model = get_model()
    model.load_weights(weight_file)
    img = image
    input_img = image
    # img_h, img_w, _ = np.shape(input_img)
    faces = np.empty((1, img_size, img_size, 3))
    faces[0] = cv2.resize(img,(img_size,img_size))
    # detect faces using dlib detector
    # detected = detector(input_img, 1)
    # print(detected)
    # faces = np.empty((len(detected), img_size, img_size, 3))

    # if len(detected) > 0:
    #     for i, d in enumerate(detected):
    #         x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
    #         xw1 = max(int(x1 - margin * w), 0)
    #         yw1 = max(int(y1 - margin * h), 0)
    #         xw2 = min(int(x2 + margin * w), img_w - 1)
    #         yw2 = min(int(y2 + margin * h), img_h - 1)
    #         cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    #         # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
    #         faces[i] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1], (img_size, img_size))

        # predict ages and genders of the detected faces
    results = model.predict(faces)
    predicted_genders = results[0]
    ages = np.arange(0, 101).reshape(101, 1)
    predicted_ages = results[1].dot(ages).flatten()
    return predicted_genders,predicted_ages

        #     # draw results
        #     for i, d in enumerate(detected):
        #         label = "{}, {}".format(int(predicted_ages[i]),
        #                                 "M" if predicted_genders[i][0] < 0.5 else "F")
        #         draw_label(img, (d.left(), d.top()), label)
        #
        # cv2.imshow("result", img)
        # key = cv2.waitKey(30)
        #
        # if key == 27:  # ESC
        #     break
