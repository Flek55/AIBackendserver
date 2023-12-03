import enum
import json
import os
from random import randint

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, make_response
from flask_ngrok import run_with_ngrok

stroka = ""

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
run_with_ngrok(app)


# class Sort(enum.Enum):
#     Osot = 0
#     Bodyak = 1
#     Shavel = 2


class Stage(enum.Enum):
    розетка = 0
    стеблевание = 1
    цветение = 2


height = 500
width = 500


class Cnt_data:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


def randomword(n):
    result = ''
    for i in range(n):
        result += chr(randint(97, 122))
    return result


# функция, предварительно обрабатывающая изображение, прежде чем отправить его в нейросеть
def image_preprocessing_v2(pathToImage):
    out_img_dim = (width, height)
    lower_green = np.array([35, 35, 35])
    upper_green = np.array([85, 255, 255])
    image = cv2.imread(pathToImage)
    input_img_area = image.shape[0] * image.shape[1]

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_green, upper_green)
    blurred = cv2.GaussianBlur(mask, (3, 3), 0)
    contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sep_contours = []
    cnt_data = []

    if len(contours) != 0:
        for cnt in contours:
            if cv2.contourArea(cnt) / input_img_area > 0.012:
                brect = cv2.boundingRect(cnt)
                x, y, w, h = brect
                cv2.rectangle(mask, brect, (255, 255, 255), 3)  # DEMO
                sep_contours.append(cnt)
                cnt_data.append(Cnt_data(x, y, w, h))

    output_img_arr = []
    output_mask_arr = []

    if len(sep_contours) != 0:
        for cnt in sep_contours:
            brect = cv2.boundingRect(cnt)
            x, y, w, h = brect
            cropped_image = image[y:y + h, x:x + w]
            output_image = cv2.resize(cropped_image, out_img_dim, interpolation=cv2.INTER_AREA)
            output_img_arr.append(output_image)
            stencil = np.zeros(mask.shape).astype(mask.dtype)
            cv2.fillPoly(stencil, [cnt], (255, 255, 255))
            output_mask_arr.append(cv2.bitwise_and(mask, stencil))

    # cv2.imshow("Original image", image) #DEMO
    # cv2.imshow("Mask", mask) #DEMO

    # DEMO
    # i=0
    # for img in output_img_arr:
    #     i+=1
    #     cv2.imshow("IMG - " + str(i), img)

    return cnt_data, output_img_arr, mask, output_mask_arr


model1 = tf.keras.models.load_model('AI/cnn1.0(0.65).h5', compile=False)
model2 = tf.keras.models.load_model('AI/cnn2.0(0.60)-BW.h5', compile=False)
model3 = tf.keras.models.load_model('AI/cnn3.0(0.67)-BW.h5', compile=False)
models = [model1, model2, model3]
ensemble_input = tf.keras.Input(shape=(width, height, 3))
model_outputs = [model(ensemble_input) for model in models]  # type: ignore
ensemble_output = tf.keras.layers.Average()(model_outputs)
ensemble_model = tf.keras.Model(inputs=ensemble_input, outputs=ensemble_output)

BodyakS = tf.keras.models.load_model('AI/Bodyak(0.94).h5', compile=False)
OsotS = tf.keras.models.load_model('AI/Osot(0.97).h5', compile=False)
ShavelS = tf.keras.models.load_model('AI/Shavel(0.99).h5', compile=False)


@app.route('/cookie/')
def cookie():
    global stroka
    name = request.cookies.get('user')
    if name == None:
        res = make_response("Setting a cookie")
        stroka = randomword(10)
        while stroka in os.listdir():
            stroka = randomword(10)
        res.set_cookie("user", stroka)
        if not os.path.isdir("outputs\\" + stroka):
            os.mkdir("outputs\\" + stroka)
            os.mkdir("outputs\\" + stroka + "\\1")
        return res
    else:
        return "OK"


@app.route('/', methods=['GET', 'POST'])
def start():
    name = request.cookies.get('user')
    uploaded_file = request.files['file']
    uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], "data.jpg"))
    data, images, mask, masks = image_preprocessing_v2('uploads/data.jpg')

    cv2.imwrite("outputs\\" + stroka + "\\" + str(int(os.listdir("outputs\\" + stroka)[-1])) + "\\Mask.jpg", mask)
    mask = cv2.cvtColor(
        cv2.imread("outputs\\" + stroka + "\\" + str(int(os.listdir("outputs\\" + stroka)[-1])) + "\\Mask.jpg"),
        cv2.COLOR_RGB2GRAY)
    Y, X = mask.shape

    sort_predictions = []
    stage_predictions = []
    predictions = []
    for i in range(0, len(images)):
        cv2.imwrite("outputs\\" + stroka + "\\" + str(int(os.listdir("outputs\\" + stroka)[-1])) + "\\img.jpg",
                    images[i])
        prediction = ensemble_model.predict(cv2.imread(
            "outputs\\" + stroka + "\\" + str(int(os.listdir("outputs\\" + stroka)[-1])) + "\\img.jpg").reshape(-1,
                                                                                                                width,
                                                                                                                height,
                                                                                                                3))
        ans = np.argmax(prediction)
        img_array = np.array(masks[i]).tolist()
        for y1 in range(len(img_array)):
            for x1 in range(len(img_array[y1])):
                if img_array[y1][x1] >= 150 and data[i].x <= x1 <= data[i].x + data[i].w and data[i].y <= y1 <= data[
                    i].y + data[i].h:
                    img_array[y1][x1] = 1
                else:
                    img_array[y1][x1] = 0
        if ans == 0:
            print("bodyak")
            sort = "1"
            sort_predictions.append(sort)
            prediction = BodyakS.predict(cv2.imread(
                "outputs\\" + stroka + "\\" + str(int(os.listdir("outputs\\" + stroka)[-1])) + "\\img.jpg").reshape(-1,
                                                                                                                    width,
                                                                                                                    height,
                                                                                                                    3))
            ans = np.argmax(prediction) % 10
            print(sort, Stage(ans).name)
            mask = cv2.putText(mask, (sort + " - " + str(ans)), (data[i].x, data[i].y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                               2.5, (255, 255, 255), 5)
            predictions.append({"bounding_box": [(data[i].x + data[i].w / 2) / X, (data[i].y + data[i].h / 2) / Y,
                                                 data[i].w / X, data[i].h / Y],
                                "class": sort,
                                "development_stage": int(Stage(ans).value),
                                "mask": img_array})
            cv2.imwrite(
                "outputs\\" + stroka + "\\" + str(int(os.listdir("outputs\\" + stroka)[-1])) + f"\\img[{i}].jpg",
                cv2.imread("outputs\\" + stroka + "\\" + str(int(os.listdir("outputs\\" + stroka)[-1])) + "\\img.jpg"))
            cv2.imwrite(
                "outputs\\" + stroka + "\\" + str(int(os.listdir("outputs\\" + stroka)[-1])) + f"\\mask[{i}].jpg",
                masks[i])
        elif ans == 2:
            print("Shavel")
            sort = "2"
            sort_predictions.append(sort)
            prediction = ShavelS.predict(cv2.imread("img.jpg").reshape(-1, width, height, 3))
            ans = np.argmax(prediction) % 10
            print(sort, Stage(ans).name)
            mask = cv2.putText(mask, (sort + " - " + str(ans)), (data[i].x, data[i].y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                               2.5, (255, 255, 255), 5)
            predictions.append({"bounding_box": [(data[i].x + data[i].w / 2) / X, (data[i].y + data[i].h / 2) / Y,
                                                 data[i].w / X, data[i].h / Y],
                                "class": sort,
                                "development_stage": int(Stage(ans).value),
                                "mask": img_array})
            cv2.imwrite(
                "outputs\\" + stroka + "\\" + str(int(os.listdir("outputs\\" + stroka)[-1])) + f"\\img[{i}].jpg",
                cv2.imread("img.jpg"))
            cv2.imwrite(
                "outputs\\" + stroka + "\\" + str(int(os.listdir("outputs\\" + stroka)[-1])) + f"\\mask[{i}].jpg",
                masks[i])
        else:
            print("Osot")
            sort = "0"
            sort_predictions.append(sort)
            prediction = OsotS.predict(cv2.imread(
                "outputs\\" + stroka + "\\" + str(int(os.listdir("outputs\\" + stroka)[-1])) + "\\img.jpg").reshape(-1,
                                                                                                                    width,
                                                                                                                    height,
                                                                                                                    3))
            ans = np.argmax(prediction) % 10
            print(sort, Stage(ans).name)
            mask = cv2.putText(mask, (sort + " - " + str(ans)), (data[i].x, data[i].y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                               2.5, (255, 255, 255), 5)
            predictions.append({"bounding_box": [(data[i].x + data[i].w / 2) / X, (data[i].y + data[i].h / 2) / Y,
                                                 data[i].w / X, data[i].h / Y],
                                "class": sort,
                                "development_stage": int(Stage(ans).value),
                                "mask": img_array})
            cv2.imwrite(
                "outputs\\" + stroka + "\\" + str(int(os.listdir("outputs\\" + stroka)[-1])) + f"\\img[{i}].jpg",
                cv2.imread("img.jpg"))
            cv2.imwrite(
                "outputs\\" + stroka + "\\" + str(int(os.listdir("outputs\\" + stroka)[-1])) + f"\\mask[{i}].jpg",
                masks[i])

    osot_n = 0
    bodyak_n = 0
    shavel_n = 0
    for s in sort_predictions:
        if s == "1":
            bodyak_n += 1
        elif s == "2":
            shavel_n += 1
        else:
            osot_n += 1
    print("Бодяк:", bodyak_n, "\nЩавель:", shavel_n, "\nОсот:", osot_n)
    # print(predictions)
    answer = {"predictions": predictions,
              "osot_num": osot_n,
              "bodyak_num": bodyak_n,
              "horse_sorrel_num": shavel_n}

    # print(answer)
    # cv2.imshow("Mask", mask)
    cv2.imwrite("Mask.jpg", mask)

    with open("outputs\\" + stroka + "\\" + str(int(os.listdir("outputs\\" + stroka)[-1])) + "\\data.json", 'w') as f:
        json.dump(answer, f)
    cv2.imwrite("outputs\\" + stroka + "\\" + str(int(os.listdir("outputs\\" + stroka)[-1])) + "\\mask.jpg", mask)
    return jsonify(answer)


@app.route('/fotochka', methods=['GET', 'POST'])
def send_photo():
    file = open(
        "outputs\\" + stroka + "\\" + str(int(int(os.listdir("outputs\\" + stroka)[-1]))) + "\\" + "mask.jpg", 'rb')
    return file


if __name__ == '__main__':
    app.run()
