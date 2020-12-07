import numpy as np
# from pygame import mixer
# import time
import cv2
from tkinter import *
# import tkinter.messagebox
from itertools import chain

import PIL
from PIL import Image, ImageTk

root = Tk()
root.geometry('1000x1080')
width, height = 800, 600
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
frame = Frame(root, relief=RIDGE, borderwidth=2)
frame.pack(side=LEFT, fill=Y, expand=1)
root.title('Yolo Live Detection')
frame.config(background='light blue')
label = Label(frame, text="Yolo Detection", bg='light blue', font=('Times 35 bold'))
label.pack()

root.bind('<Escape>', lambda e: root.quit())
lmain = Label(root)
lmain.pack(side=RIGHT)


def exitt():
    exit()


ref_point = []
crop = False

net = cv2.dnn.readNet('yolo-coco/yolov3.weights', 'yolo-coco/yolov3.cfg')
classes = []
with open('yolo-coco/coco.names', 'r') as f:
    classes = f.read().splitlines()


def shape_selection(event, x, y, flags, param):
    # grab references to the global variables
    global ref_point, crop

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being performed
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        rectangles_list.append(ref_point)
        print(ref_point)
        # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        ref_point.append((x, y))
        print(ref_point)
        # draw a rectangle around the region of interest

        # cv2.imshow("image", img)


def draw():
    rect_img = img[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    # for b in blob:
    #     for n, img_blob in enumerate(b):
    #         cv2.imshow(str(n), img_blob)

    net.setInput(blob)

    layerOutputs = net.forward(getOutputsNames(net))

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    print(len(boxes))

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    # print(indexes.flatten())

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))

    flatten2 = list(chain(*indexes))
    print("Method 2 =", flatten2)

    for i in flatten2:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label + " " + confidence, (x, y + 20), font, 2, (255, 255, 255), 2)

    print(rectangles_list)
    for i in rectangles_list:
        print(i)
        cv2.rectangle(img, ref_point[0], ref_point[1], (0, 255, 0), 2)
    print("RECT IMAGE", rect_img)
    cv2.imshow('ROI', rect_img)
    return rect_img


def getOutputsNames(net):
    layersNames = net.getLayerNames()

    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


cap = cv2.VideoCapture(0)

global rectangles_list
rectangles_list = []


def show_frame():
    _, img = cap.read()
    cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    imga = PIL.Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=imga)
    lmain.imgtk = imgtk
    # cv2.setMouseCallback(lmain,shape_selection)
    lmain.configure(image=imgtk)
    if len(ref_point) == 2 and ref_point[0] != ref_point[1]:
        draw()

    lmain.after(10, show_frame)


def webdet():
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", shape_selection)
    while True:
        global img
        _, img = cap.read()

        cv2.imshow("image", img)

        print("LEN OF POINT", len(ref_point))

        if len(ref_point) == 2 and ref_point[0] != ref_point[1]:
            draw()

        # cv2.imshow("ss", img)
        key = cv2.waitKey(1)
        if key == 27:
            break
    # cap.release()
    # cv2.destroyAllWindows()


but1 = Button(frame, padx=5, pady=5, width=20, bg='white', fg='black', relief=GROOVE, command=show_frame,
              text='Open Cam',
              font=('helvetica 15 bold'))
but1.place(x=5, y=104)

but3 = Button(frame, padx=5, pady=5, width=20, bg='white', fg='black', relief=GROOVE, command=webdet,
              text='Open Cam & Detect', font=('helvetica 15 bold'))
but3.place(x=5, y=250)

but5 = Button(frame, padx=5, pady=5, width=5, bg='white', fg='black', relief=GROOVE, text='EXIT', command=exitt,
              font=('helvetica 15 bold'))
but5.place(x=210, y=478)

root.mainloop()
