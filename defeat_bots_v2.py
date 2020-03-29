######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/20/18
# Description:
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on a webcam feed.
# It draws boxes and scores around the objects of interest in each frame from
# the webcam.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.


# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import time
import mss
import pyautogui
from pyautogui import tweens
import ctypes
import win32gui
import wave
import pyaudio
import subprocess
from pynput import keyboard
import os
from multiprocessing import Process

# # Speed-up using multithreads
# cv2.setUseOptimized(True)
# cv2.setNumThreads(4)


def audiohandler(filename):

    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 2
    fs = 44100  # Record at 44100 samples per second

    def on_press(key):
        try:
            print('alphanumeric key {0} pressed'.format(
                key.char))
            if key.char == 'q':
                # Stop listener
                return False
        except AttributeError:
            print('special key {0} pressed'.format(
                key))

    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if 'Stereo Mix' in dev['name']:
            dev_index = dev['index']
            break

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    input_device_index=dev_index,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []  # Initialize array to store frames

    # Store data in chunks for 3 seconds
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    while True:
        if not listener.running:
            break
        data = stream.read(chunk)
        frames.append(data)
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()
    # Save the recorded data as a WAV file
    wf = wave.open(f'{filename}.wav', 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()


def gameplay_video_handler(filename):
    frame_width = 1920
    frame_height = 1080
    frame_rate = 19.0

    def on_press(key):
        try:
            print('alphanumeric key {0} pressed'.format(
                key.char))
            if key.char == 'q':
                # Stop listener
                return False
        except AttributeError:
            print('special key {0} pressed'.format(
                key))

    # Store data in chunks for 3 seconds
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'{filename}.avi', fourcc, frame_rate,
                          (frame_width, frame_height))
    with mss.mss() as sct:
        # Part of the screen to capture
        # hwnd = win32gui.FindWindow(None, 'Overwatch')
        # rect = win32gui.GetWindowRect(hwnd)
        # x = rect[0] + 8
        # y = rect[1] + 32
        # w = rect[2] - x - 16
        # h = rect[3] - y - 8
        monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
        start_time = time.time()
        while True:
            if not listener.running:
                break
            # Time which frame has been captured, need to show how many seconds screen is recorded
            last_time = time.time()
            # Get raw pixels from the screen, save it to a Numpy array
            img = np.array(sct.grab(monitor))
            # Without this converting opencv cannot save each frame in a video!
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # Track seconds for screenplay
            how_much_seconds = int(time.time() - start_time)
            cv2.putText(frame,  f"How much seconds: {how_much_seconds}", (10, 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, "FPS: %f" % (1.0 / (time.time() - last_time)),
                        (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Save frame to video
            out.write(frame)
            # Show frame to you
            # cv2.imshow('frame', frame)
            # Clean up
        out.release()
        cv2.destroyAllWindows()


def ai_video_handler(filename):

    # This is needed since the notebook is stored in the object_detection folder.
    sys.path.append("..")

    # Import utilites
    from utils import label_map_util
    from utils import visualization_utils as vis_util

    # Name of the directory containing the object detection module we're using
    MODEL_NAME = 'inference_graph'

    # Grab path to current working directory
    CWD_PATH = os.getcwd()

    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'labelmap.pbtxt')

    # Path to output file
    OUTPUT_NAME = 'output4.avi'
    PATH_TO_OUTPUT = os.path.join(CWD_PATH, OUTPUT_NAME)

    # Number of classes the object detector can identify
    NUM_CLASSES = 1

    ## Load the label map.
    # Label maps map indices to category names, so that when our convolution
    # network predicts `5`, we know that this corresponds to `king`.
    # Here we use internal utility functions, but anything that returns a
    # dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    # Because running game already used some fraction of GPU memory, specify to use only n% of GPU RAM.
    # For example, in my case using 70% (and remain 30% for actually game) of memory is possible
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph, config=config)

    # Define input and output tensors (i.e. data) for the object detection classifier

    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Initialize webcam feed
    # video = cv2.VideoCapture(0)
    # ret = video.set(3,1280)
    # ret = video.set(4,720)

    # Define the codec and create VideoWriter object
    # frame_width = int(video.get(3))
    # frame_height = int(video.get(4))

    frame_width = 1280
    frame_height = 720
    frame_rate = 3.0

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'{filename}.avi', fourcc, frame_rate,
                          (frame_width, frame_height))
    with mss.mss() as sct:
        # Part of the screen to capture
        hwnd = win32gui.FindWindow(None, 'Overwatch')
        rect = win32gui.GetWindowRect(hwnd)
        x = rect[0] + 8
        y = rect[1] + 32
        w = rect[2] - x - 16
        h = rect[3] - y - 8
        monitor = {"top": y, "left": x, "width": w, "height": h}
        del x, y, w, h
        start_time = time.time()
        total_frags = 0
        number_of_frags = 0
        action_until_defeated_a_bot = False
        while True:
            # Time which frame has been captured, need to show how many seconds screen is recorded
            last_time = time.time()
            # Get raw pixels from the screen, save it to a Numpy array
            img = np.array(sct.grab(monitor))
            # Resize to better performance
            img = cv2.resize(img, (1280, 720))
            # Without this converting opencv cannot save each frame in a video!
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            rows = frame.shape[0]
            cols = frame.shape[1]
            inp = cv2.resize(frame, (300, 300))
            # inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

            # Run the model
            out1 = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                             sess.graph.get_tensor_by_name('detection_scores:0'),
                             sess.graph.get_tensor_by_name('detection_boxes:0'),
                             sess.graph.get_tensor_by_name('detection_classes:0')],
                            feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
            pyautogui.FAILSAFE = False
            # Visualize detected bounding boxes.
            num_detections = int(out1[0][0])

            xl, yl, rightl, bottoml, namel, wl, hl = [], [], [], [], [], [], []
            # centerX, centerY = 0, 0
            for i in range(num_detections):
                classId = int(out1[3][0][i])
                score = float(out1[1][0][i])
                bbox = [float(v) for v in out1[2][0][i]]
                if score > 0.9:

                    x = bbox[1] * cols
                    y = bbox[0] * rows
                    right = bbox[3] * cols
                    bottom = bbox[2] * rows

                    names = list(category_index.values())
                    name = 0
                    for i in names:
                        if i['id'] == classId:
                            name = i['name']
                            break
                    if name == 0:
                        name = 'not Found'
                    screenWidth, screenHeight = pyautogui.size()
                    currentMouseX, currentMouseY = pyautogui.position()
                    print("Width:{}, height:{}".format(screenWidth, screenHeight))
                    print("Current X:{}, Y:{}".format(currentMouseX, currentMouseY))
                    currentMouseX, currentMouseY = 960, 540
                    W, H = int(currentMouseX / 1.5), int(currentMouseY / 1.5)
                    xl.append(x)
                    yl.append(y)
                    rightl.append(right)
                    bottoml.append(bottom)
                    namel.append(name)
                    wl.append(W)
                    hl.append(H)
                    textboxes = "{}: {}".format(name, round(score, 4))
                    cv2.rectangle(frame, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
            if xl != []:
                x, y, right, bottom, name, W, H = xl[0], yl[0], rightl[0], bottoml[0], namel[0], wl[0], hl[0]
                del xl, yl, rightl, bottoml, namel, wl, hl

                centerX, centerY = int((x + right) / 2 - abs(x - right) * 0.1), int(
                    (y + bottom) / 2 - abs(y - bottom) * 0.45)
                print(centerX, centerY)
                moveexp = ""
                moveX, moveY = abs(W - centerX), abs(H - centerY)
                if moveX > 50:
                    moveX = 50
                if moveY > 50:
                    moveY = 50
                if moveX <= 5:
                    moveX = 0
                if moveY <= 5:
                    moveY = 0

                print("Moves: {}, {}".format(moveX, moveY))
                if moveX > 5 and moveY > 5 and action_until_defeated_a_bot is True:
                    action_until_defeated_a_bot = False
                    pyautogui.mouseUp(button='left')
                if moveX <= 5 and moveY <= 5 and action_until_defeated_a_bot is False:
                    moveexp += "Center"
                    # pyautogui.mouseDown(button='left')
                    # time.sleep(1.0)
                    # pyautogui.mouseUp(button='left')
                    # pyautogui.PAUSE = 1.0
                    # pyautogui.mouseDown(button='right')
                    # time.sleep(1.0)
                    # pyautogui.PAUSE = 0.01
                    # pyautogui.click()
                    action_until_defeated_a_bot = True
                    pyautogui.mouseDown(button='left')
                    # pyautogui.mouseUp(button='right')

                elif centerX > W and centerY > H:
                    ctypes.windll.user32.mouse_event(1, moveX, moveY, 0, 0)
                    pyautogui.move(moveX / 2, moveY / 2, duration=0.1, tween=pyautogui.tweens.easeInOutQuad)
                    moveexp += "DownRight"
                    # pyautogui.press("s")
                    # pyautogui.press("d")
                elif centerX > W and centerY < H:
                    ctypes.windll.user32.mouse_event(1, moveX, -moveY, 0, 0)
                    pyautogui.move(moveX / 2, -moveY / 2, duration=0.1, tween=pyautogui.tweens.easeInOutQuad)
                    moveexp += "UpRight"
                    # pyautogui.press("w")
                    # pyautogui.press("d")
                elif centerX < W and centerY > H:
                    ctypes.windll.user32.mouse_event(1, -moveX, moveY, 0, 0)
                    pyautogui.move(-moveX / 2, moveY / 2, duration=0.1, tween=pyautogui.tweens.easeInOutQuad)
                    moveexp += "DownLeft"
                    # pyautogui.press("s")
                    # pyautogui.press("a")
                elif centerX < W and centerY < H:
                    ctypes.windll.user32.mouse_event(1, -moveX, -moveY, 0, 0)
                    pyautogui.move(-moveX / 2, -moveY / 2, duration=0.1, tween=pyautogui.tweens.easeInOutQuad)
                    moveexp += "UPLeft"
                    # pyautogui.press("w")
                    # pyautogui.press("a")
                cv2.circle(frame, (centerX, centerY), 5, (0, 255, 0), -1)
                textcoord = "Coord: X: {}, Y: {}".format(int(centerX), int(centerY))
                textcurrmouse = "Currmouse: X: {}, Y: {}".format(int(W), int(H))
                textdistance = "Distance: X: {}, Y: {}".format(int(centerX - W), int(centerY - H))
                textmoves = "Moves: X: {}, Y: {}".format(moveX, moveY)
                cv2.putText(frame, textboxes, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, textcoord, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, textcurrmouse, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, textdistance, (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, textmoves, (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, moveexp, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, "FPS: %f" % (1.0 / (time.time() - last_time)),
                            (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # Convert to gray to better recognizing
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Logic for "moving" window, in which opencv looks for recognition
            # If have more than one recognized icon, increase window so it can handle next icon
            h = 50
            if number_of_frags != 0:
                h = 50 * (1 + number_of_frags)
            else:
                h = 50
            # Set coordinates for "moving" recognition window
            x, y, h, w = 1030, 0, h, 100
            # Show "moving" window
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # Crop recognition window from frame
            img_gray = img_gray[y:y + h, x:x + w]
            # Load image, which will be trying to find
            template = cv2.imread('icon_player.PNG', 0)
            w, h = template.shape[::-1]
            # Opencv will try to find pixels, which are similar ot given icon
            res = cv2.matchTemplate(img_gray, template, cv2.TM_CCORR_NORMED)
            # You may to play with this value a bit, cause results are sensitive to this value
            threshold = 0.93
            # Remain only those pixels, which contain icon at given threshold
            loc = np.where(res >= threshold)
            unique_icons = []
            number_of_all_detected_icons = len(loc[0])
            # Remain unique icon in a such way: consider unique if x and y coordinates at least 10 pixel different
            for detected_icon_index in range(number_of_all_detected_icons):
                x_of_detected_icon, y_of_detected_icon = loc[1][detected_icon_index], loc[0][detected_icon_index]
                if detected_icon_index > 0:
                    if abs(x_of_detected_icon - previous_icon_x) > 10 or \
                            abs(y_of_detected_icon - previous_icon_y) > 10:
                        previous_icon_x, previous_icon_y = x_of_detected_icon, y_of_detected_icon
                        unique_icons.append((previous_icon_x, previous_icon_y))
                else:
                    previous_icon_x, previous_icon_y = x_of_detected_icon, y_of_detected_icon
                    unique_icons.append((previous_icon_x, previous_icon_y))
            # Add coordinates of moving window to unique icon coordinates when make rectangle on frame
            number_of_unique_icons = 0
            for unique_icon in unique_icons:
                number_of_unique_icons += 1
                unique_icon_x, unique_icon_y = unique_icon[0], unique_icon[1]
                cv2.rectangle(frame, (unique_icon_x + x, unique_icon_y + y),
                              (unique_icon_x + x + w, unique_icon_y + y + h), (0, 0, 255), 2)
            # Track seconds for screenplay
            how_much_seconds = int(time.time() - start_time)
            # Increase frags if number of detected icons more than current number of frags else made equal
            if number_of_frags < number_of_unique_icons:
                number_of_frags = number_of_unique_icons
                total_frags += 1
            if number_of_frags > number_of_unique_icons:
                number_of_frags = number_of_unique_icons
            # Add variables to screen for easy monitoring
            cv2.putText(frame, "FPS: %f" % (1.0 / (time.time() - last_time)),
                        (10, 130),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame,  f"Number of detection: {number_of_unique_icons}", (30, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame,  f"Total frags: {total_frags}", (50, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame,  f"How much seconds: {how_much_seconds}", (70, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Save frame to video
            out.write(frame)
            # Show frame to you
            cv2.imshow('frame', frame)
            # Press "q" to quit
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
    # Clean up
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    filename_for_gameplay = "output3"
    filename_for_ai = "output4"
    proc1 = Process(target=audiohandler, args=(filename_for_ai,))
    proc1.start()
    proc2 = Process(target=ai_video_handler, args=(filename_for_ai,))
    proc2.start()
    proc3 = Process(target=gameplay_video_handler, args=(filename_for_gameplay,))
    proc3.start()
    proc1.join()
    proc2.join()
    proc3.join()

    # merge_into_gameplay_movie = f'ffmpeg -y -i {filename_for_gameplay}.avi -i {filename_for_ai}.wav -c copy {filename_for_gameplay}.mkv'
    merge_into_gameplay_movie = f'ffmpeg -y -i {filename_for_gameplay}.avi -i {filename_for_ai}.wav -c:v copy -c:a aac -strict experimental {filename_for_gameplay}.mkv'
    cmd_merge_gameplay = subprocess.Popen(merge_into_gameplay_movie)
    output_merge_gameplay, _ = cmd_merge_gameplay.communicate()
    os.remove(f'{filename_for_gameplay}.avi')
    merge_into_ai_movie = f'ffmpeg -y -i {filename_for_ai}.avi -i {filename_for_ai}.wav -c copy {filename_for_ai}.mkv'
    # merge_into_ai_movie = f'ffmpeg -y -i {filename_for_ai}.avi -i {filename_for_ai}.wav -c:v copy -c:a aac -strict experimental {filename_for_ai}.mkv'
    cmd_merge_ai = subprocess.Popen(merge_into_ai_movie)
    output_merge_ai, _ = cmd_merge_ai.communicate()
    print(output_merge_ai)
    os.remove(f'{filename_for_ai}.avi')
    os.remove(f'{filename_for_ai}.wav')
