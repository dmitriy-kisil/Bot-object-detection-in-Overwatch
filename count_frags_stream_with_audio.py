import numpy as np
import cv2
import time
import mss
import win32gui
import wave
import pyaudio
import subprocess
from pynput import keyboard
import os
from multiprocessing import Process


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


def videohandler(filename):

    frame_width = 1280
    frame_height = 720
    frame_rate = 15.0

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
        start_time = time.time()
        total_frags = 0
        number_of_frags = 0
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
            x, y, h, w = 1030, 10, h, 100
            # Show "moving" window
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # Crop recognition window from frame
            img_gray = img_gray[y:y+h, x:x+w]
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
                        (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame,  f"Number of detection: {number_of_unique_icons}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame,  f"Total frags: {total_frags}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame,  f"How much seconds: {how_much_seconds}", (70, 70),
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
    filename = "output4"
    proc1 = Process(target=audiohandler, args=(filename,))
    proc1.start()
    proc2 = Process(target=videohandler, args=(filename,))
    proc2.start()
    proc1.join()
    proc2.join()

    merge_into_movie = f'ffmpeg -y -i {filename}.avi -i {filename}.wav -c copy {filename}.mkv'
    # merge_into_movie = f'ffmpeg -y -i {filename}.avi -i {filename}.wav -c:v copy -c:a aac -strict experimental {filename}.mkv'
    p = subprocess.Popen(merge_into_movie)
    output, _ = p.communicate()
    print(output)
    os.remove(f'{filename}.avi')
    os.remove(f'{filename}.wav')
