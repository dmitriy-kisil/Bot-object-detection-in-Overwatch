import numpy as np
import cv2
import time
import mss


frame_width = 1280
frame_height = 720
frame_rate = 10.0
OUTPUT_FILE = "output.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(OUTPUT_FILE, fourcc, frame_rate,
                      (frame_width, frame_height))

with mss.mss() as sct:
    # Part of the screen to capture
    monitor = {"top": 120, "left": 280, "width": 1368, "height": 770}

    while True:
        last_time = time.time()

        # Get raw pixels from the screen, save it to a Numpy array
        img = np.array(sct.grab(monitor))
        img = cv2.resize(img, (1280, 720))

        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # gray = cv2.GaussianBlur(frame, (5, 5), 0)
        # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # frame = cv2.Canny(gray, 35, 125)
        # frame = gray

        cv2.putText(frame, "FPS: %f" % (1.0 / (time.time() - last_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # print("fps: {}".format(1 / (time.time() - last_time)))
        out.write(frame)
        cv2.imshow('frame', frame)
        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break


# Clean up
out.release()
cv2.destroyAllWindows()
