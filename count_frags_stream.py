import numpy as np
import cv2
import time
import mss


frame_width = 1280
frame_height = 720
frame_rate = 10.0
OUTPUT_FILE = "output4.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(OUTPUT_FILE, fourcc, frame_rate,
                      (frame_width, frame_height))

with mss.mss() as sct:
    # Part of the screen to capture
    monitor = {"top": 120, "left": 280, "width": 1368, "height": 770}
    start_time = time.time()
    total_frags = 0
    number_of_frags = 0
    while True:
        last_time = time.time()

        # Get raw pixels from the screen, save it to a Numpy array
        img = np.array(sct.grab(monitor))
        img = cv2.resize(img, (1280, 720))

        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rgb = frame
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        # x, y, h, w = 500, 440, 130, 110
        h = 50
        if number_of_frags != 0:
            h = 50 * (1 + number_of_frags)
        else:
            h = 50
        x, y, h, w = 1000, 0, h, 200
        cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 0, 255), 2)
        img_gray = img_gray[y:y+h, x:x+w]
        template = cv2.imread('icon_player.PNG', 0)
        w, h = template.shape[::-1]
        # 'cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
        # 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED'
        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCORR_NORMED)
        threshold = 0.93
        loc = np.where(res >= threshold)
        loc1 = loc[0]
        loc1_prep = []
        for i in range(len(loc[0])):
            if i > 0:
                if abs(loc[0][i]-prev[1]) > 10 or abs(loc[1][i]-prev[0]) > 10:
                    prev = (loc[1][i], loc[0][i])
                    loc1_prep.append(prev)
            else:
                prev = (loc[1][i], loc[0][i])
                loc1_prep.append(prev)

        # print(f"Before: {list(zip(*loc[::-1]))}, after: {loc1_prep}")

        num_of_detect = 0
        for pt in loc1_prep:
            num_of_detect += 1
            cv2.rectangle(img_rgb, (pt[0] + x, pt[1] + y), (pt[0] + x + w, pt[1] + y + h), (0, 0, 255), 2)
            # cv2.rectangle(img_rgb, (pt[0], pt[1]), (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

        frame = img_rgb
        # gray = cv2.GaussianBlur(frame, (5, 5), 0)
        # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # frame = cv2.Canny(gray, 35, 125)
        # frame = gray
        how_much_seconds = int(time.time() - start_time)
        if number_of_frags < num_of_detect:
            number_of_frags = num_of_detect
            total_frags += 1
        if number_of_frags > num_of_detect:
            number_of_frags = num_of_detect

        cv2.putText(frame, "FPS: %f" % (1.0 / (time.time() - last_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame,  f"Number of detection: {num_of_detect}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame,  f"Total frags: {total_frags}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame,  f"How much seconds: {how_much_seconds}", (70, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # print("fps: {}".format(1 / (time.time() - last_time)))
        out.write(frame)
        cv2.imshow('frame', frame)
        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break


# Clean up
out.release()
cv2.destroyAllWindows()
