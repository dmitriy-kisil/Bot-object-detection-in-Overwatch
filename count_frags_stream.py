import numpy as np
import cv2
import time
import mss

frame_width = 1280
frame_height = 720
frame_rate = 15.0
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
        x, y, h, w = 1000, 0, h, 200
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
