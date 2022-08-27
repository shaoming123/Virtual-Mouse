from threading import Thread

import cv2
import numpy as np
import time
import math
import mediapipe as mp


import autopy
import pyautogui


class ThreadedCamera(object):
    def __init__(self, src):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 780)

        # FPS = 1/X
        # X = desired FPS
        self.FPS = 1 / 60
        self.FPS_MS = int(self.FPS * 1000)

        # Start frame retrieval thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()

                return self.status, self.frame
            time.sleep(self.FPS)

    def show_frame(self):
        cv2.imshow('Virtual Mouse', self.frame)
        cv2.waitKey(self.FPS_MS)


class hand_detector:
    def __init__(self, static_image_mode=False, hand=1, detection_confidence=0.8, track_confidence=0.8):

        # Initialize the mediapipe hands class.
        self.mp_hands = mp.solutions.hands

        # Set up the Hands function.
        self.static_image_mode = static_image_mode
        self.hand = hand
        self.detection_confidence = detection_confidence
        self.track_confidence = track_confidence

        self.hands = self.mp_hands.Hands(self.static_image_mode, self.hand,
                                         self.detection_confidence, self.track_confidence)

        # Initialize the mediapipe drawing class.
        self.mp_draw = mp.solutions.drawing_utils

        # Initialize the fingertip hand landmark that used to get the finger
        self.tip_id = [4, 8, 12, 16, 20]

    #  This function performs hands landmarks detection on an image
    def get_hands(self, image, draw=True):

        """
       Args:
               image: The input image with prominent hand(s) whose landmarks need to be detected.
               draw: A boolean value that is if set to true the function displays detected hand with the line connection
        """

        myHand = ""

        # Convert the image from BGR into RGB format.
        image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_RGB.flags.writeable = False

        # Perform the Hands Landmarks Detection.
        self.results = self.hands.process(image_RGB)

        # Check if landmarks are found and are specified to be drawn.
        if self.results.multi_hand_landmarks:

            # Iterate over the found hands.
            for hand_type, hand_landmarks in zip(self.results.multi_handedness,
                                                 self.results.multi_hand_landmarks):

                # Draw the hand landmarks on the copy of the input image.
                if draw:
                    self.mp_draw.draw_landmarks(image, hand_landmarks,
                                                self.mp_hands.HAND_CONNECTIONS)

                # To check the hand is left or right side
                if hand_type.classification[0].label == "Right":
                    myHand = "Left"
                else:
                    myHand = "Right"

        # Return the output image of hands landmarks detection and hand type
        return image, myHand

    #  This function get the hand landmarks point location on an image
    def get_hand_position(self, image, draw=True):
        xList = []
        yList = []
        self.lmList = []

        # Check if there is hand(s) in the image.
        if self.results.multi_hand_landmarks:

            # Iterate over the hand type and the x and y coordinates of the hand landmarks
            for hand_landmark_coordinate in self.results.multi_hand_landmarks:

                # Iterate over the detected hands in the image.
                for id, lm in enumerate(hand_landmark_coordinate.landmark):

                    # Get the height and width of the input image.
                    h, w, c = image.shape

                    # Get the x and y current position of the 21 hand landmark point.
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    xList.append(cx)
                    yList.append(cy)

                    # Append the landmark position into the list with the hand id
                    self.lmList.append([id, cx, cy])

                    # Draw a circle on the cx and cy coordinate of the hand.
                    if draw:
                        cv2.circle(image, (cx, cy), 3, (0, 0, 255), cv2.FILLED)

                # Get the bounding box coordinates for the hand with the specified padding.
                x1, x2 = min(xList), max(xList)
                y1, y2 = min(yList), max(yList)

                # Draw the bounding box around the hand on the output image.
                if draw:
                    cv2.rectangle(image, (x1 - 20, y1 - 20), (x2 + 20, y2 + 20),
                                  (0, 255, 0), 2)

        # Return the all the landmark in the list and the hand type
        return self.lmList

    # This function will get the number of fingers up for each hand in the image
    def get_fingers(self, hand_type):

        # The array for store the 5 fingertips in 0 and 1 format
        fingers = []

        if self.results.multi_hand_landmarks:

            # Check the hand type
            if hand_type == "Right":

                # From the lmList to get the thumb landmarks x position to check if fingertip is up then append the 1.
                if self.lmList[self.tip_id[0]][1] > self.lmList[self.tip_id[0] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if self.lmList[self.tip_id[0]][1] < self.lmList[self.tip_id[0] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # four fingers except the thumb
            for id in range(1, 5):

                # From the lmList to check the four fingertip landmarks y position if fingertip is up then append the 1.
                if self.lmList[self.tip_id[id]][2] < self.lmList[self.tip_id[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

        total_fingers = fingers.count(1)

        print(fingers)

        # Return the 5 fingertips condition and total number of fingers
        return fingers, total_fingers

    # This function will get the between two fingers distance for each hand in the image
    def get_fingers_distance(self, p1, p2, image):

        # From the lmList to get the two finger landmark x and y coordinates
        x_one, y_one = self.lmList[p1][1:]
        x_two, y_two = self.lmList[p2][1:]

        # Get the length between two fingers
        length = math.hypot(x_two - x_one, y_two - y_one)

        return length, image, [x_one, y_one, x_two, y_two]


def main():
    # Initialize the variable
    width_cam = 1280
    height_cam = 720
    frame_reduction = 100
    extra_width = 600
    extra_height = 200

    smoothening = 4
    previous_locationX = 0
    previous_locationY = 0


    # Initialize the frames per second (FPS) variable
    fps = 0
    tau = time.time()

    threaded_camera = ThreadedCamera(src=0)

    # Perform Hands landmarks detection.
    detector = hand_detector(hand=1)



    # get the size of the computer screen
    width_screen, height_screen = autopy.screen.size()

    x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
    y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

    # To finding the least square polynomial fit
    coefficient = np.polyfit(x, y, 2)  # y = Ax^2 + Bx + C

    # Initialize the boolean of the grab and check action used to avoid the call some mouse action in multiple time
    grab = False
    check_action = False
    hori_scroll = False
    zoom_scroll = False

    # Iterate until the webcam is accessed successfully.
    while True:

        # Initialize the distance value
        distanceCM = 0
        current_distance = 0

        # Read a frame.
        success, image = threaded_camera.update()

        # Perform Hands landmarks detection and get hand type
        image, hand_type = detector.get_hands(image)

        # Get all the hand landmarks x and y position
        lmList = detector.get_hand_position(image)

        # Check if frame is not read properly then continue to the next iteration to read the next frame.
        if not success:
            continue

        #  Check the landmark list is not zero, zero means hand is not detected
        if len(lmList) != 0:

            # Get the index fingertip x and y coordinates
            index_tipX, index_tipY = lmList[8][1:]

            # Get the pinky fingertip x and y coordinates
            pinky_tipX, pinky_tipY = lmList[20][1:]

            # Get the thumb fingertip x and y coordinates
            thumbX, thumbY = lmList[4][1:]

            # Check which fingers are up based on detected hand type
            fingers, total_fingers = detector.get_fingers(hand_type)

            # Draw bounding boxes for the computer screen
            if hand_type == "Right":

                width_box = width_cam - frame_reduction - extra_width
                height_box = height_cam - frame_reduction - extra_height

                cv2.rectangle(image, (frame_reduction, 50),(width_box, height_box),(0, 0, 255), 2)

            else:

                width_box = width_cam - extra_width
                height_box = height_cam - frame_reduction - extra_height

                cv2.rectangle(image, (width_cam - frame_reduction, 50), (width_box, height_box), (0, 0, 255), 2)

            # Get the 5 and 17 hand landmark x and y coordinates
            index_finger_mcpX, index_finger_mcpY = lmList[5][1:]
            pinky_mcpX, pinky_mcpY = lmList[17][1:]

            # calculate the hand distance
            distance = int(math.sqrt((pinky_mcpY - index_finger_mcpY) ** 2 + (pinky_mcpX - index_finger_mcpX) ** 2))
            A, B, C = coefficient
            distanceCM = A * distance ** 2 + B * distance + C

            finger = all(element == fingers[0] for element in fingers)

            if finger:

                # release the shift button after horizontal scroll done
                if hori_scroll:
                    hori_scroll = False
                    pyautogui.keyUp('shift')

                # release the shift button after zoom scroll done
                elif zoom_scroll:
                    zoom_scroll = False
                    pyautogui.keyUp('ctrl')

            # Start to recognize the different hand gesture
            # Moving mouse gesture
            elif fingers[0] == 0 and fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:

                check_action = True

                #  Convert x and y Coordinates from cam size to computer screen
                if hand_type == "Right":

                    x3 = np.interp(index_tipX, (frame_reduction, width_cam - frame_reduction - extra_width),
                                   (0, width_screen))
                else:
                    x3 = np.interp(index_tipX, (width_cam - extra_width, width_cam - frame_reduction),
                                   (0, width_screen))

                y3 = np.interp(index_tipY, (frame_reduction, height_cam - frame_reduction - extra_height),
                               (0, height_screen))

                # Calculation for make the mouse move more smoothing
                smoothen_X = previous_locationX + (x3 - previous_locationX) / smoothening
                smoothen_Y = previous_locationY + (y3 - previous_locationY) / smoothening

                # Perform the mouse move action
                autopy.mouse.move(width_screen - smoothen_X, smoothen_Y)

                previous_locationX, previous_locationY = smoothen_X, smoothen_Y

                # Draw a circle on moving mouse gesture
                cv2.circle(image, (index_tipX, index_tipY), 12, (0, 0, 255), cv2.FILLED)

                # release the left mouse button after drag is apply
                if grab:
                    grab = False
                    pyautogui.mouseUp(button="left")

                # release the shift button after horizontal scroll or zoom scroll done
                elif hori_scroll:
                    hori_scroll = False
                    pyautogui.keyUp('shift')
                elif zoom_scroll:
                    zoom_scroll = False
                    pyautogui.keyUp('ctrl')

            # Right-click mouse gesture
            elif fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:

                # Access the get fingers distance function to get the 4 and 8 hand landmark distance
                length, image, lineInfo = detector.get_fingers_distance(4, 8, image)

                # The length and the check action must be true to perform right click action
                if length > 210 and check_action and distanceCM <= 60:
                    check_action = False
                    autopy.mouse.click(autopy.mouse.Button.RIGHT)
                elif length > 90 and check_action and 60 < distanceCM <= 100:
                    check_action = False
                    autopy.mouse.click(autopy.mouse.Button.RIGHT)
                elif 43 <= length <= 90 and check_action and 100 < distanceCM <= 120:
                    check_action = False
                    autopy.mouse.click(autopy.mouse.Button.RIGHT)

                # Draw two circle on right-click mouse gesture
                cv2.circle(image, (lineInfo[0], lineInfo[1]), 12, (0, 0, 255), cv2.FILLED)
                cv2.circle(image, (lineInfo[2], lineInfo[3]), 12, (0, 0, 255), cv2.FILLED)

            # Left-click and double-click the mouse gestures
            elif fingers[0] == 0 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:

                # Access the get fingers distance function to get the 8 and 12 hand landmark distance
                length, image, lineInfo = detector.get_fingers_distance(8, 12, image)

                # The length and the check action must be true to perform double click action
                if length < 60 and check_action and distanceCM < 40:
                    check_action = False
                    pyautogui.click(clicks=2)
                elif length < 40 and check_action and 40 <= distanceCM < 60:
                    check_action = False
                    pyautogui.click(clicks=2)
                elif length < 30 and check_action and distanceCM >= 60:
                    check_action = False
                    pyautogui.click(clicks=2)

                # The length and the check action must be true to perform left click action
                if length > 115 and check_action and distanceCM <= 60:
                    check_action = False
                    autopy.mouse.click(autopy.mouse.Button.LEFT)
                elif length > 90 and check_action and 60 < distanceCM <= 80:
                    check_action = False
                    autopy.mouse.click(autopy.mouse.Button.LEFT)
                elif length > 55 and check_action and 80 < distanceCM <= 100:
                    check_action = False
                    autopy.mouse.click(autopy.mouse.Button.LEFT)

                # Draw two circle on left-click gesture
                cv2.circle(image, (lineInfo[0], lineInfo[1]), 12, (0, 0, 255), cv2.FILLED)
                cv2.circle(image, (lineInfo[2], lineInfo[3]), 12, (0, 0, 255), cv2.FILLED)

            # vertical scroll up and down gestures
            elif fingers[0] == 0 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:

                # Access the get fingers distance function to get the 4 and 12 hand landmark distance
                length, image, lineInfo = detector.get_fingers_distance(9, 12, image)

                # The length must be true to perform scroll down action
                if length < 120 and distanceCM < 60:
                    pyautogui.scroll(-120)
                elif length < 85 and 60 <= distanceCM <= 80:
                    pyautogui.scroll(-120)
                elif length < 60 and 80 < distanceCM <= 100:
                    pyautogui.scroll(-120)

                if length > 140 and distanceCM < 60:
                    pyautogui.scroll(120)
                elif length > 100 and 60 <= distanceCM <= 80:
                    pyautogui.scroll(120)
                elif length > 65 and 80 < distanceCM <= 100:
                    pyautogui.scroll(120)

            # Scroll horizontal gestures
            elif fingers[0] == 1 and fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:

                # To check the horizontal scroll boolean in order to hold down the shift button
                if not hori_scroll:
                    hori_scroll = True
                    pyautogui.keyDown('shift')

                # perform the action
                if hand_type == "Right":
                    pyautogui.scroll(120)
                else:
                    pyautogui.scroll(-120)

                cv2.circle(image, (thumbX, thumbY), 12, (0, 0, 255), cv2.FILLED)

            # Scroll horizontal gestures
            elif fingers[0] == 0 and fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 1:

                # To check the horizontal scroll boolean in order to hold down the shift button
                if not hori_scroll:
                    hori_scroll = True
                    pyautogui.keyDown('shift')

                # perform the action
                if hand_type == "Right":
                    pyautogui.scroll(-120)
                else:
                    pyautogui.scroll(120)

                cv2.circle(image, (pinky_tipX, pinky_tipY), 12, (0, 0, 255), cv2.FILLED)

            # reset the zoom level
            elif fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 0:
                pyautogui.hotkey('ctrl', '0')

            # Zoom in and Zoom out
            elif fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:

                length, image, lineInfo = detector.get_fingers_distance(9, 12, image)

                # To check the zoom scroll boolean in order to hold down the shift button
                if not zoom_scroll:
                    zoom_scroll = True
                    pyautogui.keyDown('ctrl')

                # perform the action
                if length < 150 and distanceCM < 60:
                    pyautogui.scroll(-80)
                elif length < 100 and 60 <= distanceCM <= 80:
                    pyautogui.scroll(-80)
                elif length < 65 and 80 < distanceCM <= 100:
                    pyautogui.scroll(-80)

                if length > 150 and distanceCM < 60:
                    pyautogui.scroll(80)
                elif length > 105 and 60 <= distanceCM <= 80:
                    pyautogui.scroll(80)
                elif length > 65 and 80 < distanceCM <= 100:
                    pyautogui.scroll(80)

            # Drag the item gesture
            elif fingers[0] == 0 and fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 1:

                # To check the grab boolean in order to perform the hold down the left button
                if not grab:
                    grab = True
                    pyautogui.mouseDown(button="left")

                    #  Convert x and y Coordinates from cam size to computer screen
                if hand_type == "Right":

                    x3 = np.interp(index_tipX, (frame_reduction, width_cam - frame_reduction - extra_width),
                                   (0, width_screen))
                else:

                    x3 = np.interp(index_tipX, (width_cam - extra_width, width_cam - frame_reduction),
                                   (0, width_screen))

                y3 = np.interp(index_tipY, (frame_reduction, height_cam - frame_reduction - extra_height),
                               (0, height_screen))

                current_locationX = previous_locationX + (x3 - previous_locationX) / smoothening
                current_locationY = previous_locationY + (y3 - previous_locationY) / smoothening
                autopy.mouse.move(width_screen - current_locationX, current_locationY)
                previous_locationX, previous_locationY = current_locationX, current_locationY

                # Draw two circle on drag gesture
                cv2.circle(image, (index_tipX, index_tipY), 12, (0, 0, 255), cv2.FILLED)
                cv2.circle(image, (pinky_tipX, pinky_tipY), 12, (0, 0, 255), cv2.FILLED)

        box_x, box_y, box_w, box_h = 5, 5, 300, 80

        # Check the distance cannot same as the current hand distance
        if distanceCM != current_distance:
            current_distance = distanceCM

        # draw the rectangle box to display the FPS and distance value
        image = cv2.rectangle(image, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 255, 0), -1)

        # show the distance result on the screen
        cv2.putText(
            image, f"Distance: {int(current_distance)} CM", (10, 75),
            cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)

        # Frame rate
        # Set the time for this frame to the current time.
        now = time.time()

        # Check if the difference between the previous and this frame time to avoid division by zero.
        if now > tau:
            # Calculate the number of FPS.
            fps = 1 / (now - tau)

        # Update the previous frame time to this frame time.
        # As this frame will become previous frame in next iteration.
        tau = now

        # Write the calculated number of FPS on the frame.
        cv2.putText(image, "FPS : " + str(int(fps)), (10, 40), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 255, 255), 3, cv2.LINE_AA)

        # 12. Display
        threaded_camera.show_frame()

        # Wait for 1ms. Check If q key is pressed then break the loop.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture Object and close the windows.
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
