import csv
import copy
import itertools
import serial

import cv2 as cv
import numpy as np
import mediapipe as mp

from model import KeyPointClassifier

COM = 'COM3'
BAUDRATE = 9600
VIDEO_SRC = 0
WIDTH = 700
HEIGHT = 500
MAX_HAND = 2
OUTOFBOUND = 20


def main():
    # Serial COM preparation
    arduino = None
    try:
        arduino = serial.Serial(port=COM, baudrate=BAUDRATE)
    except:
        pass

    # Camera preparation
    cap = cv.VideoCapture(VIDEO_SRC)

    # Model load
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    keypoint_classifier = KeyPointClassifier()

    # Read labels
    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0]
                                      for row in keypoint_classifier_labels]

    while True:
        # ESC: end program
        if cv.waitKey(10) == 27:  # ESC
            break

        # Camera capture
        ret, image = cap.read()
        if not ret:
            break

        image = cv.resize(image, (WIDTH, HEIGHT))
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        # motors values to send to arduino
        motors_speed = [0, 0, 0, 0]

        # process only with two hands
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == MAX_HAND:

            r_brect = None
            r_hand_sign_id = None

            l_brect = None
            l_hand_sign_id = None

            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                # Drawing part
                debug_image = draw_bounding_rect(debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image, brect, handedness, keypoint_classifier_labels[hand_sign_id])

                # Saving Left and Right hand information
                info_text = handedness.classification[0].label[0:]
                if info_text == "Right":
                    r_brect = brect
                    r_hand_sign_id = hand_sign_id
                elif info_text == "Left":
                    l_brect = brect
                    l_hand_sign_id = hand_sign_id

            if (r_brect and l_brect):
                debug_image = draw_connection(debug_image, r_brect, l_brect)
                # data to send to arduino
                motors_speed = evaluate(
                    r_brect, r_hand_sign_id, l_brect, l_hand_sign_id)

        # send data to arduino
        print(motors_speed)

        # Screen reflection
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


# Coordinates normalization
def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 255), 2)

    # Key Points
    for landmark in landmark_point:
        cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 255), -1)

    return image


def draw_bounding_rect(image, brect):
    cv.rectangle(image, (brect[0], brect[1]),
                 (brect[2], brect[3]), (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text):
    cv.rectangle(image, (brect[0], brect[1]),
                 (brect[2], brect[1] - 22), (255, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text

    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
               cv.LINE_AA)

    return image


def draw_connection(image, r_brect, l_brect):
    # right hand barycenter
    rm = middle_point((r_brect[0], r_brect[1]), (r_brect[2], r_brect[3]))
    # left hand barycenter
    lm = middle_point((l_brect[0], l_brect[1]), (l_brect[2], l_brect[3]))
    m = middle_point(rm, lm)

    cv.line(image, rm, lm, (0, 255, 0), 2)
    cv.circle(image, rm, 5, (0, 255, 0), -1)
    cv.circle(image, lm, 5, (0, 255, 0), -1)
    cv.circle(image, m, 5, (0, 255, 0), -1)

    return image


def evaluate(r_brect, r_hand_sign_id, l_brect, l_hand_sign_id):
    # right hand barycenter
    rm = middle_point((r_brect[0], r_brect[1]), (r_brect[2], r_brect[3]))
    # left hand barycenter
    lm = middle_point((l_brect[0], l_brect[1]), (l_brect[2], l_brect[3]))
    m = middle_point(rm, lm)

    slope = get_slope(rm, lm)

    # forward right, backwards right, forward left, backwards left
    motor = [0, 0, 0, 0]

    if 0.1 > slope and slope > -0.1:  # center
        speed = map_range(m[1], OUTOFBOUND, HEIGHT - OUTOFBOUND, 255, 0)
        if r_hand_sign_id == 1 and l_hand_sign_id == 1:  # forward
            motor[0] = motor[2] = speed
        elif r_hand_sign_id == 0 and l_hand_sign_id == 0:  # backwards
            motor[1] = motor[3] = speed
    elif slope > 0:  # left
        rotation_speed = map_range(slope, 0.1, 1, 0, 255)
        motor[1] = motor[3] = rotation_speed
    elif slope < 0:  # right
        rotation_speed = map_range(slope, -1, -0.1, 0, 255)
        motor[0] = motor[2] = rotation_speed

    return motor

#    text = f'{slope:.2f}' + " "
#
#    # remember y axis is inverted!!
#    if 0.1 > slope and slope > -0.1:
#        text += "Center "
#    elif slope > 0:
#        text += "Left "
#    elif slope < 0:
#        text += "Right "
#
#    if r_hand_sign_id == 1 and l_hand_sign_id == 1:
#        text += "Forwards "
#
#    if r_hand_sign_id == 0 and l_hand_sign_id == 0:
#        text += "Backwards "


def get_slope(rm, lm):
    if rm[0] == lm[0] or rm[1] == lm[1]:
        return 0

    return (rm[1] - lm[1]) / (rm[0] - lm[0])


def middle_point(p0, p1):
    mx = round((p0[0] + p1[0]) / 2)
    my = round((p0[1] + p1[1]) / 2)

    return (int(mx), int(my))


def map_range(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) // (in_max - in_min) + out_min


if __name__ == '__main__':
    main()
