from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os
import logging
import sys


class DrowsinessYawnDetector:
    # Constants for gaze detection
    GAZE_THRESHOLD = 0.2

    # Constants for YOLO object detection
    PHONE_CLASS_ID = 67
    SMOKING_CLASS_ID = 68
    EATING_CLASS_ID = 69
    CIGARETTE_CLASS_ID = 70
    FOOD_CLASS_ID = 71
    YOLO_CONFIDENCE_THRESHOLD = 0.4
    PHONE_DETECTION_SCALE_FACTOR = 1.2
    PHONE_DETECTION_MIN_NEIGHBORS = 7
    PHONE_DETECTION_MIN_SIZE = (40, 40)

    # Define additional class IDs for distracting objects
    YOLO_CLASS_IDS = [PHONE_CLASS_ID, SMOKING_CLASS_ID,
                      EATING_CLASS_ID, CIGARETTE_CLASS_ID, FOOD_CLASS_ID]

    def __init__(self, webcam_index):
        self.webcam_index = webcam_index
        self.EYE_AR_CONSEC_FRAMES = 30
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(
            'shape_predictor_68_face_landmarks.dat')
        self.alarm_status = False
        self.alarm_status2 = False
        self.saying = False
        self.COUNTER = 0

        # Initialize vs with VideoStream object
        self.vs = VideoStream(src=webcam_index).start()

        # Variables to store initial measurements
        self.initial_ear_sum = 0
        self.initial_yawn_sum = 0
        self.initial_ear_count = 0
        self.initial_yawn_count = 0

        # Initialize thresholds
        self.EYE_AR_THRESH = 0
        self.YAWN_THRESH = 0

        # Load YOLO model
        print("Loading YOLO model...")
        if not self.load_yolo_model():
            print("Error loading YOLO model. Exiting...")
            sys.exit(1)
        self.WIDTH, self.HEIGHT = 416, 416

        # Collect initial measurements (including initial threshold calculation)
        self.collect_initial_measurements()

        # Calculate initial thresholds
        # self.calculate_initial_thresholds()

    def alarm(self, msg):
        while self.alarm_status:
            logging.info('Calling alarm')
            s = 'espeak "' + msg + '"'
            os.system(s)

        if self.alarm_status2:
            logging.info('Calling alarm')
            self.saying = True
            s = 'espeak "' + msg + '"'
            os.system(s)
            self.saying = False

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def final_ear(self, shape):
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = self.eye_aspect_ratio(leftEye)
        rightEAR = self.eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        return (ear, leftEye, rightEye)

    def lip_distance(self, shape):
        top_lip = shape[50:53]
        top_lip = np.concatenate((top_lip, shape[61:64]))
        low_lip = shape[56:59]
        low_lip = np.concatenate((low_lip, shape[65:68]))
        top_mean = np.mean(top_lip, axis=0)
        low_mean = np.mean(low_lip, axis=0)
        distance = abs(top_mean[1] - low_mean[1])
        return distance

    def collect_initial_measurements(self):
        # Collect initial measurements for EAR and Yawn threshold
        while self.initial_ear_count < 30 or self.initial_yawn_count < 30:
            frame = self.vs.read()
            frame = imutils.resize(frame, width=450)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Drowsiness measurements
            rects = self.detector(gray, 0)
            for rect in rects:
                shape = self.predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                eye = self.final_ear(shape)
                ear = eye[0]
                distance = self.lip_distance(shape)
                if ear > self.EYE_AR_THRESH:
                    self.initial_ear_sum += ear
                    self.initial_ear_count += 1
                if distance < self.YAWN_THRESH:
                    self.initial_yawn_sum += distance
                    self.initial_yawn_count += 1
            # Calculate initial thresholds for EAR and Yawn
            if self.initial_ear_count > 0:
                self.EYE_AR_THRESH = self.initial_ear_sum / self.initial_ear_count
            if self.initial_yawn_count > 0:
                self.YAWN_THRESH = self.initial_yawn_sum / self.initial_yawn_count

    def calculate_gaze_deviation(self, shape):
        left_eye_center = shape[37]
        right_eye_center = shape[43]
        nose_bridge = shape[27]

        left_eye_direction = np.array(
            [right_eye_center[0] - left_eye_center[0], right_eye_center[1] - left_eye_center[1]])
        reference_vector = np.array([1, 0])

        left_eye_direction = left_eye_direction / \
            np.linalg.norm(left_eye_direction)
        reference_vector = reference_vector / np.linalg.norm(reference_vector)

        gaze_deviation = np.arccos(
            np.clip(np.dot(left_eye_direction, reference_vector), -1.0, 1.0))

        # Calculate the distance between eye centers and nose bridge
        eye_center_to_nose_distance = dist.euclidean(
            left_eye_center, nose_bridge) + dist.euclidean(right_eye_center, nose_bridge)

        # Calculate a dynamic threshold based on eye-nose distance
        dynamic_threshold = 0.15 * eye_center_to_nose_distance

        gaze_deviation = np.degrees(gaze_deviation)
        return gaze_deviation, dynamic_threshold

    def distracted_driving_alert(self):
        gaze_deviation, dynamic_threshold = self.calculate_gaze_deviation(
            self.shape)

        # YOLO object detection for phones
        blob = cv2.dnn.blobFromImage(
            self.frame, scalefactor=1/255.0, size=(self.WIDTH, self.HEIGHT), swapRB=True, crop=False)
        self.net.setInput(blob)
        detections = self.net.forward()

        detected_phones = []
        detected_cigarettes = []
        detected_food = []

        for detection in detections:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > self.YOLO_CONFIDENCE_THRESHOLD:
                box = detection[0:4] * \
                    np.array([self.WIDTH, self.HEIGHT,
                              self.WIDTH, self.HEIGHT])
                (center_x, center_y, width, height) = box.astype("int")
                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))

                # Check for phones
                if class_id == self.PHONE_CLASS_ID:
                    detected_phones.append((x, y, int(width), int(height)))

                # Check for cigarettes
                elif class_id == self.CIGARETTE_CLASS_ID:
                    detected_cigarettes.append((x, y, int(width), int(height)))

                # Check for food (e.g., eating)
                elif class_id == self.FOOD_CLASS_ID:
                    detected_food.append((x, y, int(width), int(height)))

        # Draw rectangles around detected phones
        for (x, y, w, h) in detected_phones:
            cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw rectangles around detected cigarettes
        for (x, y, w, h) in detected_cigarettes:
            cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Draw rectangles around detected food
        for (x, y, w, h) in detected_food:
            cv2.rectangle(self.frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if gaze_deviation > dynamic_threshold:
            cv2.putText(self.frame, "DISTRACTED ALERT!", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if not self.alarm_status:
                self.alarm_status = True
                t = Thread(target=self.alarm, args=(
                    'Pay attention to the road!',))
                t.daemon = True
                t.start()

        if detected_phones:
            cv2.putText(self.frame, "PHONE USAGE ALERT!", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if not self.alarm_status:
                self.alarm_status = True
                t = Thread(target=self.alarm, args=(
                    'Avoid using phone while driving!',))
                t.daemon = True
                t.start()

        if detected_cigarettes:
            cv2.putText(self.frame, "SMOKING ALERT!", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if not self.alarm_status:
                self.alarm_status = True
                t = Thread(target=self.alarm, args=(
                    'Avoid smoking while driving!',))
                t.daemon = True
                t.start()

        if detected_food:
            cv2.putText(self.frame, "EATING ALERT!", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            if not self.alarm_status:
                self.alarm_status = True
                t = Thread(target=self.alarm, args=(
                    'Avoid eating while driving!',))
                t.daemon = True
                t.start()

    def start_detection(self):
        if self.vs is None:
            logging.error("Video stream is not initialized.")
            return

        logging.info("Starting Video Stream")
        self.vs.start()
        time.sleep(1.0)

        while True:
            self.frame = self.vs.read()
            self.frame = imutils.resize(self.frame, width=450)
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

            rects = self.detector(gray, 0)

            for rect in rects:
                self.shape = self.predictor(gray, rect)
                self.shape = face_utils.shape_to_np(self.shape)

                self.distracted_driving_alert()

                eye = self.final_ear(self.shape)
                ear = eye[0]
                distance = self.lip_distance(self.shape)

                leftEyeHull = cv2.convexHull(eye[1])
                rightEyeHull = cv2.convexHull(eye[2])
                cv2.drawContours(self.frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(
                    self.frame, [rightEyeHull], -1, (0, 255, 0), 1)

                lip = self.shape[48:60]
                cv2.drawContours(self.frame, [lip], -1, (0, 255, 0), 1)

                if ear < self.EYE_AR_THRESH:
                    self.COUNTER += 1
                    if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                        if not self.alarm_status:
                            self.alarm_status = True
                            t = Thread(target=self.alarm,
                                       args=('wake up sir',))
                            t.daemon = True
                            t.start()
                        cv2.putText(self.frame, "DROWSINESS ALERT!", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    self.COUNTER = 0
                    self.alarm_status = False

                if distance > self.YAWN_THRESH:
                    cv2.putText(self.frame, "Yawn Alert", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if not self.alarm_status2 and not self.saying:
                        self.alarm_status2 = True
                        t = Thread(target=self.alarm, args=(
                            'take some fresh air sir',))
                        t.daemon = True
                        t.start()
                else:
                    self.alarm_status2 = False

                cv2.putText(self.frame, "EAR: {:.2f}".format(ear), (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(self.frame, "YAWN: {:.2f}".format(distance), (300, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Frame", self.frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cv2.destroyAllWindows()
        self.vs.stop()

    def load_yolo_model(self):
        try:
            self.net = cv2.dnn.readNet('yolov3.cfg', 'yolov3.weights')
            return True
        except cv2.error as e:
            print("Error loading YOLO model:", e)
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Drowsiness and Yawn Detection Script")
    parser.add_argument("-w", "--webcam", type=int,
                        default=0, help="index of webcam on system")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logging.info("Loading the predictor and detector...")

    detector = DrowsinessYawnDetector(args.webcam)
    detector.start_detection()


if __name__ == "__main__":
    main()
