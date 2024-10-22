# DrowsyDefender

This project implements a system to detect driver fatigue by monitoring eye movements and identifying yawning. It uses Python, OpenCV, dlib, and YOLO object detection to track facial landmarks and detect potential distractions.

## Requirements

- Python 3.x
- OpenCV
- dlib
- imutils
- numpy
- scipy

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/ashwani-anand/Driver-Fatigue-Monitoring.git
    cd Driver-Fatigue-Monitoring
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    ```

3. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Download the pre-trained shape predictor file `shape_predictor_68_face_landmarks.dat` from the [dlib shape predictor page](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and place it in the project directory.

5. Download the YOLOv3 configuration file (`yolov3.cfg`) and weights file (`yolov3.weights`) from the [official YOLO website](https://pjreddie.com/darknet/yolo/) and place them in the project directory.

## Usage

Run the `main.py` script with the `-w` or `--webcam` argument to specify the index of the webcam on your system. For example:
```bash
python main.py -w 0
```

This will start the drowsiness and yawn detection system using the webcam with index 0.

Press 'q' to quit the application.

## How it Works

The system continuously captures frames from the webcam feed and analyzes facial landmarks to determine eye closure (indicating drowsiness) and lip distance (indicating yawning). It also employs YOLO object detection to identify potential distractions such as phone usage, smoking, and eating. If drowsiness, yawning, or distractions are detected, appropriate alerts are displayed on the screen.

## Contributing

Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## About

A Python project to detect faces and track eye movements to identify yawning and gaze detection.
