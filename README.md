# Drowsiness and Yawn Detection System

This project implements a drowsiness and yawn detection system using Python, OpenCV, dlib, and YOLO object detection. It utilizes facial landmarks detection to monitor eye aspect ratio (EAR) and lip distance to detect drowsiness and yawning, respectively. Additionally, it employs YOLO object detection to identify potential distractions such as phone usage, smoking, and eating while driving.

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
git clone https://github.com/your-username/drowsiness-yawn-detection.git
```

2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

3. Download the pre-trained shape predictor file (`shape_predictor_68_face_landmarks.dat`) from the [dlib shape predictor page](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and place it in the project directory.

4. Download the YOLOv3 configuration file (`yolov3.cfg`) and weights file (`yolov3.weights`) from the official YOLO website and place them in the project directory.

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
```
