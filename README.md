Real-Time Face Recognition using Deep Learning CNN - A Python Project
=====================================

This project is a real-time face recognition application. This Python application uses the OpenCV library and the pre-trained model VGGFace for running and training. This recognizes faces appearing on the camera and will run through the face database saved locally to display the name on the frame of the person recognized on the camera.

Requirements
------------

* Python 3.x
* OpenCV 3.x or 4.x
* Numpy
* PyQt5

Installation
------------

1. Clone the repository:
```
git clone https://github.com/hudsonletuan/realtime-face-recognition.git
```
2. Install the required packages:
```
pip install keras
pip install opencv-python
pip install opencv-python-headless
pip install numpy
pip install PyQt5
```
3. Download a pre-trained face recognition model:
Usually, you will use pre-trained models which have the name like *_model.h5. However, since the pre-trained model I used is a weights.h5 file, I put a function base_model to read the weights.h5 file.
You can download the vgg_face_weights.h5 file from this GG Drive link: https://drive.google.com/file/d/1yYORAKckT-RW-Sg6gWeAOu-UOtl4hGoP/view?usp=sharing
4. Structure the database
To structure the face database, you have to put each person's face images inside a folder that is labeled by their name. These folders are saved inside the folder "images/".
Usage
-----

1. Run the script `main.py`:
```
python main.py
```
2. The script will open a PyQt5 window displaying the video stream from your webcam.
3. The script will recognize and display the names of the people in the video stream in real time.
4. For training improvement, press the `LET ME IN` button to capture a face-only picture of the person if the script recognizes correctly, the picture will be added to the folder under that person's name in the database.
5. If it recognizes incorrectly, press the `WRONG PERSON` button, a list of existing people in the database will be shown and you can choose the right person to capture a face-only picture of the person, then press `SEND` to add the picture to the folder under that person's name in the database.
5. Close the window to quit the program.

Note: The script assumes that the pre-trained face recognition model is in the same directory as the script. If it is not, you will need to modify the `weights_path` variable in the script to point to the correct location of the model file.

License
-------

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
