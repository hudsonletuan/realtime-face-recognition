from keras.models import Sequential
from keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Dropout, Flatten, Activation
import cv2
import os
import datetime
import numpy as np
from scipy.spatial.distance import cosine
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QGridLayout, QPushButton, QComboBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

def base_model(weights_path=None) -> Sequential:
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    model.add(Convolution2D(64, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Convolution2D(4096, (7, 7), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation("softmax"))
    if weights_path:
        model.load_weights(weights_path)

    return model

weights_path = 'vgg_face_weights.h5'
model = base_model(weights_path=weights_path)

def detect_faces(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
    return faces

def find_closest_match(face_embedding, known_embeddings, threshold=0.5):
    min_dist = float("inf")
    name = None
    for (key, value) in known_embeddings.items():
        dist = cosine(np.squeeze(face_embedding), np.squeeze(value))
        if dist < min_dist and dist < threshold:
            min_dist = dist
            name = key
    return name

def preprocess_image(image):
    img = cv2.resize(image, (224,224))
    img = img.astype('float32')
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

def get_embeddings(model, dataset_path):
    embeddings = {}
    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_folder):
            continue

        person_embeddings = []
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            if os.path.isdir(image_path) or not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            image = cv2.imread(image_path)
            if image is None:
                print(f"Unable to read image {image_path}")
                continue
            img = preprocess_image(image)
            embedding_vector = model.predict(img)
            # Normalize the embedding vector
            embedding_vector = embedding_vector / np.linalg.norm(embedding_vector)
            person_embeddings.append(embedding_vector)

        if person_embeddings:
            embeddings[person_name] = np.mean(person_embeddings, axis=0)
    
    return embeddings

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

dataset_path = 'images/'

folders = os.listdir(dataset_path)
folders = [folder for folder in folders if os.path.isdir(os.path.join(dataset_path, folder))]

# Get the embeddings for all images
known_embeddings = get_embeddings(model, dataset_path)
np.save('known_embeddings.npy', known_embeddings)

video_capture = cv2.VideoCapture(0)
common_resolutions = [
    (3840, 2160), # 4K
    (2560, 1440), # QHD
    (1920, 1080), # FHD
    (1600, 1200),
    (1280, 720),  # HD
    (1024, 768),
    (800, 600),
    (640, 480),   # VGA
    (320, 240)    # QVGA
]

for resolution in common_resolutions:
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    if video_capture.get(cv2.CAP_PROP_FRAME_WIDTH) == resolution[0] and video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT) == resolution[1]:
        break

class VideoWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setStyleSheet("background-color: #1d2a35")

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.video_label, 0, 0)

        self.button_frame = QWidget(self)
        self.button_layout = QGridLayout()
        self.button_frame.setLayout(self.button_layout)
        self.layout.addWidget(self.button_frame, 1, 0)

        right_stylesheet = """
            QPushButton {
                border: 2px solid #04AA6D;
                border-radius: 30px;
                font-family: Google Sans,Helvetica Neue,sans-serif;
                font-size: 35px;
                font-weight: bold;
                color: white;
                background-color: #04AA6D;
            }
            QPushButton:hover {
                background-color: white;
                color: #04AA6D;
            }
        """
        wrong_stylesheet = """
            QPushButton {
                border: 2px solid #f44336;
                border-radius: 20px;
                font-family: Google Sans,Helvetica Neue,sans-serif;
                font-size: 35px;
                font-weight: bold;
                color: white;
                background-color: #f44336;
            }
            QPushButton:hover {
                background-color: white;
                color: #f44336;
            }
        """
        question_stylesheet = """
            QLabel {
                font-family: Google Sans,Helvetica Neue,sans-serif;
                font-size: 35px;
                font-weight: bold;
                color: white;
            }
        """
        menu_stylesheet = """
            QComboBox {
                border-radius: 30px;
                font-family: Google Sans,Helvetica Neue,sans-serif;
                font-size: 35px;
                color: black;
                background-color: #E7E7E7;
            }
            QComboBox QAbstractItemView {
                color: white;
            }
        """
        send_stylesheet = """
            QPushButton {
                border: 2px solid #008CBA;
                border-radius: 30px;
                font-family: Google Sans,Helvetica Neue,sans-serif;
                font-size: 35px;
                font-weight: bold;
                color: white;
                background-color: #008CBA;
            }
            QPushButton:hover {
                background-color: white;
                color: #008CBA;
            }
        """

        self.right_button = QPushButton('LET ME IN', self)
        self.right_button.clicked.connect(self.on_right_click)
        self.right_button.setFixedWidth(300)
        self.right_button.setFixedHeight(100)
        self.right_button.setStyleSheet(right_stylesheet)
        self.button_layout.addWidget(self.right_button, 0, 0)
        
        self.wrong_button = QPushButton('WRONG PERSON', self)
        self.wrong_button.clicked.connect(self.on_wrong_click)
        self.wrong_button.setFixedWidth(300)
        self.wrong_button.setFixedHeight(100)
        self.wrong_button.setStyleSheet(wrong_stylesheet)
        self.button_layout.addWidget(self.wrong_button, 0, 1)

        self.question_title = QLabel("Who are you?", self)
        self.question_title.hide()
        self.question_title.setStyleSheet(question_stylesheet)
        self.button_layout.addWidget(self.question_title, 1, 0, 1, 2, Qt.AlignCenter)

        self.option_menu = QComboBox(self)
        self.option_menu.hide()
        self.option_menu.insertItem(0, "Select")
        self.option_menu.addItems(folders)
        self.option_menu.currentIndexChanged.connect(self.on_selected_name_change)
        # Find the longest name
        longest_name = max(folders, key=len)

        # Set the minimum width of the view to the width of the longest name
        width = self.option_menu.fontMetrics().horizontalAdvance(longest_name)
        self.option_menu.view().setMinimumWidth(width)
        self.option_menu.setFixedHeight(80)
        self.option_menu.setStyleSheet(menu_stylesheet)
        self.button_layout.addWidget(self.option_menu, 2, 0, 1, 2, Qt.AlignCenter)

        self.send_button = QPushButton('SEND', self)
        self.send_button.hide()
        self.send_button.clicked.connect(self.on_send_click)
        self.send_button.setFixedWidth(300)
        self.send_button.setFixedHeight(100)
        self.send_button.setStyleSheet(send_stylesheet)
        self.button_layout.addWidget(self.send_button, 3, 0, 1, 2, Qt.AlignCenter)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_video)
        self.timer.start(1)

    def update_video(self):
        ret, frame = video_capture.read()
        flipped_frame = cv2.flip(frame, 1)

        display_width = 1080
        aspect_ratio = flipped_frame.shape[1] / flipped_frame.shape[0]
        display_height = int(display_width / aspect_ratio)
        display_frame = cv2.resize(flipped_frame, (display_width, display_height))
        
        faces = detect_faces(flipped_frame, face_cascade)

        self.frame_copy = flipped_frame.copy()

        for (x, y, w, h) in faces:
            face = flipped_frame[y:y+h, x:x+w]
            face = preprocess_image(face)
            
            face_embedding = model.predict(face)
            # Normalize the embedding
            normalized_embedding = face_embedding / np.linalg.norm(face_embedding)
            name = find_closest_match(normalized_embedding, known_embeddings)
            
            if name:
                new_x = int(x * display_width / flipped_frame.shape[1])
                new_y = int(y * display_height / flipped_frame.shape[0])
                new_w = int(w * display_width / flipped_frame.shape[1])
                new_h = int(h * display_height / flipped_frame.shape[0])

                cv2.rectangle(display_frame, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 255, 0), 2)
                cv2.putText(display_frame, name, (new_x, new_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

                self.last_face_x = x
                self.last_face_y = y
                self.last_face_w = w
                self.last_face_h = h
                self.last_face_correct = name

        # Convert the frame to a format that can be displayed by PyQt
        qimg = QImage(display_frame.data, display_frame.shape[1], display_frame.shape[0], QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pixmap)

    def on_right_click(self):
        if self.last_face_correct is not None:
            face = self.frame_copy[self.last_face_y:self.last_face_y+self.last_face_h, self.last_face_x:self.last_face_x+self.last_face_w]
            # cv2.imwrite(os.path.join('images/' + self.last_face_correct, f"{len(os.listdir('images/' + self.last_face_correct))}.jpg"), face)
            cv2.imwrite(os.path.join('images/' + self.last_face_correct, f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')[:-4]}.jpg"), face)

    def on_wrong_click(self):
        self.question_title.show()
        self.option_menu.show()
        self.send_button.show()

    def on_send_click(self):
        if self.option_menu.currentText() != "Unknown":
            face = self.frame_copy[self.last_face_y:self.last_face_y+self.last_face_h, self.last_face_x:self.last_face_x+self.last_face_w]
            # cv2.imwrite(os.path.join('images/' + self.option_menu.currentText(), f"{len(os.listdir('images/' + self.option_menu.currentText()))}.jpg"), face)
            cv2.imwrite(os.path.join('images/' + self.option_menu.currentText(), f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')[:-4]}.jpg"), face)

    def on_selected_name_change(self):
        self.option_menu.currentText()

app = QApplication([])
window = VideoWindow()
window.show()
app.exec_()