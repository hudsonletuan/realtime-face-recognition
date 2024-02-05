# Real-time Face Recognition using Deep Learning CNN

### This Python application uses the OpenCV library and the pre-trained model VGGFace for running and training. This recognizes faces appearing on the camera and will run through the face database saved locally to display the name on the frame of the person recognized on the camera.
### This application uses PyQt for GUI.
### This application is still under finetuning, more training, and optimization for better performance.
### This model can be manually improved. When capturing faces, the user will have two options either the face is true or wrong. By selecting the option, this application will capture the face and save it to the right person's folder, then the model will have more data to train itself.

##### Usually, you will use pre-trained models which have the name like *_model.h5. However, since the pre-trained model I used is a weights.h5 file, I put a function base_model to read the weights.h5 file.
##### You can download the vgg_face_weights.h5 file from this GG Drive link: https://drive.google.com/file/d/1yYORAKckT-RW-Sg6gWeAOu-UOtl4hGoP/view?usp=sharing

To structure the face database, you have to put each person's face images inside a folder that is labeled by their name. These folders are saved inside the folder "images/".

* Please do not use this for any commercial purposes for now since it is still under development. For any questions or concerns, please contact tuanquocle.contact@gmail.com or info@tuanle.top.
