# ImageClassification
An EfficientNetB4 based Image classifier for indoor and outdoor images of some categories selected from YouTube-8m dataset

# Installation
Numpy

Tensorflow ==2.3

keras

pandas

sklearn

ffmpeg

youtube-dl

$ pip3 install -r requirements.txt

# Download the video from selected categories and extract the frames
The dataset has 6470 images with the following categories such as:

Classroom - Indoor

Kitchen   - Indoor

library   - Indoor

school    - Indoor

sea       - Outdoor

sky       - Outdoor

mountain  - Outdoor

airplanes - Outdoor


# To train the model
$ python train.py

# Run the inference script(CLI)
$ python pred.py --input ./test_data/im1.jpg

# Run the evaluation script
$ python run_evaluation.py --class_1 ./test_data/indoor/ --class_2 ./test_data/outdoor/

the values of the evaluation script will be saved in evaluation.csv for reference.

# Run the unit test
$ python unit_test.py --indoor ./test_data/benchmark_in.jpg --outdoor ./test_data/benchmark_out.jpg

# Recommendations to improve the model

 

