# AIRR Gate Detection
My individual assignment for AMAV course

### Dependencies
You will need the following libraries, in addition to your Python 3.8.5 interpreter:
- numpy 1.19.2
- matplotlib 3.3.2
- opencv_contrib_python 4.5.1.48
The script has only been tested with the above specified versions. You can install these automatically via pip install -r requirements.txt

### Usage
To run the gate detection on a single frame, use single_detect from main.py. As arguments, it takes a grey scale openCV image, reference (SIFT) feature descriptions, the number of rows and columns to divide the image, the Lowe's test ratio, and the minimum number of good matches to promote a block to Region of Interest.

To test the algorithm, you can run it on a set of randomized images, via the main function in main.py. As arguments, it takes an integer representing the test type, the size of the random sample, the number of rows and columns to divide the image, the Lowe's test ratio, and the minimum number of good matches to promote a block to Region of Interest. The test types may be
- 1 ->    Manually count the True Positive and False Positive Regions of Interest on each image. Used for measuring the TP and FP rate.
- 2 ->    Gather statistical data on the cost.
- 3 ->    Measure the compute time.
