# LabelDetectorNode
Catkin package to detect labels on images of industrial boxes.

## Initial setup
This repo should be cloned into a folder called label_detector in the src/ folder of your catkin workspace top-level directory.

## What's Inside
The project has been organized using the catkin workspace approach and catkin's CMake macros.
### Python jupyter notebook rough proto
`LabelDetection.ipynb`: Rough prototype of the computer vision pipeline algorithm itself; used for rapid experimentation with opencv. Path to image file directory may need to be modified to `test/imagefiles/` in order to run this after cloning the repo.
### C++ Source files
1. `src/main.cpp`: Entry point for primary output executable `label_detector_node`.
2. `src/dynamic_label_detector_node.cpp` and `include/dynamic_label_detector_node.h`: Wraps cv_bridge and message handling around dynamic label detector.
3. `src/dynamic_label_detector.cpp` and `include/dynamic_label_detector.h`: Wraps dynamic_reconfigure tunability around label detector. This is included in an exported project library.
4. `src/label_detector.cpp` and `include/label_detector.h`: Implements the computer vision pipeline algorithms.
5. `include/label_detector_debug.h`: For popping up lots of debug opencv image windows and messages depending on DEBUG #define flag.
6. `src/image_publisher.cpp`: Entry point (only file) for `test_image_publisher` executable.
7. `test/test_label_detector.cpp`: Entry point for gtest unit test of `label_detector.cpp`.
8. `test/csvrow.h`: CSV file reading helper class for reading in hand-annotated ground truth rectangle vertices for test.
### Executables Generated
1. `label_detector_node`: Primary node that subscribes to a (hardcoded) image topic, detects labels, posts a custom topic of detection parameters, and shows an opencv window with the image annotated. The node may be tuned using rqt_reconfigure.
2. `test_image_publisher`: Roughly written image publishing node hardcoded to publish the two test images back and forth at a 0.5Hz rate. Used to demonstrate a basic system.

To build just this package's executables, type this at the catkin workspace top-level directory:
```
$ catkin_make -DCATKIN_WHITELIST_PACKAGES="label_detector"
```

### Unit Test 
A gtest unit test of label_detector.h functionality. It checks for the expected number of label detections on each image.  It also computes an Intersection Over Union metric and checks that the performance exceeds 0.9 on each image. To build and run this test, type this at the catkin workspace top-level directory:

```
$ catkin_make run_tests
```

### Run a basic system
A launch file has been created to bring up a basic system using two test images. To run:

```
$ roslaunch label_detector basic_system.launch
```

### Tune the parameters
To tune the algorithmic parameters of the computer vision algorithm use rqt\_reconfigure:

```
$ rosrun rqt_reconfigure rqt_reconfigure
```

## Potential Further Work
1. Go through TODO comments to swap in identified cv::cuda versions of opencv routines.
2. Add file/function headers with doxygen tags if desired.
3. Figure out why the detections are slightly biased away from the origin as scaled_height_ decreases.
4. Add a rostest+gtest for automated testing of label_detector_node.
5. Profile the cpu cycle/time contribution of major algorithm subroutines; confirm python jupyter notebook analysis' preliminary relative running time analysis.
6. Consider swapping findContours for a Hough based approach.
7. Consider additional confidence measurement components: rectangularity, size, color histogram, etc.
8. Apply random relevant random transformations (rotation, scale, illumination) along with transformations to the ground truth to perform further robustness testing. Maybe add dynamic_reconfigure rotation parameter for experimentation.
