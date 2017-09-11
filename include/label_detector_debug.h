#ifndef LABEL_DETECTOR_DEBUG_H_
#define LABEL_DETECTOR_DEBUG_H_

#if DEBUG < 2

#define DEBUG_MSG(str) do { } while ( false )
#define DEBUG_PLOT(str, img) do { } while ( false )

#else // DEBUG >= 2

#include <iostream>
#include <opencv2/highgui.hpp>
#define DEBUG_PLOT(str, img) do {             \
	cv::namedWindow(str, cv::WINDOW_AUTOSIZE);  \
	cv::imshow(str, img); } while ( false )
#define DEBUG_MSG(str) do { std::cout << str << "\n"; } while( false )

#endif // DEBUG < 2


#endif // LABEL_DETECTOR_DEBUG_H_
