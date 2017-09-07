#include <vector>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h> // for sensor_msgs topic subscriptions

#include <opencv2/highgui/highgui.hpp>

#include "label_detector/Position.h"
#include "label_detector/Label.h"
#include "label_detector/Labels.h"

#include "label_detector.h" // jyb::LabelDetector::Label
#include "dynamic_label_detector_node.h"

namespace jyb {


static const std::string OPENCV_WINDOW = "label_detector_node Window";


DynamicLabelDetectorNode::DynamicLabelDetectorNode() : it_(nh_)
{
	image_sub_ = it_.subscribe("/static_image", 1,
														 &DynamicLabelDetectorNode::imageCb, this);
	image_pub_ = it_.advertise("/label_detector/output_image", 1);
	labels_pub_ = nh_.advertise<label_detector::Labels>("/label_detector/labels", 100);

	cv::namedWindow(OPENCV_WINDOW);
}


DynamicLabelDetectorNode::~DynamicLabelDetectorNode()
{
	cv::destroyWindow(OPENCV_WINDOW);
}


void DynamicLabelDetectorNode::imageCb(const sensor_msgs::ImageConstPtr& msg)
{
	cv_bridge::CvImagePtr cv_ptr;
	try
	{
		cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
	}
	catch (cv_bridge::Exception& e)
	{
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return;
	}

	std::vector<jyb::LabelDetector::Label> labels;
	dynamic_detector_.FindLabels(cv_ptr->image, labels);

	// Populates and publishes a label detections message
	label_detector::Labels labels_msg;
	for (auto &label : labels)
	{
		label_detector::Position position_msg;
		position_msg.u = label.position.u;
		position_msg.v = label.position.v;
		label_detector::Label label_msg;
		label_msg.position = position_msg;
		label_msg.orientation = label.orientation;
		label_msg.perimeter = label.perimeter;
		label_msg.area = label.area;
		label_msg.confidence = label.confidence;

		labels_msg.labels.push_back(label_msg);
	}
	labels_pub_.publish(labels_msg);

	// Update GUI Window
	cv::imshow(OPENCV_WINDOW, cv_ptr->image);
	cv::waitKey(3);

	// Output modified video stream
	image_pub_.publish(cv_ptr->toImageMsg());
}


} // namespace jyb
