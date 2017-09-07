#include <iostream>
#include <vector>

#include <boost/bind.hpp>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <dynamic_reconfigure/server.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "label_detector/DetectorConfig.h"
#include "label_detector/Position.h"
#include "label_detector/Label.h"
#include "label_detector/Labels.h"

#include "label_detector.h"


static const std::string OPENCV_WINDOW = "label_detector_node Window";

class DynamicLabelDetector
{
	dynamic_reconfigure::Server<label_detector::DetectorConfig> server_;

public:
	jyb::LabelDetector detector_;

	DynamicLabelDetector()
	{
		dynamic_reconfigure::Server<label_detector::DetectorConfig>::CallbackType f;
		f = boost::bind(&DynamicLabelDetector::reconfigureCb, this, _1, _2);
		server_.setCallback(f);
	}

	void reconfigureCb(label_detector::DetectorConfig &config, uint32_t level)
	{
		ROS_INFO("Reconfigure Request: %d %d %d %f %s %f %f %f %f",
	           config.scaled_height, config.v_thresh, config.s_thresh,
	           config.median_ksize_ratio, config.use_rectangles?"True":"False",
	           config.tophat_ratio, config.close_ratio,
	           config.open_width_ratio, config.open_height_ratio);

		detector_.scaled_height_ = config.scaled_height;
		detector_.v_thresh_ = config.v_thresh;
		detector_.s_thresh_ = config.s_thresh;
		detector_.median_ksize_ratio_ = config.median_ksize_ratio;
		if (config.use_rectangles)
		{
			detector_.tophat_param_.shape = cv::MORPH_RECT;
			detector_.close_param_.shape = cv::MORPH_RECT;
			detector_.open_param_.shape = cv::MORPH_RECT;
		}
		else
		{
			detector_.tophat_param_.shape = cv::MORPH_ELLIPSE;
			detector_.close_param_.shape = cv::MORPH_ELLIPSE;
			detector_.open_param_.shape = cv::MORPH_ELLIPSE;
		}
		detector_.tophat_param_.width_ratio = config.tophat_ratio;
		detector_.tophat_param_.height_ratio = config.tophat_ratio;
		detector_.close_param_.width_ratio = config.close_ratio;
		detector_.close_param_.height_ratio = config.close_ratio;
		detector_.open_param_.width_ratio = config.open_width_ratio;
		detector_.open_param_.height_ratio = config.open_height_ratio;
	}

};


class LabelDetectorNode
{
	ros::NodeHandle nh_;
	image_transport::ImageTransport it_;
	image_transport::Subscriber image_sub_;
	image_transport::Publisher image_pub_;
	ros::Publisher labels_pub_;
	DynamicLabelDetector dynamic_detector_;

public:

	LabelDetectorNode() : it_(nh_)
	{
		image_sub_ = it_.subscribe("/static_image", 1,
                               &LabelDetectorNode::imageCb, this);

    image_pub_ = it_.advertise("/label_detector/output_image", 1);

		labels_pub_ = nh_.advertise<label_detector::Labels>("/label_detector/labels", 100);

		cv::namedWindow(OPENCV_WINDOW);
	}

	~LabelDetectorNode()
	{
		cv::destroyWindow(OPENCV_WINDOW);
	}

	void imageCb(const sensor_msgs::ImageConstPtr& msg)
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
		dynamic_detector_.detector_.FindLabels(cv_ptr->image, labels);

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

};

int main(int argc, char** argv)
{
	ros::init(argc, argv, "label_detector_node");

	LabelDetectorNode ic;

	ros::spin();
	return 0;
}
