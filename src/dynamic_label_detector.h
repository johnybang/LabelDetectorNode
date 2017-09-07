#include <ros/ros.h>
#include <dynamic_reconfigure/server.h>

#include "label_detector/DetectorConfig.h"

#include "label_detector.h"


namespace jyb {

class DynamicLabelDetector
{
public:
	DynamicLabelDetector()
	{
		dynamic_reconfigure::Server<label_detector::DetectorConfig>::CallbackType f;
		f = boost::bind(&DynamicLabelDetector::reconfigureCb, this, _1, _2);
		server_.setCallback(f);
	}

	void FindLabels(cv::Mat &image,
									std::vector<jyb::LabelDetector::Label> &labels)
	{
		detector_.FindLabels(image, labels);
	}

private:
	dynamic_reconfigure::Server<label_detector::DetectorConfig> server_;
	jyb::LabelDetector detector_;

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

} // namespace jyb
