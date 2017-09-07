#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>

#include "dynamic_label_detector.h"


namespace jyb {


class DynamicLabelDetectorNode
{
public:
	DynamicLabelDetectorNode();
	~DynamicLabelDetectorNode();

private:
	ros::NodeHandle nh_;
	image_transport::ImageTransport it_;
	image_transport::Subscriber image_sub_;
	image_transport::Publisher image_pub_;
	ros::Publisher labels_pub_;
	jyb::DynamicLabelDetector dynamic_detector_;

	void imageCb(const sensor_msgs::ImageConstPtr& msg);
};

} // namespace jyb
