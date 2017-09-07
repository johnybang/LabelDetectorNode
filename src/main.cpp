#include <ros/ros.h>

#include "dynamic_label_detector_node.h"

int main(int argc, char** argv)
{
	ros::init(argc, argv, "node");

	jyb::DynamicLabelDetectorNode node;

	ros::spin();
	return 0;
}
