#include <ros/ros.h>
#include <ros/package.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>

static const std::string kPackageName = "label_detector";

int main(int argc, char** argv)
{
	ros::init(argc, argv, "image_publisher");
	ros::NodeHandle nh;

	image_transport::ImageTransport it(nh);
	image_transport::Publisher image_pub = it.advertise("/static_image", 1);

	// read image file and create image message
	cv_bridge::CvImage cv_image;
	std::string root = ros::package::getPath(kPackageName);
	std::vector<std::string> image_paths = {root + "/test/imagefiles/216.jpg",
	                                        root + "/test/imagefiles/506.jpg"};

	// publish every 5 seconds
	cv_image.encoding = "bgr8";
	sensor_msgs::Image ros_image;
	ros::Rate loop_rate(0.5);
	int image_idx = 0;
	while (nh.ok())
	{
		cv_image.image = cv::imread(image_paths[(image_idx++)%2]);
		cv_image.toImageMsg(ros_image);

		image_pub.publish(ros_image);
		loop_rate.sleep();
	}

	return 0;
}
