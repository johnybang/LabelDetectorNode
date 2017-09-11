#ifndef DYNAMIC_LABEL_DETECTOR_H_
#define DYNAMIC_LABEL_DETECTOR_H_

#include <dynamic_reconfigure/server.h>

#include "label_detector/DetectorConfig.h"

#include "label_detector.h"


namespace jyb {

class DynamicLabelDetector
{
public:
	DynamicLabelDetector();
	void FindLabels(cv::Mat &image,
	                std::vector<jyb::LabelDetector::Label> &labels);

private:
	dynamic_reconfigure::Server<label_detector::DetectorConfig> server_;
	jyb::LabelDetector detector_;

	void reconfigureCb(label_detector::DetectorConfig &config, uint32_t level);
};

} // namespace jyb

#endif // DYNAMIC_LABEL_DETECTOR_H_
