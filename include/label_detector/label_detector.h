#ifndef LABEL_DETECTOR_H_
#define LABEL_DETECTOR_H_

#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace jyb {

class LabelDetector
{
public:
	struct MorphParam
	{
		cv::MorphShapes shape;
		double width_ratio; // scaled_height_ * width_ratio = width
		double height_ratio; // scaled_height_ * height_ratio = height
		uchar iters;
	};

	std::size_t scaled_height_;
	MorphParam tophat_param_;
	uchar v_thresh_;
	uchar s_thresh_;
	double median_ksize_ratio_; // scaled_height_ * median_ksize_ratio = kernel_size
	MorphParam close_param_;
	MorphParam open_param_;

	struct Label
	{
		struct Position
		{
			float u;
			float v;
		} position;
		float orientation;
		float perimeter;
		float area;
		float confidence;
		float width;
		float height;
	};

	LabelDetector();
	virtual ~LabelDetector();
	void FindLabels(cv::Mat &image, std::vector<Label> &labels);

private:
	double scale_ratio_;
	void ResizeImage(const cv::Mat &image, cv::Mat &scaled);
	void ComputeHsvMask(const cv::Mat &image, cv::Mat &mask_image);
	void DenoiseBlobs(const cv::Mat &binary_image, cv::Mat &denoised);
	float ComputeConfidence(const cv::Mat &binary_image, const cv::RotatedRect &box);
	void ApplyMorphology(cv::Mat &image, cv::MorphTypes op, MorphParam param);
	void AnnotateImage(cv::Mat &image, const std::vector<cv::RotatedRect> &boxes,
										 const std::vector<float> &confidences);
	void DrawBox(cv::Mat &image, const cv::RotatedRect &box);

};

} // namespace jyb

#endif // LABEL_DETECTOR_H_
