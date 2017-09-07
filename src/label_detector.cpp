#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "label_detector.h"
#include "label_detector_debug.h"


namespace jyb {

using namespace cv;

LabelDetector::LabelDetector() : scaled_height_(400),
																 tophat_param_{MORPH_ELLIPSE, 0.125, 0.125, 1},
																 v_thresh_(90),
																 s_thresh_(130),
																 median_ksize_ratio_(0.01),
																 close_param_{MORPH_ELLIPSE, 0.04, 0.04, 1},
																 open_param_{MORPH_ELLIPSE, 0.075, 0.060, 1},
																 scale_ratio_(0.0) {}

LabelDetector::~LabelDetector() {}

void LabelDetector::FindLabels(Mat &image, std::vector<Label> &labels)
{
	Mat scaled;
	ResizeImage(image, scaled);

	Mat img_bin;
	ComputeHsvMask(scaled, img_bin);

	Mat denoised;
	DenoiseBlobs(img_bin, denoised);

	// Finds blob outlines
	std::vector< std::vector<Point> > contours;
	findContours(denoised, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

#if DEBUG == 0
#else
	drawContours(scaled, contours, -1, Scalar(0,255,0), 1);
	DEBUG_PLOT("contours", scaled);
#endif

	// Computes minimum bounding rectangle for each blob
	std::vector<RotatedRect> boxes;
	for (auto &contour : contours)
	{
		RotatedRect box = minAreaRect(contour);
		boxes.push_back(box);
	}

	// Computes confidence of each rectangle's labelness
	std::vector<float> confidences;
	for (auto &box : boxes)
	{
		float confidence = ComputeConfidence(img_bin, box);
		confidences.push_back(confidence);
		DEBUG_MSG("confidence="<<confidence);
	}

#if DEBUG == 0
#else
	AnnotateImage(scaled, boxes, confidences);
	namedWindow("rectangles", WINDOW_AUTOSIZE);
	imshow("rectangles", scaled);
#endif

	// Unscale box parameters to original image scaling
	// Append unscaled parameters to labels output vector
	std::vector<RotatedRect> unscaled_boxes;
	for (std::size_t i = 0; i < boxes.size(); ++i)
	{
		Point2f unscaled_center(boxes[i].center.x / scale_ratio_,
														boxes[i].center.y / scale_ratio_);
		Size2f unscaled_size(boxes[i].size.width / scale_ratio_,
												 boxes[i].size.height / scale_ratio_);
		RotatedRect unscaled_box(unscaled_center, unscaled_size, boxes[i].angle);
		unscaled_boxes.push_back(unscaled_box);

		Label label;
		label.position.u = unscaled_box.center.x;
		label.position.v = unscaled_box.center.y;
		label.orientation = unscaled_box.angle;
		label.area = unscaled_box.size.area();
		label.perimeter = 2.0*(unscaled_box.size.height + unscaled_box.size.width);
		label.confidence = confidences[i];
		label.width = unscaled_box.size.width;
		label.height = unscaled_box.size.height;

		labels.push_back(label);
	}

	AnnotateImage(image, unscaled_boxes, confidences);
#if DEBUG < 2
#else
	namedWindow("rectangles_on_original", WINDOW_AUTOSIZE);
	imshow("rectangles_on_original", image_copy);
#endif

}


// Resizes the image according to scaled_height_, preserving aspect ratio.
void LabelDetector::ResizeImage(const Mat &image, Mat &scaled)
{
	if ((scaled_height_ == 0) || (scaled_height_ == image.size().height))
	{
		scaled = image;
		scale_ratio_ = 1.0;
		DEBUG_MSG("No image scaling applied. 'scaled' set to shallow copy of 'image'.");
		return;
	}

	scale_ratio_ = (double)scaled_height_ / (double)image.size().height;
	std::size_t scaled_width = std::round((double)image.size().width * scale_ratio_);

	InterpolationFlags mode;
	if (scale_ratio_ < 1.0)
		mode = INTER_AREA; // recommended by opencv for shrinking images
	else
		mode = INTER_LINEAR; // recommended by opencv for expanding images

	// TODO: consider cv::cuda::resize()
	resize(image, scaled, Size(scaled_width, scaled_height_), 0, 0, mode);
	DEBUG_MSG("Image scaled with scale_ratio_="<<scale_ratio_);
}


void LabelDetector::ComputeHsvMask(const Mat &image, Mat &mask_image)
{
	// Convert to Hue, Saturation, and Value images
	Mat hsv[3];
	{
		Mat image_hsv;
		// TODO: consider cv::cuda::cvtColor()
		cvtColor(image, image_hsv, COLOR_BGR2HSV);
		// TODO: consier cv::cuda::split()
		split(image_hsv, hsv);
	}
	DEBUG_PLOT("Saturation channel", hsv[1]);
	DEBUG_PLOT("Value channel", hsv[2]);

	// Subtract background estimate using morphological tophat operation:
	// 1) estimate background using morph open on V channel
	// 2) subtract background estimate from V channel
#if DEBUG == 0
#else
	{
		Mat v_open = hsv[2].clone();
		ApplyMorphology(v_open, MORPH_OPEN, tophat_param_);
		DEBUG_PLOT("v_open", v_open);
	}
#endif
	ApplyMorphology(hsv[2], MORPH_TOPHAT, tophat_param_);
	DEBUG_PLOT("tophat", hsv[2]);

	// Normalize background suppressed V image
	// TODO: consider cv::cuda::normalize()
	normalize(hsv[2], hsv[2], 0, 255, NORM_MINMAX, CV_8UC1);
  DEBUG_PLOT("tophat_norm", hsv[2]);

	// Threshold background suppressed V image
	// TODO: consider cv::cuda::threshold()
	threshold(hsv[2], hsv[2], v_thresh_, 255, THRESH_BINARY);
	DEBUG_PLOT("th_tophat", hsv[2]);

	// Threshold S image
	// TODO: consider cv::cuda::threshold()
	threshold(hsv[1], hsv[1], s_thresh_, 255, THRESH_BINARY_INV);
	DEBUG_PLOT("th_S", hsv[1]);

	// Create composite mask (binary) image
	// TODO: consider cv::cuda::bitwise_and()
	bitwise_and(hsv[1], hsv[2], mask_image);
	DEBUG_PLOT("th_tophat & th_S", mask_image);

}


void LabelDetector::DenoiseBlobs(const Mat &binary_image, Mat &denoised)
{
	// Apply median filter to binary image
	int med_ksize = scaled_height_ * median_ksize_ratio_;
	med_ksize += (med_ksize % 2) == 0; // enforce odd number
	// TODO: consider cv::cuda::createMedianFilter()
	medianBlur(binary_image, denoised, med_ksize);
	DEBUG_PLOT("median", denoised);

	// Morph Close to fill holes in label masks
	ApplyMorphology(denoised, MORPH_CLOSE, close_param_);
	DEBUG_PLOT("close", denoised);

	// Morph Open to eliminate non-label spurious pixels
	ApplyMorphology(denoised, MORPH_OPEN, open_param_);
	DEBUG_PLOT("open", denoised);
}


float LabelDetector::ComputeConfidence(const Mat &binary_image, const RotatedRect &box)
{
	// Extract a rectangular patch just big enough to enclose the RotatedRect
	Rect roi = box.boundingRect();
	roi &= Rect(0, 0, binary_image.size().width, binary_image.size().height);
	Mat patch = binary_image(roi);

	// Adjust for new patch origin
	RotatedRect patch_box(box);
	patch_box.center.x -= roi.tl().x;
	patch_box.center.y -= roi.tl().y;

	// Rotate image to unrotate rectangle region
	Mat M = getRotationMatrix2D(patch_box.center, -patch_box.angle, 1.0);
	// TODO: consider cv::cuda::warpAffine()
	warpAffine(patch, patch, M, patch.size()); // cuda?

  // Set roi to the now unrotated rectangle
  roi.x = patch_box.center.x - (patch_box.size.width / 2);
  roi.y = patch_box.center.y - (patch_box.size.height / 2);
  roi.width = patch_box.size.width;
  roi.height = patch_box.size.height;
	roi &= Rect(0, 0, patch.size().width, patch.size().height);

	// Store total number of pixels
	float num_total = roi.area();

	// Count hot pixels in the rectangle
	// TODO: consider cv::cuda::countNonZero()
	float num_hot_inside = countNonZero(patch(roi));

	// Compute confidence as proportion of hot pixels in rectangle
	float confidence = num_hot_inside / num_total;

	return confidence;
}


inline void LabelDetector::ApplyMorphology(Mat &image, MorphTypes op, MorphParam param)
{
	// TODO: consider cv::cuda::createMorphologyFilter()
	Size kernel_size(scaled_height_ * param.width_ratio,
									 scaled_height_ * param.height_ratio);
	Mat element = getStructuringElement(param.shape, kernel_size);
	morphologyEx(image, image, op, element);
}


// Draw the rectangles on the scaled image
// Annotate confidences for each rectangle
inline void LabelDetector::AnnotateImage(Mat &image,
																				 const std::vector<RotatedRect> &boxes,
																				 const std::vector<float> &confidences)
{
	for (std::size_t i = 0; i < boxes.size(); ++i)
	{
		DrawBox(image, boxes[i]);

		Rect bounding = boxes[i].boundingRect();
		double fontscale = scaled_height_ * 0.0015;
		int thickness = std::max(1, (int)std::round(scaled_height_ * 0.003));
		// Point center(bounding.x + bounding.width/2,
		// 						 bounding.y + bounding.height/2);
		Point topright(bounding.br().x, bounding.tl().y);
		putText(image, std::to_string(int(confidences[i] * 100)), topright,
						FONT_HERSHEY_SIMPLEX, fontscale, Scalar(0,0,255), thickness);
	}
}


inline void LabelDetector::DrawBox(Mat &image, const RotatedRect &box)
{
	Point2f vtx[4];
	box.points(vtx);
	for (int j = 0; j < 4; ++j)
	{
		line(image, vtx[j], vtx[(j+1)%4], Scalar(255,0,0), 1, LINE_AA);
	}
}

} // namespace jyb
