#include "label_detector.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <numeric> // std::accumulate()
#include <algorithm> // std::sort()
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp> // cv::imread()

#include <gtest/gtest.h>

#include "csvrow.h"


namespace {

// Test Fixture for gtests
class LabelDetectorTest : public ::testing::Test
{
protected:
	jyb::LabelDetector detector_;
	std::vector<jyb::LabelDetector::Label> sharp_labels_;
	std::vector<jyb::LabelDetector::Label> blurry_labels_;
	std::vector<cv::RotatedRect> sharp_truths_;
	std::vector<cv::RotatedRect> blurry_truths_;

	virtual void SetUp()
	{
		// Detect labels on sharp and blurry images
		cv::Mat sharp = cv::imread("imagefiles/216.jpg", cv::IMREAD_COLOR);
		detector_.FindLabels(sharp, sharp_labels_);
		cv::Mat blurry = cv::imread("imagefiles/506.jpg", cv::IMREAD_COLOR);
		detector_.FindLabels(blurry, blurry_labels_);

		// Read ground truth from csv
		std::ifstream file("imagefiles/ground_truth.csv");
		SetTruthFromFile(file);
		file.close();
	}

	// Computes Intersection Over Union object bounding box accuracy metric
	// where the intersection is computed as the intersection of the set of
	// detected label boxes and the set of truth boxes and the union is the
	// union of all boxes in both sets. It is a ratio always in the range
	// 0.0 to 1.0 (unless there's a bug). 1.0 reflects ideal performance.
	static float ComputeIoU(const std::vector<jyb::LabelDetector::Label> &labels,
	                        const std::vector<cv::RotatedRect> &truths)
	{
		// Accumulates the area of detection boxes.
		// Converts jyb::LabelDetector::Label to cv::RotatedRect for later.
		double labels_area = 0;
		std::vector<cv::RotatedRect> rect_labels;
		for (auto &label : labels)
		{
		  labels_area += label.area;

			cv::Point2f center(label.position.u, label.position.v);
			cv::Size2f rect_size(label.width, label.height);
			cv::RotatedRect rect_label(center, rect_size, label.orientation);
			rect_labels.push_back(rect_label);
		}

		// Accumulates the area of truth boxes.
		double truths_area;
		for (auto &truth : truths)
		{
			truths_area += truth.size.area();
		}

		// Accumulates intersecting (overlapping) region areas.
		double intersection_area = 0;
		for (auto &rect_label : rect_labels)
			for (auto &truth : truths)
			{
				std::vector<cv::Point2f> overlap;
				int result;
				result = cv::rotatedRectangleIntersection(rect_label, truth, overlap);

				if (result != cv::INTERSECT_NONE)
				{
					SortPointsClockwise(overlap);
					intersection_area += cv::contourArea(overlap);
				}
			}

		double union_area = labels_area + truths_area - intersection_area;

		return intersection_area / union_area;
	}

private:
	void SetTruthFromFile(std::ifstream &file)
	{
		jyb::CSVRow row;
		std::vector<cv::Point2f> points;
		while(file >> row)
		{
			// Store one of three points defining ground truth rectangle
			points.push_back(cv::Point2f(std::stof(row[2]), std::stof(row[3])));

			if (points.size() == 3)
			{
				cv::RotatedRect box(points[0], points[1], points[2]);
				if (row[0].compare("216.jpg") == 0)
					sharp_truths_.push_back(box);
				else if (row[0].compare("506.jpg") == 0)
					blurry_truths_.push_back(box);
				// Ready points vector for next three points
				points.clear();
			}
		}
	}

	// Sorts points by angle around the centroid to be suitable for opencv
	// functions like contourArea().
	static void SortPointsClockwise(std::vector<cv::Point2f> &points)
	{
		// Calculates centroid point
		// https://stackoverflow.com/questions/11567307/calculate-mean-for-vector-of-points
		cv::Point2f zero(0.0f, 0.0f);
		cv::Point2f sum = std::accumulate(points.begin(), points.end(), zero);
		cv::Point2f centroid(sum * (1.0f / points.size()));

		// Sorts points clockwise around the centroid
		// 1) Shifts the centroid to the origin
		// 2) Sorts around the origin
		// 3) Shifts the centroid back to original position
		for (auto &point : points)
			point -= centroid;
		std::sort(points.begin(), points.end(), ClockwiseLessThan);
		for (auto &point : points)
			point += centroid;
	}

	// Performs clockwise angle comparison where positive y axis is "zero" angle.
	// From SO adapted to be clockwise for positive y down coordinate system:
	// https://stackoverflow.com/questions/6989100/sort-points-in-clockwise-order
	static bool ClockwiseLessThan(const cv::Point2f &a, const cv::Point2f &b)
	{
		if (a.x >= 0 && b.x < 0)
			return false;
		if (a.x < 0 && b.x >= 0)
			return true;
		if (a.x == 0 && b.x == 0) {
			if (a.y >= 0 || b.y >= 0)
				return a.y < b.y;
			return b.y < a.y;
		}

		// compute the cross product of vectors
		float det = a.cross(b);
		if (det < 0)
			return false;
		if (det > 0)
			return true;

		// points a and b are on the same line through the origin
		// check which point is closer to the origin
		float d1 = a.dot(a);
		float d2 = b.dot(b);
		return d1 < d2;
	}

};


TEST_F(LabelDetectorTest, NumLabels)
{
	EXPECT_EQ(blurry_truths_.size(), blurry_labels_.size());
	EXPECT_EQ(sharp_truths_.size(), sharp_labels_.size());
}

TEST_F(LabelDetectorTest, IntersectionOverUnion)
{
	float blurry_iou = ComputeIoU(blurry_labels_, blurry_truths_);
	EXPECT_GT(blurry_iou, 0.9);
	float sharp_iou = ComputeIoU(sharp_labels_, sharp_truths_);
	EXPECT_GT(sharp_iou, 0.9);
}


} // namespace

int main(int argc, char **argv){
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
