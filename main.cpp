#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <assert.h>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

double getPSNR(const Mat& I1, const Mat& I2)
{
	Mat s1;
	absdiff(I1, I2, s1);       // |I1 - I2|
	s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
	s1 = s1.mul(s1);           // |I1 - I2|^2

	Scalar s = sum(s1);         // sum elements per channel

	double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

	if (sse <= 1e-10) // for small values return zero
		return 0;
	else
	{
		double  mse = sse / (double)(I1.channels() * I1.total());
		double psnr = 10.0*log10((255 * 255) / mse);
		return psnr;
	}
}

double PSNRShift(int shift, const Mat& I1, const Mat& I2) {
	if (shift >= 0) {
		Mat cut2left = cv::Mat(I2, Range::all(), Range(shift, I2.size().width - 1));
		Mat cut1right = cv::Mat(I1, Range::all(), Range(0, I1.size().width - 1 - shift));
		return getPSNR(cut1right, cut2left);
	} else {
		shift *= -1;
		Mat cut1left = cv::Mat(I1, Range::all(), Range(shift, I1.size().width - 1));
		Mat cut2right = cv::Mat(I2, Range::all(), Range(0, I2.size().width - 1 - shift));
		return getPSNR(cut1left, cut2right);
	}
}

void analyze_pair(const Mat& part1, const Mat& part2, double& fullShift, double& avgPSNR, double& maxPSNR)
{
	// Scaling by 2
	std::vector<Mat> scaledPart1, scaledPart2;
	scaledPart1.push_back(part1);
	scaledPart2.push_back(part2);
	
	int dw = part1.size().width, dh = part1.size().height;
	const int factor = 2;
	while (dw > 32 && dh > 32) {
		dw /= factor; dh /= factor;
		Mat newP1, newP2;
		resize(scaledPart1.back(), newP1, Size(dw, dh));
		resize(scaledPart2.back(), newP2, Size(dw, dh));
		scaledPart1.push_back(newP1);
		scaledPart2.push_back(newP2);
	}

	assert(scaledPart1.size() == scaledPart2.size());
	int moved = 0;
	int p = 1;
	vector<double> maxPSNRValues;
	vector<int> shifts;
	for (auto iter1 = scaledPart1.rbegin(), iter2 = scaledPart2.rbegin(); iter1 != scaledPart1.rend(); iter1++, iter2++) {
		int diff = 0;
		double maxPSNR = 0;
		for (int shift = -10; shift <= 10; shift++) {
			double psnr = PSNRShift(moved + shift, *iter1, *iter2);
//			cout << p*(moved + shift) << " " << psnr << endl;
			if (psnr > maxPSNR) {
				maxPSNR = psnr;
				diff = shift;
			}
		}
		p *= factor;
		shifts.push_back(moved + diff);
		moved = (moved + diff) * factor;
//		cout << endl;
		maxPSNRValues.push_back(maxPSNR);
	}

	// Averaging the results
	avgPSNR = 0;
	maxPSNR = 0;
	fullShift = shifts.back();
	for (size_t i = 0; i < scaledPart1.size(); i++) {
		avgPSNR += maxPSNRValues[i];
		if (maxPSNR < maxPSNRValues[i]) {
			maxPSNR = maxPSNRValues[i];
		}
	}
	fullShift /= (part1.size().width);
	avgPSNR /= maxPSNRValues.size();
}

int main(int argc, char** argv)
{
	if (argc != 2)
	{
		cout << " Usage: detect3d <image_file>" << endl;
		return -1;
	}

	Mat image;
	image = imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file
	if (!image.data)                              // Check for invalid input
	{
		cerr << "Could not open or find the image" << std::endl;
		return -1;
	}

	cv::Range part1HRange(0, image.size().height / 2 - 1);
	cv::Range part2HRange(image.size().height / 2, image.size().height - 1);

	Mat part1 = cv::Mat(image, part1HRange, Range::all());
	Mat part2 = cv::Mat(image, part2HRange, Range::all());

	double fullShift, avgPSNR, maxPSNR;
	analyze_pair(part1, part2, fullShift, avgPSNR, maxPSNR);

	std::cout << "{ \"avgShift\": \"" << fullShift << "\", \"avgPSNR\": \"" << avgPSNR << "\", \"maxPSNR\": \"" << maxPSNR << "\" }" << std::endl;

	return 0;
}
