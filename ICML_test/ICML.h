#ifndef __ICML_AR__
#define __ICML_AR__

#include "opencv2\core\core.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\video\tracking.hpp"
#include "opencv2\features2d\features2d.hpp"
#include "opencv2\flann\flann.hpp"
#include "opencv2\calib3d\calib3d.hpp"
#include "opencv2\objdetect\objdetect.hpp"
#include "opencv2\legacy\compat.hpp"
#include "opencv2\core\internal.hpp"
#include "opencv2\core\operations.hpp"
#include"opencv2\legacy\legacy.hpp"
#include <iostream>

using namespace cv;
using namespace std;


class CFeatures_2D
{
public:
	CFeatures_2D(void);
	~CFeatures_2D(void);

	bool init_DetectorDescriptorMatcher();
	bool init_DetectorDescriptorMatcherTracker();
	void matches2points(const vector<KeyPoint>& train, const vector<KeyPoint>& query, const std::vector<cv::DMatch>& matches, std::vector<cv::Point2f>& pts_train, std::vector<Point2f>& pts_query);
	bool read_TrainingImages(vector <Mat>& trainImages, const string& trainFilename, vector<string>& trainImageNames);
	bool recognise_ImageDatabase(Mat queryImage);
	bool train_ImageDatabase(const string& trainingImgsFilename);
	bool track_Object(Mat queryImage);

	Ptr<FeatureDetector> Detector, Detector_tracker;
	Ptr<DescriptorExtractor> Descriptor, Descriptor_tracker;
	Ptr<DescriptorMatcher> Matcher, Matcher_tracker;

	vector <Mat> imageDatabase;
	vector<vector<KeyPoint> > databaseKeypoints;    
	vector<Mat> databaseDescriptors;             
	vector<KeyPoint> trackedKeypoints;
	Mat trackedDescriptors;
	Mat H_transf;
	Mat    databaseRecognisedImg;
	string databaseRecognisedName;
	int    databaseRecognisedIndx;
	vector<string> trainImageNames;
	bool init_flag;
};

#endif