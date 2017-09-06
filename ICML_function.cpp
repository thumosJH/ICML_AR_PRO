#include"ICML.h"

using namespace cv;
using namespace std;

CFeatures_2D::CFeatures_2D(void)
{
	this->init_flag = false;
}


CFeatures_2D::~CFeatures_2D(void)
{
}


bool CFeatures_2D::init_DetectorDescriptorMatcher()
{
	this->Detector = FeatureDetector::create("ORB");
	this->Descriptor = DescriptorExtractor::create("ORB");
	this->Matcher = DescriptorMatcher::create("FlannBased");

	if ((this->Detector.empty() || this->Descriptor.empty() || this->Matcher.empty()))
	{
		cout << "Can not create detector or descriptor extractor or descriptor matcher of given types:" << endl;
		return false;
	}

	this->init_flag = true;

	return true;
}
bool CFeatures_2D::init_DetectorDescriptorMatcherTracker()
{
	this->Descriptor_tracker = DescriptorExtractor::create("ORB");
	this->Matcher_tracker = DescriptorMatcher::create("BruteForce-HammingLUT");

	this->Detector_tracker = new OrbFeatureDetector();

	if ((this->Detector_tracker.empty() || this->Descriptor_tracker.empty() || this->Matcher_tracker.empty()))
	{
		cout << "Can not create detector or descriptor extractor or descriptor matcher of given types:" << endl;
		return false;
	}

	this->init_flag = true;

	return true;
}
bool CFeatures_2D::read_TrainingImages(vector <Mat>& trainImages, const string& trainFilename, vector<string>& trainImageNames)
{
	cout << "< Reading the images..." << endl;
	
	string trainDirName;
	
	trainImageNames.clear();

	trainImageNames.push_back("pattern1.jpg");
	trainImageNames.push_back("pattern2.jpg");
	trainImageNames.push_back("pattern3.jpg");
	trainImageNames.push_back("pokemon.jpg");
	trainImageNames.push_back("pattern4.jpg");
	cout << trainImageNames[0] <<endl;
	cout << trainImageNames[1] << endl;
	cout << trainImageNames[2] << endl;
	cout << trainImageNames[3] << endl;
	cout << trainImageNames[4] << endl;


	if (trainImageNames.empty())
	{
		cout << "Train image filenames can not be read." << endl << ">" << endl;
		return false;
	}
	int readImageCount = 0;
	for (size_t i = 0; i < trainImageNames.size(); i++)
	{
		Mat img = imread(trainImageNames[i], CV_LOAD_IMAGE_GRAYSCALE);
		resize(img, img, Size(380, 460));
		if (img.empty())
			cout << "Train image " << trainImageNames[i]  << " can not be read." << endl;
		else
			readImageCount++;
		trainImages.push_back(img);
		imshow("train Image"+i, trainImages[i]);
	}
	if (!readImageCount)
	{
		cout << "All train images can not be read." << endl << ">" << endl;
		return false;
	}
	else
		cout << readImageCount << " train images were read." << endl;
	cout << ">" << endl;
	return true;
}
void CFeatures_2D::matches2points(const vector<KeyPoint>& train, const vector<KeyPoint>& query,	const std::vector<cv::DMatch>& matches, std::vector<cv::Point2f>& pts_train, std::vector<Point2f>& pts_query)
{

	pts_train.clear();
	pts_query.clear();
	pts_train.reserve(matches.size());
	pts_query.reserve(matches.size());

	size_t i = 0;

	for (; i < matches.size(); i++)
	{

		const DMatch & dmatch = matches[i];

		pts_query.push_back(query[dmatch.queryIdx].pt);
		pts_train.push_back(train[dmatch.trainIdx].pt);

	}

}
bool CFeatures_2D::recognise_ImageDatabase(Mat queryImage)
{
	int i;// k = 0;

	vector<KeyPoint> queryKeypoints;
	Mat queryDescriptors;

	vector<KeyPoint> queryKeypoints_NN1, queryKeypoints_NN2, queryKeypoints_NN3;
	Mat queryDescriptors_NN1, queryDescriptors_NN2, queryDescriptors_NN3;


	vector<DMatch> matches;
	vector<char> mask;

	this->Detector->detect(queryImage, queryKeypoints);
	this->Descriptor->compute(queryImage, queryKeypoints, queryDescriptors);
	queryDescriptors.convertTo(queryDescriptors, CV_32F);
	
	cout << "HERE1~!\n";
	cout << "Query Image type = " << queryDescriptors.type() <<", dim = " << queryDescriptors.dims<< endl;
	cout << "Train Image 0 type = "<<this->databaseDescriptors[0].type()<< ", dim = "<< this->databaseDescriptors[0].dims <<  endl;
	cout << "Train Image 1 type = " << this->databaseDescriptors[1].type() << ", dim = " << this->databaseDescriptors[1].dims << endl;
	cout << "Train Image 2 type = " << this->databaseDescriptors[2].type() << ", dim = " << this->databaseDescriptors[2].dims << endl;
	cout << "Train Image 3 type = " << this->databaseDescriptors[3].type() << ", dim = " << this->databaseDescriptors[3].dims << endl;
	cout << "Train Image 4 type = " << this->databaseDescriptors[4].type() << ", dim = " << this->databaseDescriptors[4].dims << endl;

	this->Matcher->match(queryDescriptors, matches);
	
	mask.resize(matches.size());
	fill(mask.begin(), mask.end(), 0);
	cout << "HERE2~!\n";
	std::vector<float> matches_cnt(this->imageDatabase.size());
	for (i = 0; i< this->imageDatabase.size(); i++)
		matches_cnt[i] = 0;

	for (i = 0; i< this->imageDatabase.size(); i++)
	{
		for (size_t j = 0; j < matches.size(); j++)
		{
			if (matches[j].imgIdx == i)
			{
				mask[j] = 1;
				matches_cnt[i]++;

			}
		}
	}
	cout << "HERE3~!\n";
	float max1 = 0, max2 = 0, max3 = 0; int max_indx1 = 0, max_indx2 = 0, max_indx3 = 0, matched_indx = -1;

	// Getting the first 3 Nearest Matches
	for (i = 0; i< matches_cnt.size(); i++)
	{
		matches_cnt[i] = matches_cnt[i] / this->databaseKeypoints[i].size();

		if (matches_cnt[i]> max1)
		{
			max1 = matches_cnt[i];
			max_indx1 = i;
		}
		else if ((matches_cnt[i]> max2) && (matches_cnt[i]< max1))
		{
			max2 = matches_cnt[i];
			max_indx2 = i;
		}
		else if ((matches_cnt[i]> max3) && (matches_cnt[i]< max2))
		{
			max3 = matches_cnt[i];
			max_indx3 = i;
		}
	}
	
	// Finding Incorrect Nearest matches with homography calculation 
	vector<Point2f> points1, points2, points3, trainpts1, trainpts2, trainpts3;

	vector<DMatch> matches_NN1, matches_NN2, matches_NN3;
	for (size_t j = 0; j < matches.size(); j++)
	{
		if (matches[j].imgIdx == max_indx1)
		{
			const DMatch & dmatch1 = matches[j];
			points1.push_back(queryKeypoints[dmatch1.queryIdx].pt);
			trainpts1.push_back(this->databaseKeypoints[max_indx1][dmatch1.trainIdx].pt);
		}
		if (matches[j].imgIdx == max_indx2)
		{
			const DMatch & dmatch2 = matches[j];
			points2.push_back(queryKeypoints[dmatch2.queryIdx].pt);
			trainpts2.push_back(this->databaseKeypoints[max_indx2][dmatch2.trainIdx].pt);
		}
		if (matches[j].imgIdx == max_indx3)
		{
			const DMatch & dmatch3 = matches[j];
			points3.push_back(queryKeypoints[dmatch3.queryIdx].pt);
			trainpts3.push_back(this->databaseKeypoints[max_indx3][dmatch3.trainIdx].pt);
		}
	}
	
	Mat test_H1, test_H2, test_H3;

	Mat mask1, mask2, mask3;

	if (trainpts1.size() > 8)
		test_H1 = findHomography(Mat(trainpts1), Mat(points1), RANSAC, 3, mask1);
	if (trainpts2.size() > 8)
		test_H2 = findHomography(Mat(trainpts2), Mat(points2), RANSAC, 3, mask2);
	if (trainpts3.size() > 8)
		test_H3 = findHomography(Mat(trainpts3), Mat(points3), RANSAC, 3, mask3);
	int NN1_cnt1 = 0, NN1_cnt2 = 0, NN1_cnt3 = 0;

	Scalar sum1 = sum(mask1);   Scalar sum2 = sum(mask2);   Scalar sum3 = sum(mask3);

	cout << "HERE4~!\n";

	if ((sum1[0]>sum2[0]) && (sum1[0]>sum3[0]))
		matched_indx = max_indx1;
	else if (sum2[0]>sum3[0])
		matched_indx = max_indx2;
	else
		matched_indx = max_indx3;

	if (matched_indx>0)
	{
		this->databaseRecognisedImg = this->imageDatabase[matched_indx];
		this->databaseRecognisedName = this->trainImageNames[matched_indx];
		this->databaseRecognisedIndx = matched_indx;
		cout << "HERE5~!\n";
		return true;
	}
	else
		return false;
	
}

bool CFeatures_2D::train_ImageDatabase(const string& trainingImgsFilename)
{
	if (this->init_flag == false)
	{
		cout << "\n--- ERROR ---\n" << endl;
		cout << ":init_DetectorDescriptorMatcher has NOT been called. \n\n\n" << endl;
		return false;
	}

	this->read_TrainingImages(this->imageDatabase, trainingImgsFilename, this->trainImageNames);

	cout << endl << "< Extracting keypoints from the database of images..." << endl;
	this->Detector->detect(this->imageDatabase, this->databaseKeypoints);
	cout << ">" << endl;


	cout << "< Computing descriptors for keypoints..." << endl;
	this->Descriptor->compute(this->imageDatabase, this->databaseKeypoints, this->databaseDescriptors);
	cout << ">" << endl;
	this->databaseDescriptors[0].convertTo(databaseDescriptors[0], CV_32F);
	this->databaseDescriptors[1].convertTo(databaseDescriptors[1], CV_32F);
	this->databaseDescriptors[2].convertTo(databaseDescriptors[2], CV_32F);
	this->databaseDescriptors[3].convertTo(databaseDescriptors[3], CV_32F);
	this->databaseDescriptors[4].convertTo(databaseDescriptors[4], CV_32F);
	this->Matcher->add(this->databaseDescriptors);
	return true;
}
bool Train_fern_detector(const string& class_name, Mat class_image)
{
	string model_filename = format("%s_model.xml.gz", class_name);

	Mat image;

	double imgscale = 1;

	Size patchSize(32, 32);
	LDetector ldetector(7, 20, 2, 2000, patchSize.width, 2);
	ldetector.setVerbose(true);
	PlanarObjectDetector detector;

	vector<Mat> objpyr;
	int blurKSize = 3;
	double sigma = 0;
	GaussianBlur(class_image, class_image, Size(blurKSize, blurKSize), sigma, sigma);
	buildPyramid(class_image, objpyr, ldetector.nOctaves - 1);


	vector<KeyPoint> objKeypoints;
	PatchGenerator gen(0, 256, 5, true, 0.8, 1.2, -CV_PI / 2, CV_PI / 2, -CV_PI / 2, CV_PI / 2);

	//   printf("Trying to load %s ...\n", model_filename.c_str());
	FileStorage fs(model_filename, FileStorage::READ);
	cout << "Training the class:" << class_name << endl;

	cout << "Step 1. Finding the robust keypoints ...\n" << endl;
	ldetector.setVerbose(true);
	ldetector.getMostStable2D(class_image, objKeypoints, 100, gen);
	cout << "Done.\nStep 2. Training ferns-based planar object detector ...\n" << endl;
	detector.setVerbose(true);

	detector.train(objpyr, objKeypoints, patchSize.width, 100, 11, 10000, ldetector, gen);
	cout << "Done.\nStep 3. Saving the model to: \n .. \n" << model_filename.c_str() << endl;
	if (fs.open(model_filename, FileStorage::WRITE))
		detector.write(fs, "ferns_model");

	fs.release();


	return true;
}

bool CFeatures_2D::track_Object(Mat queryImage)
{
	vector<KeyPoint> queryKeypoints;
	Mat queryDescriptors;
	vector<DMatch> matches;

	this->Detector_tracker->detect(queryImage, queryKeypoints);
	this->Descriptor_tracker->compute(queryImage, queryKeypoints, queryDescriptors);
	queryDescriptors.convertTo(queryDescriptors, CV_32F);
	
	// @@_여기서 에러 발생_@@
	cout << "tracked Index = " << this->databaseRecognisedIndx<<endl;
	this->trackedDescriptors = this->databaseDescriptors[this->databaseRecognisedIndx];
	imshow("computed Image by Matcher", this->imageDatabase[this->databaseRecognisedIndx]);
	this->Matcher_tracker->match(queryDescriptors, this->trackedDescriptors, matches);
	cout << "HERE7~!\n";
	if (queryDescriptors.type() == this->trackedDescriptors.type())
		this->Matcher_tracker->match(queryDescriptors, this->trackedDescriptors, matches);
	else{
		cout << "Do not match descriptors\n";
		return -1;		
	}

	vector<Point2f> points1, trainpts;

	this->matches2points(this->trackedKeypoints, queryKeypoints, matches, trainpts, points1);

	if (matches.size() > 5)
	{
		this->H_transf = findHomography(Mat(trainpts), Mat(points1), RANSAC, 3);
		return true;
	}

}