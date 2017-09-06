// Object_Recognition_Tracking.cpp 
//
//---------------------------------------------------------------------------------------
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//o-o-o-o-                     created by Tony Marrero Barroso                   -o-o-o-o   
//                                     28th of July 2011
//
//---------------------------------------------------------------------------------------


//#include "stdafx.h"

#include "ICML.h"
//#include "ROI_Listener.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/flann/flann.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/legacy/compat.hpp"
#include "opencv2/core/internal.hpp"
#include "opencv2/core/operations.hpp"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;
using namespace cv;


int main()
{
	Mat frame, gray;
	vector<Point2f> origCorners(4);
	vector<Point>   warpedPoints(4);
	VideoCapture capture(0);

	CFeatures_2D* Feat_2d_ptr = new CFeatures_2D;
	
	Feat_2d_ptr->init_DetectorDescriptorMatcher();
	Feat_2d_ptr->init_DetectorDescriptorMatcherTracker();

	Feat_2d_ptr->train_ImageDatabase("training_imgs.txt");

	char key; bool print_recog = false;

	float h11, h12, h13, h21, h22, h23, h31, h32, h33, Z, X, Y;
	capture.open(0);
	if (!capture.isOpened())
	{

		cout << "capture device " << -1 << " failed to open!" << endl;
		cout << "press any button to continue \n" << endl;
		getchar();
		return -1;
	}



	for (;;)
	{
		capture >> frame;
		if (frame.empty())
			continue;

		cvtColor(frame, gray, CV_RGB2GRAY);

		if (print_recog)
		{
			//if(Feat_2d_ptr->match_CalonderClassifier(gray,calond_desc))

			
			if (Feat_2d_ptr->track_Object(gray))	// 여기까지 진행하면 gray이미지와 특징점 계산 및 매칭 
			{
				for (int i = 0; i < 4; i++)
				{
					double x = origCorners[i].x, y = origCorners[i].y;
					cout << "HERE8_"<<i<<"~!\n";
					h11 = Feat_2d_ptr->H_transf.at<double>(0, 0);  h12 = Feat_2d_ptr->H_transf.at<double>(0, 1);  h13 = Feat_2d_ptr->H_transf.at<double>(0, 2);
					h21 = Feat_2d_ptr->H_transf.at<double>(1, 0);  h22 = Feat_2d_ptr->H_transf.at<double>(1, 1);  h23 = Feat_2d_ptr->H_transf.at<double>(1, 2);
					h31 = Feat_2d_ptr->H_transf.at<double>(2, 0);  h32 = Feat_2d_ptr->H_transf.at<double>(2, 1);  h33 = Feat_2d_ptr->H_transf.at<double>(2, 2);


					Z = 1. / (h31*x + h32*y + h33);
					X = (h11*x + h12*y + h13)*Z;
					Y = (h21*x + h22*y + h23)*Z;

					warpedPoints[i] = Point(cvRound(X), cvRound(Y));

				}
				
				line(frame, warpedPoints[0], warpedPoints[1], Scalar(255, 50, 200), 3);
				line(frame, warpedPoints[1], warpedPoints[2], Scalar(255, 50, 200), 3);
				line(frame, warpedPoints[2], warpedPoints[3], Scalar(255, 50, 200), 3);
				line(frame, warpedPoints[3], warpedPoints[0], Scalar(255, 50, 200), 3);
				cout << "HERE9~!\n";
			}

			Size textsize = getTextSize(Feat_2d_ptr->databaseRecognisedName, CV_FONT_HERSHEY_COMPLEX, 1.5, 2, 0);

			Point org((frame.cols - textsize.width) / 2, (frame.rows - textsize.height));
			
			putText(frame, Feat_2d_ptr->databaseRecognisedName, org, CV_FONT_HERSHEY_COMPLEX, 1.5, Scalar(50, 240, 240), 2, 8);
			cout << "HERE10~!\n";
		}
		
		imshow("frame", frame);



		key = waitKey(20);
		switch (key)
		{
		case 'r':
			if (Feat_2d_ptr->recognise_ImageDatabase(gray))
			{
				//  putText(frame, Feat_2d_ptr->databaseRecognisedName, Point(frame.cols/3,frame.rows - 20), 
				//	             CV_FONT_HERSHEY_COMPLEX, 1,  Scalar(50, 240, 240), 2, 8);


				//				  Feat_2d_ptr->Detector->detect( Feat_2d_ptr->databaseRecognisedImg, Feat_2d_ptr->databaseCalondKeypoints );

				//				  calond_desc.compute( Feat_2d_ptr->databaseRecognisedImg, Feat_2d_ptr->databaseCalondKeypoints, Feat_2d_ptr->databaseCalondDescriptors );

				origCorners[0].x = 0;
				origCorners[0].y = 0;
				origCorners[1].x = Feat_2d_ptr->databaseRecognisedImg.cols;
				origCorners[1].y = 0;
				origCorners[2].x = Feat_2d_ptr->databaseRecognisedImg.cols;
				origCorners[2].y = Feat_2d_ptr->databaseRecognisedImg.rows;
				origCorners[3].x = 0;
				origCorners[3].y = Feat_2d_ptr->databaseRecognisedImg.rows;
				
				
				Feat_2d_ptr->Detector_tracker->detect(Feat_2d_ptr->databaseRecognisedImg
					, Feat_2d_ptr->trackedKeypoints);
				Feat_2d_ptr->Descriptor_tracker->compute(Feat_2d_ptr->databaseRecognisedImg
					, Feat_2d_ptr->trackedKeypoints, Feat_2d_ptr->trackedDescriptors);
				Feat_2d_ptr->trackedDescriptors.convertTo(Feat_2d_ptr->trackedDescriptors, CV_32F);
				
				print_recog = true;
				cout << "HERE6~!\n";
			}
			break;
		case 's':
			print_recog = false;
			break;
		case 27:
			return -1;
			break;
		}

	}

	return 0;
}


