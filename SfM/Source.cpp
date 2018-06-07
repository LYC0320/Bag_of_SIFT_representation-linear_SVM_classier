#include "opencv2\opencv.hpp"
#include "opencv2/nonfree/nonfree.hpp"  
#include "opencv2/legacy/legacy.hpp"
#include <opencv2/ml/ml.hpp>
#include "math.h"
#include <io.h>

#include "dirent.h"

using namespace cv;
using namespace std;

#define clusterNum (50)
#define descriptorNum (200)

int main()
{
	cout << "Process start" << endl;

	DIR *dir;
	struct dirent *ent;
	vector<string> trainingFolder;
	int folderFilter = 0;
	if ((dir = opendir("../bin/hw4_data/train")) != NULL)
	{

		while ((ent = readdir(dir)) != NULL)
		{
			if (folderFilter >= 3)
			{
				trainingFolder.push_back(ent->d_name);
			}
			folderFilter++;
		}
		closedir(dir);
	}

	Mat featuresUnclustered;
	vector<int>KeypointNum;

	

	// read class folder
	for (int folderNum = 0; folderNum < trainingFolder.size(); folderNum++)
	{

		vector<string> trainingDataName;
		string tempPath = "hw4_data/train/";
		tempPath = tempPath + trainingFolder[folderNum];

		const char *tempC = tempPath.c_str();

		if ((dir = opendir(tempC)) != NULL)
		{
			while ((ent = readdir(dir)) != NULL)
			{
				if (ent->d_name[ent->d_namlen - 1] == 'g')
				{
					trainingDataName.push_back(ent->d_name);
				}
			}
			closedir(dir);
		}

		// read image file
		for (int dataNum = 0; dataNum < trainingDataName.size(); dataNum++)
		{
			string tempPath2 = tempPath + "/" + trainingDataName[dataNum];

			Mat src = imread(tempPath2);
			Mat dst, descriptor;

			resize(src, dst, Size(256, 256), 0, 0, INTER_LINEAR);

			vector<KeyPoint> keyPoint;

			SiftDescriptorExtractor extractor(descriptorNum);
			extractor.detect(dst, keyPoint);

			KeypointNum.push_back(keyPoint.size());

			extractor.compute(dst, keyPoint, descriptor);

			featuresUnclustered.push_back(descriptor);
		}
	}

	for (int i = 0; i < KeypointNum.size(); i++)
	{
		cout << KeypointNum[i] << endl;
	}

	cout << "descriptor done" << endl;

	Mat clusters;
	Mat center(clusterNum, 1, CV_32FC3);

	// kmeans
	kmeans(featuresUnclustered, clusterNum, clusters, cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 0.1), clusterNum, KMEANS_PP_CENTERS, center);

	cout << "K-means done" << endl;

	Mat histogram(100, clusterNum, CV_32FC1);

	// initial histogram
	for (int i = 0; i < histogram.rows; i++)
	{
		for (int j = 0; j < histogram.cols; j++)
		{
			histogram.at<int>(i,j) = 0;
		}
	}

	int pointCount = 0;
	int pictureCount = 0;

	float histagramF[1500][clusterNum];
	//float length = 0;

	
	// compute histogram // labelNum
	for (int k = 0; k < trainingFolder.size(); k++)
	{						// pictureNum
		for (int i = 0; i < 100; i++)
		{						// pointNum
			for (int j = 0; j < KeypointNum[i + pictureCount]; j++)
			{
				histogram.at<int>(i, clusters.at<int>(j + pointCount, 0))++;
			}

			// normalized
			//for (int jF = 0; jF < clusterNum; jF++)
			//{
			//	length += histogram.at<int>(i, jF)*histogram.at<int>(i, jF);
			//}

			// normalized
			//length = sqrtf(length);

			pointCount += KeypointNum[i + pictureCount];

			for (int jF = 0; jF < clusterNum; jF++)
			{																
				histagramF[i + pictureCount][jF] = histogram.at<int>(i, jF);
			}
		}

		// initial histogram
		for (int ii = 0; ii < histogram.rows; ii++)
		{
			for (int jj = 0; jj < histogram.cols; jj++)
			{
				histogram.at<int>(ii, jj) = 0;
			}
		}

		pictureCount += 100;

		// normalized
		//length = 0;
	}

	cout << "histogram done" << endl;

	float label[1500];

	// initial label
	for (int i = 0; i < 1500; i++)
	{
		if (i < 100)
		{
			label[i] = 1;
		}
		else if (i >= 100 && i < 200)
		{
			label[i] = 2;
		}
		else if (i >= 200 && i < 300)
		{
			label[i] = 3;
		}
		else if (i >= 300 && i < 400)
		{
			label[i] = 4;
		}
		else if (i >= 400 && i < 500)
		{
			label[i] = 5;
		}
		else if (i >= 500 && i < 600)
		{
			label[i] = 6;
		}
		else if (i >= 600 && i < 700)
		{
			label[i] = 7;
		}
		else if (i >= 700 && i < 800)
		{
			label[i] = 8;
		}
		else if (i >= 800 && i < 900)
		{
			label[i] = 9;
		}
		else if (i >= 900 && i < 1000)
		{
			label[i] = 10;
		}
		else if (i >= 1000 && i < 1100)
		{
			label[i] = 11;
		}
		else if (i >= 1100 && i < 1200)
		{
			label[i] = 12;
		}
		else if (i >= 1200 && i < 1300)
		{
			label[i] = 13;
		}
		else if (i >= 1300 && i < 1400)
		{
			label[i] = 14;
		}
		else if (i >= 1400 && i < 1500)
		{
			label[i] = 15;
		}
	}

	Mat labelMat = Mat(1500, 1, CV_32FC1, label);

	Mat histogramFMat = Mat(1500, clusterNum, CV_32FC1, histagramF);

	CvSVM svm;
	
	CvSVMParams params;
	params.svm_type = SVM::C_SVC;
	params.C = 0.1;
	params.kernel_type = SVM::LINEAR;
	params.term_crit = TermCriteria(CV_TERMCRIT_ITER, (int)1e7, 1e-6);

	// compute SVM
	svm.train(histogramFMat, labelMat, Mat(), Mat(), params);
	
	cout << "SVM done" << endl;

	float correctNum = 0;

	// compute testing data
	for (int folderNum = 0; folderNum < trainingFolder.size(); folderNum++)
	{
		vector<string> testingDataName;
		string tempPath = "hw4_data/test/";
		tempPath = tempPath + trainingFolder[folderNum];

		const char *tempC = tempPath.c_str();

		if ((dir = opendir(tempC)) != NULL)
		{
			while ((ent = readdir(dir)) != NULL)
			{
				if (ent->d_name[ent->d_namlen - 1] == 'g')
				{
					testingDataName.push_back(ent->d_name);
				}
			}
			closedir(dir);
		}
		
		for (int dataNum = 0; dataNum < testingDataName.size(); dataNum++)
		{
			string tempPath2 = tempPath + "/" + testingDataName[dataNum];

			Mat src = imread(tempPath2);
			Mat dst, descriptor;

			resize(src, dst, Size(256, 256), 0, 0, INTER_LINEAR);

			vector<KeyPoint> keyPoint;

			SiftDescriptorExtractor extractor(descriptorNum);
			extractor.detect(dst, keyPoint);

			extractor.compute(dst, keyPoint, descriptor);


			float minDistance = 100000000;
			int minCenterIndex[descriptorNum];
			float testDataH[clusterNum];

			for (int i = 0; i < clusterNum; i++)
			{
				testDataH[i] = 0;
			}

			for (int i = 0; i < descriptorNum; i++)
			{
				minCenterIndex[i] = 0;
			}

			// compute minCenterIndex
			for (int k = 0; k < descriptor.rows; k++)
			{
				for (int i = 0; i < center.rows; i++)
				{
					float tempDistance = 0;

					for (int j = 0; j < center.cols; j++)
					{
						tempDistance += (descriptor.at<float>(k, j) - center.at<float>(i, j))*(descriptor.at<float>(k, j) - center.at<float>(i, j));
					}

					if (tempDistance < minDistance && k < descriptorNum)
					{
						minDistance = tempDistance;
						minCenterIndex[k] = i;
					}
				}
				minDistance = 100000000;
			}

			// compute histogram
			for (int i = 0; i < descriptor.rows; i++)
			{
				if (i < descriptorNum)
				{
					testDataH[minCenterIndex[i]]++;
				}
			}



			// normalized
			//float length2 = 0;

			//for (int i = 0; i < clusterNum; i++)
			//{
			//	length2 += testDataH[i] * testDataH[i];
			//}

			//length2 = sqrtf(length2);

			//for (int i = 0; i < clusterNum; i++)
			//{
			//	testDataH[i] = testDataH[i] / length2;
			//}
			// normalized



			Mat testHistogram(1, clusterNum, CV_32FC1, testDataH);

			// predict svm
			float response = svm.predict(testHistogram);

			cout << trainingFolder[folderNum] << "/" << testingDataName[dataNum] << ":" << trainingFolder[response - 1] << endl;
			
			if (response == folderNum + 1)
			{
				correctNum++;
			}
		}
	}
	cout << correctNum << endl;

	cout << "Accuracy:" << correctNum /150.0 << endl;

	system("Pause");
	waitKey(0);
	return 0;
}
