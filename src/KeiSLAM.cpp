/*
 * KeiSLAM.cpp
 *
 *  Created on: 27 Aug, 2015
 *      Author: root
 */
#include <opencv.hpp>
#include <camera_calibration.hpp>
#include <inttypes.h>
#include <iostream>
#include <cmath>
#include <vector>

using namespace cv;
using namespace std;

//class NormalizeAspectRatio{
//	public:
//		NormalizeAspectRatio(double width, double high){
//			Width = width;
//			High = high;
//			aspect3D = Mat::zeros(3, 3, CV_64F);
//			aspect3D.at<double>(0,0) = 1.0 / Width;
//			aspect3D.at<double>(1,1) = 1.0 / High;
//			aspect3D.at<double>(2,2) = 1.0;
//			aspect2D = Mat::zeros(2, 2, CV_64F);
//			aspect2D.at<double>(0,0) = 1.0 / Width;
//			aspect2D.at<double>(1,1) = 1.0 / High;
//		}
//		Mat Mat3D(Mat mat){
//			return aspect3D * mat;
//		}
//		Mat Mat2D(Mat mat){
//			return aspect3D * mat;
//		}
//		Point3d Point3D(Point3d p){
//			Mat m(3, 1, CV_64F);
//			m.at<double>(0,0) = p.x;
//			m.at<double>(1,0) = p.y;
//			m.at<double>(2,0) = p.z;
//			Point3d _p;
//			m = aspect3D * m;
//			_p.x = m.at<double>(0,0);
//			_p.y = m.at<double>(1,0);
//			_p.z = m.at<double>(2,0);
//			return _p;
//		}
//		Point2d Point2D(Point2d p){
//			Mat m(2, 1, CV_64F);
//			m.at<double>(0,0) = p.x;
//			m.at<double>(1,0) = p.y;
//			Point2d _p;
//			m = aspect2D * m;
//			_p.x = m.at<double>(0,0);
//			_p.y = m.at<double>(1,0);
//			return _p;
//		}
//
//	private:
//		double Width;
//		double High;
//		Mat aspect3D;
//		Mat aspect2D;
//};

//class RecoverDistortionPoint{
//	public:
//		RecoverDistortionPoint(Mat intrinsicMat, Mat distCoeffMat){
//			intrinsic = intrinsicMat;
//			distCoeff = distCoeffMat;
//		}
//
//		Point2d Recover(Point2d p){
//			double r2 = p.x * p.x + p.y * p.y;
//			double k1 = distCoeff.at<double>(0,0);
//			double k2 = distCoeff.at<double>(0,1);
//			double p1 = distCoeff.at<double>(0,2);
//			double p2 = distCoeff.at<double>(0,3);
//			Point2d corrPt;
//			corrPt.x = p.x * (1.0 + k1 * r2 + k2 * r2 * r2);
//			corrPt.y = p.y * (1.0 + k1 * r2 + k2 * r2 * r2);
//			corrPt.x += 2 * p1 * p.x * p.y + p2 * (r2 + 2 * p.x * p.x);
//			corrPt.y += 2 * p2 * p.x * p.y + p1 * (r2 + 2 * p.y * p.y);
//			return corrPt;
//		}
//	private:
//		Mat intrinsic;
//		Mat distCoeff;
//};

uint16_t maxNumOfFeatures = 500;
uint16_t minNumOfFeatures = 200;
uint16_t goodNumOfFeatures = 20;
uint16_t distanceThreshold = 20;

double calcDeterminant(Mat M){
	Vec3d r1 = M.col(0);
	Vec3d r2 = M.col(1);
	Vec3d r3 = M.col(2);
	return r1.dot(r2.cross(r3));
}

double calcVecNorm(Vec3d V){
	return sqrt(V.dot(V));
}

double calcMatNorm(Mat M){
	Mat A = M;
	Vec3f r1 = M.col(0);
	Vec3f r2 = M.col(1);
	Vec3f r3 = M.col(2);
	double sum[3] = {};
	for(int i = 0; i < 3; i++){
		sum[0] += r1[i];
		sum[1] += r2[i];
		sum[2] += r3[i];
	}
	return max(sum[0], max(sum[1], sum[2]));
}

//Mat calcNormalizedMat3D(Mat M){
//	Vec3d r1 = M.col(0);
//	Vec3d r2 = M.col(1);
//	Vec3d r3 = M.col(2);
//
////	for(int i = 0; i < 3; i++){
////		double norm = calcVecNorm(M.col(i));
////		for(int j = 0; j < 3; j++){
////			M.at<double>(j, i) /= norm;
////		}
////	}
//	return M;
//}

void Mat2Eulers(Mat R, Vec3d& eulers){
	eulers[0] = atan2(R.at<double>(0,1),R.at<double>(0,0)) * 180.0f / 3.1415926f;
	eulers[1] = asin(-R.at<double>(2,0)) * 180.0f / 3.1415926f;
	eulers[2] = atan2(R.at<double>(2,1),R.at<double>(2,2)) * 180.0f / 3.1415926f;
}

void calcRigidTransform(vector<Point3f> p1, vector<Point3f> p2, Mat& R, Point3f& t){
	Mat centroid1 = Mat::zeros(3, 1, CV_64F);
	Mat centroid2 = Mat::zeros(3, 1, CV_64F);
	for(int i = 0; i < p1.size(); i++){
		centroid1.at<double>(0,0) += p1.at(i).x;
		centroid1.at<double>(1,0) += p1.at(i).y;
		centroid1.at<double>(2,0) += p1.at(i).z;
		centroid2.at<double>(0,0) += p2.at(i).x;
		centroid2.at<double>(1,0) += p2.at(i).y;
		centroid2.at<double>(2,0) += p2.at(i).z;
	}
	centroid1.at<double>(0,0) /= p1.size();
	centroid1.at<double>(1,0) /= p1.size();
	centroid1.at<double>(2,0) /= p1.size();
	centroid2.at<double>(0,0) /= p1.size();
	centroid2.at<double>(1,0) /= p1.size();
	centroid2.at<double>(2,0) /= p1.size();
	Mat H = Mat::zeros(3, 3, CV_64F);
	for(int i = 0; i < p1.size(); i++){
		Mat v1 = Mat::zeros(3,1, CV_64F);
		Mat v2 = Mat::zeros(3,1, CV_64F);
		v1.at<double>(0, 0) = p1.at(i).x - centroid1.at<double>(0, 0);
		v1.at<double>(1, 0) = p1.at(i).y - centroid1.at<double>(1, 0);
		v1.at<double>(2, 0) = p1.at(i).z - centroid1.at<double>(2, 0);
		v2.at<double>(0, 0) = p2.at(i).x - centroid2.at<double>(0, 0);
		v2.at<double>(1, 0) = p2.at(i).y - centroid2.at<double>(1, 0);
		v2.at<double>(2, 0) = p2.at(i).z - centroid2.at<double>(2, 0);
		H += v1 * v2.t();
	}
	SVD decompose = SVD(H);
	R = decompose.vt.t() * decompose.u.t();
	if(calcDeterminant(R) < 0){
		R.at<double>(0,2) *= -1.0;
		R.at<double>(1,2) *= -1.0;
		R.at<double>(2,2) *= -1.0;
	}

	Mat translation = -R * centroid1 + centroid2;
	t.x = translation.at<double>(0,0);
	t.y = translation.at<double>(1,0);
	t.z = translation.at<double>(2,0);
}

void calcRigidTransformRANSAC(vector<Point3f> p1, vector<Point3f> p2, Mat3d R, Vec3d t, double n = 3, double p = 0.99, int k = 100){
	int count = 0;
	vector<double> Prob;
	vector<Mat> _R;
	vector<Point3f> _t;
	do{
		vector<Point3f> d1;
		vector<Point3f> d2;
		vector<int> rnd;
		for(int i = 0; i < n; i++){
			rnd.push_back(rand() % p1.size());
			d1.push_back(p1.at(rnd.at(i)));
			d2.push_back(p2.at(rnd.at(i)));
		}
		_R.push_back(Mat(3,3,CV_64F));
		_t.push_back(Point3f());
		calcRigidTransform(d1, d2, _R.at(count), _t.at(count));
		double w = 0;
		for(int i = 0; i < p1.size(); i++){
			if(i != rnd.at(count)){
				Mat _p1(3,1,CV_64F);
				Mat _p2(3,1,CV_64F);
				_p1.at<double>(0,0) = p1.at(i).x;
				_p1.at<double>(1,0) = p1.at(i).y;
				_p1.at<double>(2,0) = p1.at(i).z;
				_p2.at<double>(0,0) = p2.at(i).x;
				_p2.at<double>(1,0) = p2.at(i).y;
				_p2.at<double>(2,0) = p2.at(i).z;
				Mat __t(3,1,CV_64F);
				__t.at<double>(0,0) = _t.at(count).x;
				__t.at<double>(1,0) = _t.at(count).y;
				__t.at<double>(2,0) = _t.at(count).z;
				Mat v = _R.at(count) * _p1 + __t - _p2;
				if(v.dot(v) < 0.001){
					w++;
				}
			}
		}
		Prob.push_back(w / p1.size());
	}while(Prob.at(count) < p && count++ < k);
	if(Prob.at(count-1) >= p){
		R = _R.at(count-1);
		t[0] = _t.at(count-1).x;
		t[1] = _t.at(count-1).y;
		t[2] = _t.at(count-1).z;
	}
	else{
		double prob = 0;
		int index = 0;
		for(int i = 0; i < Prob.size(); i++){
			for(int j = 0; j < Prob.size(); j++){
				prob = max(Prob.at(i), Prob.at(j));
				if(prob == Prob.at(i)){
					index = i;
				}
				else if(prob == Prob.at(j)){
					index = j;
				}
			}
		}
		R = _R.at(index);
		t[0] = _t.at(index).x;
		t[1] = _t.at(index).y;
		t[2] = _t.at(index).z;
	}
}


int main(int argc, char **argv)
{
	Mat intrinsic1;
	Mat intrinsic2;
	Mat extrinsic1;
	Mat extrinsic2;
	Mat distCoeff1;
	Mat distCoeff2;
	Mat rectMatrix1;
	Mat rectMatrix2;
	Mat projMatrix1;
	Mat projMatrix2;

//	NormalizeAspectRatio normalizeAspectRatio(320.0, 240.0);

	string inputSettingsFile = "left_camera.xml";
	FileStorage fs(inputSettingsFile, FileStorage::READ); // Read the settings
	if (!fs.isOpened())
	{
		cout << "Could not open the left camera configuration file: \"" << inputSettingsFile << "\"" << endl;
		return -1;
	}
	fs["Camera_Matrix"] >> intrinsic1;
	fs["Distortion_Coefficients"] >> distCoeff1;
	fs["Projection_Matrix"] >> projMatrix1;
	fs["Per_View_Reprojection_Errors"] >> rectMatrix1;

	fs.release();

	inputSettingsFile = "right_camera.xml";
	fs.open(inputSettingsFile, FileStorage::READ); // Read the settings
	if (!fs.isOpened())
	{
		cout << "Could not open the right camera configuration file: \"" << inputSettingsFile << "\"" << endl;
		return -1;
	}

	fs["Camera_Matrix"] >> intrinsic2;
	fs["Distortion_Coefficients"] >> distCoeff2;
	fs["Projection_Matrix"] >> projMatrix2;
	fs["Per_View_Reprojection_Errors"] >> rectMatrix2;
	fs.release();

	extrinsic1 = intrinsic1.inv() * projMatrix1;
//	double norm = calcMatNorm(extrinsic1.colRange(0,3));
//	extrinsic1 /= norm;
	Mat translation = Mat::zeros(3, 4, CV_64F);
	translation.at<double>(0, 3) = projMatrix2.at<double>(0, 3) / intrinsic1.at<double>(0, 0);// / norm;
	extrinsic2 = extrinsic1 + translation;
//	intrinsic1 = normalizeAspectRatio.Mat3D(intrinsic1);
//	intrinsic2 = normalizeAspectRatio.Mat3D(intrinsic2);
//	projMatrix1 = normalizeAspectRatio.Mat3D(projMatrix1);
//	projMatrix2 = normalizeAspectRatio.Mat3D(projMatrix2);

//	extrinsic1 = Mat::zeros(3, 4, CV_64F);
//	extrinsic1.at<double>(0, 0) = 1.0;
//	extrinsic1.at<double>(1, 1) = 1.0;
//	extrinsic1.at<double>(2, 2) = 1.0;
//
//	extrinsic2 = Mat::zeros(3, 4, CV_64F);
//	extrinsic2.at<double>(0, 0) = 1.0;
//	extrinsic2.at<double>(1, 1) = 1.0;
//	extrinsic2.at<double>(2, 2) = 1.0;

//	intrinsic1 = normalizeAspectRatio.Mat3D(intrinsic1);
//	intrinsic2 = normalizeAspectRatio.Mat3D(intrinsic2);
//	RecoverDistortionPoint recover1(intrinsic1, distCoeff1);
//	RecoverDistortionPoint recover2(intrinsic2, distCoeff2);

//	extrinsic1 = Mat::zeros(3, 4, CV_64F);
//	extrinsic2 = Mat::zeros(3, 4, CV_64F);

//	extrinsic1.at<double>(0, 0) = 1.0;
//	extrinsic1.at<double>(1, 1) = 1.0;
//	extrinsic1.at<double>(2, 2) = 1.0;
//	extrinsic2.at<double>(0, 0) = 1.0;
//	extrinsic2.at<double>(1, 1) = 1.0;
//	extrinsic2.at<double>(2, 2) = 1.0;
//
//
//	projMatrix1 = rectMatrix1 * projMatrix1;
//	projMatrix2 = rectMatrix2 * projMatrix2;

	VideoCapture cap1(1);
	VideoCapture cap2(2);
    if(!cap1.isOpened()){
        return -1;
	}
	if(!cap2.isOpened()){
        return -1;
	}

	cap1.set(CV_CAP_PROP_FRAME_WIDTH, 320);
	cap1.set(CV_CAP_PROP_FRAME_HEIGHT, 240);

	cap2.set(CV_CAP_PROP_FRAME_WIDTH, 320);
	cap2.set(CV_CAP_PROP_FRAME_HEIGHT, 240);

	namedWindow("Test");
	namedWindow("Left");
	namedWindow("Right");

	moveWindow("Test", 0, 0);
	moveWindow("Left", 0, 300);
	moveWindow("Right", 400, 300);

	Mat frame;
	Mat temp[3] = {255 * Mat::ones(240, 640, CV_8U), 255 * Mat::ones(240, 640, CV_8U), 255 * Mat::ones(240, 640, CV_8U)};
	merge(temp, 3, frame);

	Ptr<AdjusterAdapter> AdjusterAdapter1 = AdjusterAdapter::create("FAST");
	Ptr<AdjusterAdapter> AdjusterAdapter2 = AdjusterAdapter::create("FAST");
	Ptr<DescriptorExtractor> descriptorExtractor1 = DescriptorExtractor::create("ORB");
	Ptr<DescriptorExtractor> descriptorExtractor2 = DescriptorExtractor::create("ORB");
	Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create("BruteForce-Hamming");
	vector<DMatch> dMatch;
	int count = 0;

	vector<Point2f> p1;
	vector<Point2f> p2;
	vector<int> x1;
	vector<int> x2;
	vector<int> y1;
	vector<int> y2;
	Mat R = Mat::zeros(3,3,CV_64F);
	R.at<double>(0,0) = 1.0;
	R.at<double>(1,1) = 1.0;
	R.at<double>(2,2) = 1.0;
	Mat T = Mat::zeros(3,3,CV_64F);
	Vec3d t;
	Vec3d euler;
	vector<Point3f> points3D;
	vector<Point3f> prevPoints3D;
//	RANSAC ransac(3, 0.99, 1000, )

	while (true){
		count++;
        if(waitKey(20) == 27){
			break;
		}

		Mat frame1;
		Mat frame2;

		vector<KeyPoint> keypoints1;
		vector<KeyPoint> keypoints2;
		vector<KeyPoint> goodkeypoints1;
		vector<KeyPoint> goodkeypoints2;
		vector<DMatch> goodDMatch;

		Mat descriptors1;
		Mat descriptors2;

		Mat goodDescriptors1;
		Mat goodDescriptors2;

		cap1 >> frame1;
		AdjusterAdapter1->detect(frame1, keypoints1);

		cap2 >> frame2;
		AdjusterAdapter2->detect(frame2, keypoints2);

		if(keypoints1.size() > maxNumOfFeatures){
			AdjusterAdapter1->tooMany(maxNumOfFeatures, keypoints1.size());
		}

		if(keypoints1.size() < minNumOfFeatures){
			AdjusterAdapter1->tooFew(minNumOfFeatures, keypoints2.size());
		}

		if(keypoints2.size() > maxNumOfFeatures){
			AdjusterAdapter2->tooMany(maxNumOfFeatures, keypoints2.size());
		}

		if(keypoints2.size() < minNumOfFeatures){
			AdjusterAdapter2->tooFew(minNumOfFeatures, keypoints2.size());
		}

		if(keypoints1.size() > minNumOfFeatures && keypoints2.size() > minNumOfFeatures){
			descriptorExtractor1->compute(frame1, keypoints1, descriptors1);
			descriptorExtractor2->compute(frame2, keypoints2, descriptors2);
			descriptorMatcher->match(descriptors1, descriptors2, dMatch);
			p1.clear();
			p2.clear();
			goodkeypoints1.clear();
			goodkeypoints2.clear();
			for(int i = 0; i < dMatch.size(); i++){
				if(dMatch.at(i).distance < distanceThreshold){
					p1.push_back(keypoints1.at(dMatch.at(i).queryIdx).pt);
					p2.push_back(keypoints2.at(dMatch.at(i).trainIdx).pt);
					goodkeypoints1.push_back(keypoints1.at(dMatch.at(i).queryIdx));
					goodkeypoints2.push_back(keypoints2.at(dMatch.at(i).trainIdx));
				}
			}

			descriptorExtractor1->compute(frame1, goodkeypoints1, goodDescriptors1);
			descriptorExtractor2->compute(frame2, goodkeypoints2, goodDescriptors2);
			descriptorMatcher->match(goodDescriptors1, goodDescriptors2, goodDMatch);

			drawKeypoints(frame1, goodkeypoints1, frame1, Scalar(0,0,255));
			drawKeypoints(frame2, goodkeypoints2, frame2, Scalar(0,0,255));
			drawMatches(frame1, goodkeypoints1, frame2, goodkeypoints2, goodDMatch, frame);
//			if(count % 25 == 0){
				cout << "Number of Matched Pairs: " << dMatch.size()
					<< " Number of P1: " << p1.size()
					<< " Number of P2: " << p2.size() << endl;


				if(goodkeypoints1.size() > goodNumOfFeatures){
//					Mat F = findFundamentalMat(p1, p2);
//					Mat E = intrinsic2.t() * F * intrinsic1;
//					SVD decompose = SVD(E);
//					Mat U = decompose.u;
//					Mat W = Mat::zeros(3, 3, CV_64F);
//					W.at<double>(0,0) = decompose.w.at<double>(0, 0);
//					W.at<double>(1,1) = decompose.w.at<double>(0, 1);
//					W.at<double>(2,2) = decompose.w.at<double>(0, 2);
//					Mat VT = decompose.vt;
//					Mat S = Mat::zeros(3, 3, CV_64F);
//					S.at<double>(0,1) = -1.0;
//					S.at<double>(1,0) = 1.0;
//					S.at<double>(2,2) = 1.0;
//					R = U * S.t() * VT;
//					T = U* S * W * U.t();
//					t[0] = T.at<double>(0,2);
//					t[1] = T.at<double>(1,2);
//					t[2] = T.at<double>(2,2);

					Mat points4D(4, p1.size(), CV_64F);
					triangulatePoints(projMatrix1, projMatrix2, p1, p2, points4D);
					vector<Point2f> validedPoints;
					prevPoints3D = points3D;
					points3D.clear();
					for(int i = 0; i < p1.size(); i++){
						float x = points4D.at<float>(0, i) / points4D.at<float>(3, i);
						float y = points4D.at<float>(1, i) / points4D.at<float>(3, i);
						float z = points4D.at<float>(2, i) / points4D.at<float>(3, i);
						if(fabs(x) < 10000.0f && fabs(y) < 10000.0f && fabs(z) < 10000.0f){
							points3D.push_back(Point3f(x,y,z));
							validedPoints.push_back(p1.at(i));
						}
					}
					Mat AffineTransformMat(3, 4, CV_64F);
					Mat inliers;
					vector<Point3f> t;
					vector<Mat> _R;
					if(points3D.size() > 0 && prevPoints3D.size() > 0){
						vector<Point3f> P3D;
						vector<Point3f> prevP3D;
						for(int i = 0; i < min(points3D.size(), prevPoints3D.size()); i++){
							P3D.push_back(points3D.at(i));
							prevP3D.push_back(prevPoints3D.at(i));
						}
						for(int i = 0; i < P3D.size() / 3; i++){
							vector<Point3f> __p1;
							vector<Point3f> __p2;
							__p1.push_back(P3D.at(i * 3));
							__p1.push_back(P3D.at(i * 3 + 1));
							__p1.push_back(P3D.at(i * 3 + 2));
							__p2.push_back(prevP3D.at(i * 3));
							__p2.push_back(prevP3D.at(i * 3 + 1));
							__p2.push_back(prevP3D.at(i * 3 + 2));
							t.push_back(Point3f());
							_R.push_back(Mat(3,3,CV_64F));
							calcRigidTransform(__p1, __p2, _R.at(i), t.at(i));
//							delR *= R;
//							R += delR;
//							R = calcNormalizedMat3D(R);
						}
//						estimateAffine3D(P3D, prevP3D, AffineTransformMat, inliers);
					}
					for(int i = 0; i < t.size(); i++){
						cout << i << ":" <<  endl;
						cout << "t: " << t.at(i) << endl;
						cout << "R: " << _R.at(i) << endl;
						cout << endl;
					}
//					vector<double> rvec;
//					vector<double> tvec;
//					solvePnPRansac(points3D, validedPoints, intrinsic1, distCoeff1, rvec, tvec);
//					Rodrigues(rvec, R);
//					R = AffineTransformMat.colRange(0, 3);
//					R = calcNormalizedMat3D(R);
//					float norm = calcMatNorm(R);
//					cout << norm << endl;
//					cout << R.t()*R << endl;
//					T = AffineTransformMat.colRange(3, 4);
//					euler[0] = atan2(R.at<double>(0,1),R.at<double>(0,0)) * 180.0f / 3.1415926f;
//					euler[1] = asin(-R.at<double>(2,0)) * 180.0f / 3.1415926f;
//					euler[2] = atan2(R.at<double>(2,1),R.at<double>(2,2)) * 180.0f / 3.1415926f;
//					cout << "R: " << R << endl;
//					cout << "T: " << T << endl;
//					cout << euler << endl;
//					cout << endl;
				}
//			}

//			if(count % 25 == 0){
//				x1.clear();
//				y1.clear();
//				x2.clear();
//				y2.clear();
//				p1.clear();
//				p2.clear();
//				for(int i = 0; i < dMatch.size(); i++){
//					if(dMatch.at(i).distance < 20){
//						x1.push_back(keypoints1.at(dMatch.at(i).queryIdx).pt.x);
//						y1.push_back(keypoints1.at(dMatch.at(i).queryIdx).pt.y);
//						x2.push_back(keypoints2.at(dMatch.at(i).trainIdx).pt.x);
//						y2.push_back(keypoints2.at(dMatch.at(i).trainIdx).pt.y);
//						p1.push_back(keypoints1.at(dMatch.at(i).queryIdx).pt);
//						p2.push_back(keypoints2.at(dMatch.at(i).trainIdx).pt);
//					}
//				}
//			}
//
//			stringstream ss1;
//			stringstream ss2;
//			for(int i = 0; i < p1.size(); i++){
////				ss1.str("");
////				ss2.str("");
////				ss1 << x1.at(i) << " " << y1.at(i);
////				ss2 << x2.at(i) << " " << y2.at(i);
////				putText(frame1, ss1.str(), Point(10,20+i*10), FONT_HERSHEY_COMPLEX_SMALL, 0.5, Scalar(255,0,0));
////				putText(frame2, ss2.str(), Point(10,20+i*10), FONT_HERSHEY_COMPLEX_SMALL, 0.5, Scalar(255,0,0));
//				circle(frame1, p1.at(i), 5, Scalar(0,0,i*25), 2);
//				circle(frame2, p2.at(i), 5, Scalar(0,0,i*25), 2);
//
//				Mat F = findFundamentalMat(p1,p2);
//				ss1.str("");
//				ss1 << F.at<double>(0,0) << " " << F.at<double>(0,1) << " " << F.at<double>(0,2);
//				cout << ss1.str() << endl;
//				ss1.str("");
//				ss1 << F.at<double>(1,0) << " " << F.at<double>(1,1) << " " << F.at<double>(1,2);
//				cout << ss1.str() << endl;
//				ss1.str("");
//				ss1 << F.at<double>(2,0) << " " << F.at<double>(2,1) << " " << F.at<double>(2,2);
//				cout << ss1.str() << endl;
//				cout << endl;
//			}

			imshow("Test", frame);
		}
		imshow("Left", frame1);
		imshow("Right", frame2);
	}

	return 0;
}

