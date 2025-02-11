#ifndef GLOBALS_H
#define GLOBALS_H

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <iostream>
#include <cstring>
#include <sstream>
#include <opencv2/highgui.hpp>
#include <filesystem>
#include <opencv2/calib3d.hpp>
#include <bitset>
#include <random>
#include <cstdlib>
#include <cmath>
#include <ctime>
// Per levare i warning generati dalle Trackbar
#include <opencv2/core/utils/logger.hpp>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

bool checkImage(Mat& img, const string& imgPath);
void mergeImages(const Mat& img1, const Mat& img2, const vector<KeyPoint>& keypoints1, const vector<KeyPoint>& keypoints2, const vector<DMatch>& matches, double threshold, int maxIterations);
void drawCorners(Mat &img, const vector<KeyPoint> &keypoints);
vector<DMatch> ransac(const vector<KeyPoint>& keypoints1, const vector<KeyPoint>& keypoints2, const vector<DMatch>& matches, double threshold, int maxIterations);
void saveImageKP(const Mat &img, const string &imgPath, const string &command);
void saveImageM(const Mat &img, const string &imgPath1, const string &imgPath2, const string &command);
void saveImageP(const Mat &img, const string &imgPath1, const string &imgPath2, const string &command);

// Dichiarazione delle variabili globali
extern Mat img, imgWithCorners, img1, img2, imgMatches, descriptors1, descriptors2, imgCopy, panorama;
extern vector<KeyPoint> keypoints, keypoints1, keypoints2;
extern vector<DMatch> matches;

extern int MAX_SIZE;

// Parametri di default per Harris
extern int window_harris;
extern int sigma_bar_harris;
extern int threshold_bar_harris;

// Parametri di default per FAST
extern int threshold_bar_fast;
extern int n_bar_fast;
extern int dist_bar_fast;

// Parametri di default per Shi-Tomasi
extern int threshold_bar_shi_tomasi;
extern int window_size_bar_shi_tomasi;

// Parametri di default per SIFT
extern int threshold_bar_ransac_sift;
extern int maxIterations_bar_ransac_sift;

// Parametri di default per ORB
extern int threshold_fast_bar_orb;
extern int patch_size_bar_orb;
extern int n_bits_bar_orb;
extern int threshold_bar_ransac_orb;
extern int maxIterations_bar_ransac_orb;

// Parametri di default per match Harris
extern int window_bar_match_harris;
extern double sigma_bar_match_harris;
extern float threshold_bar_match_harris;
extern int patch_size_bar_match_harris;
extern int n_bits_bar_match_harris;
extern int threshold_bar_ransac_harris;
extern int maxIterations_bar_ransac_harris;
extern string descriptorType;

// Parametri di default per match Fast
extern int threshold_bar_match_fast;
extern int n_bar_match_fast;
extern int dist_bar_match_fast;
extern int threshold_bar_ransac_fast;
extern int maxIterations_bar_ransac_fast;

// Parametri di default per Shi_Tomasi
extern int threshold_bar_match_shi_tomasi;
extern int window_size_bar_match_shi_tomasi;
extern int patch_size_bar_match_shi_tomasi;
extern int n_bits_bar_match_shi_tomasi;
extern int threshold_bar_ransac_shi_tomasi;
extern int maxIterations_bar_ransac_shi_tomasi;

// Parametro per fare merge se richiesto
extern bool merge_after_match;
extern bool merge_tot;

#endif