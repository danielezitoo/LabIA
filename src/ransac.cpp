#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>

using namespace cv;
using namespace std;

/*
RANSAC stima un modello robusto iterando tra selezione casuale di punti e verifica degli inlier, scartando gli outlier dovuti a rumore o mismatch.
*/
vector<DMatch> ransac(const vector<KeyPoint>& keypoints1, const vector<KeyPoint>& keypoints2, const vector<DMatch>& matches, double threshold = 5.0, int maxIterations = 10000) {
    vector<Point2f> pts1, pts2;
    for (int i = 0; i < matches.size(); i++) {
        pts1.push_back(keypoints1[matches[i].queryIdx].pt);
        pts2.push_back(keypoints2[matches[i].trainIdx].pt);
    }

    // Utilizzo RANSAC implementato da openCV così da esssere sicuramente più efficiente
    vector<uchar> inliersVec;
    Mat H = findHomography(pts1, pts2, LMEDS, threshold, inliersVec, maxIterations);

    vector<DMatch> inlier;
    for (int i = 0; i < matches.size(); i++) {
        if (inliersVec[i]) {
            inlier.push_back(matches[i]);
        }
    }

    return inlier;
}