#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>

using namespace cv;
using namespace std;

// Fa una X sui corners
void drawCorners(Mat &img, const vector<KeyPoint> &keypoints) {
    for (int i = 0; i < keypoints.size(); i++) {
        Point pt(cvRound(keypoints[i].pt.x), cvRound(keypoints[i].pt.y));

        line(img, Point(pt.x - 5, pt.y - 5), Point(pt.x + 5, pt.y + 5), Scalar(0, 255, 0));
        line(img, Point(pt.x - 5, pt.y + 5), Point(pt.x + 5, pt.y - 5), Scalar(0, 255, 0));
    }
}

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
    Mat H = findHomography(pts1, pts2, RANSAC, threshold, inliersVec, maxIterations);

    vector<DMatch> inlier;
    for (int i = 0; i < matches.size(); i++) {
        if (inliersVec[i]) {
            inlier.push_back(matches[i]);
        }
    }

    return inlier;
}

// Salva immagini per KeyPoints
void saveImageKP(const Mat &img, const string &imgPath, const string &command) {
    // Estrazione del nome del file senza estensione
    int start = imgPath.find_last_of("/\\") + 1;
    int end = imgPath.find_last_of(".");
    string filename = imgPath.substr(start, end - start);

    string savePath = "output/" + filename + "_" + command + ".jpg";
    imwrite(savePath, img);
}

// Salva immagini per Matching
void saveImageM(const Mat &img, const string &imgPath1, const string &imgPath2, const string &command) {
    // Estrazione del nome del file senza estensione
    int start = imgPath1.find_last_of("/\\") + 1;
    int end = imgPath1.find_last_of(".");
    string filename1 = imgPath1.substr(start, end - start);

    // Estrazione del nome del file senza estensione
    start = imgPath2.find_last_of("/\\") + 1;
    end = imgPath2.find_last_of(".");
    string filename2 = imgPath2.substr(start, end - start);
    int imgIndex2 = stoi(filename2);

    string savePath = "output/" + filename1 + "_" + filename2 + command + ".jpg";
    imwrite(savePath, img);
}