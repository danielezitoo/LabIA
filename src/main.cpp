#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <cstring>
#include <sstream>

#include "harris.cpp"
#include "fast.cpp"
#include "draw_corners.cpp"
#include "patch_descriptors.cpp"
#include "ransac.cpp"

using namespace cv;
using namespace std;

Mat img, imgWithCorners, img1, img2, imgMatches;
vector<KeyPoint> keypoints, keypoints1, keypoints2;

// Parametri di default per Harris ("int" perchè è il tipo che vuole createTrackBar(), li convertirò successivamente)
int window_harris = 20;
int sigma_bar_harris = 34;
int k_bar_harris = 4;
int threshold_bar_harris = 100;

// Harris corner detection
void updateCornersHarris(int, void*) {
    imgWithCorners = img.clone();

    if (window_harris % 2 == 0) {
        window_harris++;
    }
    if (window_harris <= 1) {
        window_harris = 3;
    }
    double sigma = sigma_bar_harris / 10.0;
    float k = 0.04f + (k_bar_harris / 100.0f) * (0.06f - 0.04f);
    float threshold = threshold_bar_harris / 10.0f;

    keypoints = harrisCornerDetection(img, window_harris, sigma, k, threshold);

    drawCorners(imgWithCorners, keypoints);

    // Cerco di non far creare una schermata troppo grande per farlo entrare nello schermo
    /*Mat resizedImg;
    resize(imgWithCorners, resizedImg, Size(imgWithCorners.cols / 2, imgWithCorners.rows / 2));

    imshow("Harris Corner Detection", resizedImg);*/
    imshow("Harris Corner Detection", imgWithCorners);
}

void do_harris(const string& imgPath) {
    img = imread(imgPath);

    if (img.empty()) {
        cout << "Errore nel caricare l'immagine!" << endl;
        return;
    }

    updateCornersHarris(0, 0);

    // Estraggo il nome del file (senza estensione)
    size_t start = imgPath.find_last_of("/\\") + 1;
    size_t end = imgPath.find_last_of(".");
    string filename = imgPath.substr(start, end - start);
    int imgIndex = stoi(filename);
    string savePath = "output/" + to_string(imgIndex) + "_corners_harris.jpg";
    imwrite(savePath, imgWithCorners);

    // Crea le trackbar per cambiare i parametri in tempo reale
    createTrackbar("Window Size", "Harris Corner Detection", &window_harris, 20, updateCornersHarris);
    createTrackbar("Sigma", "Harris Corner Detection", &sigma_bar_harris, 200, updateCornersHarris);
    // createTrackbar("K", "Harris Corner Detection", &k_bar_harris, 100, updateCornersHarris);
    createTrackbar("Threshold", "Harris Corner Detection", &threshold_bar_harris, 100, updateCornersHarris);

    waitKey(0);
}

// Parametri di default per FAST
int threshold_bar_fast = 50;
int n_bar_fast = 12;
int dist_bar_fast = 3;

// Harris corner detection
void updateCornersFast(int, void*) {
    imgWithCorners = img.clone();
    cvtColor(imgWithCorners, imgWithCorners, COLOR_GRAY2BGR);

    keypoints = fastCornerDetection(img, threshold_bar_fast, n_bar_fast, dist_bar_fast);

    drawCorners(imgWithCorners, keypoints);

    // Cerco di non far creare una schermata troppo grande per farlo entrare nello schermo
    /*Mat resizedImg;
    resize(imgWithCorners, resizedImg, Size(imgWithCorners.cols / 2, imgWithCorners.rows / 2));

    imshow("FAST Corner Detection", resizedImg);*/
    imshow("FAST Corner Detection", imgWithCorners);
}

void do_fast(const string& imgPath) {
    img = imread(imgPath, IMREAD_GRAYSCALE);

    if (img.empty()) {
        cout << "Errore nel caricare l'immagine!" << endl;
        return;
    }

    updateCornersFast(0, 0);

    // Estraggo il nome del file (senza estensione)
    size_t start = imgPath.find_last_of("/\\") + 1;
    size_t end = imgPath.find_last_of(".");
    string filename = imgPath.substr(start, end - start);
    int imgIndex = stoi(filename);
    string savePath = "output/" + to_string(imgIndex) + "_corners_fast.jpg";
    imwrite(savePath, imgWithCorners);

    // Crea le trackbar per cambiare i parametri in tempo reale
    createTrackbar("Threshold", "FAST Corner Detection", &threshold_bar_fast, 255, updateCornersFast);
    createTrackbar("N", "FAST Corner Detection", &n_bar_fast, 20, updateCornersFast);
    createTrackbar("Dist", "FAST Corner Detection", &dist_bar_fast, 15, updateCornersFast);

    waitKey(0);
}

// Parametri di default per match e Ransac
int threshold_bar_ransac = 5;
int maxIterations_bar_ransac = 1000;
int patchSize_bar_patchDescriptor = 32;
int scale_thresh_bar_matches = 75;
int threshold_bar_matches = 3000;

void updateMatch(int, void*) {
    float scale_thresh = scale_thresh_bar_matches / 100.0f;
    float threshold = static_cast<float>(threshold_bar_matches);
    
    vector<Mat> descriptors1 = computePatchDescriptors(img1, keypoints1, patchSize_bar_patchDescriptor);
    vector<Mat> descriptors2 = computePatchDescriptors(img2, keypoints2, patchSize_bar_patchDescriptor);
    
    vector<DMatch> matches = matchDescriptors(descriptors1, descriptors2, "lowe", scale_thresh, threshold);
    
    vector<DMatch> inlierMatches = ransac(keypoints1, keypoints2, matches, threshold_bar_ransac, maxIterations_bar_ransac);

    drawMatches(img1, keypoints1, img2, keypoints2, inlierMatches, imgMatches);
    imshow("Matches", imgMatches);
}

void do_match(const Mat& img1, const vector<KeyPoint>& keypoints1, const Mat& img2, const vector<KeyPoint>& keypoints2) {
    updateMatch(0, 0);

    // Crea le trackbar per cambiare i parametri in tempo reale
    createTrackbar("Patch Size", "Matches", &patchSize_bar_patchDescriptor, 64, updateMatch);
    createTrackbar("RANSAC Threshold", "Matches", &threshold_bar_ransac, 20, updateMatch);
    createTrackbar("Max Iterations", "Matches", &maxIterations_bar_ransac, 10000, updateMatch);
    createTrackbar("Scale Thresh", "Matches", &scale_thresh_bar_matches, 100, updateMatch);
    createTrackbar("Threshold", "Matches", &threshold_bar_matches, 10000, updateMatch);
    
    waitKey(0);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Uso: ./progetto <harris | fast | match | ...>" << endl;
        return -1;
    }

    string command = argv[1];

    if (command == "harris") {
        const string& imgPath1="immagini/7.jpg";
        do_harris(imgPath1);
        const string& imgPath2="immagini/8.jpg";
        do_harris(imgPath2);
    }
    else if (command == "fast") {
        const string& imgPath1="immagini/7.jpg";
        do_fast(imgPath1);
        const string& imgPath2="immagini/8.jpg";
        do_fast(imgPath2);
    }
    else if (command == "match") {
        if (argc < 3) {
            cout << "Uso: ./progetto <match harris | fast | ...>" << endl;
            return -1;
        }

        string type_kp = argv[2];

        if (type_kp == "harris") { 
            const string& imgPath1 = "immagini/7.jpg";
            do_harris(imgPath1);
            img1 = imread(imgPath1);
            keypoints1 = keypoints;

            const string& imgPath2 = "immagini/8.jpg";
            do_harris(imgPath2);
            img2 = imread(imgPath2);
            keypoints2 = keypoints;

            do_match(img1, keypoints1, img2, keypoints2);
        }
        else if (type_kp == "fast") {
            const string& imgPath1="immagini/0.jpg";
            do_fast(imgPath1);
            img1=imread(imgPath1);
            keypoints1=keypoints;

            const string& imgPath2="immagini/1.jpg";
            do_fast(imgPath2);
            img2=imread(imgPath2);
            keypoints2=keypoints;

            do_match(img1, keypoints1, img2, keypoints2);
        }
        else {
            cout << "Uso: ./progetto <match harris | fast | ...>" << endl;
            return -1;
        }
    }
    else {
        cout << "Uso: ./progetto <harris | fast | ...>" << endl;
        return -1;
    }

    return 0;
}