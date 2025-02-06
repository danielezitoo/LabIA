#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <cstring>
#include <sstream>

#include "harris.cpp"
#include "fast.cpp"
#include "shi_tomasi.cpp"
#include "utils.cpp"
#include "patch_descriptors.cpp"
#include "brief.cpp"
#include "hog.cpp"

using namespace cv;
using namespace std;

Mat img, imgWithCorners, img1, img2, imgMatches, descriptors1, descriptors2, imgCopy;
vector<KeyPoint> keypoints, keypoints1, keypoints2;

// Parametri di default per Harris ("int" perchè è il tipo che vuole createTrackBar(), li convertirò successivamente)
int window_harris = 20;
int sigma_bar_harris = 34;
int threshold_bar_harris = 100;

// Funzione per aggiornare la rilevazione dei corner Harris
void updateCornersHarris(int, void*) {
    imgWithCorners = img.clone();

    if (window_harris % 2 == 0) {
        window_harris++;
    }
    if (window_harris <= 1) {
        window_harris = 3;
    }
    double sigma = sigma_bar_harris / 10.0;
    float threshold = threshold_bar_harris / 10.0f;

    keypoints = harrisCornerDetection(img, window_harris, sigma, threshold);
    drawCorners(imgWithCorners, keypoints);

    namedWindow("Harris Corner Detection", WINDOW_NORMAL);
    // Cerco di non far creare una schermata troppo grande per farlo entrare nello schermo
    resizeWindow("Harris Corner Detection", 1300, 750);
    imshow("Harris Corner Detection", imgWithCorners);
}

// Funzione per eseguire Harris sull'immagine ritagliata
void do_harris(const string& imgPath) {
    img = imread(imgPath);

    if (img.empty()) {
        cout << "Errore nel caricare l'immagine!" << endl;
        return;
    }

    updateCornersHarris(0, 0);

    // Salva l'immagine con i corner rilevati
    saveImageKP(imgWithCorners, imgPath, "corners_harris");

    // Crea le trackbar per cambiare i parametri in tempo reale
    createTrackbar("Window Size", "Harris Corner Detection", &window_harris, 20, updateCornersHarris);
    createTrackbar("Sigma", "Harris Corner Detection", &sigma_bar_harris, 200, updateCornersHarris);
    createTrackbar("Threshold", "Harris Corner Detection", &threshold_bar_harris, 100, updateCornersHarris);

    waitKey(0);
}

// Parametri di default per FAST
int threshold_bar_fast = 50;
int n_bar_fast = 12;
int dist_bar_fast = 3;

void updateCornersFast(int, void*) {
    imgWithCorners = img.clone();
    cvtColor(imgWithCorners, imgWithCorners, COLOR_GRAY2BGR);

    keypoints = fastCornerDetection(img, threshold_bar_fast, n_bar_fast, dist_bar_fast);

    drawCorners(imgWithCorners, keypoints);

    namedWindow("FAST Corner Detection", WINDOW_NORMAL);
    // Cerco di non far creare una schermata troppo grande per farlo entrare nello schermo
    resizeWindow("FAST Corner Detection", 1300, 750);
    imshow("FAST Corner Detection", imgWithCorners);
}

void do_fast(const string& imgPath) {
    img = imread(imgPath, IMREAD_GRAYSCALE);

    if (img.empty()) {
        cout << "Errore nel caricare l'immagine!" << endl;
        return;
    }

    updateCornersFast(0, 0);

    saveImageKP(imgWithCorners, imgPath, "_corners_fast");

    // Crea le trackbar per cambiare i parametri in tempo reale
    createTrackbar("Threshold", "FAST Corner Detection", &threshold_bar_fast, 255, updateCornersFast);
    createTrackbar("N", "FAST Corner Detection", &n_bar_fast, 20, updateCornersFast);
    createTrackbar("Dist", "FAST Corner Detection", &dist_bar_fast, 15, updateCornersFast);

    waitKey(0);
}

int threshold_bar_shi_tomasi = 10000;
int window_size_bar_shi_tomasi = 3;

void updateCornersShiTomasi(int, void*) {
    imgWithCorners = img.clone();

    double threshold = threshold_bar_shi_tomasi / 10.0;

    keypoints = shiTomasiCornerDetection(img, threshold, window_size_bar_shi_tomasi);

    drawCorners(imgWithCorners, keypoints);

    namedWindow("Shi-Tomasi Corner Detection", WINDOW_NORMAL);
    // Cerco di non far creare una schermata troppo grande per farlo entrare nello schermo
    resizeWindow("Shi-Tomasi Corner Detection", 1300, 750);
    imshow("Shi-Tomasi Corner Detection", imgWithCorners);
}

void do_shi_tomasi(const string& imgPath) {
    img = imread(imgPath);

    if (img.empty()) {
        cout << "Errore nel caricare l'immagine!" << endl;
        return;
    }

    updateCornersShiTomasi(0, 0);

    saveImageKP(imgWithCorners, imgPath, "_corners_shi_tomasi");

    // Crea le trackbar per cambiare i parametri in tempo reale
    createTrackbar("Threshold", "Shi-Tomasi Corner Detection", &threshold_bar_shi_tomasi, 100000, updateCornersShiTomasi);
    createTrackbar("Window Size", "Shi-Tomasi Corner Detection", &window_size_bar_shi_tomasi, 20, updateCornersShiTomasi);

    waitKey(0);
}

// Parametri di default per SIFT
int threshold_bar_ransac_sift = 3;
int maxIterations_bar_ransac_sift = 20000;

void updateSIFT(int, void*) {
    Ptr<SIFT> sift = SIFT::create();
    sift->detect(img1, keypoints1);
    sift->detect(img2, keypoints2);

    descriptors1 = computeHOG(img1, keypoints1);
    descriptors2 = computeHOG(img2, keypoints2);

    vector<DMatch> matches = matchHOG(descriptors1, descriptors2);

    matches = ransac(keypoints1, keypoints2, matches, threshold_bar_ransac_sift, maxIterations_bar_ransac_sift);

    drawMatches(img1, keypoints1, img2, keypoints2, matches, imgMatches);

    namedWindow("SIFT Matches", WINDOW_NORMAL);
    // Cerco di non far creare una schermata troppo grande per farlo entrare nello schermo
    resizeWindow("SIFT Matches", 1300, 750);
    imshow("SIFT Matches", imgMatches);
}

void do_sift(const string& imgPath1, const string& imgPath2) {
    img1 = imread(imgPath1);
    img2 = imread(imgPath2);

    // Ridimensiono più per diminuire la mole di calcolo che per la dimensione dell'immagine
    resize(img1, img1, Size(img1.cols / 3, img1.rows / 3));
    resize(img2, img2, Size(img2.cols / 3, img2.rows / 3));

    if (img1.empty() || img2.empty()) {
        cout << "Errore nel caricare le immagini!" << endl;
        return;
    }

    updateSIFT(0, 0);

    saveImageM(imgMatches, imgPath1, imgPath2, "_match_sift");

    createTrackbar("Threshold RANSAC", "SIFT Matches", &threshold_bar_ransac_sift, 20, updateSIFT);
    createTrackbar("Max Iter RANSAC", "SIFT Matches", &maxIterations_bar_ransac_sift, 20000, updateSIFT);

    waitKey(0);
}

// Parametri di default per ORB
int threshold_fast_bar_orb = 100;
int patch_size_bar_orb = 64;
int n_bits_bar_orb = 256;
int threshold_bar_ransac_orb = 3;
int maxIterations_bar_ransac_orb = 5000;

void updateORB(int, void*) {
    Ptr<FastFeatureDetector> fast = FastFeatureDetector::create(threshold_fast_bar_orb);
    fast->detect(img1, keypoints1);
    fast->detect(img2, keypoints2);

    descriptors1 = computeBRIEF(img1, keypoints1, patch_size_bar_orb, n_bits_bar_orb);
    descriptors2 = computeBRIEF(img2, keypoints2, patch_size_bar_orb, n_bits_bar_orb);

    vector<DMatch> matches = matchBRIEF(descriptors1, descriptors2);

    matches = ransac(keypoints1, keypoints2, matches, threshold_bar_ransac_orb, maxIterations_bar_ransac_orb);

    drawMatches(img1, keypoints1, img2, keypoints2, matches, imgMatches);

    namedWindow("Matching ORB", WINDOW_NORMAL);
    // Cerco di non far creare una schermata troppo grande per farlo entrare nello schermo
    resizeWindow("Matching ORB", 1300, 750);
    imshow("Matching ORB", imgMatches);
}

void do_orb(const string& imgPath1, const string& imgPath2) {
    img1 = imread(imgPath1);
    img2 = imread(imgPath2);

    resize(img1, img1, Size(img1.cols / 4, img1.rows / 4));
    resize(img2, img2, Size(img2.cols / 4, img2.rows / 4));

    if (img1.empty() || img2.empty()) {
        cout << "Errore nel caricare le immagini!" << endl;
        return;
    }

    updateORB(0, 0);

    saveImageM(imgMatches, imgPath1, imgPath2, "_match_orb");

    createTrackbar("Threshold FAST", "Matching ORB", &threshold_fast_bar_orb, 150, updateORB);
    createTrackbar("Patch Size BRIEF", "Matching ORB", &patch_size_bar_orb, 128, updateORB);
    createTrackbar("Num Bits BRIEF", "Matching ORB", &n_bits_bar_orb, 256, updateORB);
    createTrackbar("Threshold RANSAC", "Matching ORB", &threshold_bar_ransac_orb, 20, updateORB);
    createTrackbar("Max Iter RANSAC", "Matching ORB", &maxIterations_bar_ransac_orb, 20000, updateORB);

    waitKey(0);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Uso: ./progetto <harris | fast | match | ...>" << endl;
        return -1;
    }

    string command = argv[1];

    if (command == "harris") {
        const string& imgPath1="immagini/15.jpg";
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
    else if (command == "shitomasi") {
        const string& imgPath1="immagini/7.jpg";
        do_shi_tomasi(imgPath1);
        const string& imgPath2="immagini/8.jpg";
        do_shi_tomasi(imgPath2);
    }
    else if (command == "match") {
        if (argc < 3) {
            cout << "Uso: ./progetto <match harris | fast | ...>" << endl;
            return -1;
        }

        string type_kp = argv[2];

        if (type_kp == "sift") { 
            const string& imgPath1="immagini/0.jpg";
            const string& imgPath2="immagini/1.jpg";
            do_sift(imgPath1, imgPath2);
        }
        else if (type_kp == "orb") {
            const string& imgPath1="immagini/0.jpg";
            const string& imgPath2="immagini/1.jpg";
            do_orb(imgPath1, imgPath2);
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