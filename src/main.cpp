#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <cstring>
#include <sstream>

#include "harris.cpp"
#include "fast.cpp"
#include "draw_corners.cpp"

using namespace cv;
using namespace std;

Mat img, imgWithCorners;
vector<KeyPoint> keypoints;

// Parametri di default per Harris ("int" perchè è il tipo che vuole createTrackBar(), li convertirò successivamente)
int window_harris = 11;
int sigma_bar_harris = 10;
int k_bar_harris = 4;
int threshold_bar_harris = 90;

// Harris corner detection
void updateCornersHarris(int, void*) {
    imgWithCorners = img.clone();

    if (window_harris % 2 == 0) {
        window_harris++;
    }
    if (window_harris <= 1) {
        window_harris = 3;
    }
    double sigma = sigma_bar_harris / 10.0;  // Sigma va tra 0 e 20
    float k = 0.04f + (k_bar_harris / 100.0f) * (0.06f - 0.04f);  // K va tra 0.04 e 0.06
    float threshold = threshold_bar_harris / 100.0f;  // Threshold va tra 0 e 1

    keypoints = harrisCornerDetection(img, window_harris, sigma, k, threshold);

    drawCorners(imgWithCorners, keypoints);

    // Cerco di non far creare una schermata troppo grande per farlo entrare nello schermo
    Mat resizedImg;
    resize(imgWithCorners, resizedImg, Size(imgWithCorners.cols / 2, imgWithCorners.rows / 2));

    imshow("Harris Corner Detection", resizedImg);
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
    createTrackbar("K", "Harris Corner Detection", &k_bar_harris, 100, updateCornersHarris);
    createTrackbar("Threshold", "Harris Corner Detection", &threshold_bar_harris, 100, updateCornersHarris);

    waitKey(0);
}

// Parametri di default per FAST ("int" perchè è il tipo che vuole createTrackBar(), li convertirò successivamente)
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
    Mat resizedImg;
    resize(imgWithCorners, resizedImg, Size(imgWithCorners.cols / 2, imgWithCorners.rows / 2));

    imshow("FAST Corner Detection", resizedImg);
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

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Uso: ./progetto <harris|orb|...>" << endl;
        return -1;
    }

    string command = argv[1];

    if (command == "harris") {
        do_harris("immagini/7.jpg");
        do_harris("immagini/8.jpg");
    }
    else if (command == "fast") {
        do_fast("immagini/7.jpg");
        do_fast("immagini/8.jpg");
    }
    else {
        cout << "Uso: ./progetto <harris|orb|...>" << endl;
    }

    return 0;
}