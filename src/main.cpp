#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstring>

#include "harris.cpp"

using namespace cv;
using namespace std;

Mat img, imgWithCorners;

// Parametri di default per Harris ("int" perchè è il tipo che vuole createTrackBar(), li convertirò poi)
int window = 3;
int sigma_bar = 20;
int k_bar = 4;
int threshold_bar = 30;

// Harris corner detection
void updateCorners(int, void*) {
    double sigma = sigma_bar / 10.0;  // Sigma va tra 0 e 20
    float k = 0.04f + (k_bar / 100.0f) * (0.06f - 0.04f);  // K va tra 0.04 e 0.06
    float threshold = threshold_bar / 100.0f;  // Threshold va tra 0 e 1
    Mat corners = harrisCornerDetection(img, window, sigma, k, threshold);

    imgWithCorners = img.clone();
    drawCorners(imgWithCorners, corners, threshold);

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

    updateCorners(0, 0);

    // Crea le trackbar per cambiare i parametri in tempo reale
    createTrackbar("Window Size", "Harris Corner Detection", &window, 20, updateCorners);
    createTrackbar("Sigma", "Harris Corner Detection", &sigma_bar, 200, updateCorners);
    createTrackbar("K", "Harris Corner Detection", &k_bar, 100, updateCorners);
    createTrackbar("Threshold", "Harris Corner Detection", &threshold_bar, 100, updateCorners);

    waitKey(0);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Uso: ./progetto <harris|...>" << endl;
        return -1;
    }

    string command = argv[1];

    if (command == "harris") {
        do_harris("immagini/0.jpg");
        do_harris("immagini/1.jpg");
    }
    else {
        cout << "Uso: ./progetto <harris|...>" << endl;
    }

    return 0;
}
