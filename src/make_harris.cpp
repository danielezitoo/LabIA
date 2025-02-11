#include "globals.h"

void updateCornersHarris(int, void*) {
    imgWithCorners = img.clone();

    if (window_harris % 2 == 0) {
        window_harris++;
    }
    if (window_harris <= 1) {
        window_harris = 3;
    }

    // Converto a float o double
    float threshold = threshold_bar_harris / 10.0f;
    double sigma = sigma_bar_harris / 10.0;

    int64 start = getTickCount();

    keypoints = harrisCornerDetection(img, window_harris, sigma, threshold);

    // Stampa il numero di corner trovati
    cout << "Numero di corner rilevati: " << keypoints.size() << endl;

    // Misura prestazioni
    int64 end = getTickCount();
    double tEsec = (end - start) / getTickFrequency();
    cout << "Tempo di esecuzione per Harris Corner Detection: " << tEsec << " secondi\n" << endl;

    drawCorners(imgWithCorners, keypoints);

    namedWindow("Harris Corner Detection", WINDOW_NORMAL);
    // Cerco di non far creare una schermata troppo grande per farla entrare nello schermo
    if (imgWithCorners.cols > MAX_SIZE || imgWithCorners.rows > MAX_SIZE) {
        resizeWindow("Harris Corner Detection", 1300, 700);
    }
    imshow("Harris Corner Detection", imgWithCorners);
}

void do_harris(const string& imgPath) {
    img = imread(imgPath);

    if (!checkImage(img, imgPath)) {
        return;
    }

    updateCornersHarris(0, 0);

    saveImageKP(imgWithCorners, imgPath, "corners_harris");

    // Crea le trackbar per cambiare i parametri in tempo reale
    createTrackbar("Window Size", "Harris Corner Detection", &window_harris, 30, updateCornersHarris);
    createTrackbar("Sigma", "Harris Corner Detection", &sigma_bar_harris, 400, updateCornersHarris);
    createTrackbar("Threshold", "Harris Corner Detection", &threshold_bar_harris, 250, updateCornersHarris);

    waitKey(0);
}