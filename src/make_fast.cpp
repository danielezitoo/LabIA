#include "globals.h"

void updateCornersFast(int, void*) {
    imgWithCorners = img.clone();
    cvtColor(imgWithCorners, imgWithCorners, COLOR_GRAY2BGR);

    keypoints = fastCornerDetection(img, threshold_bar_match_fast, n_bar_fast, dist_bar_fast);

    drawCorners(imgWithCorners, keypoints);

    namedWindow("FAST Corner Detection", WINDOW_NORMAL);
    // Cerco di non far creare una schermata troppo grande per farla entrare nello schermo
    if (imgWithCorners.cols > MAX_SIZE || imgWithCorners.rows > MAX_SIZE) {
        resizeWindow("FAST Corner Detection", 1300, 700);
    }
    imshow("FAST Corner Detection", imgWithCorners);
}

void do_fast(const string& imgPath) {
    img = imread(imgPath, IMREAD_GRAYSCALE);

    if (!checkImage(img, imgPath)) {
        return;
    }

    updateCornersFast(0, 0);

    saveImageKP(imgWithCorners, imgPath, "_corners_fast");

    // Crea le trackbar per cambiare i parametri in tempo reale
    createTrackbar("Threshold", "FAST Corner Detection", &threshold_bar_match_fast, 255, updateCornersFast);
    createTrackbar("N", "FAST Corner Detection", &n_bar_fast, 20, updateCornersFast);
    createTrackbar("Dist", "FAST Corner Detection", &dist_bar_fast, 15, updateCornersFast);

    waitKey(0);
}