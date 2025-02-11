#include "globals.h"

void updateCornersShiTomasi(int, void*) {
    imgWithCorners = img.clone();

    int64 start = getTickCount();

    keypoints = shiTomasiCornerDetection(img, threshold_bar_shi_tomasi / 10.0f, window_size_bar_shi_tomasi);

    // Stampa il numero di corner trovati
    cout << "Numero di corner rilevati: " << keypoints.size() << endl;

    // Misura prestazioni
    int64 end = getTickCount();
    double tEsec = (end - start) / getTickFrequency();
    cout << "Tempo di esecuzione per Shi Tomasi Corner Detection: " << tEsec << " secondi\n" << endl;

    drawCorners(imgWithCorners, keypoints);

    namedWindow("Shi-Tomasi Corner Detection", WINDOW_NORMAL);
    // Cerco di non far creare una schermata troppo grande per farla entrare nello schermo
    if (imgWithCorners.cols > MAX_SIZE || imgWithCorners.rows > MAX_SIZE) {
        resizeWindow("Shi-Tomasi Corner Detection", 1300, 700);
    }
    imshow("Shi-Tomasi Corner Detection", imgWithCorners);
}

void do_shi_tomasi(const string& imgPath) {
    img = imread(imgPath);

    if (!checkImage(img, imgPath)) {
        return;
    }

    updateCornersShiTomasi(0, 0);

    saveImageKP(imgWithCorners, imgPath, "_corners_shi_tomasi");

    // Crea le trackbar per cambiare i parametri in tempo reale
    createTrackbar("Threshold", "Shi-Tomasi Corner Detection", &threshold_bar_shi_tomasi, 100000, updateCornersShiTomasi);
    createTrackbar("Window Size", "Shi-Tomasi Corner Detection", &window_size_bar_shi_tomasi, 30, updateCornersShiTomasi);

    waitKey(0);
}