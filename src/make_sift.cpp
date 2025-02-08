#include "globals.h"

void updateSIFT(int, void*) {
    Ptr<SIFT> sift = SIFT::create();
    sift->detect(img1, keypoints1);
    sift->detect(img2, keypoints2);

    descriptors1 = computeHOG(img1, keypoints1);
    descriptors2 = computeHOG(img2, keypoints2);

    matches = matchHOG(descriptors1, descriptors2);

    matches = ransac(keypoints1, keypoints2, matches, threshold_bar_ransac_sift, maxIterations_bar_ransac_sift);

    drawMatches(img1, keypoints1, img2, keypoints2, matches, imgMatches);

    namedWindow("SIFT Matches", WINDOW_NORMAL);
    // Cerco di non far creare una schermata troppo grande per farla entrare nello schermo
    if (imgMatches.cols > MAX_SIZE || imgMatches.rows > MAX_SIZE) {
        resizeWindow("SIFT Matches", 1300, 700);
    }
    resizeWindow("SIFT Matches", 1300, 750);
    imshow("SIFT Matches", imgMatches);

    //mergeImages(img1, img2, keypoints1, keypoints2, matches);
}

void do_sift(const string& imgPath1, const string& imgPath2) {
    img1 = imread(imgPath1);
    img2 = imread(imgPath2);

    if (!checkImage(img1, imgPath1) || !checkImage(img2, imgPath2)) {
        return;
    }

    updateSIFT(0, 0);

    saveImageM(imgMatches, imgPath1, imgPath2, "_match_sift");

    // Crea le trackbar per cambiare i parametri in tempo reale
    createTrackbar("Threshold RANSAC", "SIFT Matches", &threshold_bar_ransac_sift, 20, updateSIFT);
    createTrackbar("Max Iter RANSAC", "SIFT Matches", &maxIterations_bar_ransac_sift, 20000, updateSIFT);

    waitKey(0);
}