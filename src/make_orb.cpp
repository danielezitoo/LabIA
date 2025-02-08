#include "globals.h"

void updateORB(int, void*) {
    Ptr<FastFeatureDetector> fast = FastFeatureDetector::create(threshold_fast_bar_orb);
    fast->detect(img1, keypoints1);
    fast->detect(img2, keypoints2);

    descriptors1 = computeBRIEF(img1, keypoints1, patch_size_bar_orb, n_bits_bar_orb);
    descriptors2 = computeBRIEF(img2, keypoints2, patch_size_bar_orb, n_bits_bar_orb);

    matches = matchBRIEF(descriptors1, descriptors2);

    matches = ransac(keypoints1, keypoints2, matches, threshold_bar_ransac_orb, maxIterations_bar_ransac_orb);

    drawMatches(img1, keypoints1, img2, keypoints2, matches, imgMatches);

    namedWindow("Matching ORB", WINDOW_NORMAL);
    // Cerco di non far creare una schermata troppo grande per farla entrare nello schermo
    if (imgMatches.cols > MAX_SIZE || imgMatches.rows > MAX_SIZE) {
        resizeWindow("Matching ORB", 1300, 700);
    }
    imshow("Matching ORB", imgMatches);

    //mergeImages(img1, img2, keypoints1, keypoints2, matches);
}

void do_orb(const string& imgPath1, const string& imgPath2) {
    img1 = imread(imgPath1);
    img2 = imread(imgPath2);

    if (!checkImage(img1, imgPath1) || !checkImage(img2, imgPath2)) {
        return;
    }

    updateORB(0, 0);

    saveImageM(imgMatches, imgPath1, imgPath2, "_match_orb");

    // Crea le trackbar per cambiare i parametri in tempo reale
    createTrackbar("Threshold FAST", "Matching ORB", &threshold_fast_bar_orb, 150, updateORB);
    createTrackbar("Patch Size BRIEF", "Matching ORB", &patch_size_bar_orb, 128, updateORB);
    createTrackbar("Num Bits BRIEF", "Matching ORB", &n_bits_bar_orb, 256, updateORB);
    createTrackbar("Threshold RANSAC", "Matching ORB", &threshold_bar_ransac_orb, 20, updateORB);
    createTrackbar("Max Iter RANSAC", "Matching ORB", &maxIterations_bar_ransac_orb, 20000, updateORB);

    waitKey(0);
}