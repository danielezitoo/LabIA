#include "globals.h"

void updateMatchHarris(int, void*) {
    if (window_bar_match_harris % 2 == 0) {
        window_bar_match_harris += 1;
    }
    if (window_bar_match_harris <= 1) {
        window_bar_match_harris = 3;
    }

    keypoints1 = harrisCornerDetection(img1, window_bar_match_harris, sigma_bar_match_harris, threshold_bar_match_harris);
    keypoints2 = harrisCornerDetection(img2, window_bar_match_harris, sigma_bar_match_harris, threshold_bar_match_harris);

    if (descriptorType == "hog") {
        descriptors1 = computeHOG(img1, keypoints1);
        descriptors2 = computeHOG(img2, keypoints2);
    }
    else {
        descriptors1 = computeBRIEF(img1, keypoints1, patch_size_bar_match_harris, n_bits_bar_match_harris);
        descriptors2 = computeBRIEF(img2, keypoints2, patch_size_bar_match_harris, n_bits_bar_match_harris);
    }

    matches = (descriptorType == "hog") ? matchHOG(descriptors1, descriptors2) : matchBRIEF(descriptors1, descriptors2);

    matches = ransac(keypoints1, keypoints2, matches, threshold_bar_ransac_harris, maxIterations_bar_ransac_harris);

    drawMatches(img1, keypoints1, img2, keypoints2, matches, imgMatches);

    namedWindow("Harris Matches", WINDOW_NORMAL);
    // Cerco di non far creare una schermata troppo grande per farla entrare nello schermo
    if (imgMatches.cols > MAX_SIZE || imgMatches.rows > MAX_SIZE) {
        resizeWindow("Harris Matches", 1300, 700);
    }
    imshow("Harris Matches", imgMatches);

    //mergeImages(img1, img2, keypoints1, keypoints2, matches);
}

void do_match_harris(const string& imgPath1, const string& imgPath2, const string& descriptor) {
    descriptorType = descriptor;

    img1 = imread(imgPath1);
    img2 = imread(imgPath2);

    if (!checkImage(img1, imgPath1) || !checkImage(img2, imgPath2)) {
        return;
    }

    updateMatchHarris(0, 0);

    saveImageM(imgMatches, imgPath1, imgPath2, "_match_harris_" + descriptor);

    // Crea le trackbar per cambiare i parametri in tempo reale
    if (descriptorType == "brief") {
        createTrackbar("Patch Size BRIEF", "Harris Matches", &patch_size_bar_match_harris, 128, updateMatchHarris);
        createTrackbar("NBits BRIEF", "Harris Matches", &n_bits_bar_match_harris, 256, updateMatchHarris);
    }
    createTrackbar("Threshold RANSAC", "Harris Matches", &threshold_bar_ransac_harris, 20, updateMatchHarris);
    createTrackbar("Max Iter RANSAC", "Harris Matches", &maxIterations_bar_ransac_harris, 20000, updateMatchHarris);

    waitKey(0);
}