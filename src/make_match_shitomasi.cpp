#include "globals.h"

void updateMatchShiTomasi(int, void*) {
    keypoints1 = shiTomasiCornerDetection(img1, threshold_bar_match_shi_tomasi / 10.0f, window_size_bar_match_shi_tomasi);
    keypoints2 = shiTomasiCornerDetection(img2, threshold_bar_match_shi_tomasi / 10.0f, window_size_bar_match_shi_tomasi);

    if (descriptorType == "hog") {
        descriptors1 = computeHOG(img1, keypoints1);
        descriptors2 = computeHOG(img2, keypoints2);
    }
    else {
        descriptors1 = computeBRIEF(img1, keypoints1, patch_size_bar_match_shi_tomasi, n_bits_bar_match_shi_tomasi);
        descriptors2 = computeBRIEF(img2, keypoints2, patch_size_bar_match_shi_tomasi, n_bits_bar_match_shi_tomasi);
    }

    matches = (descriptorType == "hog") ? matchHOG(descriptors1, descriptors2) : matchBRIEF(descriptors1, descriptors2);

    matches = ransac(keypoints1, keypoints2, matches, threshold_bar_ransac_shi_tomasi, maxIterations_bar_ransac_shi_tomasi);

    drawMatches(img1, keypoints1, img2, keypoints2, matches, imgMatches);

    namedWindow("Shi-Tomasi Matches", WINDOW_NORMAL);
    // Cerco di non far creare una schermata troppo grande per farla entrare nello schermo
    if (imgMatches.cols > MAX_SIZE || imgMatches.rows > MAX_SIZE) {
        resizeWindow("Shi-Tomasi Matches", 1300, 700);
    }
    imshow("Shi-Tomasi Matches", imgMatches);

    //mergeImages(img1, img2, keypoints1, keypoints2, matches);
}

void do_match_shitomasi(const string& imgPath1, const string& imgPath2, const string& descriptor) {
    descriptorType = descriptor;

    img1 = imread(imgPath1);
    img2 = imread(imgPath2);

    if (!checkImage(img1, imgPath1) || !checkImage(img2, imgPath2)) {
        return;
    }

    updateMatchShiTomasi(0, 0);

    saveImageM(imgMatches, imgPath1, imgPath2, "_match_shitomasi_" + descriptor);

    // Crea le trackbar per cambiare i parametri in tempo reale
    if (descriptorType == "brief") {
        createTrackbar("Patch Size BRIEF", "Shi-Tomasi Matches", &patch_size_bar_match_shi_tomasi, 128, updateMatchShiTomasi);
        createTrackbar("NBits BRIEF", "Shi-Tomasi Matches", &n_bits_bar_match_shi_tomasi, 256, updateMatchShiTomasi);
    }
    createTrackbar("Threshold RANSAC", "Shi-Tomasi Matches", &threshold_bar_ransac_shi_tomasi, 20, updateMatchShiTomasi);
    createTrackbar("Max Iter RANSAC", "Shi-Tomasi Matches", &maxIterations_bar_ransac_shi_tomasi, 20000, updateMatchShiTomasi);

    waitKey(0);
}