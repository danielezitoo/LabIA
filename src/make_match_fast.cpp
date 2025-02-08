#include "globals.h"

void updateMatchFast(int, void*) {
    keypoints1 = fastCornerDetection(img1, threshold_bar_match_fast, n_bar_match_fast, dist_bar_match_fast);
    keypoints2 = fastCornerDetection(img2, threshold_bar_match_fast, n_bar_match_fast, dist_bar_match_fast);

    descriptors1 = computeHOG(img1, keypoints1);
    descriptors2 = computeHOG(img2, keypoints2);

    matches = matchHOG(descriptors1, descriptors2);

    matches = ransac(keypoints1, keypoints2, matches, threshold_bar_ransac_fast, maxIterations_bar_ransac_fast);

    drawMatches(img1, keypoints1, img2, keypoints2, matches, imgMatches);

    namedWindow("Fast Matches", WINDOW_NORMAL);
    // Cerco di non far creare una schermata troppo grande per farla entrare nello schermo
    if (imgMatches.cols > MAX_SIZE || imgMatches.rows > MAX_SIZE) {
        resizeWindow("Fast Matches", 1300, 700);
    }
    imshow("Fast Matches", imgMatches);

    //mergeImages(img1, img2, keypoints1, keypoints2, matches);
}

void do_match_fast(const string& imgPath1, const string& imgPath2) {
    img1 = imread(imgPath1);
    img2 = imread(imgPath2);

    if (!checkImage(img1, imgPath1) || !checkImage(img2, imgPath2)) {
        return;
    }

    updateMatchFast(0, 0);

    saveImageM(imgMatches, imgPath1, imgPath2, "_match_fast_hog");

    // Crea le trackbar per cambiare i parametri in tempo reale
    createTrackbar("Threshold Fast", "Fast Matches", &threshold_bar_match_fast, 255, updateMatchFast);
    createTrackbar("N Fast", "Fast Matches", &n_bar_match_fast, 20, updateMatchFast);
    createTrackbar("Dist Fast", "Fast Matches", &dist_bar_match_fast, 15, updateMatchFast);
    createTrackbar("Threshold RANSAC", "Fast Matches", &threshold_bar_ransac_fast, 20, updateMatchFast);
    createTrackbar("Max Iter RANSAC", "Fast Matches", &maxIterations_bar_ransac_fast, 20000, updateMatchFast);

    waitKey(0);
}