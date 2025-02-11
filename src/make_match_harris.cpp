#include "globals.h"

void updateMatchHarris(int, void*) {
    if (window_bar_match_harris % 2 == 0) {
        window_bar_match_harris += 1;
    }
    if (window_bar_match_harris <= 1) {
        window_bar_match_harris = 3;
    }

    int64 start = getTickCount();

    keypoints1 = harrisCornerDetection(img1, window_bar_match_harris, sigma_bar_match_harris, threshold_bar_match_harris);
    keypoints2 = harrisCornerDetection(img2, window_bar_match_harris, sigma_bar_match_harris, threshold_bar_match_harris);

    // Stampa il numero di corner trovati
    cout << "Numero di corner rilevati per L'immagine 1: " << keypoints1.size() << endl;
    cout << "Numero di corner rilevati per L'immagine 2: " << keypoints2.size() << endl;

    // Misura prestazioni
    int64 end = getTickCount();
    double tEsec = (end - start) / getTickFrequency();
    cout << "Tempo di esecuzione per Harris Corner Detection: " << tEsec << " secondi\n" << endl;

    start = getTickCount();

    if (descriptorType == "hog") {
        descriptors1 = computeHOG(img1, keypoints1);
        descriptors2 = computeHOG(img2, keypoints2);
    }
    else {
        descriptors1 = computeBRIEF(img1, keypoints1, patch_size_bar_match_harris, n_bits_bar_match_harris);
        descriptors2 = computeBRIEF(img2, keypoints2, patch_size_bar_match_harris, n_bits_bar_match_harris);
    }

    matches = (descriptorType == "hog") ? matchHOG(descriptors1, descriptors2) : matchBRIEF(descriptors1, descriptors2);

    int sizeM = matches.size();

    cout << "Numero di matches prima di RANSAC: " << sizeM << endl;

    matches = ransac(keypoints1, keypoints2, matches, threshold_bar_ransac_harris, maxIterations_bar_ransac_harris);

    // Conta Outliers e Inliers
    cout << "Numero di matches dopo RANSAC (INLIERS): " << matches.size() << endl;
    cout << "Matches eliminati (OUTLIERS): " << sizeM - matches.size() << endl;

    // Misura prestazioni
    end = getTickCount();
    tEsec = (end - start) / getTickFrequency();
    cout << "Tempo di esecuzione per " << descriptorType << " Descriptor Computing and (right with RANSAC) Matching: " << tEsec << " secondi" <<  ((merge_after_match == true) ? "" : "\n") << endl;

    drawMatches(img1, keypoints1, img2, keypoints2, matches, imgMatches);

    namedWindow("Harris Matches", WINDOW_NORMAL);
    // Cerco di non far creare una schermata troppo grande per farla entrare nello schermo
    if (imgMatches.cols > MAX_SIZE || imgMatches.rows > MAX_SIZE) {
        resizeWindow("Harris Matches", 1300, 700);
    }
    imshow("Harris Matches", imgMatches);

    if (merge_after_match) {
        start = getTickCount();

        mergeImages(img1, img2, keypoints1, keypoints2, matches, threshold_bar_ransac_harris, maxIterations_bar_ransac_harris);

        // Misura prestazioni
        end = getTickCount();
        tEsec = (end - start) / getTickFrequency();
        cout << "Tempo di esecuzione per Harris " << descriptorType << " Merging: " << tEsec << " secondi" << endl;
    }
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
    createTrackbar("Threshold RANSAC", "Harris Matches", &threshold_bar_ransac_harris, 50, updateMatchHarris);
    createTrackbar("Max Iter RANSAC", "Harris Matches", &maxIterations_bar_ransac_harris, 50000, updateMatchHarris);

    waitKey(0);

    if (merge_after_match) {
        saveImageP(panorama, imgPath1, imgPath2, "_merged_harris_" + descriptorType);
    }
}