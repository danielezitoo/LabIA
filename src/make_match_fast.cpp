#include "globals.h"

void updateMatchFast(int, void*) {
    int64 start = getTickCount();

    keypoints1 = fastCornerDetection(img1, threshold_bar_match_fast, n_bar_match_fast, dist_bar_match_fast);
    keypoints2 = fastCornerDetection(img2, threshold_bar_match_fast, n_bar_match_fast, dist_bar_match_fast);

    // Stampa il numero di corner trovati
    cout << "Numero di corner rilevati per L'immagine 1: " << keypoints1.size() << endl;
    cout << "Numero di corner rilevati per L'immagine 2: " << keypoints2.size() << endl;

    // Misura prestazioni
    int64 end = getTickCount();
    double tEsec = (end - start) / getTickFrequency();
    cout << "Tempo di esecuzione per Fast Corner Detection: " << tEsec << " secondi" << endl;

    start = getTickCount();

    descriptors1 = computeHOG(img1, keypoints1);
    descriptors2 = computeHOG(img2, keypoints2);

    matches = matchHOG(descriptors1, descriptors2);

    int sizeM = matches.size();

    cout << "Numero di matches prima di RANSAC: " << sizeM << endl;

    matches = ransac(keypoints1, keypoints2, matches, threshold_bar_ransac_fast, maxIterations_bar_ransac_fast);

    // Conta Outliers e Inliers
    cout << "Numero di matches dopo RANSAC (INLIERS): " << matches.size() << endl;
    cout << "Matches eliminati (OUTLIERS): " << sizeM - matches.size() << endl;

    // Misura prestazioni
    end = getTickCount();
    tEsec = (end - start) / getTickFrequency();
    cout << "Tempo di esecuzione per HOG Descriptor Computing and (right with RANSAC) Matching: " << tEsec << " secondi" <<  ((merge_after_match == true) ? "" : "\n") << endl;

    drawMatches(img1, keypoints1, img2, keypoints2, matches, imgMatches);

    namedWindow("Fast Matches", WINDOW_NORMAL);
    // Cerco di non far creare una schermata troppo grande per farla entrare nello schermo
    if (imgMatches.cols > MAX_SIZE || imgMatches.rows > MAX_SIZE) {
        resizeWindow("Fast Matches", 1300, 700);
    }
    imshow("Fast Matches", imgMatches);

    if (merge_after_match) {
        start = getTickCount();

        mergeImages(img1, img2, keypoints1, keypoints2, matches, threshold_bar_ransac_fast, maxIterations_bar_ransac_fast);

        // Misura prestazioni
        end = getTickCount();
        tEsec = (end - start) / getTickFrequency();
        cout << "Tempo di esecuzione per Fast HOG Merging: " << tEsec << " secondi\n" << endl;
    }
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
    createTrackbar("Dist Fast", "Fast Matches", &dist_bar_match_fast, 20, updateMatchFast);
    createTrackbar("Threshold RANSAC", "Fast Matches", &threshold_bar_ransac_fast, 50, updateMatchFast);
    createTrackbar("Max Iter RANSAC", "Fast Matches", &maxIterations_bar_ransac_fast, 50000, updateMatchFast);

    waitKey(0);

    if (merge_after_match) {
        saveImageP(panorama, imgPath1, imgPath2, "_merged_fast_hog");
    }
}