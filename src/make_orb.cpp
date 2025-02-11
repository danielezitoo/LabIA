#include "globals.h"

void updateORB(int, void*) {
    Ptr<FastFeatureDetector> fast = FastFeatureDetector::create(threshold_fast_bar_orb);

    int64 start = getTickCount();

    fast->detect(img1, keypoints1);
    fast->detect(img2, keypoints2);

    // Stampa il numero di corner trovati
    cout << "Numero di corner rilevati per L'immagine 1: " << keypoints1.size() << endl;
    cout << "Numero di corner rilevati per L'immagine 2: " << keypoints2.size() << endl;

    // Misura prestazioni
    int64 end = getTickCount();
    double tEsec = (end - start) / getTickFrequency();
    cout << "Tempo di esecuzione per Orb Corner Detection (Fast): " << tEsec << " secondi" << endl;

    start = getTickCount();

    descriptors1 = computeBRIEF(img1, keypoints1, patch_size_bar_orb, n_bits_bar_orb);
    descriptors2 = computeBRIEF(img2, keypoints2, patch_size_bar_orb, n_bits_bar_orb);

    matches = matchBRIEF(descriptors1, descriptors2);

    int sizeM = matches.size();

    cout << "Numero di matches prima di RANSAC: " << sizeM << endl;

    matches = ransac(keypoints1, keypoints2, matches, threshold_bar_ransac_orb, maxIterations_bar_ransac_orb);

    // Conta Outliers e Inliers
    cout << "Numero di matches dopo RANSAC (INLIERS): " << matches.size() << endl;
    cout << "Matches eliminati (OUTLIERS): " << sizeM - matches.size() << endl;

    // Misura prestazioni
    end = getTickCount();
    tEsec = (end - start) / getTickFrequency();
    cout << "Tempo di esecuzione per BRIEF Descriptor Computing and (right with RANSAC) Matching: " << tEsec << " secondi" <<  ((merge_after_match == true) ? "" : "\n") << endl;

    drawMatches(img1, keypoints1, img2, keypoints2, matches, imgMatches);

    namedWindow("Matching ORB", WINDOW_NORMAL);
    // Cerco di non far creare una schermata troppo grande per farla entrare nello schermo
    if (imgMatches.cols > MAX_SIZE || imgMatches.rows > MAX_SIZE) {
        resizeWindow("Matching ORB", 1300, 700);
    }
    imshow("Matching ORB", imgMatches);

    if (merge_after_match) {
        start = getTickCount();

        mergeImages(img1, img2, keypoints1, keypoints2, matches, threshold_bar_ransac_orb, maxIterations_bar_ransac_orb);

        // Misura prestazioni
        end = getTickCount();
        tEsec = (end - start) / getTickFrequency();
        cout << "Tempo di esecuzione per Orb Merging: " << tEsec << " secondi\n" << endl;
    }
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
    createTrackbar("Threshold FAST", "Matching ORB", &threshold_fast_bar_orb, 255, updateORB);
    createTrackbar("Patch Size BRIEF", "Matching ORB", &patch_size_bar_orb, 128, updateORB);
    createTrackbar("Num Bits BRIEF", "Matching ORB", &n_bits_bar_orb, 256, updateORB);
    createTrackbar("Threshold RANSAC", "Matching ORB", &threshold_bar_ransac_orb, 50, updateORB);
    createTrackbar("Max Iter RANSAC", "Matching ORB", &maxIterations_bar_ransac_orb, 50000, updateORB);

    waitKey(0);

    if (merge_after_match) {
        saveImageP(panorama, imgPath1, imgPath2, "_merged_orb");
    }
}