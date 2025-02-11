#include "globals.h"

void updateSIFT(int, void*) {
    Ptr<SIFT> sift = SIFT::create();

    int64 start = getTickCount();

    sift->detect(img1, keypoints1);
    sift->detect(img2, keypoints2);

    // Stampa il numero di corner trovati
    cout << "Numero di corner rilevati per L'immagine 1: " << keypoints1.size() << endl;
    cout << "Numero di corner rilevati per L'immagine 2: " << keypoints2.size() << endl;

    // Misura prestazioni
    int64 end = getTickCount();
    double tEsec = (end - start) / getTickFrequency();
    cout << "Tempo di esecuzione per Sift Corner Detection (Dog): " << tEsec << " secondi" << endl;

    start = getTickCount();

    descriptors1 = computeHOG(img1, keypoints1);
    descriptors2 = computeHOG(img2, keypoints2);

    matches = matchHOG(descriptors1, descriptors2);

    int sizeM = matches.size();

    cout << "Numero di matches prima di RANSAC: " << sizeM << endl;

    matches = ransac(keypoints1, keypoints2, matches, threshold_bar_ransac_sift, maxIterations_bar_ransac_sift);

    // Conta Outliers e Inliers
    cout << "Numero di matches dopo RANSAC (INLIERS): " << matches.size() << endl;
    cout << "Matches eliminati (OUTLIERS): " << sizeM - matches.size() << endl;

    // Misura prestazioni
    end = getTickCount();
    tEsec = (end - start) / getTickFrequency();
    cout << "Tempo di esecuzione per HOG Descriptor Computing and (right with RANSAC) Matching: " << tEsec << " secondi" <<  ((merge_after_match == true) ? "" : "\n") << endl;

    drawMatches(img1, keypoints1, img2, keypoints2, matches, imgMatches);

    namedWindow("SIFT Matches", WINDOW_NORMAL);
    // Cerco di non far creare una schermata troppo grande per farla entrare nello schermo
    if (imgMatches.cols > MAX_SIZE || imgMatches.rows > MAX_SIZE) {
        resizeWindow("SIFT Matches", 1300, 700);
    }
    imshow("SIFT Matches", imgMatches);

    if (merge_after_match) {
        start = getTickCount();

        mergeImages(img1, img2, keypoints1, keypoints2, matches, threshold_bar_ransac_sift, maxIterations_bar_ransac_sift);
    
        // Misura prestazioni
        end = getTickCount();
        tEsec = (end - start) / getTickFrequency();
        cout << "Tempo di esecuzione per Sift Merging: " << tEsec << " secondi\n" << endl;
    }
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
    createTrackbar("Threshold RANSAC", "SIFT Matches", &threshold_bar_ransac_sift, 50, updateSIFT);
    createTrackbar("Max Iter RANSAC", "SIFT Matches", &maxIterations_bar_ransac_sift, 50000, updateSIFT);

    waitKey(0);

    if (merge_after_match) {
        saveImageP(panorama, imgPath1, imgPath2, "_merged_sift");
    }
    else if (merge_tot) {
        
    }
}