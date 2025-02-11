#include "globals.h"

void removeBlackBorders(Mat& panorama) {
    Mat gray;
    cvtColor(panorama, gray, COLOR_BGR2GRAY);
    
    // Trova i bordi non neri
    Mat mask = gray > 1;

    // Trova i contorni e il rettangolo minimo che li contiene
    vector<vector<Point>> contorni;
    findContours(mask, contorni, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Unisce tutti i contorni per ottenere il rettangolo totale e ritaglia l'immagine
    if (!contorni.empty()) {
        Rect roi = boundingRect(contorni[0]);
        for (int i = 1; i < contorni.size(); i++) {
            roi = roi | boundingRect(contorni[i]);
        }
        panorama = panorama(roi);
    }
}

void blendImages(Mat& panorama, const Mat& img1) {
    Mat mask1 = Mat::zeros(panorama.size(), CV_8UC1);
    img1.copyTo(panorama(Rect(0, 0, img1.cols, img1.rows)));

    // Crea una maschera dell'immagine e la sfuma per rendere il blending più naturale
    Mat mask2;
    cvtColor(panorama, mask2, COLOR_BGR2GRAY);
    threshold(mask2, mask2, 1, 255, THRESH_BINARY);
    GaussianBlur(mask2, mask2, Size(21, 21), 7); // Sfuma il bordo

    // Applica il blending
    Mat blended;
    panorama.copyTo(blended);
    addWeighted(panorama, 0.5, blended, 0.5, 0, panorama);
}

void mergeImages(const Mat& img1, const Mat& img2, const vector<KeyPoint>& keypoints1, const vector<KeyPoint>& keypoints2, const vector<DMatch>& matches, double threshold = 5.0, int maxIterations = 10000) {
    if (img1.empty() || img2.empty()) {
        cerr << "Errore: una o entrambe le immagini sono vuote." << endl;
        return;
    }

    vector<Point2f> points1, points2;
    for (int i = 0; i < matches.size(); i++) {
        points1.push_back(keypoints1[matches[i].queryIdx].pt);
        points2.push_back(keypoints2[matches[i].trainIdx].pt);
    }

    if (points1.size() < 4 || points2.size() < 4) {
        cerr << "Errore: non ci sono abbastanza match per fare merging." << endl;
        return;
    }

    vector<uchar> inliers;
    Mat H = findHomography(points2, points1, RANSAC, threshold, inliers, maxIterations);

    /* 
    Trasformazione omografica :
        La trasformazione prospettica applica una matrice H a ogni punto dell'immagine per modificarne la posizione.
        Ogni punto, inizialmente rappresentato con coordinate (x,y), viene trasformato utilizzando H,
        che combina traslazione, rotazione, scala e prospettiva. Il risultato è un nuovo punto 
        calcolato come funzione lineare dei valori originali, con una normalizzazione finale.

    */
    Mat img2Warped;
    warpPerspective(img2, img2Warped, H, Size(img1.cols + img2.cols, img1.rows));

    panorama = img2Warped.clone();
    img1.copyTo(panorama(Rect(0, 0, img1.cols, img1.rows)));

    removeBlackBorders(panorama);
    blendImages(panorama, img1);

    if (!merge_tot) {
        namedWindow("Merge", WINDOW_NORMAL);
        imshow("Merge", panorama);
        waitKey(1);
    }
}

Mat mergeMultipleImages(const vector<Mat>& images) {
    if (images.empty()) {
        cerr << "Errore: il vettore di immagini è vuoto." << endl;
        return Mat();
    }

    Mat panorama_f = images[0].clone();

    Ptr<SIFT> sift = SIFT::create();

    for (int i = 1; i < images.size(); ++i) {
        sift->detect(panorama_f, keypoints1);
        sift->detect(images[i], keypoints2);

        descriptors1 = computeHOG(panorama_f, keypoints1);
        descriptors2 = computeHOG(images[i], keypoints2);

        matches = matchHOG(descriptors1, descriptors2);

        matches = ransac(keypoints1, keypoints2, matches, 10.0, 20000);

        mergeImages(panorama_f, images[i], keypoints1, keypoints2, matches);
    }

    return panorama_f;
}
