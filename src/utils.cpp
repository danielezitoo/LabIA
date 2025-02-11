#include "globals.h"

// Funzione per controllare e ridimensionare l'immagine se è troppo grande o se è vuota
bool checkImage(Mat& img, const string& imgPath) {
    if (img.empty()) {
        cout << "Errore nel caricare l'immagine!" << endl;
        return false;
    }

    while (img.cols > MAX_SIZE || img.rows > MAX_SIZE) {
        resize(img, img, Size(img.cols / 2, img.rows / 2));
        cout << "Immagine troppo grande!\nImmagine ridotta a: " << img.cols << "x" << img.rows << "\n" << endl;
    }

    return true;
}


// Fa una X sui corners
void drawCorners(Mat &img, const vector<KeyPoint> &keypoints) {
    for (int i = 0; i < keypoints.size(); i++) {
        Point pt(cvRound(keypoints[i].pt.x), cvRound(keypoints[i].pt.y));

        line(img, Point(pt.x - 5, pt.y - 5), Point(pt.x + 5, pt.y + 5), Scalar(0, 255, 0));
        line(img, Point(pt.x - 5, pt.y + 5), Point(pt.x + 5, pt.y - 5), Scalar(0, 255, 0));
    }
}


// RANSAC stima un modello robusto iterando tra selezione casuale di punti e verifica degli inlier, scartando gli outlier dovuti a rumore o mismatch.
vector<DMatch> ransac(const vector<KeyPoint>& keypoints1, const vector<KeyPoint>& keypoints2, const vector<DMatch>& matches, double threshold = 10.0, int maxIterations = 20000) {
    vector<Point2f> pts1, pts2;
    for (int i = 0; i < matches.size(); i++) {
        pts1.push_back(keypoints1[matches[i].queryIdx].pt);
        pts2.push_back(keypoints2[matches[i].trainIdx].pt);
    }

    // Utilizzo RANSAC implementato da openCV così da esssere sicuramente più efficiente
    vector<uchar> inliers;
    Mat H = findHomography(pts1, pts2, RANSAC, threshold, inliers, maxIterations);

    vector<DMatch> inlier;
    for (int i = 0; i < matches.size(); i++) {
        if (inliers[i]) {
            inlier.push_back(matches[i]);
        }
    }

    return inlier;
}

// Salva immagini per KeyPoints
void saveImageKP(const Mat &img, const string &imgPath, const string &command) {
    // Estrazione del nome del file senza estensione
    int start = imgPath.find_last_of("/\\") + 1;
    int end = imgPath.find_last_of(".");
    string filename = imgPath.substr(start, end - start);

    string savePath = "output/" + filename + "_" + command + ".jpg";
    imwrite(savePath, img);
}

// Salva immagini per Matching
void saveImageM(const Mat &img, const string &imgPath1, const string &imgPath2, const string &command) {
    // Estrazione del nome del file senza estensione
    int start = imgPath1.find_last_of("/\\") + 1;
    int end = imgPath1.find_last_of(".");
    string filename1 = imgPath1.substr(start, end - start);

    // Estrazione del nome del file senza estensione
    start = imgPath2.find_last_of("/\\") + 1;
    end = imgPath2.find_last_of(".");
    string filename2 = imgPath2.substr(start, end - start);
    int imgIndex2 = stoi(filename2);

    string savePath = "output/" + filename1 + "_" + filename2 + command + ".jpg";
    imwrite(savePath, img);
}

// Salva immagini per Merging
void saveImageP(const Mat &img, const string &imgPath1, const string &imgPath2, const string &command) {
    // Estrazione del nome del file senza estensione
    int start = imgPath1.find_last_of("/\\") + 1;
    int end = imgPath1.find_last_of(".");
    string filename1 = imgPath1.substr(start, end - start);

    // Estrazione del nome del file senza estensione
    start = imgPath2.find_last_of("/\\") + 1;
    end = imgPath2.find_last_of(".");
    string filename2 = imgPath2.substr(start, end - start);
    int imgIndex2 = stoi(filename2);

    string savePath = "output/merged/" + filename1 + "_" + filename2 + command + ".jpg";
    imwrite(savePath, img);
}