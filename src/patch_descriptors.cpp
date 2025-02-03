#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <ctime>

using namespace cv;
using namespace std;

Mat extractPatchDescriptor(const Mat& img, const KeyPoint& kp, int patchSize = 64) {
    int half = patchSize / 2;
    int x = cvRound(kp.pt.x);
    int y = cvRound(kp.pt.y);

    // Controlla se il patch è completamente dentro l'immagine
    if (x - half < 0 || x + half >= img.cols || y - half < 0 || y + half >= img.rows) {
        return Mat();
    }

    // Crea un quadrato di dimensione patchSize centrato sul keypoint e lo converte in vettore riga
    Mat patch = img(Rect(x - half, y - half, patchSize, patchSize)).clone().reshape(1, 1);
    return patch;
}

// Estrae descrittori per un insieme di keypoint in un'immagine
vector<Mat> computePatchDescriptors(const Mat& img, const vector<KeyPoint>& keypoints, int patchSize = 64) {
    vector<Mat> descriptors;
    for (size_t i = 0; i < keypoints.size(); i++) {
        Mat desc = extractPatchDescriptor(img, keypoints[i], patchSize);
        // Se il patch è dentro l'immagine, aggiunge il descrittore
        if (!desc.empty()) {
            descriptors.push_back(desc);
        }
    }
    return descriptors;
}

/*
Patch Descriptor con Lowe:
Confronta le distanze dei descrittori e accetta una corrispondenza solo se il miglior match
è significativamente più vicino rispetto al secondo miglior match, secondo un rapporto di soglia
*/
vector<DMatch> matchDescriptorsLowe(const vector<Mat>& descriptors1, const vector<Mat>& descriptors2, float scale_thresh = 0.5) {
    vector<DMatch> matches;

    for (int i = 0; i < descriptors1.size(); i++) {
        double bestDist = 1e10;
        double secondBestDist = 1e10;
        int bestIndex = -1;

        for (int j = 0; j < descriptors2.size(); j++) {
            double dist = norm(descriptors1[i], descriptors2[j], NORM_L2);  // Calcola la distanza tra i due descrittori

            if (dist < bestDist) {
                secondBestDist = bestDist;
                bestDist = dist;
                bestIndex = j;
            }
            else if (dist < secondBestDist) {  // Trova il secondo miglior descrittore
                secondBestDist = dist;
            }
        }

        // Se il miglior match supera il test di Lowe, aggiunge il match
        if (bestIndex != -1 && bestDist < scale_thresh * secondBestDist) {
            matches.push_back(DMatch(i, bestIndex, bestDist));
        }
    }

    return matches;
}

/*
Patch Descriptor con Soglia Fissa:
Accetta solo le corrispondenze in cui la distanza tra i descrittori è inferiore a una soglia predefinita (threshold)
*/
vector<DMatch> matchDescriptorsThreshold(const vector<Mat>& descriptors1, const vector<Mat>& descriptors2, float threshold) {
    vector<DMatch> matches;

    for (int i = 0; i < descriptors1.size(); i++) {
        double bestDist = 1e10;
        int bestIndex = -1;

        for (int j = 0; j < descriptors2.size(); j++) {
            double dist = norm(descriptors1[i], descriptors2[j], NORM_L2);  // Calcola la distanza tra i due descrittori

            if (dist < bestDist) {
                bestDist = dist;
                bestIndex = j;
            }
        }

        // Se la distanza è inferiore alla threshold aggiungi il match
        if (bestIndex != -1 && bestDist < threshold) {
            matches.push_back(DMatch(i, bestIndex, bestDist));
        }
    }

    return matches;
}

vector<DMatch> matchDescriptors(const vector<Mat>& descriptors1, const vector<Mat>& descriptors2, const string& command = "threshold", float scale_thresh = 0.75, float threshold = 3000.0) {
    vector<DMatch> matches;

    if (command == "lowe") {
        matches = matchDescriptorsLowe(descriptors1, descriptors2, scale_thresh);
    }
    else if (command == "threshold") {
        matches = matchDescriptorsThreshold(descriptors1, descriptors2, threshold);
    }

    return matches;
}