#include "globals.h"

// Funzione per calcolare i gradienti
pair<Mat, Mat> gradients(const Mat& img) {
    Mat Ix, Iy;
    Sobel(img, Ix, CV_64F, 1, 0, 3);
    Sobel(img, Iy, CV_64F, 0, 1, 3);
    return {Ix, Iy};
}

// Funzione per calcolare la structure matrix per ogni pixel
Matx22d computeMMatrix(int x, int y, const Mat& Ix, const Mat& Iy, int window) {
    double Ix2 = 0, Iy2 = 0, IxIy = 0;

    for (int dy = -window / 2; dy <= window / 2; dy++) {
        for (int dx = -window / 2; dx <= window / 2; dx++) {
            Ix2 += Ix.at<double>(y + dy, x + dx) * Ix.at<double>(y + dy, x + dx);
            Iy2 += Iy.at<double>(y + dy, x + dx) * Iy.at<double>(y + dy, x + dx);
            IxIy += Ix.at<double>(y + dy, x + dx) * Iy.at<double>(y + dy, x + dx);
        }
    }
    
    return Matx22d(Ix2, IxIy, IxIy, Iy2);
}

// Funzione per calcolare gli autovalori
vector<double> computeLambda(const Matx22d& M) {
    Mat eigenvalues;
    eigen(M, eigenvalues);
    return {eigenvalues.at<double>(0), eigenvalues.at<double>(1)};
}

Mat nonMaximumSuppression(const Mat& img, int window) {
    Mat ret = Mat::zeros(img.size(), CV_64F);
    
    for (int y = window / 2; y < img.rows - window / 2; y++) {
        for (int x = window / 2; x < img.cols - window / 2; x++) {
            double pixel = img.at<double>(y, x);
            bool localMax = true;
            
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (pixel < img.at<double>(y + dy, x + dx)) {
                        localMax = false;
                        break;
                    }
                }
                if (!localMax) break;
            }

            if (localMax) {
                ret.at<double>(y, x) = pixel;
            }
        }
    }
    
    return ret;
}

/*
Cit: (medium.com)
    Shi-Tomasi is almost similar to Harris Corner detector, apart from the way the score (R) is calculated.
    This gives a better result. Moreover, in this method, we can find the top N corners,
    which might be useful in case we don’t want to detect each and every corner
    In Shi-Tomasi, R is calculated in the following way:

        R = min(lambda_1 , lambda_2)

    If R is greater than a threshold, its classified as a corner.
*/
vector<KeyPoint> shiTomasiCornerDetection(const Mat& img, double threshold = 0.01, int window = 3) {
    Mat imgGray;
    cvtColor(img, imgGray, COLOR_BGR2GRAY);
    
    auto [Ix, Iy] = gradients(imgGray);

    Mat responseMap = Mat::zeros(imgGray.size(), CV_64F);
    
    for (int y = window / 2; y < imgGray.rows - window / 2; y++) {
        for (int x = window / 2; x < imgGray.cols - window / 2; x++) {
            Matx22d M = computeMMatrix(x, y, Ix, Iy, window);
            auto lambda = computeLambda(M);
            
            double lambda1 = lambda[0];
            double lambda2 = lambda[1];

            // Scalo la tresh per non dover settare la threshold molto alta
            int scale_thresh = 100;

            if (lambda1 > threshold * scale_thresh && lambda2 > threshold * scale_thresh) {
                responseMap.at<double>(y, x) = min(lambda1, lambda2);
            }
        }
    }

    Mat suppressedCorners = nonMaximumSuppression(responseMap, window);

    // Estraggo i keypoints
    vector<KeyPoint> keypoints;

    for (int y = 0; y < suppressedCorners.rows; y++) {
        for (int x = 0; x < suppressedCorners.cols; x++) {
            if (suppressedCorners.at<double>(y, x) > 0) {
                keypoints.push_back(KeyPoint(x, y, 1)); 
            }
        }
    }

    return keypoints;
}

/*
✅ Migliore precisione rispetto a Harris: l'uso degli autovalori riduce i falsi positivi.
✅ Meno sensibile al parametro k: evita problemi di scelta del parametro arbitrario.
❌ Computazionalmente costoso: calcolare gli autovalori è più oneroso di Harris.
❌ Sensibile alla scala.
*/