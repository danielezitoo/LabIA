#include "globals.h"

/*
FAST utilizza un cerchio di 16 pixel, ossia un cerchio di Bresenham di raggio 3,
algoritmo utilizzato per disegnare cerchi su una griglia di pixel.
*/
vector<Point> cerchio = {
        {0, -3}, {1, -3}, {2, -2}, {3, -1}, {3, 0}, {3, 1}, {2, 2}, {1, 3},
        {0, 3}, {-1, 3}, {-2, 2}, {-3, 1}, {-3, 0}, {-3, -1}, {-2, -2}, {-1, -3}
    };

bool fastSegmentTest(const Mat &img, int x, int y, int threshold = 50, int n = 12) {
    uchar pixel = img.at<uchar>(y, x);

    int countHigh = 0, countLow = 0;

    for (int i = 0; i < 16; i++) {
        int dx = x + cerchio[i].x;
        int dy = y + cerchio[i].y;

        // Controlla se il pixel è fuori l'immagine
        if (dx < 0 || dx >= img.cols || dy < 0 || dy >= img.rows)
            continue;

        uchar vicino = img.at<uchar>(dy, dx);

        if (vicino > pixel + threshold) {
            countHigh++;
        } else if (vicino < pixel - threshold) {
            countLow++;
        }
    }

    return (countHigh >= n || countLow >= n);
}

bool fastHighSpeedTest(const Mat &img, int x, int y, int threshold) {
    uchar pixel = img.at<uchar>(y, x);

    // Indici dei 4 pixel da testare: 1, 9, 5, 13 (in un cerchio di 16 pixel)
    vector<int> test_pixel = {1, 9, 5, 13};
    
    int countHigh = 0, countLow = 0;

    for (int i = 0; i < test_pixel.size(); i++) {
        int j = test_pixel[i];
        int dx = x + (cerchio[j].x);
        int dy = y + (cerchio[j].y);

        // Controlla se il pixel è fuori l'immagine
        if (dx < 0 || dx >= img.cols || dy < 0 || dy >= img.rows)
            continue;

        uchar vicino = img.at<uchar>(dy, dx);

        if (vicino > pixel + threshold) {
            countHigh++;
        } else if (vicino < pixel - threshold) {
            countLow++;
        }
    }

    return (countHigh >= 3 || countLow >= 3);
}

void nonMaximumSuppression(vector<KeyPoint> &keypoints, const Mat &img, int dist = 3) {
    vector<float> V_keypoints(keypoints.size());

    // Calcola V per ogni keypoint
    for (int i = 0; i < keypoints.size(); i++) {
        KeyPoint kp = keypoints[i];
        int x = kp.pt.x;
        int y = kp.pt.y;

        int sum = 0;

        // Calcola V come somma delle differenze assolute tra il pixel centrale e i circostanti ()
        uchar pixel = img.at<uchar>(y, x);
        for (const auto& pt : cerchio) {
            int dx = x + pt.x;
            int dy = y + pt.y;
            if (dx >= 0 && dx < img.cols && dy >= 0 && dy < img.rows) {
                uchar vicino = img.at<uchar>(dy, dx);
                sum += abs(pixel - vicino);
            }
        }
        
        V_keypoints[i] = sum;
    }

    // Rimuove i keypoint con V più basso rispetto ai vicini
    for (int i = 0; i < keypoints.size(); i++) {
        KeyPoint kp = keypoints[i];
        int x = kp.pt.x;
        int y = kp.pt.y;

        // Confronta con i keypoint vicini
        for (int j = 0; j < keypoints.size(); j++) {
            if (i != j) {
                KeyPoint kp2 = keypoints[j];
                int x2 = kp2.pt.x;
                int y2 = kp2.pt.y;

                // Se i keypoint sono vicini (ad esempio a distanza di 'dist' pixel)
                if (abs(x - x2) <= dist && abs(y - y2) <= dist) {
                    // Se V del primo keypoint è inferiore a quello del secondo, lo rimuove
                    if (V_keypoints[i] < V_keypoints[j]) {
                        keypoints.erase(keypoints.begin() + i);
                        i--;
                        break;
                    }
                }
            }
        }
    }
}

/*
Cit: (docs.opencv.org)
    A basic summary of the algorithm is presented below:

    1)  (Segment test)  
        Select a pixel p in the image which is to be identified as an interest point or not. Let its intensity be Ip.
        Select appropriate threshold value t.
        Consider a circle of 16 pixels around the pixel under test.
        Now the pixel p is a corner if there exists a set of n contiguous pixels in the circle (of 16 pixels) which are all brighter than Ip+t,
        or all darker than Ip−t.
    
    2)  (High speed test)
        A high-speed test was proposed to exclude a large number of non-corners. This test examines only the four pixels at 1, 9, 5 and 13
        (First 1 and 9 are tested if they are too brighter or darker. If so, then checks 5 and 13).
        If p is a corner, then at least three of these must all be brighter than Ip+t or darker than Ip−t. If neither of these is the case,
        then p cannot be a corner. The full segment test criterion can then be applied to the passed candidates by examining all pixels in the circle.
        This detector in itself exhibits high performance, but there are several weaknesses:
        .   It does not reject as many candidates for n < 12.
        .   The choice of pixels is not optimal because its efficiency depends on ordering of the questions and distribution of corner appearances.
        .   Results of high-speed tests are thrown away.
        .   Multiple features are detected adjacent to one another.
    
    3)  (Non maximum suppression)
        .   Compute a score function, V for all the detected feature points. V is the sum of absolute difference between p and 16 surrounding pixels values.
        .   Consider two adjacent keypoints and compute their V values.
        .   Discard the one with lower V value.
*/
vector<KeyPoint> fastCornerDetection(const Mat &img, int threshold = 50, int n = 12, int dist = 3) {
    vector<KeyPoint> keypoints;

    for (int y = 1; y < img.rows - 1; ++y) {
        for (int x = 1; x < img.cols - 1; ++x) {
            if (fastHighSpeedTest(img, x, y, threshold)) {
                if (fastSegmentTest(img, x, y, threshold, n)) {
                    keypoints.push_back(KeyPoint(x, y, 1.f));
                }
            }
        }
    }

    nonMaximumSuppression(keypoints, img, dist);

    return keypoints;
}

/*
✅ Molto veloce: non richiede il calcolo di derivate o matrici.
✅ Efficace su immagini in movimento o tempo reale.
❌ Non robusto alla rotazione.
❌ Non fornisce una misura della risposta del corner: solo una classificazione binaria, non "quanto forte".
❌ Non invariante alla scala: il raggio fisso del cerchio lo rende inadatto a scale diverse.
*/