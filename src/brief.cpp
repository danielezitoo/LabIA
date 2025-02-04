#include <opencv2/opencv.hpp>
#include <vector>
#include <bitset>
#include <random>

using namespace cv;
using namespace std;

int n_bits = 256;
vector<Point2i> pattern;

void generatePattern(int patch_size = 31) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(-patch_size / 2, patch_size / 2);
    for (int i = 0; i < n_bits * 2; i++) {
        pattern.emplace_back(dist(gen), dist(gen));
    }
}

Mat computeBRIEF(const Mat &img, const vector<KeyPoint> &keypoints) {
    Mat descriptors(keypoints.size(), n_bits / 8, CV_8U);
    Mat gray;
    if (img.channels() > 1)
        cvtColor(img, gray, COLOR_BGR2GRAY);
    else
        gray = img;

    for (size_t i = 0; i < keypoints.size(); i++) {
        int x = keypoints[i].pt.x;
        int y = keypoints[i].pt.y;
        bitset<256> descriptor;

        for (int j = 0; j < n_bits; j++) {
            Point2i p1 = pattern[2 * j] + Point2i(x, y);
            Point2i p2 = pattern[2 * j + 1] + Point2i(x, y);
            if (p1.x >= 0 && p1.y >= 0 && p1.x < gray.cols && p1.y < gray.rows &&
                p2.x >= 0 && p2.y >= 0 && p2.x < gray.cols && p2.y < gray.rows) {
                descriptor[j] = gray.at<uchar>(p1) < gray.at<uchar>(p2);
            }
        }

        for (int j = 0; j < n_bits / 8; j++) {
            uchar byte = 0;
            for (int k = 0; k < 8; k++) {
                byte |= descriptor[j * 8 + k] << k;
            }
            descriptors.at<uchar>(i, j) = byte;
        }
    }
    return descriptors;
}

vector<DMatch> matchBRIEF(const Mat &desc1, const Mat &desc2) {
    vector<DMatch> matches;
    for (int i = 0; i < desc1.rows; i++) {
        int best_dist = numeric_limits<int>::max();
        int best_idx = -1;
        for (int j = 0; j < desc2.rows; j++) {
            int distance = 0;
            for (int k = 0; k < desc1.cols; k++) {
                distance += bitset<8>(desc1.at<uchar>(i, k) ^ desc2.at<uchar>(j, k)).count();
            }
            if (distance < best_dist) {
                best_dist = distance;
                best_idx = j;
            }
        }
        if (best_idx != -1) {
            matches.emplace_back(i, best_idx, best_dist);
        }
    }
    return matches;
}