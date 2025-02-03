#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

// Fa una X sui corners
void drawCorners(Mat &img, const vector<KeyPoint> &keypoints) {
    for (size_t i = 0; i < keypoints.size(); i++) {
        Point pt(cvRound(keypoints[i].pt.x), cvRound(keypoints[i].pt.y));

        line(img, Point(pt.x - 5, pt.y - 5), Point(pt.x + 5, pt.y + 5), Scalar(0, 255, 0));
        line(img, Point(pt.x - 5, pt.y + 5), Point(pt.x + 5, pt.y - 5), Scalar(0, 255, 0));
    }
}