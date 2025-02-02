#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

// Fa una X sui corners
void drawCorners(Mat &img, const vector<KeyPoint> &keypoints) {
    for (const auto &kp : keypoints) {
        line(img, 
             Point(cvRound(kp.pt.x) - 5, cvRound(kp.pt.y) - 5), 
             Point(cvRound(kp.pt.x) + 5, cvRound(kp.pt.y) + 5), 
             Scalar(0, 255, 0));

        line(img, 
             Point(cvRound(kp.pt.x) - 5, cvRound(kp.pt.y) + 5), 
             Point(cvRound(kp.pt.x) + 5, cvRound(kp.pt.y) - 5), 
             Scalar(0, 255, 0));
    }
}