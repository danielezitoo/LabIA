#include "globals.h"

Mat rgbToGrayscale(const Mat& imgRGB) {
    Mat imgGray(imgRGB.size(), CV_32F);
    for (int y = 0; y < imgRGB.rows; y++) {
        for (int x = 0; x < imgRGB.cols; x++) {
            Vec3b pixel = imgRGB.at<Vec3b>(y, x);
            imgGray.at<float>(y, x) = 0.299f * pixel[2] + 0.587f * pixel[1] + 0.114f * pixel[0]; // openCV non considera RGB ma BGR (SI PUÒ PROVARE A NORMALIZZARE (?))
        }
    }
    return imgGray;
}

pair<Mat, Mat> gradient(const Mat& img) {
    Mat Ix, Iy;

    // Calcola il gradiente usando la funzione Scharr di OpenCV, che è ottimizzato per gradienti più precisi, al posto di calcolarlo con Sobel.
    Scharr(img, Ix, CV_32F, 1, 0);
    Scharr(img, Iy, CV_32F, 0, 1);

    return {Ix, Iy};
}

tuple<Mat, Mat, Mat> structureTensorSetup(const Mat& Ix, const Mat& Iy, int window, double sigma) {
    /*
    Cit: (www.baeldung.com)
        Compute the products of the derivatives I_x I_x , I_x I_y, I_y I_y.
        Convolve the images I_x I_x , I_x I_y, I_y I_y  with a Gaussian filter or a mean filter. Define the structure tensor for each pixel.
    */

    // La funzione mul() esegue una moltiplicazione elemento per elemento tra due matrici.
    // Quindi crea una matrice in cui ogni elemento è il quadrato dell'elemento corrispondente
    // (o Ix * Iy)

    Mat IxIx = Ix.mul(Ix); 
    Mat IyIy = Iy.mul(Iy);
    Mat IxIy = Ix.mul(Iy);
    
    Mat SxSx, SySy, SxSy;
    GaussianBlur(IxIx, SxSx, Size(window, window), sigma);
    GaussianBlur(IyIy, SySy, Size(window, window), sigma);
    GaussianBlur(IxIy, SxSy, Size(window, window), sigma);
    
    return {SxSx, SySy, SxSy};
}

Mat harrisResponse(const Mat& SxSx, const Mat& SySy, const Mat& SxSy) {
    Mat corners = Mat::zeros(SxSx.size(), CV_32F);
    
    for (int y = 0; y < SxSx.rows; y++) {
        for (int x = 0; x < SxSx.cols; x++) {
            float a = SxSx.at<float>(y, x);
            float b = SxSy.at<float>(y, x);
            float c = SySy.at<float>(y, x);
            
            float det = a * c - b * b;
            float trace = a + c;

            if (trace != 0) {
                corners.at<float>(y, x) = det / trace;  // Avrei potuto anche settare ogni elemento come det - k * trace * trace.
                                                        // Cit: (en.wikipedia.org)
                                                        //    k is an empirically determined constant: k in [0.04, 0.06].
            }
            else {
                corners.at<float>(y, x) = 0;
            }

        }
    }

    normalize(corners, corners, 0, 255, NORM_MINMAX);
    
    return corners;
}

// IMPLEMENTAZIONE FROM SCRATCH
Mat nonms(const Mat& corners, float threshold) {
    Mat cornersSuppressed = Mat::zeros(corners.size(), CV_32F);

    for (int y = 1; y < corners.rows - 1; y++) {
        for (int x = 1; x < corners.cols - 1; x++) {
            float pixel = corners.at<float>(y, x);
            bool localMax = true;
            
            /*
            Cit: (en.wikipedia.org)
                In order to pick up the optimal values to indicate corners, we find the local maxima as corners within the window which is a 3 by 3 filter.
            */
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (pixel < corners.at<float>(y + dy, x + dx)) {
                        localMax = false;
                        break;
                    }
                }
                if (!localMax) break;
            }
            
            if (localMax && pixel > threshold) {
                cornersSuppressed.at<float>(y, x) = pixel;
            }
        }
    }
    
    return cornersSuppressed;
}

/*
// Implementazione usando funzioni di openCV fatta per curiosità
Mat nonms(const Mat& corners, float threshold) {
    Mat dilated, localMax;
    
    dilate(corners, dilated, Mat(3, 3, CV_8U));  
    compare(corners, dilated, localMax, CMP_GE); 
    
    Mat cornersSuppressed = Mat::zeros(corners.size(), CV_32F);
    corners.copyTo(cornersSuppressed, localMax & (corners > threshold)); 
    
    return cornersSuppressed;
}
*/

vector<KeyPoint> extractKeypoints(const Mat& corners) {
    vector<KeyPoint> keypoints;
    
    for (int y = 0; y < corners.rows; y++) {
        for (int x = 0; x < corners.cols; x++) {
            if (corners.at<float>(y, x) > 0) {
                KeyPoint keypoint(Point2f(x, y), 1);
                keypoints.push_back(keypoint);
            }
        }
    }
    
    return keypoints;
}

/*
Cit: (en.wikipedia.org)
    Commonly, Harris corner detector algorithm can be divided into five steps:
*/
vector<KeyPoint> harrisCornerDetection(const Mat& imgRGB, int window = 3, double sigma = 2.0, float threshold = 0.2) {
    // 1. Color to grayscale
    Mat img = rgbToGrayscale(imgRGB);
    
    // 2. Spatial derivative calculation
    auto [Ix, Iy] = gradient(img);
    
    // 3. Structure tensor setup
    auto [SxSx, SySy, SxSy] = structureTensorSetup(Ix, Iy, window, sigma);
    
    // 4. Harris response calculation
    Mat cornersNonSuppressed = harrisResponse(SxSx, SySy, SxSy);

    // 5. Non-maximum suppression
    Mat corners = nonms(cornersNonSuppressed, threshold);
    
    return extractKeypoints(corners);
}

/*
✅ Robustezza: invarianza alla rotazione e alle variazioni di luminosità.
✅ Precisione: identifica bene i corner.
❌ Sensibile a variazioni di scala: non funziona bene se un oggetto viene ridimensionato.
❌ Computationally expensive.
*/