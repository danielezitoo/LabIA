#include "globals.h"

/*
Cit: (en.wikipedia.org)
    As Dalal and Triggs point out image pre-processing thus provides little impact on performance.
    Instead, the first step of calculation is the computation of the gradient values. The most common method is to apply the 
    derivative mask in one or both of the horizontal and vertical directions.
*/
void computeGradients(const Mat &img, Mat &mag, Mat &angle) {
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY, CV_32F);
    
    Mat gx, gy;
    Sobel(gray, gx, CV_32F, 1, 0, 1);
    Sobel(gray, gy, CV_32F, 0, 1, 1);
    
    cartToPolar(gx, gy, mag, angle, true);
}

/*
Cit: (en.wikipedia.org)
    The second step of calculation is creating the cell histograms. Each pixel within the cell casts a weighted vote for an orientation-based histogram bin
    based on the values found in the gradient computation. The cells themselves can either be rectangular or radial in shape,
    and the histogram channels are evenly spread over 0 to 180 degrees or 0 to 360 degrees, depending on whether the gradient is “unsigned” or “signed”.
    Dalal and Triggs found that unsigned gradients used in conjunction with 9 histogram channels performed best in their human detection experiments,
    while noting that signed gradients lead to significant improvements in the recognition of some other object classes, like cars or motorbikes.
    As for the vote weight, pixel contribution can either be the gradient magnitude itself, or some function of the magnitude.

    To account for changes in illumination and contrast, the gradient strengths must be locally normalized, which requires grouping the cells together.
    The HOG descriptor is then the concatenated vector of the components of the normalized cell histograms from all of the block regions.
    Two main block geometries exist: rectangular R-HOG blocks and circular C-HOG blocks. R-HOG (utilizzato da me) blocks are generally square grids
    The optimal parameters were found to be four 8x8 pixels cells per block (16x16 pixels per block) with 9 histogram channels.
*/
vector<float> computeHistogram(const Mat &magBlock, const Mat &angleBlock, int nBins, int cellSize, int blockSize) {
    // Creo un vettore descriptor che conterrà tutti gli istogrammi delle celle nel blocco.
    vector<float> descriptor(nBins * 4);
    
    for (int i = 0; i < blockSize; i += cellSize) {
        for (int j = 0; j < blockSize; j += cellSize) {
            vector<float> hist(nBins, 0.0f);
            
            for (int y = 0; y < cellSize; y++) {
                for (int x = 0; x < cellSize; x++) {
                    float theta = angleBlock.at<float>(i + y, j + x);
                    float m = magBlock.at<float>(i + y, j + x);
                    
                    // Convertiamo theta in un indice di bin.
                    // Poiché abbiamo 9 bin in 360°, ogni bin copre 40°.
                    // Sottraiamo 20° per centrare il primo bin attorno a 0°.
                    float ci = (theta - 20.0f) / 40.0f;
                    int leftBin = floor(ci);
                    //float weight = ci - leftBin;
                    
                    // Correggo eventuali indici negativi di leftBin.
                    leftBin = (leftBin + nBins) % nBins;
                    int rightBin = (leftBin + 1) % nBins;

                    hist[leftBin] += m/* * (1.0f - weight)*/;
                    hist[rightBin] += m/* * weight*/;
                }
            }
            
            int cellIdx = (i / cellSize) * (blockSize / cellSize) + (j / cellSize);
            copy(hist.begin(), hist.end(), descriptor.begin() + cellIdx * nBins);
        }
    }
    return descriptor;
}

/*
Cit (en.wikipedia.org)
    Dalal and Triggs explored four different methods for block normalization:
    .   L2-norm
    .   L2-hys
    .   L1-norm
    .   L1-sqrt
    In their experiments, Dalal and Triggs found the L2-hys, L2-norm, and L1-sqrt schemes provide similar performance,
    while the L1-norm provides slightly less reliable performance; however, all four methods showed very significant improvement over the non-normalized data.
*/
void normHOG(vector<float> &descriptor, float epsilon) {
    float normValue = sqrt(epsilon + norm(descriptor, NORM_L2));    // Epsilon serve per evitare successivamente divisioni per 0
    
    for (int i = 0; i < descriptor.size(); ++i) {
        descriptor[i] /= normValue;
    }
}

Mat computeHOG(const Mat &img, const vector<KeyPoint> &keypoints, int cellSize = 8, int blockSize = 16, int nBins = 9, float epsilon = 1e-6f) {
    Mat mag, angle;
    computeGradients(img, mag, angle);
    
    Mat descriptors(keypoints.size(), nBins * 4, CV_32F);
    
    for (int k = 0; k < keypoints.size(); k++) {
        Mat magBlock, angleBlock;
        getRectSubPix(mag, Size(blockSize, blockSize), keypoints[k].pt, magBlock);
        getRectSubPix(angle, Size(blockSize, blockSize), keypoints[k].pt, angleBlock);
        
        vector<float> descriptor = computeHistogram(magBlock, angleBlock, nBins, cellSize, blockSize);
        normHOG(descriptor, epsilon);
        
        Mat(descriptor).reshape(1,1).convertTo(descriptors.row(k), CV_32F);
    }
    return descriptors;
}

// Il matching avviene confrontando i descrittori con la distanza di Euclide.
vector<DMatch> matchHOG(const Mat &descriptor1, const Mat &descriptor2) {
    vector<DMatch> matches;
    
    for(int i=0; i<descriptor1.rows; i++){
        int bestIndex = -1;
        float bestDist = 1e10f;
        
        for(int j=0; j<descriptor2.rows; j++){
            float dist = 0.0f;
            for(int k=0; k<descriptor1.cols; k++){
                float d = descriptor1.at<float>(i,k) - descriptor2.at<float>(j,k);
                dist += d*d;
            }
            dist = sqrt(dist);
            
            if(dist < bestDist){
                bestDist = dist;
                bestIndex = j;
            }
        }
        
        if(bestIndex != -1)
            matches.emplace_back(i, bestIndex, bestDist);
    }
    
    return matches;
}

/*
✅ Robusto a variazioni di illuminazione e contrasto (grazie alla normalizzazione).
✅ Descrive bene bordi e contorni → molto utile per il riconoscimento di oggetti.
✅ Invariante a piccole traslazioni e rotazioni (ma non completamente).
❌ Sensibile a cambiamenti di scala e grandi rotazioni (non è completamente invariante).
❌ Computazionalmente pesante rispetto a BRIEF (richiede calcoli sui gradienti e normalizzazione).
*/