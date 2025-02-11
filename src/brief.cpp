#include "globals.h"

int n_bits;
vector<Point2i> pattern;

/*
    Viene generato un numero pari di punti (2 per ogni bit, quindi n_bits * 2 elementi) perché ogni test binario confronta l’intensità di due pixel.
    Le coordinate vengono scelte in modo casuale, il che corrisponde alla scelta casuale delle coppie di pixel all’interno della patch.
*/
void generatePattern(int patch_size = 64) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(-patch_size / 2, patch_size / 2);
    for (int i = 0; i < n_bits * 2; i++) {
        pattern.emplace_back(dist(gen), dist(gen));
    }
}

/*
Cit: (medium.com)
    BRIEF definisce un insieme di test binari da applicare alla patch.
    Ogni test consiste nel confrontare l’intensità di due pixel all’interno della patch.
    Se l’intensità del primo pixel è minore di quella del secondo, il test restituisce 1, 0 altrimenti.
    La sequenza di risultati (bit) viene concatenata per formare il descrittore binario del keypoint.
    Il numero totale di test (e quindi di bit) è un parametro dell’algoritmo. Ad esempio, si può decidere
    di eseguire 128 o 256 test per ottenere un descrittore di 128 o 256 bit.
    Le coppie (p,q) possono essere selezionate in modo casuale oppure seguendo una distribuzione
    (spesso gaussiana centrata sulla patch per dare maggiore peso ai pixel vicini al centro).
*/
Mat computeBRIEF(const Mat &img, const vector<KeyPoint> &keypoints, int patch_size = 64, int n_bits_par = 256) {
    n_bits = n_bits_par;
    generatePattern(patch_size);

    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    Mat descriptors(keypoints.size(), n_bits / 8, CV_8U);

    for (int i = 0; i < keypoints.size(); i++) {
        int x = keypoints[i].pt.x;
        int y = keypoints[i].pt.y;

        bitset<256> descriptor;

        for (int j = 0; j < n_bits; j++) {
            Point2i p1 = pattern[2 * j] + Point2i(x, y);
            Point2i p2 = pattern[2 * j + 1] + Point2i(x, y);
            // Controllo dei limiti dell'immagine
            if (p1.x >= 0 && p1.y >= 0 && p1.x < gray.cols && p1.y < gray.rows &&
                p2.x >= 0 && p2.y >= 0 && p2.x < gray.cols && p2.y < gray.rows) {
                descriptor[j] = gray.at<uchar>(p1) < gray.at<uchar>(p2);
            }
        }

        // Dopo aver costruito il bitset, lo converto in un vettore di byte poichè
        // memorizzare ogni bit singolarmente sarebbe inefficiente dal punto di vista dello spazio.
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


// Il matching avviene confrontando i descrittori con la distanza di Hamming.
vector<DMatch> matchBRIEF(const Mat &descriptor1, const Mat &descriptor2) {
    vector<DMatch> matches;

    for (int i = 0; i < descriptor1.rows; i++) {
        int best_dist = 9999999;
        int index = -1;

        for (int j = 0; j < descriptor2.rows; j++) {
            int dist = 0;

            for (int k = 0; k < descriptor1.cols; k++) {
                uchar byte1 = descriptor1.at<uchar>(i, k);
                uchar byte2 = descriptor2.at<uchar>(j, k);

                // Confronta i bit uno per uno
                for (int b = 0; b < 8; b++) {
                    bool bit1 = (byte1 & (1 << b)) != 0; // Estrae il bit b-esimo di byte1
                    bool bit2 = (byte2 & (1 << b)) != 0; // Estrae il bit b-esimo di byte2
                    if (bit1 != bit2) {
                        dist++; // Conta il numero di bit diversi
                    }
                }
            }

            if (dist < best_dist) {
                best_dist = dist;
                index = j;
            }
        }

        if (index != -1) {
            matches.emplace_back(i, index, best_dist);
        }
    }
    return matches;
}

/*
✅ Estremamente veloce (usa solo confronti di intensità).
✅ Vettori descrittori compatti (molto più piccoli rispetto a HOG o SIFT).
✅ Robusto al rumore e alla variazione di contrasto (grazie alla differenza di intensità tra coppie).
❌ Non invariante alla rotazione (se l'oggetto ruota, il descrittore cambia).
❌ Meno distintivo di HOG o SIFT.
*/