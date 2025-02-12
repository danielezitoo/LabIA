#include "harris.cpp"
#include "fast.cpp"
#include "shi_tomasi.cpp"
#include "patch_descriptors.cpp"
#include "brief.cpp"
#include "hog.cpp"
#include "make_harris.cpp"
#include "make_fast.cpp"
#include "make_shitomasi.cpp"
#include "make_orb.cpp"
#include "make_sift.cpp"
#include "make_match_harris.cpp"
#include "make_match_fast.cpp"
#include "make_match_shitomasi.cpp"
#include "make_panorama.cpp"
#include "utils.cpp"
#include "globals.h"

Mat img, imgWithCorners, img1, img2, imgMatches, descriptors1, descriptors2, imgCopy, panorama;
vector<KeyPoint> keypoints, keypoints1, keypoints2;
vector<DMatch> matches;

int MAX_SIZE = 1000;

// Parametri di default per Harris
// ("int" perchè è il tipo che vuole createTrackBar(), li convertirò successivamente)
int window_harris = 20;
int sigma_bar_harris = 34;
int threshold_bar_harris = 100;

// Parametri di default per FAST
int threshold_bar_fast = 50;
int n_bar_fast = 12;
int dist_bar_fast = 3;

// Parametri di default per Shi-Tomasi
int threshold_bar_shi_tomasi = 10000;
int window_size_bar_shi_tomasi = 3;

// Parametri di default per SIFT
int threshold_bar_ransac_sift = 13;
int maxIterations_bar_ransac_sift = 20000;

// Parametri di default per ORB
int threshold_fast_bar_orb = 70;
int patch_size_bar_orb = 64;
int n_bits_bar_orb = 256;
int threshold_bar_ransac_orb = 12;
int maxIterations_bar_ransac_orb = 5000;

// Parametri di default per match Harris
int window_bar_match_harris = 20;
double sigma_bar_match_harris = 3.4f;
float threshold_bar_match_harris = 10.0f;
int patch_size_bar_match_harris = 64;
int n_bits_bar_match_harris = 256;
int threshold_bar_ransac_harris = 3;
int maxIterations_bar_ransac_harris = 20000;
string descriptorType = "hog"; // Utile anche per Shi-Tomasi Matching

// Parametri di default per match Fast
int threshold_bar_match_fast = 44;
int n_bar_match_fast = 9;
int dist_bar_match_fast = 3;
int threshold_bar_ransac_fast = 9;
int maxIterations_bar_ransac_fast = 20000;

// Parametri di default per match Shi-Tomasi
int threshold_bar_match_shi_tomasi = 10000;
int window_size_bar_match_shi_tomasi = 3;
int patch_size_bar_match_shi_tomasi = 64;
int n_bits_bar_match_shi_tomasi = 256;
int threshold_bar_ransac_shi_tomasi = 3;
int maxIterations_bar_ransac_shi_tomasi = 15000;

// Parametro per fare merge se richiesto
bool merge_after_match = false;
bool merge_tot = false;

int main(int argc, char** argv) {
    // Leva i warning generati dalle Trackbar
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    if (argc < 2) {
        cout << "Uso: ./progetto <harris | fast | shitomasi | match>" << endl;
        return -1;
    }

    string command = argv[1];

    string imgPath1;
    string imgPath2;

    cout << "\nInserisci il percorso della prima immagine:\n";

    string filename;
    cin >> filename;
    string path = "immagini/" + filename + ".jpg";
    if (!fs::exists(path)) {
        cout << "File non trovato: " << filename << endl;
        return -1;
    }
    else {
        imgPath1=path;
    }

    cout << "Inserisci il percorso della seconda immagine:\n";

    cin >> filename;
    path = "immagini/" + filename + ".jpg";
    if (!fs::exists(path)) {
        cout << "File non trovato: " << filename << endl;
        return -1;
    }
    else {
        imgPath2=path;
    }

    if (command == "harris") {
        do_harris(imgPath1);
        do_harris(imgPath2);
    }
    else if (command == "fast") {
        do_fast(imgPath1);
        do_fast(imgPath2);
    }
    else if (command == "shitomasi") {
        do_shi_tomasi(imgPath1);
        do_shi_tomasi(imgPath2);
    }
    else if (command == "match") {
        if (argc < 3) {
            cout << "Uso: ./progetto match <harris | fast | shitomasi | sift | orb>" << endl;
            return -1;
        }

        const string& merge_command=argv[argc - 1];

        if (merge_command == "merge") {
            merge_after_match = true;
        }

        string type_kp = argv[2];
        
        if (type_kp == "sift") { 
            do_sift(imgPath1, imgPath2);
        }
        else if (type_kp == "orb") {
            do_orb(imgPath1, imgPath2);
        }
        else if (type_kp == "harris") {
            if (argc < 4) {
                cout << "Uso: ./progetto match harris <brief | hog>" << endl;
                return -1;
            }

            string descriptor = argv[3];

            if (descriptor == "brief" || descriptor == "hog") {
                do_match_harris(imgPath1, imgPath2, descriptor);
            } 
            else {
                cout << "Descrittore non valido. Usa: brief | hog" << endl;
                return -1;
            }
        }
        else if (type_kp == "fast") {
            if (argc < 4 || string(argv[3]) != "hog") {
                cout << "Uso: ./progetto match fast hog (fast + brief è già ORB)" << endl;
                return -1;
            }

            do_match_fast(imgPath1, imgPath2);
        }
        else if (type_kp == "shitomasi") {
            if (argc < 4) {
                cout << "Uso: ./progetto match shitomasi <brief | hog>" << endl;
                return -1;
            }

            string descriptor = argv[3];

            if (descriptor == "brief" || descriptor == "hog") {
                do_match_shitomasi(imgPath1, imgPath2, descriptor);
            }
            else {
                cout << "Descrittore non valido. Usa: brief | hog" << endl;
                return -1;
            }
        }
        else {
            cout << "Metodo di matching non valido. Usa: harris | fast | shitomasi | sift | orb" << endl;
            return -1;
        }
    }
    // TODO BETTER //
    else if (command == "mergetot") {
        merge_tot = true;

        vector<Mat> images;
        string folder_path = "immagini/";
        vector<string> image_names;

        cout << "Inserisci i nomi delle immagini da unire ('end' per terminare):\n";
        string filename;
        while (true) {
            cin >> filename;
            if (filename == "end") break;
            string path = folder_path + filename;
            if (fs::exists(path)) {
                image_names.push_back(path);
            } else {
                cout << "File non trovato: " << filename << endl;
            }
        }

        for (const string& name : image_names) {
            Mat imgM = imread(name);
            if (!checkImage(imgM, name)) {
                return -1;
            }
            images.push_back(imgM);
        }

        if (images.size() < 2) {
            cerr << "Errore: servono almeno due immagini per il merging." << endl;
            return -1;
        }

        panorama = mergeMultipleImages(images);

        if (!panorama.empty()) {
            imwrite("panorama.jpg", panorama);
            imshow("Panorama", panorama);
            waitKey(1);
        } else {
            cerr << "Errore nella creazione del panorama." << endl;
        }
    }
    else {
        cout << "Uso: ./progetto <harris | fast | shitomasi | match>" << endl;
        return -1;
    }
    
    return 0;
}