#include <iostream>
#include <vector>
#include <dirent.h>

// DBoW2
#include <DBoW2/DBoW2.h> // defines Surf64Vocabulary and Surf64Database

#include <DUtils/DUtils.h>
#include <DUtilsCV/DUtilsCV.h> // defines macros CVXX
#include <DVision/DVision.h>

// OpenCV
#include <opencv/cv.h>
#include <opencv/highgui.h>
#if CV24
#include <opencv2/nonfree/features2d.hpp>
#endif

#include <mrpt/utils.h>
#include <mrpt/obs/CRawlog.h>
#include <mrpt/obs/CObservation3DRangeScan.h>

//#include <Miscellaneous.h>

using namespace DBoW2;
using namespace DUtils;
using namespace std;

using namespace mrpt;
using namespace mrpt::obs;
using namespace mrpt::utils;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

// number of training images
const int NIMAGES = 4;

// extended surf gives 128-dimensional vectors
const bool EXTENDED_SURF = false;

//void loadFeatures(vector<vector<vector<float> > > &features, const string & filename);
//void changeStructure(const vector<float> &plain, vector<vector<float> > &out, int L);
//void testVocCreation(const vector<vector<vector<float> > > &features);
//void testDatabase(const vector<vector<vector<float> > > &features);


void getListOfFiles(const std::string & path_folder, std::vector<std::string> & path_files)
{
    DIR *dir = opendir (path_folder.c_str());
    struct dirent *file;
    if( dir != NULL )
    {
        /* print all the files and directories within directory */
        while ((file = readdir (dir)) != NULL)
        {
            path_files.push_back(file->d_name);
            printf ("%s\n", file->d_name);
        }
        closedir (dir);
    } else
    {
        /* could not open directory */
        perror ("");
        return;
    }
}

void getListOfFilesByType(const std::string & path_folder, const std::string & file_type, std::vector<std::string> & path_files)
{
    DIR *dir = opendir (path_folder.c_str());
    struct dirent *file;
    if( dir != NULL )
    {
        /* print all the files and directories within directory */
        while((file = readdir (dir)) != NULL)
        {
            //printf("%s\n", file->d_name);
            std::string file_name = file->d_name;
            if( file_name.length() > file_type.length() &&
                file_type.compare( file_name.substr(file_name.length()-file_type.length()) ) == 0  )
            {
                path_files.push_back(file->d_name);
                //printf("%s\n", file->d_name);
            }
        }
        closedir (dir);
    } else
    {
        /* could not open directory */
        perror ("");
        return;
    }
}


// ----------------------------------------------------------------------------

void changeStructure(const vector<float> &plain, vector<vector<float> > &out, int L)
{
    out.resize(plain.size() / L);

    unsigned int j = 0;
    for(unsigned int i = 0; i < plain.size(); i += L, ++j)
    {
        out[j].resize(L);
        std::copy(plain.begin() + i, plain.begin() + i + L, out[j].begin());
    }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

//void loadFeatures(vector<vector<vector<float> > > &features, const string & filename)
//{
//    mrpt::obs::CRawlog dataset;
//    if (!dataset.loadFromRawLogFile(filename))
//        throw std::runtime_error("\nCouldn't open dataset dataset file for input...");
//    cout << "dataset size " << dataset.size() << "\n";

//    // Set external images directory:
//    const string imgsPath = CRawlog::detectImagesDirectory(filename);
//    CImage::IMAGES_PATH_BASE = imgsPath;

//    features.clear();
//    features.reserve(dataset.size());

//    cv::SURF surf(400, 4, 2, EXTENDED_SURF);

//    cout << "Extracting SURF features..." << endl;
//    size_t n_RGBD = 0, n_obs = 0;
//    mrpt::obs::CObservationPtr observation;
//    mrpt::obs::CObservation3DRangeScanPtr obsRGBD;  // The RGBD observation
//    while ( n_obs < dataset.size() )
//    {
//        observation = dataset.getAsObservation(n_obs);
//        ++n_obs;
//        if(!IS_CLASS(observation, CObservation3DRangeScan))
//        {
//            continue;
//        }

//        cout << n_obs << " observation: " << observation->sensorLabel << ". Timestamp " << observation->timestamp << endl;
//        ++n_RGBD;

//        obsRGBD = mrpt::obs::CObservation3DRangeScanPtr(observation);
//        obsRGBD->load();

//        ASSERT_( obsRGBD->hasIntensityImage ); // Check that the images are really RGBD

//        cv::Mat image = cv::Mat(obsRGBD->intensityImage.getAs<IplImage>());
//        cv::Mat mask;
//        vector<cv::KeyPoint> keypoints;
//        vector<float> descriptors;

//        surf(image, mask, keypoints, descriptors);

//        features.push_back(vector<vector<float> >());
//        changeStructure(descriptors, features.back(), surf.descriptorSize());
//    }
//    features.resize(n_RGBD);
//}

void loadFeatures(vector<vector<vector<float> > > &features, const std::string path_folder, const std::vector<std::string> & img_names)
{
    features.clear();
    features.reserve(img_names.size());
    cv::SURF surf(400, 4, 2, EXTENDED_SURF);

    cout << "Extracting SURF features..." << endl;
    for(size_t i=0; i < img_names.size(); i++ )
    {
        //cout << "img: " << img_names[i] << endl;
        string img_path = path_folder + "/" + img_names[i];
        cv::Mat image = cv::imread(img_names[i], 0);
        cv::Mat mask;
        vector<cv::KeyPoint> keypoints;
        vector<float> descriptors;

        surf(image, mask, keypoints, descriptors);

        features.push_back(vector<vector<float> >());
        changeStructure(descriptors, features.back(), surf.descriptorSize());
    }
}

// ----------------------------------------------------------------------------

void testVocCreation(const vector<vector<vector<float> > > &features)
{
    // branching factor and depth levels
    const int k = 10;
    const int L = 4; //[3 - 6]
    const WeightingType weight = TF_IDF;
    const ScoringType score = L1_NORM;

    Surf64Vocabulary voc(k, L, weight, score);

    cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
    voc.create(features);
    cout << "... done!" << endl;

    cout << "Vocabulary information: " << endl
         << voc << endl << endl;

    // lets do something with this vocabulary
    cout << "Matching images against themselves (0 low, 1 high): " << endl;
    BowVector v1, v2;
    for(int i = 0; i < NIMAGES; i++)
    {
        voc.transform(features[i], v1);
        for(int j = 0; j < NIMAGES; j++)
        {
            voc.transform(features[j], v2);

            double score = voc.score(v1, v2);
            cout << "Image " << i << " vs Image " << j << ": " << score << endl;
        }
    }

    // save the vocabulary to disk
    cout << endl << "Saving vocabulary..." << endl;
    voc.save("small_voc.yml.gz");
    cout << "Done" << endl;
}

// ----------------------------------------------------------------------------

void testDatabase(const vector<vector<vector<float> > > &features)
{
    cout << "Creating a small database..." << endl;

    // load the vocabulary from disk
    Surf64Vocabulary voc("small_voc.yml.gz");

    Surf64Database db(voc, false, 0); // false = do not use direct index
    // (so ignore the last param)
    // The direct index is useful if we want to retrieve the features that
    // belong to some vocabulary node.
    // db creates a copy of the vocabulary, we may get rid of "voc" now

    // add images to the database
    for(size_t i = 0; i < features.size(); i++)
    {
        db.add(features[i]);
    }

    cout << "... done!" << endl;

    cout << "Database information: " << endl << db << endl;

    // and query the database
    cout << "Querying the database: " << endl;

    QueryResults ret;
    for(size_t i = 0; i < features.size(); i++)
    {
        db.query(features[i], ret, 4);

        // ret[0] is always the same image in this case, because we added it to the
        // database. ret[1] is the second best match.

        cout << "Searching for Image " << i << ". " << ret << endl;
    }

    cout << endl;

    // we can save the database. The created file includes the vocabulary
    // and the entries added
    cout << "Saving database..." << endl;
    db.save("small_db.yml.gz");
    cout << "... done!" << endl;

    // once saved, we can load it again
    cout << "Retrieving database once again..." << endl;
    Surf64Database db2("small_db.yml.gz");
    cout << "... done! This is: " << endl << db2 << endl;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

void wait()
{
    cout << endl << "Press enter to continue" << endl;
    getchar();
}

// ----------------------------------------------------------------------------


int main (int argc, char ** argv)
{

    if(argc != 3 )
    {
        cout << "Provide 3 paths containing the images for the vocabulary, the database images and the query sequence." << endl;
        return 0;
    }

    const string path_img_folder_voc = string( argv[1] );
    const string path_img_folder_database = string( argv[2] );
    //const string path_img_folder_query = string( argv[3] );

    string file_type = ".png";
    std::vector<std::string> images_voc, images_datab;
    getListOfFilesByType(path_img_folder_voc, file_type, images_voc);

    vector<vector<vector<float> > > features_voc, features_database;
    loadFeatures(features_voc, path_img_folder_voc, images_voc);

    wait();

    testVocCreation(features_voc);

    wait();

    getListOfFilesByType(path_img_folder_voc, file_type, images_voc);
    loadFeatures(features_database, path_img_folder_database, images_datab);
    testDatabase(features_database);

    return 0;
}
