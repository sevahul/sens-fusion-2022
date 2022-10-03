#include <jsoncpp/json/reader.h>
#include <jsoncpp/json/value.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include "main.h"
#include <cmath>
#include <omp.h>
#include <boost/program_options.hpp> // COMMENT IN CASE YOU DONT HAVE BOOST
#include <chrono>
#include <jsoncpp/json/json.h>

namespace po = boost::program_options;
namespace ch = std::chrono;

int main(int argc, char** argv) {

    // Optionally set nProcessors (maximum by default)
    int nProcessors = 0;
    // And window size
    int window_size = 3;

    // COMMENT THE NEXT BLOCK IN CASE YOU DONT HAVE BOOST
    po::options_description desc("Options for my program");
    desc.add_options()
    ("jobs,j", po::value<int>(& nProcessors)->default_value(omp_get_max_threads()), "Number of Threads")
    ("window-size,w", po::value<int>(& window_size)->default_value(3), "Window Size");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);


    ////////////////
    // Parameters //
    ////////////////

    // setup njobs
    if (!nProcessors) {
        nProcessors = omp_get_max_threads();
    }
    omp_set_num_threads(nProcessors);

    // camera setup parameters
    std::ifstream ifs("data/config.json");
    Json::Reader reader;
    Json::Value obj;
    reader.parse(ifs, obj);

    const double focal_length = obj["focal_length"].asDouble();
    const double baseline = obj["baseline"].asDouble();

    // stereo estimation parameters
    const int dmin = obj["dmin"].asInt();

    ///////////////////////////
    // Commandline arguments //
    ///////////////////////////

    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " IMAGE1 IMAGE2 OUTPUT_FILE [-j<njobs> -w<window-size>]" << std::endl;
        return 1;
    }

    cv::Mat image1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    cv::Mat image2 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
    const std::string output_file = argv[3];

    if (!image1.data) {
        std::cerr << "No image1 data" << std::endl;
        return EXIT_FAILURE;
    }

    if (!image2.data) {
        std::cerr << "No image2 data" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "------------------ Parameters -------------------" << std::endl;
    std::cout << "focal_length = " << focal_length << std::endl;
    std::cout << "baseline = " << baseline << std::endl;
    std::cout << "window_size = " << window_size << std::endl;
    std::cout << "disparity added due to image cropping = " << dmin << std::endl;
    std::cout << "output filename = " << argv[3] << std::endl;
    std::cout << "nProcessors = " << nProcessors << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;

    int height = image1.size().height;
    int width = image1.size().width;

    ////////////////////
    // Reconstruction //
    ////////////////////

    // Naive disparity image
    //cv::Mat naive_disparities = cv::Mat::zeros(height - window_size, width - window_size, CV_8UC1);
    cv::Mat naive_disparities = cv::Mat::zeros(height, width, CV_8UC1);

    StereoEstimation_Naive(
        window_size, dmin, height, width,
        image1, image2,
        naive_disparities);

    ////////////
    // Output //
    ////////////

    // reconstruction
    Disparity2PointCloud(
        output_file,
        height, width, naive_disparities,
        window_size, dmin, baseline, focal_length);

    // save / display images
    std::stringstream out1;
    out1 << output_file << "_naive.png";
    cv::imwrite(out1.str(), naive_disparities);

    cv::namedWindow("Naive", cv::WINDOW_AUTOSIZE);
    cv::imshow("Naive", naive_disparities);

    cv::waitKey(0);

    return 0;
}


void StereoEstimation_DP(
    const int& window_size,
    const int& dmin,
    int height,
    int width,
    cv::Mat& image1, cv::Mat& image2, cv::Mat& naive_disparities)
{
    std::cout << "height " << height << std::endl;
    std::cout << "width " << width << std::endl;
    int half_window_size = window_size / 2;

    auto start = ch::high_resolution_clock::now();
    cv::Mat dissim = cv::Mat::zeros(width, width, CV_32FC1);
    // for each row (scanline)
    std::cout << "calculating the disparities using DP approach" << std::endl;
    // dissimilarity(i, j) for each (i, j)

    for (int y = 0; y < height; y++) {
        for (int i = half_window_size; i < height - half_window_size; ++i) {
            for (int j = half_window_size; j < width - half_window_size; j++) {
                float sum = 0;
                for (int u = 0 ; u <= half_window_size; u++) {
                    for (int v = 0 ; v <= half_window_size; v++) {
                        float i1 = static_cast<float>(image1.at<uchar>(y + v, i + u));
                        float i2 = static_cast<float>(image2.at<uchar>(y + v, i + u));
                        sum += std::abs(i1 - i2); // SAD
                        // sum += (i1 - i2) * (i1 - i2) // SSD
                    }
                }
                dissim.at<float>(i, j) = sum;
            }


        }
    }



    cv::Mat C = cv::Mat::zeros(width, width, CV_32FC1);
    cv::Mat M = cv::Mat::zeros(width, width, CV_8UC1);
    // allocate C, M
    // populate C, M (dynamic programming ... recursive function evaluation)
    //   initialize C and M (you can do it outside of for loop in case no parallel computing)
    //     first row
    //     first column
    //     (these do not have preceding nodes)
    // for(horizontally...)
    //   for(vertically...)
    //
    // trace sink -> source (from bottom-right to top-left of C/M)
    //   fill y-th row of disparities
    //   d = j-i
    //
    //

    auto stop = ch::high_resolution_clock::now();
    auto duration = ch::duration_cast<ch::seconds>(stop - start);

    std::cout << "Calculating disparities for the naive approach... Done in " << duration.count() << " seconds.\r" << std::flush;
    std::cout << std::endl;
}




void StereoEstimation_Naive(
    const int& window_size,
    const int& dmin,
    int height,
    int width,
    cv::Mat& image1, cv::Mat& image2, cv::Mat& naive_disparities)
{
    std::cout << "height " << height << std::endl;
    std::cout << "width " << width << std::endl;
    int half_window_size = window_size / 2;

    auto start = ch::high_resolution_clock::now();

    for (int i = half_window_size; i < height - half_window_size; ++i) {
        std::cout
                << "Calculating disparities for the naive approach... "
                << std::ceil(((i - half_window_size + 1) / static_cast<double>(height - window_size + 1)) * 100) << "%\r"
                << std::flush;

        #pragma omp parallel for
        for (int j = half_window_size; j < width - half_window_size; ++j) {
            double min_ssd = INT_MAX;
            int disparity = 0;

            for (int d = -j + half_window_size; d < width - j - half_window_size; ++d) {
                double ssd = 0;

                for (int di = - half_window_size ; di <= half_window_size; di++)
                {
                    for (int dj = - half_window_size; dj <= half_window_size; dj++)
                    {
                        ssd += std::pow((int)image1.at<uchar>(i+di, j+dj) - (int)image2.at<uchar>(i+di, j+dj+d), 2);
                    }
                }

                if (ssd < min_ssd) {
                    min_ssd = ssd;
                    disparity = d;
                }
            }

            naive_disparities.at<uchar>(i - half_window_size, j - half_window_size) = std::abs(disparity);
        }
    }

    auto stop = ch::high_resolution_clock::now();
    auto duration = ch::duration_cast<ch::seconds>(stop - start);

    std::cout << "Calculating disparities for the naive approach... Done in " << duration.count() << " seconds.\r" << std::flush;
    std::cout << std::endl;
}

void Disparity2PointCloud(
    const std::string& output_file,
    int height, int width, cv::Mat& disparities,
    const int& window_size,
    const int& dmin, const double& baseline, const double& focal_length)
{
    std::stringstream out3d;
    out3d << output_file << ".xyz";
    std::ofstream outfile(out3d.str());
    for (int i = 0; i < height - window_size; ++i) {
        std::cout << "Reconstructing 3D point cloud from disparities... " << std::ceil(((i) / static_cast<double>(height - window_size + 1)) * 100) << "%\r" << std::flush;
        for (int j = 0; j < width - window_size; ++j) {
            if (disparities.at<uchar>(i, j) == 0) continue;

            int d = (int)disparities.at<uchar>(i, j) + dmin;
            int u1 = j - width/2;
            int u2 = j + d - width/2;
	    int v1 = i - height/2;

            const double Z = baseline * focal_length / d;
            const double X = -0.5 * ( baseline * (u1 + u2) ) / d;
            const double Y = baseline * v1 / d;
            outfile << X << " " << Y << " " << Z << std::endl;
        }
    }

    std::cout << "Reconstructing 3D point cloud from disparities... Done.\r" << std::flush;
    std::cout << std::endl;
}
