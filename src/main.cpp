#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <jsoncpp/json/reader.h>
#include <jsoncpp/json/value.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <ostream>
#include <string>
#include <fstream>
#include <sstream>
#include "main.h"
#include <cmath>
#include <omp.h>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <chrono>
#include <jsoncpp/json/json.h>
#include "progressbar.hpp"

namespace po = boost::program_options;
namespace fs = boost::filesystem;
namespace ch = std::chrono;

int main(int argc, char** argv) {
    
    ////////////////
    // Parameters //
    ////////////////
       // Optionally set nProcessors (maximum by default)
    int nProcessors;
    // And window size
    int window_size;
    
    fs::path default_config_path ("data");
    default_config_path /= "config.json";
    fs::path default_output_file ("output");
    default_output_file /= "output";
    fs::path default_image_file ("data");
    std::string output_file;
    std::string config_path;
    std::string image1_name;
    std::string image2_name;

    // Read parameters from command line (higher priority)
    po::options_description desc("Options for stereo program");
    desc.add_options()
    ("help,h", "produce help message")
    ("jobs,j", po::value<int>(& nProcessors)->default_value(omp_get_max_threads()), "Number of Threads (max by default)")
    ("config-path,c", po::value<std::string>(& config_path)->default_value(default_config_path.string()), "Path to the congiguration file")
    ("window-size,w", po::value<int>(& window_size), "Window Size (taken from config if not specified)")
    ("left-image,l", po::value<std::string>(& image1_name)->default_value((default_image_file/"view0.png").string()), "Image1 name")
    ("right-image,r", po::value<std::string>(& image2_name)->default_value((default_image_file/"view1.png").string()), "Image2 name")
    ("output,o", po::value<std::string>(& output_file)->default_value(default_output_file.string()), "Output files template (do not add extentions)");
   
    po::positional_options_description p;
    p.add("left-image", 1);
    p.add("right-image", 1);
    p.add("output", 1);
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
          options(desc).positional(p).run(), vm);
    po::notify(vm);
    
    if (vm.count("help")) {
	std::cout << desc << "\n";
	return 1;
    }



    // camera setup parameters
    
     
    // parameters config parser
    std::ifstream ifs(config_path);
    Json::Reader reader;
    Json::Value obj;
    reader.parse(ifs, obj);

    if(!vm.count("window-size")){
	window_size = obj.isMember("window_size") ? obj["window_size"].asInt() : 3;
    }
    if(!vm.count("output-file")){
	output_file = obj.isMember("output") ? obj["output"].asString() : default_output_file.string();
    }
    
    const double focal_length = obj.isMember("focal_length") ? obj["focal_length"].asDouble(): 3740;
    const double baseline = obj.isMember("baseline") ? obj["baseline"].asDouble(): 160;
    const bool debug = obj.isMember("debug") ? obj["debug"].asBool() : false;
    const std::string method = obj.isMember("method") ? obj["method"].asString() : "naive";
    const double lambda = obj.isMember("lambda_DP") ? obj["lambda_DP"].asDouble()*window_size*window_size : 0;
 
    // stereo estimation parameters
    const int dmin = obj.isMember("dmin") ? obj["dmin"].asInt(): 200;

    ///////////////////////////
    // Commandline arguments //
    ///////////////////////////

    cv::Mat image1 = cv::imread(image1_name, cv::IMREAD_GRAYSCALE);
    cv::Mat image2 = cv::imread(image2_name, cv::IMREAD_GRAYSCALE);

    if (!image1.data) {
        std::cerr << "No image1 data" << std::endl;
        return EXIT_FAILURE;
    }

    if (!image2.data) {
        std::cerr << "No image2 data" << std::endl;
        return EXIT_FAILURE;
    }
    
    
    int height = image1.size().height;
    int width = image1.size().width;
    
    std::cout << "------------------ Parameters -------------------" << std::endl;
    std::cout << "focal_length = " << focal_length << std::endl;
    std::cout << "baseline = " << baseline << std::endl;
    std::cout << "window_size = " << window_size << std::endl;
    std::cout << "disparity added due to image cropping = " << dmin << std::endl;
    std::cout << "output filename = " << output_file << std::endl;
    std::cout << "nProcessors = " << nProcessors << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;

    std::cout << "height " << height << std::endl;
    std::cout << "width " << width << std::endl;   
    std::cout << "method = " << method << std::endl;
    if (method == "DP"){ 
    	std::cout << "lambda = " << (int)(lambda/window_size/window_size) << std::endl;
    	std::cout << "lambda*window_size^2 = " << lambda << std::endl;
	std::cout << "debug = " << debug << std::endl; 
    }
     std::cout << "-------------------------------------------------" << std::endl;


    ////////////////////
    // Reconstruction //
    ////////////////////

    // Disparity image
    
    
    // setup njobs
    omp_set_num_threads(nProcessors);
    
    cv::Mat disparities = cv::Mat::zeros(height, width, CV_8UC1);

    if (method=="naive") {
	 StereoEstimation_Naive(
            window_size, dmin, height, width,
            image1, image2,
            disparities);
    } else if (method == "DP"){
	StereoEstimation_DP(
            window_size, dmin, height, width,
            image1, image2,
            disparities, lambda, debug);
    } else {
       std::cerr << "Unknown method!\n The method has to be either 'naive' or 'DP'" << std::endl;
       return EXIT_FAILURE;
    }

    ////////////
    // Output //
    ////////////

    // Reconstruction
    Disparity2PointCloud(
        output_file,
        height, width, disparities,
        window_size, dmin, baseline, focal_length);

    // save / display images
    std::stringstream out1;
    out1 << output_file << "_" << method << ".png";
    cv::imwrite(out1.str(), disparities);

    cv::namedWindow("Disaparities_" + method, cv::WINDOW_AUTOSIZE);
    cv::imshow("Disaparities_" + method, disparities);

    cv::waitKey(0);

    return 0;
}


void StereoEstimation_DP(
    const int& window_size,
    const int& dmin,
    int height,
    int width,
    cv::Mat& image1, cv::Mat& image2, cv::Mat& disparities, double lambda, bool debug)
{
    int half_window_size = window_size / 2;
	
    auto start = ch::high_resolution_clock::now();
    int dwidth = width - 2*half_window_size;

    progressbar bar(height);
    
    
    // for each row (scanline)
    std::cout << "Calculating disparities for the DP approach... " << std::endl;

    #pragma omp parallel for
    for (int y = 0; y < height; y++) {

	#pragma omp critical
		bar.update();

    	cv::Mat dissim = cv::Mat::zeros(dwidth, dwidth, CV_32FC1);

        for (int i = half_window_size; i < width - half_window_size; ++i) {
            for (int j = half_window_size; j < width - half_window_size; j++) {
                float sum = 0;
                for (int u = 0 ; u <= half_window_size; u++) {
                    for (int v = 0 ; v <= half_window_size; v++) {
                        float i1 = static_cast<float>(image1.at<uchar>(y + v, i + u));
                        float i2 = static_cast<float>(image2.at<uchar>(y + v, j + u));
                        sum += std::abs(i1 - i2); // SAD
                        // sum += (i1 - i2) * (i1 - i2) // SSD
                    }
                }
                dissim.at<float>(i - half_window_size, j - half_window_size) = sum;
            }
        }
	
	// allocate C, M
	
	cv::Mat C = cv::Mat::zeros(dwidth, dwidth, CV_32FC1);
    	cv::Mat M = cv::Mat::zeros(dwidth, dwidth, CV_8UC1);
	C(cv::Range(0, 1), cv::Range(0, dwidth)) = dissim(cv::Range(0, 1), cv::Range(0, dwidth));
        C(cv::Range(0, dwidth), cv::Range(0, 1)) = dissim(cv::Range(0, dwidth), cv::Range(0, 1));
	// loop over matricies
	for (int i = 1; i < dwidth; i++) {
	    for (int j = 1; j < dwidth; j++) {
		M.at<uint8_t>(i, j) = 0;
		double value = C.at<float>(i-1, j-1) + dissim.at<float>(i, j);
		if (value > (C.at<float>(i-1, j) + lambda)) {
		    value = C.at<float>(i-1, j) + lambda;
		    M.at<uint8_t>(i, j) = 1;
		}
		if (value > (C.at<float>(i, j-1) + lambda)) {	
		    value = C.at<float>(i, j-1) + lambda;
		    M.at<uint8_t>(i, j) = 2;
		}
		C.at<float>(i, j) = value;
	    }
 	}
	
	// trace back
	int j = dwidth - 1;
        int i = dwidth - 1;
	int last_defined = 0;
	cv::Mat path = cv::Mat::zeros(dwidth, dwidth, CV_8UC1);
	while ((j >= 0) && (i>=0)){
	    path.at<uchar>(i, j) = 255;
	    switch (M.at<uchar>(i, j)) {
		case 0:
		    disparities.at<uchar>(y, i + half_window_size) = std::abs(j - i);
		    last_defined = std::abs(j-i);
		    i--; j--;
		    break;
 		case 1:
		    disparities.at<uchar>(y, i + half_window_size) = 0;
		    i--;
  		    break;
		case 2:
		    j--;
		    break;
		default:
		    std::cout << "Unexpected case!" << std::endl;
		    exit(0);
	    }
	}
        if (debug) {	
		cv::namedWindow("PATH", cv::WINDOW_AUTOSIZE);
		cv::imshow("PATH", path);
  	
		cv::Mat cost_normed(C);
		double min, max;
		std::cout << C << std::endl;
		cv::minMaxLoc(cost_normed, &min, &max);
		cost_normed = cost_normed/max;
		cv::normalize(cost_normed, cost_normed, 0, 1, cv::NORM_MINMAX);
		cv::namedWindow("COST", cv::WINDOW_AUTOSIZE);
		cv::imshow("COST", C);	
			
		cv::Mat dissim_normed;
		cv::normalize(dissim, dissim_normed, 0, 1, cv::NORM_MINMAX);
		cv::namedWindow("Dissim", cv::WINDOW_AUTOSIZE);
		cv::imshow("Dissim", dissim_normed);
		
		cv::waitKey(0);
	}
    }
      
    auto stop = ch::high_resolution_clock::now();
    auto duration = ch::duration_cast<ch::seconds>(stop - start);

    std::cout << std::endl << "Done in " << duration.count() << " seconds.\r" << std::flush;
    std::cout << std::endl;
}




void StereoEstimation_Naive(
    const int& window_size,
    const int& dmin,
    int height,
    int width,
    cv::Mat& image1, cv::Mat& image2, cv::Mat& disparities)
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

            disparities.at<uchar>(i - half_window_size, j - half_window_size) = std::abs(disparity);
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
