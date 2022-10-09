#include <boost/program_options/options_description.hpp>
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
//#include <jsoncpp/json/json.h>
#include "progressbar.hpp"

namespace po = boost::program_options;
namespace fs = boost::filesystem;
namespace ch = std::chrono;

const std::string version = "1.0";

int main(int argc, char** argv) {
    
    ////////////////
    // Parameters //
    ////////////////

    // Default parameters values     
    const double default_baseline = 160;
    const double default_focal_length = 3740;
    const std::string default_method = "DP";
    const double default_fulfill_occlusions = false;
    const bool default_debug = false;
    const int default_dmin = 200;
    const int default_window_size = 1;
    const double default_lambda = 2;

    fs::path default_config_path ("params.cfg");
    fs::path default_output_file ("output");
    default_output_file /= "output";
    fs::path default_image_file ("data");

    // Parameters variables
    double focal_length;
    double baseline;
    bool fulfill_occlusions;
    bool debug;
    std::string method;
    double lambda;
    int dmin;
    int nProcessors;
    int window_size;
    // parameters that will be set by the program 
    int height;
    int width;

    std::string output_file;
    std::string config_path;
    std::string image1_name;
    std::string image2_name;

    // Parameters description
    po::options_description command_line_options("cli options");
    command_line_options.add_options()
    ("help,h", "Produce help message")
    ("version,v", "Get program version")
    ("jobs,j", po::value<int>(& nProcessors)->default_value(omp_get_max_threads()), "Number of Threads (max by default)")
    ("config,c", po::value<std::string>(& config_path)->default_value(default_config_path.string()), "Path to the congiguration file");

    po::options_description common_options("Options that can be set both in cli and in config");
    common_options.add_options()
    ("window-size,w", po::value<int>(& window_size)->default_value(default_window_size), "Window Size (taken from config if not specified)")
    ("fulfill-occlusions,f", po::value<bool>(& fulfill_occlusions)->default_value(default_fulfill_occlusions), "To fulfill occlusions or not (for DP only)")
    ("lambda-DP,l", po::value<double>(& lambda)->default_value(default_lambda), "Lambda value (for DP only)")
    ("debug,d", po::value<bool>(& debug)->default_value(default_debug), "Dispay every step of dynamic programming")
    ("method,m", po::value<std::string>(& method)->default_value(default_method), "Method: 'naive' or 'DP'");
  
    po::options_description hidden_common_options;
    hidden_common_options.add_options()
    ("left-image", po::value<std::string>(& image1_name)->default_value((default_image_file/"view0.png").string()), "Image1 name")
    ("right-image", po::value<std::string>(& image2_name)->default_value((default_image_file/"view1.png").string()), "Image2 name")
    ("output,o", po::value<std::string>(& output_file)->default_value(default_output_file.string()), "Output files template (do not add extentions)");
         
    po::options_description config_only_options("Configuration options");
    config_only_options.add_options()
    ("dmin", po::value<int>(& dmin)->default_value(default_dmin), "Dmin value due to the image cropping")
    ("focal-length", po::value<double>(& focal_length)->default_value(default_focal_length), "Focal length")
    ("baseline", po::value<double>(& baseline)->default_value(default_baseline), "Baseline");
 
    po::positional_options_description p;
    p.add("left-image", 1);
    p.add("right-image", 1);
    p.add("output", 1);

    // Read parameters from command line (higher priority)
    
    po::variables_map vm;
    po::options_description cmd_opts;
    cmd_opts.add(command_line_options).add(hidden_common_options).add(common_options);
    po::store(po::command_line_parser(argc, argv).
          options(cmd_opts).positional(p).run(), vm);
    po::notify(vm);
    
    po::store(po::command_line_parser(0, 0).options(config_only_options).run(), vm);
    notify(vm);
    
    // Read parameters from config (lower priority)

    bool ALLOW_UNREGISTERED = true;

    po::options_description config_opts;
    config_opts.add(config_only_options).add(common_options).add(hidden_common_options);

    std::ifstream cfg_file(config_path.c_str());
    if (cfg_file)
    {
       po::store(po::parse_config_file(cfg_file, config_opts, ALLOW_UNREGISTERED), vm);
       po::notify(vm);
    } 
    
    // Read parameters from command line again to ensure higher priority

    po::store(po::command_line_parser(argc, argv).
          options(cmd_opts).positional(p).run(), vm);
    po::notify(vm); 
      
    // In case user asked for information    

    if (vm.count("help")) {
	std::cout << "Usage: OpenCV_stereo [<left-image> [<right-image> [<output>]]] [<options>]\n";
	po::options_description help_opts;
	help_opts.add(command_line_options).add(common_options).add(config_only_options);
	std::cout << help_opts << "\n";
	return 1;
    }

    if (vm.count("version")) {
	std::cout << "Stereo Estimation " << version << std::endl;
	return 1;
    }

    // check validness of images
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
    
    
    // Set parameters that are automatic
    height = image1.size().height;
    width = image1.size().width;
    
    // Print parameters values 
    std::cout << "----------- Algorithm Parameters ----------------" << std::endl;
    std::cout << "window_size = " << window_size << std::endl;  
    std::cout << "method = " << method << std::endl;
    if (method == "DP"){ 
    	std::cout << "lambda = " << (int)(lambda/window_size/window_size) << std::endl;
    	std::cout << "lambda*window_size^2 = " << lambda << std::endl;
	std::cout << "debug = " << debug << std::endl; 
	std::cout << "fulfill_occlusions = " << fulfill_occlusions << std::endl; 
    }

    std::cout << "------------- Data Parameters -------------------" << std::endl;
    std::cout << "focal_length = " << focal_length << std::endl;
    std::cout << "baseline = " << baseline << std::endl;
    std::cout << "disparity added due to image cropping = " << dmin << std::endl;
    
    std::cout << "-------------  IO Parameters  -------------------" << std::endl;
    std::cout << "output filename template = " << output_file << std::endl;
    std::cout << "left-image filename = " << image1_name << std::endl;
    std::cout << "right-image filename = " << image2_name << std::endl;
    
    std::cout << "----------- Inferred Variables ------------------" << std::endl;
    std::cout << "height " << height << std::endl;
    std::cout << "width " << width << std::endl;
    
    std::cout << "------------- System Parameters -----------------" << std::endl; 
    std::cout << "nProcessors = " << nProcessors << std::endl;

    std::cout << "-------------------------------------------------" << std::endl;
 
    // setup njobs
    omp_set_num_threads(nProcessors);

    ////////////////////
    // Reconstruction //
    ////////////////////

    // Disparity image    
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
            disparities, lambda, fulfill_occlusions, debug);
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
    cv::Mat& image1, cv::Mat& image2, cv::Mat& disparities, double lambda, bool fulfill_occlusions, bool debug)
{
    int half_window_size = window_size / 2;
    int dwidth = width - 2*half_window_size; // width of matchable image region
    
    //measure execution time
    auto start = ch::high_resolution_clock::now();
    
    //init progressbar
    progressbar bar(height);
    
    // for each row (scanline)
    std::cout << "Calculating disparities for the DP approach... " << std::endl;
    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
	//update progressbar
	#pragma omp critical
		bar.update();
	// calculate dissimilarity matrix
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
	// initialize C, M
	C(cv::Range(0, 1), cv::Range(0, dwidth)) = dissim(cv::Range(0, 1), cv::Range(0, dwidth));
        C(cv::Range(0, dwidth), cv::Range(0, 1)) = dissim(cv::Range(0, dwidth), cv::Range(0, 1));
	
	// Fulfill C, M
	for (int i = 1; i < dwidth; i++) {
	    for (int j = 1; j < dwidth; j++) {
            	// match
		M.at<uint8_t>(i, j) = 0;
		double value = C.at<float>(i-1, j-1) + dissim.at<float>(i, j);
                // right occlusion
		if (value > (C.at<float>(i-1, j) + lambda)) {
		    value = C.at<float>(i-1, j) + lambda;
		    M.at<uint8_t>(i, j) = 1;
		}
		// left occlusion
		if (value > (C.at<float>(i, j-1) + lambda)) {	
		    value = C.at<float>(i, j-1) + lambda;
		    M.at<uint8_t>(i, j) = 2;
		}
		// set the value
		C.at<float>(i, j) = value;
	    }
 	}
	
	// trace back
	int j = dwidth - 1;
        int i = dwidth - 1;
	int last_defined = 0; // in case you want to Fulfill right occlusions
	cv::Mat path = cv::Mat::zeros(dwidth, dwidth, CV_8UC1);
	while ((j >= 0) && (i>=0)){
	    path.at<uchar>(i, j) = 255;
	    switch (M.at<uchar>(i, j)) {
		case 0:
		    disparities.at<uchar>(y, i + half_window_size) = std::abs(j - i);
		    if (fulfill_occlusions){
		        last_defined = std::abs(j-i);
		    }
		    i--; j--;
		    break;
 		case 1:
		    disparities.at<uchar>(y, i + half_window_size) = last_defined;
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
