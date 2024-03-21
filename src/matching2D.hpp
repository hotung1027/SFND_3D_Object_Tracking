#ifndef matching2D_hpp
#define matching2D_hpp

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <sstream>
#include <stdio.h>
#include <vector>

#include "dataStructures.h"

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img,
                        bool bVis = false);
void detKeypointsShiTomasi(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img,
                           bool bVis = false);
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img,
                        std::string detectorType, bool bVis = false);
void descKeypoints(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img,
                   cv::Mat &descriptors, std::string descriptorType);
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource,
                      std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource,
                      cv::Mat &descRef, std::vector<cv::DMatch> &matches,
                      std::string descriptorType, std::string matcherType,
                      std::string selectorType);

// clang-format off
enum class DetectorType : int {
  HARRIS = 0,
  SHITOMASI,
  BRISK,
  BRIEF,
  FAST,
  ORB,
  SIFT,
  AKAZE
};
enum class DescriptorType : int {
  BRISK,
  BRIEF,
  FREAK,
  ORB,
  SIFT,
  AKAZE
};
const std::map<std::string, DescriptorType> DescriptorTypeDict{
  { "BRISK",     DescriptorType::BRISK     },
  { "BRIEF",     DescriptorType::BRIEF     },
  { "FREAK",     DescriptorType::FREAK     },
  { "ORB",       DescriptorType::ORB       },
  { "SIFT",      DescriptorType::SIFT      },
  { "AKAZE",     DescriptorType::AKAZE     }
};

const std::map<std::string, DetectorType> DetectorTypeDict{
  { "HARRIS",    DetectorType::HARRIS    },
  { "SHITOMASI", DetectorType::SHITOMASI },
  { "BRISK",     DetectorType::BRISK     },
  { "BRIEF",     DetectorType::BRIEF     },
  { "FAST",      DetectorType::FAST      },
  { "ORB",       DetectorType::ORB       },
  { "SIFT",      DetectorType::SIFT      },
  { "AKAZE",     DetectorType::AKAZE     }
};
// clang-format on

#endif /* matching2D_hpp */
