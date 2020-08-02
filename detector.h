//
// Created by YangYuqi on 2020/7/27.
//

#ifndef LCT2_DETECTOR_H
#define LCT2_DETECTOR_H

#endif //LCT2_DETECTOR_H

#include <opencv2/opencv.hpp>
#include "util.h"

class detector
{
public:

    float ratio;
    cv::Size t_sz;
    int nbin;
    cv::Size target_sz, image_sz;
    cv::Ptr<cv::ml::SVM> det;
    cv::Mat w;
    double b;

    //intialize the detector
    void init(cv::Size target_sz ,cv::Size image_sz);

    //get the needed featuure
    cv::Mat get_feature(cv::Mat image);

    //get the needed feature and label
    std::vector<cv::Mat> get_sample(cv::Mat image, int pos_x, int pos_y, cv::Size window_sz);

    //train the detector
    void train(cv::Mat image, int pos_x, int pos_y, cv::Size window_sz, bool online);
};