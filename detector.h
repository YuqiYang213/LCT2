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
    float thresh_p, thresh_n;
    cv::Ptr<cv::ml::SVM> det;

    //constructor
    detector(cv::Size target_sz ,cv::Size image_sz);

    //get the needed featuure
    cv::Mat get_feature(cv::Mat image);

    //get the needed feature and label
    std::pair<cv::Mat, cv::Mat> get_sample(cv::Mat image, int pos_x, int pos_y, cv::Size window_sz);
};