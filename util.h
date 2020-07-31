//
// Created by YangYuqi on 2020/7/27.
//

#ifndef LCT2_UTIL_H
#define LCT2_UTIL_H

#endif //LCT2_UTIL_H

#include <opencv2/opencv.hpp>
/***************************************/
/**
 * @file calcIIF.cpp
 * @mex interface for IIF computation routine
 * @author Jianming Zhang
 * @date 2013
 */

cv::Mat doWork(cv::InputArray _src,cv::Size ksize,int nbins);
cv::Mat get_subwindow(const cv::Mat &image, int x, int y, int size_x, int size_y);//x for rows, y for cols
void circshift(cv::Mat &out, const cv::Point &delta);
