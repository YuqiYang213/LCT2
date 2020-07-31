//
// Created by YangYuqi on 2020/7/24.
//
#include <opencv2/opencv.hpp>
#ifndef LCT2_ASSIGNTOBINS1_H
#define LCT2_ASSIGNTOBINS1_H

#endif //LCT2_ASSIGNTOBINS1_H

int findBin( double x, double *edges, int nBins );
void assignToBins( double *B, double* A, double* edges, int n, int nBins );
cv::Mat hoi(const cv::Mat &input, int nbins, int window_size);