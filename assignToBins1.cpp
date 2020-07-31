/**************************************************************************
 * Piotr's Image&Video Toolbox      Version 2.2
 * Copyright 2012 Piotr Dollar.  [pdollar-at-caltech.edu]
 * Please email me if you find bugs, or have suggestions or questions!
 * Licensed under the Simplified BSD License [see external/bsd.txt]
 *************************************************************************/
#include "assignToBins1.h"

using namespace cv;
/**************************************************************************
 * Return index of bin for x. The edges are determined by the (nBins+1)
 * element vector edges. Returns an integer value k in [0,nBins-1]
 * representing the bin x falls into, or k==nBins if x does not fall
 * into any bin. if edges[k] <= x < edges[k+1], then x falls
 * into bin k (k<nBins). Additionally, if x==edges[nBins], then x falls
 * into bin k=nBins-1. Eventually, all values where k==nBins should be ingored.
 * Adapted from \MATLAB6p5\toolbox\matlab\datafun\histc.c
 *************************************************************************/
int findBin( double x, double *edges, int nBins ) {
  int k = nBins; /* NOBIN */
  int k0 = 0; int k1 = nBins;
  if( x >= edges[0] && x < edges[nBins] ) {
    k = (k0+k1)/2;
    while( k0 < k1-1 ) {
      if(x >= edges[k]) k0 = k;
      else k1 = k;
      k = (k0+k1)/2;
    }
    k = k0;
  }
  /* check for special case */
  if(x == edges[nBins]) k = nBins-1;
  //std::cout<<x<<":"<<edges[nBins]<<" "<<k<<std::endl;
  return k;
}

void assignToBins( double *B, double* A, double* edges, int n, int nBins ) {
  int j; for( j=0; j<n; j++ ) B[j]=(double) findBin( A[j], edges, nBins );
}

enum ConvolutionType {
    /* Return the full convolution, including border */
            CONVOLUTION_FULL,

    /* Return only the part that corresponds to the original image */
            CONVOLUTION_SAME,
    /* Return only the submatrix containing elements that were not influenced by the border */
            CONVOLUTION_VALID
};
Mat conv2(const Mat &img, const Mat& ikernel, ConvolutionType type)
{
    Mat dest;
    Mat kernel;
    flip(ikernel,kernel,-1);
    Mat source = img;
    if(CONVOLUTION_FULL == type)
    {
        source = Mat();
        const int additionalRows = kernel.rows-1, additionalCols = kernel.cols-1;
        copyMakeBorder(img, source, (additionalRows+1)/2, additionalRows/2, (additionalCols+1)/2, additionalCols/2, BORDER_CONSTANT, Scalar(0));
    }
    Point anchor(kernel.cols - kernel.cols/2 - 1, kernel.rows - kernel.rows/2 - 1);
    int borderMode = BORDER_CONSTANT;
    filter2D(source, dest, img.depth(), kernel, anchor, 0, borderMode);

    if(CONVOLUTION_VALID == type)
    {
        dest = dest.colRange((kernel.cols-1)/2, dest.cols - kernel.cols/2).rowRange((kernel.rows-1)/2, dest.rows - kernel.rows/2);
    }
    return dest;
}

/**********************************************************************/

cv::Mat hoi(const cv::Mat &input, int nbins, int window_size)
{
    int height = input.rows;
    int width = input.cols;
    int depth = input.channels();
    double *edges = new double[nbins + 1];
    cv::minMaxIdx(input, edges, edges + nbins);
    double gap = (edges[nbins] - edges[0])/nbins;
    for(int i = 1; i < nbins; i++)
        edges[i] = edges[i - 1] + gap;;
    std::vector<cv::Mat> ans;
    for(int i = 0; i < nbins; i++)
        ans.push_back(cv::Mat::zeros(height, width, CV_32F));
    for(size_t i = 0; i < height; i++)
    {
        for(size_t j = 0; j < width; j++)
        {
            int m = findBin(input.at<float>(i, j), edges, nbins);
            ans[m].at<float>(i, j) = 1;
        }
    }
    cv::Mat kernel = cv::Mat::ones(window_size, window_size, CV_32F);
    kernel /= window_size*window_size;
    for(int i = 0; i < nbins; i++)
    {
        ans[i] = conv2(ans[i], kernel, CONVOLUTION_SAME);
    }
    cv::Mat out;
    cv::merge(ans, out);
    delete[] edges;
    return out;
}