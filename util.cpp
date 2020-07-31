//
// Created by YangYuqi on 2020/7/27.
//

#include "util.h"

cv::Mat doWork(cv::InputArray _src,cv::Size ksize,int nbins)
{
    cv::Mat src = _src.getMat();
    CV_Assert( src.type() == CV_8UC1 && nbins > 0 );

    std::vector<cv::Mat> mv;
    cv::Mat dst = cv::Mat::zeros(src.size(),CV_8UC1);
    cv::Mat mask = cv::Mat::zeros(src.size(),CV_8UC1);

    int step = 256/nbins;
    for (int i = 0; i < nbins; i++)
    {
        cv::Mat temp, temp_blr;
        inRange(src,cv::Scalar(i*step),cv::Scalar(i*step+step),temp);
        mask += temp;

        blur(temp, temp_blr, ksize, cv::Point(-1,-1), cv::BORDER_DEFAULT);
        dst += mask.mul(temp_blr,1/double(255));
    }
    return dst;
}

cv::Mat get_subwindow(const cv::Mat &image, int x, int y, int size_x, int size_y)
{
    cv::Mat paded;
    //cv::namedWindow("image", 1);
    cv::copyMakeBorder(image, paded, size_x/2 + 1, size_x/2 + 1, size_y/2 + 1, size_y/2 + 1, cv::BORDER_REPLICATE);
    return paded(cv::Range(x, x + size_x), cv::Range(y, y + size_y)).clone();
}

void circshift(cv::Mat &out, const cv::Point &delta)
{
    cv::Size sz = out.size();

    assert(sz.height > 0 && sz.width > 0);
    if ((sz.height == 1 && sz.width == 1) || (delta.x == 0 && delta.y == 0))
        return;

    int x = delta.x;
    int y = delta.y;
    if (x > 0) x = x % sz.width;
    if (y > 0) y = y % sz.height;
    if (x < 0) x = x % sz.width + sz.width;
    if (y < 0) y = y % sz.height + sz.height;


    std::vector<cv::Mat> planes;
    cv::split(out, planes);

    for (size_t i = 0; i < planes.size(); i++)
    {
        // 竖直方向移动
        cv::Mat tmp0, tmp1, tmp2, tmp3;
        cv::Mat q0(planes[i], cv::Rect(0, 0, sz.width, sz.height - y));
        cv::Mat q1(planes[i], cv::Rect(0, sz.height - y, sz.width, y));
        q0.copyTo(tmp0);
        q1.copyTo(tmp1);
        tmp0.copyTo(planes[i](cv::Rect(0, y, sz.width, sz.height - y)));
        tmp1.copyTo(planes[i](cv::Rect(0, 0, sz.width, y)));

        // 水平方向移动
        cv::Mat q2(planes[i], cv::Rect(0, 0, sz.width - x, sz.height));
        cv::Mat q3(planes[i], cv::Rect(sz.width - x, 0, x, sz.height));
        q2.copyTo(tmp2);
        q3.copyTo(tmp3);
        tmp2.copyTo(planes[i](cv::Rect(x, 0, sz.width - x, sz.height)));
        tmp3.copyTo(planes[i](cv::Rect(0, 0, x, sz.height)));
    }

    merge(planes, out);
}