//
// Created by YangYuqi on 2020/7/27.
//

#include "detector.h"

double round(double r)
{
    return (r > 0.0) ? floor(r + 0.5) : ceil(r - 0.5);
}

detector::detector(cv::Size target_sz, cv::Size image_sz) {
    float target_max_win = 144;
    ratio = std::sqrt(target_max_win/target_sz.area());

    t_sz.width = round(target_sz.width*ratio);
    t_sz.height = round(target_sz.height*ratio);

    nbin = 32;

    this->target_sz = target_sz;
    this->image_sz = image_sz;

    thresh_p = 0.5;thresh_n = 0.1;

    det = cv::ml::SVM::create();
}

cv::Mat detector::get_feature(cv::Mat image) {
    int nth = 8;
    int ksize = 4;
    cv::Mat f_iif = 255 - doWork(image, cv::Size(ksize, ksize), nbin);
    std::vector<cv::Mat> ans;
    for(int i = 1; i <= nth; i++)
    {
        float thr = i/float(nth + 1)*255;
        ans.push_back(f_iif > thr);
    }
    for(int i = 1; i <= nth; i++)
    {
        float thr = i/float(nth + 1)*255;
        ans.push_back(image > thr);
    }
    cv::Mat out;
    cv::merge(ans, out);
    return out;
}

std::pair<cv::Mat, cv::Mat> detector::get_sample(cv::Mat image, int pos_x, int pos_y, cv::Size window_sz) {
    cv::Mat w_area = get_subwindow(image, pos_x, pos_y, floor(window_sz.height*1.2), floor(window_sz.width*1.2));
    cv::Mat feat = get_feature(w_area);

    std::vector<cv::Mat> alfeat;
    cv::Mat label((feat.rows - t_sz.height)*(feat.cols - t_sz.width),1, CV_32F);

    cv::Rect target_rect((feat.cols-t_sz.width)/2, (feat.rows - t_sz.height)/2, t_sz.width, t_sz.height);
    int truerow = feat.rows - t_sz.height;int truecol = feat.cols - t_sz.width;int truesta_i = 0;int truesta_j = 0;
    cv::resize(feat, feat, cv::Size(ceil(feat.rows*ratio), ceil(feat.cols*ratio)), 0, 0, cv::INTER_NEAREST);
    for(int i = 0; i < feat.rows - t_sz.height; i += 1)
    {
        if((i + t_sz.height -feat.rows/2)/ratio + pos_x < 0)
        {
            truesta_i++;truerow--;
            continue;
        }
        else if((i + t_sz.height -feat.rows/2)/ratio + pos_x >= image_sz.height)
        {
            truerow--;
            continue;
        }
        for(int j = 0; j < feat.cols - t_sz.width; j += 1)
        {
            if(j < truesta_j || j > truesta_j + truecol - 1)
                continue;
            if((j + t_sz.width -feat.cols/2)/ratio + pos_y < 0)
            {
                truesta_j++;truecol--;
                continue;
            }
            else if((i + t_sz.height -feat.rows/2)/ratio + pos_x >= image_sz.height)
            {
                truecol--;
                continue;
            }
            cv::Rect range(j, i, t_sz.width, t_sz.height);
            cv::Mat localfeat = feat(range).reshape(1, 1);
            alfeat.push_back(localfeat);
            label.at<float>(i*(feat.rows - t_sz.height) + j, 1) = float((range & target_rect).area())/(range.area() + target_rect.area() - (range & target_rect).area());
        }
    }
    label = label.rowRange(0, truecol*truerow);
    cv::Mat feature;
    cv::merge(alfeat, feature);
    feature = feature.reshape(1, alfeat.size());
    return std::make_pair(feature, label);
}