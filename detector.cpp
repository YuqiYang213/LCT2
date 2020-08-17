//
// Created by YangYuqi on 2020/7/27.
//

#include "detector.h"

double round(double r)
{
    return (r > 0.0) ? floor(r + 0.5) : ceil(r - 0.5);
}

void detector::init(cv::Size target_sz, cv::Size image_sz) {
    float target_max_win = 144;
    ratio = std::sqrt(target_max_win/target_sz.area());

    t_sz.width = round(target_sz.width*ratio);
    t_sz.height = round(target_sz.height*ratio);

    nbin = 32;

    this->target_sz = target_sz;
    this->image_sz = image_sz;


    det = cv::ml::SVM::create();
}

cv::Mat detector::get_feature(cv::Mat image_o) {
    int nth;
    cv::Mat image = image_o.clone(), f;
    if(image.channels() == 3)
    {
        cv::cvtColor(image, image, cv::COLOR_BGR2Lab);
        nth = 4;
        std::vector<cv::Mat> Lab;
        cv::split(image, Lab);
        f = Lab[0].clone();
    }
    else {
        nth = 8;
        f = image.clone();
    }
    int ksize = 4;
    cv::Mat f_iif = 255 - doWork(f, cv::Size(ksize, ksize), nbin);
    std::vector<cv::Mat> ans, tmp;
    cv::split(image, tmp);
    for(int i = 1; i <= nth; i++)
    {
        float thr = i/float(nth + 1)*255;
        ans.push_back(f_iif >= thr);
    }
    for(int k = 0; k < tmp.size(); k++)
    {
        for(int i = 1; i <= nth; i++)
        {
            float thr = i/float(nth + 1)*255;
            ans.push_back(tmp[k] >= thr);
        }
    }
    cv::Mat out;
    cv::merge(ans, out);
    return out;
}

std::vector<cv::Mat> detector::get_sample(cv::Mat image, int pos_x, int pos_y, cv::Size window_sz) {
    cv::Mat w_area = get_subwindow(image, pos_x, pos_y, floor(window_sz.height*1.2), floor(window_sz.width*1.2));
    //cv::imshow("in", w_area);
    //cv::waitKey();
    cv::Mat feat = get_feature(w_area);
    cv::resize(feat, feat, cv::Size(ceil(feat.cols*ratio), ceil(feat.rows*ratio)), 0, 0, cv::INTER_NEAREST);

    std::vector<cv::Mat> alfeat;
    cv::Mat label(feat.rows - t_sz.height, feat.cols - t_sz.width, CV_32F, cv::Scalar::all(0));
    cv::Mat yy(feat.rows - t_sz.height, feat.cols - t_sz.width, CV_32F), xx(feat.rows - t_sz.height, feat.cols - t_sz.width, CV_32F, cv::Scalar::all(0));
    cv::Mat weights(feat.rows - t_sz.height, feat.cols - t_sz.width, CV_32F, cv::Scalar::all(0));
    for(int i = 0; i < weights.rows; i++)
        for(int j = 0; j < weights.cols; j++)
            weights.at<float>(i, j) = std::exp(-0.5*((i - weights.rows/2)*(i - weights.rows/2) + (j - weights.cols/2)*(j - weights.cols/2))/(25.0));
    float re = cv::sum(weights)[0];
    //std::cout<<weights<<std::endl;
    cv::Rect target_rect((feat.cols-t_sz.width)/2, (feat.rows - t_sz.height)/2, t_sz.width, t_sz.height);

    int truerow = feat.rows - t_sz.height;int truecol = feat.cols - t_sz.width;int truesta_i = 0;int truesta_j = 0;
    for(int i = 0; i < feat.rows - t_sz.height; i += 1)
    {
        if((i + t_sz.height/2 - feat.rows/2)/ratio + pos_x < 0)
        {
            truesta_i++;truerow--;
            continue;
        }
        else if((i + t_sz.height/2 -feat.rows/2)/ratio + pos_x >= image_sz.height)
        {
            truerow--;
            continue;
        }
        for(int j = 0; j < feat.cols - t_sz.width; j += 1)
        {
            if(j < truesta_j || j > truesta_j + truecol - 1)
                continue;
            if((j + t_sz.width/2 -feat.cols/2)/ratio + pos_y < 0)
            {
                truesta_j++;truecol--;
                continue;
            }
            else if((j + t_sz.width/2 -feat.cols/2)/ratio + pos_y >= image_sz.width)
            {
                truecol--;
                continue;
            }
            cv::Rect range(j, i, t_sz.width, t_sz.height);
            cv::Mat localfeat = feat(range).clone().reshape(1, 1);
            alfeat.push_back(localfeat);
            //std::cout<<target_rect<<std::endl<<range<<std::endl;
            label.at<float>(i, j) = ((range & target_rect).area() + 0.0f)/(range.area() + target_rect.area() - (range & target_rect).area());
            xx.at<float>(i, j) = i;
            yy.at<float>(i, j) = j;
        }
    }
    xx = (xx + t_sz.height/2 - feat.rows/2)/ratio + pos_x;
    yy = (yy + t_sz.width/2 - feat.cols/2)/ratio + pos_y;
    xx = xx(cv::Range(truesta_i, truesta_i + truerow), cv::Range(truesta_j, truesta_j + truecol)).clone().reshape(1, 1);
    yy = yy(cv::Range(truesta_i, truesta_i + truerow), cv::Range(truesta_j, truesta_j + truecol)).clone().reshape(1, 1);
    weights = weights(cv::Range(truesta_i, truesta_i + truerow), cv::Range(truesta_j, truesta_j + truecol)).clone();
    cv::Mat feature(alfeat.size(), alfeat[0].cols, CV_32F, cv::Scalar::all(0));
    for(int i = 0; i < alfeat.size(); i++)
    {
        cv::Mat tmp;
        alfeat[i].convertTo(tmp, CV_32F);
        tmp /= 255;
        alfeat[i].copyTo(feature.row(i));
    }
    label = label.reshape(1, 1);
    std::vector<cv::Mat> ans;


    ans.push_back(feature.clone());
    ans.push_back(label.clone().t());
    ans.push_back(xx.clone());
    ans.push_back(yy.clone());
    ans.push_back(weights.clone());
    return ans;
}

void detector::train(cv::Mat image, int pos_x, int pos_y, cv::Size window_sz, bool online) {
    //std::cout<<"de"<<std::endl;
    std::vector<cv::Mat> samples = get_sample(image, pos_x, pos_y, window_sz);
    //std::cout<<samples[1]<<std::endl;
    float posi = 0.5,  nega = 0.1;
    std::vector<cv::Mat> features;
    std::vector<int> labels;
    int k = 0;
    for(int i = 0; i < samples[0].rows; i++)
    {
        if(samples[1].at<float>(0, i) > posi)
        {
            labels.push_back(1);
            features.push_back(samples[0].row(i).clone());
            k++;
        }
        else if(samples[1].at<float>(0, i) < nega)
        {
            labels.push_back(-1);
            features.push_back(samples[0].row(i).clone());
        }
    }
    cv::Mat feat(features.size(), features[0].cols, CV_32F, cv::Scalar::all(0)), labe(features.size(), 1, CV_32F);
    for(int i = 0; i < features.size(); i++)
    {
        features[i].copyTo(feat.row(i));
        labe.at<float>(i, 0) = labels[i];
    }
    if(!online)
    {
        det->setType(cv::ml::SVM::C_SVC);
        det->setKernel(cv::ml::SVM::LINEAR);
        //det->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 10, 1e-6));
        cv::Ptr<cv::ml::TrainData> tData = cv::ml::TrainData::create(feat,cv::ml::ROW_SAMPLE, labels);
        det->train(tData);
        cv::Mat svidx;
        b = det->getDecisionFunction(0, w, svidx);
        cv::Mat sv = det->getSupportVectors();
        w.convertTo(w, CV_32F);
        w = -w*sv;
        w = w.t();
        //std::cout<<b<<std::endl;
        //std::cout<<feat.rows - cv::sum((((feat*w + b) > 0) & (labe > 0)) | (((feat*w + b) < 0) & (labe < 0)) )[0]/255<<std::endl;
    }
    else{
        for(int i = 0; i < feat.rows; i++)
        {
            cv::Mat loss = 1 - labels[i]*(feat.row(i) * w + b);

            if(loss.at<float>(0, 0) > 0)
            {
                //std::cout<<loss.at<float>(0, 0)<<std::endl;
                cv::Mat cur = feat.row(i).clone();
                cv::Mat update = cur.t()*labels[i];
                update *= loss.at<float>(0, 0)/(cv::sum(cur.mul(cur))[0] + 1.f/2);
                w += update;
            }
        }
        //std::cout<<cv::sum((((feat*w + b) < 0) & (labe > 0)) )[0]/255<<std::endl;
    }
}