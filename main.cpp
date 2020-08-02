#include <iostream>
#include <fstream>
#include "LCT2tracker.h"
#include <opencv2/opencv.hpp>

using namespace std;

void printar(cv::Mat tar)
{
    for(int k = 0; k < tar.channels();k++)
    {
        for(int i = 0; i < tar.rows; i++)
        {
            for(int j = 0; j < tar.cols; j++)
            {
                float *x = (float *)(tar.data + tar.step[0]*i + tar.step[1]*j + k* sizeof(float));
                cout<<*x<<" ";
            }
            cout<<endl;
        }
        cout<<endl;
    }
}

string num_2_str(int n)
{
    string x = to_string(n);
    while(x.length() < 4)
        x = "0" + x;
    return x + ".jpg";
}

void SplitString(const string& s, vector<string>& v, const string& c)
{
    string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    while(string::npos != pos2)
    {
        v.push_back(s.substr(pos1, pos2-pos1));

        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if(pos1 != s.length())
        v.push_back(s.substr(pos1));
}

int main() {
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();

    LCT2tracker tracker;
    //int size[3] = {16, 20};
    string img_root = ".\\David\\img\\";
    ifstream fin(".\\David\\groundtruth_rect.txt");
    for(int i = 299; i <= 770; i++)
    {
        cout<<i<<endl;
        cv::Mat image;
        image = cv::imread(img_root + num_2_str(i));
        cv::Rect gt, det;
        cv::Point sit;
        string in;
        fin>>in;
        vector<string> tmp;
        SplitString(in, tmp, ",");
        gt.x = atoi(tmp[0].c_str());
        gt.y = atoi(tmp[1].c_str());
        gt.width = atoi(tmp[2].c_str());
        gt.height = atoi(tmp[3].c_str());
        //cout<<gt<<endl;
        if(i == 299)
        {
            gt.x += gt.width/2;
            gt.y += gt.height/2;
            tracker.init(gt, image);
            gt.x -= gt.width/2;
            gt.y -= gt.height/2;
            det = gt;
        }
        else
        {

            sit = tracker.train(image);
            det.x = sit.x;det.y = sit.y;
            if(tracker.resize_image)
            {
                det.width = floor(tracker.app_y*tracker.currentscalefactor)*2;
                det.height = floor(tracker.app_x*tracker.currentscalefactor)*2;
                det.x -= det.width/2;
                det.y -= det.height/2;
            }
            else {
                det.width = floor(tracker.app_y * tracker.currentscalefactor);
                det.height = floor(tracker.app_x * tracker.currentscalefactor);
                det.x -= det.width / 2;
                det.y -= det.height / 2;
            }

        }
        cv::rectangle(image, gt, cv::Scalar(255, 0, 0), 3, cv::LINE_8, 0);
        cv::rectangle(image, det, cv::Scalar(0, 255, 0), 3, cv::LINE_8, 0);
        cv::imshow("ans", image);
        cv::waitKey();
        image.release();
    }
    //cv::waitKey();
    return 0;
}