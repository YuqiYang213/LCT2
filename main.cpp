#include <iostream>
#include <fstream>
#include "LCT2tracker.h"
#include <opencv2/opencv.hpp>
#include<sstream>

using namespace std;

string num_2_str(int n)
{
    string x = to_string(n);
    while(x.length() < 4)
        x = "0" + x;
    return x + ".jpg";
}

int main(int argc, char **argv) {
    LCT2tracker tracker;
    //int size[3] = {16, 20};
    string img_root;
    //ifstream fin(".\\David\\groundtruth_rect.txt");
    cv::Rect gt;
    vector<cv::Rect> out;
    int start_frame, end_frame;
    stringstream ssin;
    img_root = argv[1];

    ssin<<argv[2];
    ssin>>start_frame;
    ssin.clear();
    ssin<<argv[3];
    ssin>>end_frame;
    ssin.clear();

    ssin<<argv[4];
    ssin>>gt.x;
    ssin.clear();
    ssin<<argv[5];
    ssin>>gt.y;
    ssin.clear();
    ssin<<argv[6];
    ssin>>gt.width;
    ssin.clear();
    ssin<<argv[7];
    ssin>>gt.height;
    ssin.clear();


    string name = argv[8];
    //ssin>>img_root>>start_frame>>end_frame>>gt.x>>gt.y>>gt.width>>gt.height;
    for(int i = start_frame; i <= end_frame; i++)
    {
        cv::Mat image;
        //cout<<"good"<<endl;
        //cout<<i<<endl;
        cv::Rect det;
        image = cv::imread(img_root + num_2_str(i));
        cv::Point sit;
        //cout<<gt<<endl;
        if(i == start_frame)
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

            sit = tracker.detect(image);
            det.x = sit.x;det.y = sit.y;
            if(tracker.resize_image)
            {
                det.width = floor(tracker._roi.width*tracker.currentscalefactor)*2;
                det.height = floor(tracker._roi.height*tracker.currentscalefactor)*2;
                det.x -= det.width/2;
                det.y -= det.height/2;
                //cout<<tracker.resize_image<<endl;
            }
            else {
                det.width = floor(tracker._roi.width * tracker.currentscalefactor);
                det.height = floor(tracker._roi.height * tracker.currentscalefactor);
                det.x -= det.width / 2;
                det.y -= det.height / 2;
            }
            det.width = min(det.width ,image.cols - det.x - 1);
            det.height = min(det.height, image.rows - det.y - 1);
            det.x = max(0, det.x);
            det.y = max(0, det.y);
        }
        out.push_back(det);
        //cout<<det<<endl;
        //cv::rectangle(image, gt, cv::Scalar(255, 0, 0), 3, cv::LINE_8, 0);
        //cv::rectangle(image, det, cv::Scalar(0, 255, 0), 3, cv::LINE_8, 0);
        //cv::imshow("ans", image);
        //cv::waitKey(10);
    }
    ofstream fout(name + "_ans.txt",ios::trunc|ios::out|ios::in );
    for(cv::Rect rec : out)
    {
        fout<<rec.x<<" "<<rec.y<<" "<<rec.width<<" "<<rec.height<<endl;
    }
    fout.close();
    //cv::waitKey();
    return 0;
}