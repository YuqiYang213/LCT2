# LCT2 c++ implementation

## LCT2 introduction
[[Paper Link](https://arxiv.org/abs/1707.02309v1)]

Object tracking is challenging as target objects often undergo drastic appearance changes over time. Recently, adaptive correlation filters have been successfully applied to object tracking. However, tracking algorithms relying on highly adaptive correlation filters are prone to drift due to noisy updates. Moreover, as these algorithms do not maintain long-term memory of target appearance, they cannot recover from tracking failures caused by heavy occlusion or target disappearance in the camera view. In this paper, we propose to learn multiple adaptive correlation filters with both long-term and short-term memory of target appearance for robust object tracking. First, we learn a kernelized correlation filter with an aggressive learning rate for locating target objects precisely. We take into account the appropriate size of surrounding context and the feature representations. Second, we learn a correlation filter over a feature pyramid centered at the estimated target position for predicting scale changes. Third, we learn a complementary correlation filter with a conservative learning rate to maintain long-term memory of target appearance. We use the output responses of this long-term filter to determine if tracking failure occurs. In the case of tracking failures, we apply an incrementally learned detector to recover the target position in a sliding window fashion.Extensive experimental results on large-scale benchmark datasets demonstrate that the proposed algorithm performs favorably against the state-of-the-art methods in terms of efficiency, accuracy, and robustness.

## Prerequisites:

Windows10

## Installation:

1. Install mingw(x86_64-8.1.0-release-posix-seh-rt_v6-rev0) from https://sourceforge.net/projects/mingw-w64/ , and add its /bin to the environment variable PATH.

2. Download the opencv 4.1.1 source code, build it with the mingw above. You can use cmake for convinience. Or you can download the built opencv from https://github.com/huihut/OpenCV-MinGW-Build . Add its /bin to the environment variable PATH.

## Demo:

We use the dataset David from OTB as the demo to test our program. To run it, you have to first download the David Datasets from http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html , and put it in the code folder. Then , you can just simply build and run the main.cpp without inputing anything and then you can see a sequence of pictures with two rectangles on it that the green one indicates our methods' result and the blue one indicates the groundtruth.