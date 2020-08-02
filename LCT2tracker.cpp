//
// Created by YangYuqi on 2020/7/22.
//

#include "LCT2tracker.h"

/***************ComplexComputation**************/
cv::Mat real(const cv::Mat &img)
{
    std::vector<cv::Mat> planes, out;
    cv::Mat ans;
    cv::split(img, planes);
    for(int i = 0; i < planes.size(); i+= 2)
        out.push_back(planes[i]);
    cv::merge(out, ans);
    return ans;
}
cv::Mat conj(const cv::Mat &img)
{
    std::vector<cv::Mat> deal;
    cv::split(img, deal);
    deal[1] = -deal[1];
    cv::Mat ans;
    cv::merge(deal, ans);
    return ans;
}
cv::Mat complexMultiplication(const cv::Mat& a, const cv::Mat &b, bool conj = false)
{
    std::vector<cv::Mat> pa;
    std::vector<cv::Mat> pb;
    cv::split(a, pa);
    cv::split(b, pb);

    if (conj)
        pb[1] *= -1.0;

    std::vector<cv::Mat> pres;
    pres.push_back(pa[0].mul(pb[0]) - pa[1].mul(pb[1]));
    pres.push_back(pa[0].mul(pb[1]) + pa[1].mul(pb[0]));

    cv::Mat ans;
    cv::merge(pres, ans);

    return ans;
}

cv::Mat complexDivisionReal(const cv::Mat &a, const cv::Mat& b)
{
    std::vector<cv::Mat> pa;
    cv::split(a, pa);

    std::vector<cv::Mat> pres;

    cv::Mat divisor = 1. / b;

    pres.push_back(pa[0].mul(divisor));
    pres.push_back(pa[1].mul(divisor));

    cv::Mat ans;
    cv::merge(pres, ans);
    return ans;
}

cv::Mat complexDivision(const cv::Mat& a, const cv::Mat &b)
{
    std::vector<cv::Mat> pa;
    std::vector<cv::Mat> pb;
    cv::split(a, pa);
    cv::split(b, pb);

    cv::Mat divisor = 1. / (pb[0].mul(pb[0]) + pb[1].mul(pb[1]));

    std::vector<cv::Mat> pres;

    pres.push_back((pa[0].mul(pb[0]) + pa[1].mul(pb[1])).mul(divisor));
    pres.push_back((pa[1].mul(pb[0]) + pa[0].mul(pb[1])).mul(divisor));

    cv::Mat ans;
    cv::merge(pres, ans);
    return ans;
}


/********************************************************/

cv::Mat multichafft(const cv::Mat &input, bool inverse)
{
    std::vector<cv::Mat> inputvec;
    cv::split(input, inputvec);
    if(inverse)
    {
        std::vector<cv::Mat>ansvec;
        if(input.channels() % 2)
        {
            for(int i = 0; i < inputvec.size(); i++)
            {
                std::vector<cv::Mat> tmp;
                tmp.push_back(inputvec[i].clone());
                tmp.push_back(cv::Mat(inputvec[i].size(), CV_32F, cv::Scalar::all(0)));
                cv::Mat in(inputvec[i].rows, inputvec[i].cols, CV_32FC2);
                cv::merge(tmp, in);
                cv::idft(in, in, cv::DFT_SCALE);
                tmp.clear();
                cv::split(in, tmp);
                ansvec.push_back(tmp[0].clone());
                ansvec.push_back(tmp[1].clone());
            }
            cv::Mat ans;
            cv::merge(ansvec, ans);
            return ans;
        }
        for(int i = 0; i < inputvec.size(); i+=2)
        {
            std::vector<cv::Mat> tmp;
            tmp.push_back(inputvec[i].clone());tmp.push_back(inputvec[i + 1].clone());
            cv::Mat in(inputvec[i].rows, inputvec[i].cols, CV_32FC2);
            cv::merge(tmp, in);
            cv::idft(in, in, cv::DFT_SCALE | cv::DFT_COMPLEX_OUTPUT);
            tmp.clear();
            cv::split(in, tmp);
            ansvec.push_back(tmp[0].clone());
        }
        cv::Mat ans;
        cv::merge(ansvec, ans);
        return ans;
    }
    else
    {
        for(int i = 0; i < inputvec.size(); i++)
        {
            cv::Mat planes[] = {cv::Mat_<float>(inputvec[i]), cv::Mat_<float>::zeros(inputvec[i].size())};
            //cv::Mat planes[] = {cv::Mat_<double> (img), cv::Mat_<double>::zeros(img.size())};
            cv::merge(planes, 2, inputvec[i]);
            cv::dft(inputvec[i], inputvec[i],  cv::DFT_COMPLEX_OUTPUT);
        }
        cv::Mat ans;
        cv::merge(inputvec, ans);
        return ans;
    }
}

LCT2tracker::LCT2tracker()
{
    padding = 1.8; //area surrounding the target
    lambda = 1e-4; //regularization
    output_sigma_factor = 0.1; //spatial bandwidth
    interp_factor = 0.01;
    kernal_sigma = 1;

    //features
    hog_orientations = 9;
    cell_size = 4; //hog grid cell
    window_size = 6; //hoi local region
    nbins = 8; //bins of HOI

    //threshold
    motion_thresh = 0.15;
    appearance_thresh = 0.38;

    nScale = 33;
    float scale_sigma_factor = 0.25;
    scale_sigma = nScale/std::sqrt(33)*scale_sigma_factor;
    scale_step = 1.02;
    for(int i = 0; i < nScale; i++)
    {
        scale_factor[i] = std::pow(scale_step, int(ceil(nScale/2.0) - i - 1));
    }
    currentscalefactor = 1;
};

void LCT2tracker::search_window(int target_x, int target_y, int image_x, int image_y) {
    if(target_x/target_y >= 2)
    {
        window_x = int(target_x*1.4);
        window_y = int(target_y*(1 + padding));
    }
    else if(target_x > 80 && target_y > 80 && image_x*image_y/(target_x*target_y) < 10)
    {
        window_x = target_x*2;window_y = target_y*2;
    }
    else
    {
        window_x = int(target_x*(1 + padding));window_y = int(target_y*(1 + padding));
    }
    app_x = target_x + 2*cell_size;
    app_y = target_y + 2*cell_size;
}

cv::Mat LCT2tracker::create_gaussian_label(float sigma, int col, int row) {
    cv::Mat ans(row, col, CV_32F);
    for(int i = 0; i < row; i++)
        for(int j = 0; j < col; j++)
        {
            ans.at<float>(i, j) = std::exp(-0.5/(sigma*sigma)*((i - row/2)*(i - row/2) + (j - col/2)*(j - col/2)));
        }
    circshift(ans, cv::Point(-col/2, -row/2));
    return ans;
}


std::pair<cv::Point, float> LCT2tracker::do_correlation(cv::Mat image, int pos_x, int pos_y, cv::Size window_sz, bool window, bool app)
{
    cv::Mat patch = get_subwindow(image, pos_x, pos_y, window_sz.height, window_sz.width);
    cv::Mat z = get_feature(patch, window);
    cv::Mat zf = multichafft(z, false);
    cv::Mat res;
    if(app)
    {
        cv::Mat kzf = gaussian_correlation(zf, app_xf, kernal_sigma);
        res = multichafft(complexMultiplication(app_alphaf, kzf), true);
    }
    else{
        cv::Mat kzf = gaussian_correlation(zf, win_xf, kernal_sigma);
        res = multichafft(complexMultiplication(_alphaf, kzf), true);
    }
    cv::Mat response = real(res);
    circshift(response, cv::Point(floor(response.cols/2.0), floor(response.rows/2.0)));

    double minVal, maxVal;
    int    minIdx[2] = {}, maxIdx[2] = {};	// minnimum Index, maximum Index
    //std::cout<<response<<std::endl;
    cv::minMaxIdx(response, &minVal, &maxVal, minIdx, maxIdx);

    maxIdx[0] = (maxIdx[0] - floor(zf.rows/2.0))*cell_size;
    maxIdx[1] = (maxIdx[1] - floor(zf.cols/2.0))*cell_size;
    //std::cout<<maxVal<<" "<<maxIdx[0]<<" "<<maxIdx[1]<<std::endl;

    return std::make_pair<cv::Point, float>(cv::Point(pos_y + maxIdx[1], pos_x + maxIdx[0]), maxVal);
}

std::pair<cv::Point, float> LCT2tracker::refine_pos(cv::Mat image, int pos_x, int pos_y, bool app) {
    std::vector<cv::Mat> samples = det.get_sample(image, pos_x, pos_y, cv::Size(window_y, window_x));
    cv::Mat scores = samples[0] * det.w + det.b;
    //samples[4] = samples[4].t();
    scores = scores.mul(samples[4].reshape(1, scores.rows));
    //std::cout<<scores<<std::endl;

    double minVal, maxVal;
    int    minIdx[2] = {}, maxIdx[2] = {};
    cv::minMaxIdx(scores, &minVal, &maxVal, minIdx, maxIdx);
    cv::Point tpos(samples[3].at<float>(0, maxIdx[0]), samples[2].at<float>(0, maxIdx[0]));
    //std::cout<<cv::Point(pos_y, pos_x)<<tpos<<" "<<maxVal<<std::endl;

    float max_reponse = do_correlation(image, tpos.y, tpos.x, cv::Size(app_y, app_x), false, app).second;
    if(max_reponse < 1.5 * m_response || max_reponse < 0)
    {
        tpos = cv::Point(pos_y, pos_x);
        max_reponse = m_response;
    }
    return std::make_pair(tpos, max_reponse);
}

cv::Mat LCT2tracker::gaussian_correlation(const cv::Mat &xf, const cv::Mat &yf, float sigma) {
    cv::Mat c(xf.rows, xf.cols, CV_32F, cv::Scalar(0));
    std::vector<cv::Mat> xfvec, yfvec;
    cv::split(xf, xfvec);
    cv::split(yf, yfvec);
    cv::Mat caux;
    cv::Scalar xx = 0, yy = 0;
    std::vector<cv::Mat> xtmp, ytmp, ctmp;
    for(int i = 0; i < xfvec.size(); i+=2)
    {
        xtmp.push_back(xfvec[i].clone());xtmp.push_back(xfvec[i + 1].clone());
        ytmp.push_back(yfvec[i].clone());ytmp.push_back(yfvec[i + 1].clone());
        cv::Mat xft, yft;
        cv::merge(xtmp, xft);cv::merge(ytmp, yft);
        xx += cv::sum(xft.mul(xft));
        yy += cv::sum(yft.mul(yft));
        xtmp.clear();ytmp.clear();
        cv::mulSpectrums(xft, yft, caux, 0, true);
        caux = multichafft(caux, true);
        cv::split(real(caux), ctmp);
        for(int j = 0; j < ctmp.size(); j++)
            c += ctmp[j];
        ctmp.clear();
    }
    cv::Mat d;
    cv::max(((xx[0] + xx[1])/(xf.cols*xf.rows)  + (yy[0] + yy[1])/(xf.cols*xf.rows) - 2 * c)/float(xf.cols*xf.rows*xf.channels()/2), 0, d);
    cv::Mat k;
    cv::exp((-d / (sigma*sigma)), k);
    return multichafft(k, false);
}

void LCT2tracker::init(const cv::Rect &roi, cv::Mat Image)
{
    resize_image = (roi.width * roi.height > 10000);
    cv::Mat im_gray;
    if(Image.channels() == 3)
         cv::cvtColor(Image, im_gray, cv::COLOR_RGB2GRAY);
    else
        im_gray = Image.clone();
    cv::Rect real_roi = roi;
    if(resize_image)
    {
        cv::resize(Image, Image, cv::Size(Image.cols/2, Image.rows/2));
        cv::resize(im_gray, im_gray, cv::Size(Image.cols/2, Image.rows/2));
        real_roi.height /= 2;
        real_roi.width /= 2;
        real_roi.x/=2;
        real_roi.y /=2;
    }
    //std::cout<<im_gray.channels()<<std::endl;
    search_window(real_roi.height, real_roi.width, im_gray.rows, im_gray.cols);
    det.init(real_roi.size(), im_gray.size());

    int scale_model_max_area = 512;
    //std::cout<<app_x*app_y<<std::endl;
    if(app_x*app_y > scale_model_max_area)
        scale_model_sz = cv::Size(floor(app_y * std::sqrt(scale_model_max_area/float(app_x*app_y))), floor(app_x * std::sqrt(scale_model_max_area/float(app_x*app_y))));
    else scale_model_sz = cv::Size(app_y, app_x);
    _roi = real_roi;
    output_sigma = std::sqrt(_roi.area())*output_sigma_factor/cell_size;

    min_scalefactor = std::pow(scale_step, std::ceil(std::log(std::max(5/window_y, 5/window_x)/std::log(scale_step))));
    max_scalefactor = std::pow(scale_step, std::floor(std::log(std::min(float(im_gray.cols)/_roi.width, float(im_gray.rows)/_roi.height)/std::log(scale_step))));

    cv::Mat patch_win = get_subwindow(im_gray, _roi.y, _roi.x, window_x, window_y);
    win_xf = multichafft(get_feature(patch_win, true), false);
    tar = multichafft(create_gaussian_label(output_sigma,size_patch[1], size_patch[0]), false);

    cv::Mat domini = gaussian_correlation(win_xf, win_xf, kernal_sigma);
    std::vector<cv::Mat> tmp;
    cv::split(domini, tmp);
    tmp[0] += lambda;
    cv::merge(tmp, domini);
    _alphaf = complexDivision(tar, domini);


    cv::Mat patch_app = get_subwindow(im_gray,_roi.y, _roi.x, app_x, app_y);
    app_xf = multichafft(get_feature(patch_app, false), false);
    app_tar = multichafft(create_gaussian_label(output_sigma,size_patch[1], size_patch[0]), false);

    domini = gaussian_correlation(app_xf, app_xf, kernal_sigma);
    tmp.clear();
    cv::split(domini, tmp);
    tmp[0] += lambda;
    cv::merge(tmp, domini);
    app_alphaf = complexDivision(app_tar, domini);

    cv::Mat xs = get_scale_sample(im_gray, cv::Rect(_roi.x, _roi.y, app_y, app_x), scale_factor, scale_model_sz, true);
    //std::cout<<xs<<std::endl;
    cv::Mat xsf(xs.size(), CV_32FC2);
    cv::dft(xs, xsf, cv::DFT_COMPLEX_OUTPUT | cv::DFT_ROWS);

    scale_tar = cv::Mat(xsf.rows, nScale, CV_32FC2);
    cv::Mat rowgau(1, nScale, CV_32F, cv::Scalar::all(0));
    for(int i = 0; i < nScale; i++)
        rowgau.at<float>(0, i) = std::exp(-0.5*((i - floor(nScale/2.0))*(i - floor(nScale/2.0))/(scale_sigma*scale_sigma)));
    cv::Mat rowgauf(1, nScale, CV_32FC2);
    cv::dft(rowgau, rowgauf, cv::DFT_COMPLEX_OUTPUT | cv::DFT_ROWS);
    //std::cout<<rowgau<<std::endl;
    for(int i = 0; i < scale_tar.rows; i++)
        rowgauf.copyTo(scale_tar.row(i));
    cv::mulSpectrums(scale_tar, xsf, sf_num, 0, true);



    cv::Mat sf_den_ori;
    cv::mulSpectrums(xsf, xsf, sf_den_ori, 0, true);
    tmp.clear();
    std::vector<cv::Mat>  ans;
    cv::split(sf_den_ori, tmp);
    cv::Mat sf_den_sp;
    for(int i = 0; i < tmp.size(); i++)
    {
        cv::reduce(tmp[i], sf_den_sp, 0, cv::REDUCE_SUM);
        ans.push_back(sf_den_sp.clone());
    }
    cv::merge(ans, sf_den);

    det.train(im_gray, _roi.y, _roi.x, cv::Size(window_y, window_x), false);
}

cv::Point LCT2tracker::train(cv::Mat Image)
{
    cv::Mat im_gray;
    if(Image.channels() == 3)
        cv::cvtColor(Image, im_gray, cv::COLOR_RGB2GRAY);
    else
        im_gray = Image.clone();
    if(resize_image)
    {
        cv::resize(Image, Image, cv::Size(Image.cols/2, Image.rows/2));
        cv::resize(im_gray, im_gray, cv::Size(Image.cols/2, Image.rows/2));
    }
    std::pair<cv::Point, float> pos = do_correlation(im_gray, _roi.y, _roi.x, cv::Size(window_y, window_x), true, false);
    std::pair<cv::Point, float> max_response = do_correlation(im_gray, pos.first.y, pos.first.x, cv::Size(app_y, app_x), false, true);
    this->m_response = std::max(this->m_response, max_response.second);

    //if(max_response.second < motion_thresh)
    //{
        pos = max_response = refine_pos(im_gray, pos.first.y, pos.first.x, true);
    //}

    float c_scalefactor[40];
    for(int i = 0; i < nScale; i++)
        c_scalefactor[i] = scale_factor[i]*currentscalefactor;
    cv::Mat xs = get_scale_sample(im_gray, cv::Rect(pos.first.x, pos.first.y, app_y, app_x), c_scalefactor, scale_model_sz, true);
    //std::cout<<xs<<std::endl;
    cv::Mat xsf;
    cv::dft(xs, xsf, cv::DFT_COMPLEX_OUTPUT | cv::DFT_ROWS);
    cv::Mat sc_response = complexMultiplication(sf_num, xsf);


    std::vector<cv::Mat> tmp, out;
    cv::split(sc_response, tmp);
    cv::Mat sum_sc_response_sp, sum_sc_response;
    for(int i = 0; i < tmp.size(); i++)
    {
        cv::reduce(tmp[i], sum_sc_response_sp, 0, cv::REDUCE_SUM);
        out.push_back(sum_sc_response_sp.clone());
    }
    cv::merge(out, sum_sc_response);


    tmp.clear();
    cv::split(sf_den, tmp);
    //std::cout<<sf_den<<std::endl;
    tmp[0] += lambda;
    cv::Mat denomi;
    cv::merge(tmp, denomi);
    sum_sc_response = complexDivision(sum_sc_response, denomi);

    cv::Mat i_sum_sc_response;
    cv::idft(sum_sc_response, i_sum_sc_response, cv::DFT_ROWS | cv::DFT_COMPLEX_OUTPUT);
    cv::Mat scale_response = real(i_sum_sc_response);
    //std::cout<<scale_response<<std::endl;

    double minVal, maxVal;
    int    minIdx[2] = {}, maxIdx[2] = {};	// minnimum Index, maximum Index
    cv::minMaxIdx(scale_response, &minVal, &maxVal, minIdx, maxIdx);
    currentscalefactor = c_scalefactor[maxIdx[1]];
    if(currentscalefactor < min_scalefactor)
        currentscalefactor = min_scalefactor;
    else if(currentscalefactor > max_scalefactor)
        currentscalefactor = max_scalefactor;
    //std::cout<<currentscalefactor<<std::endl;

    _roi.x = pos.first.x;
    _roi.y = pos.first.y;

    cv::Mat patch_win = get_subwindow(im_gray, _roi.y, _roi.x, window_x, window_y);
    cv::Mat new_win_xf = multichafft(get_feature(patch_win, true), false);
    cv::Mat domini = gaussian_correlation(new_win_xf, new_win_xf, kernal_sigma);
    tmp.clear();
    cv::split(domini, tmp);
    tmp[0] += lambda;
    cv::merge(tmp, domini);
    cv::Mat new_alphaf = complexDivision(tar, domini);
    win_xf = (1 - interp_factor)*win_xf + interp_factor*new_win_xf;
    _alphaf = (1 - interp_factor)* _alphaf + interp_factor*new_alphaf;

    cv::Mat patch_app = get_subwindow(im_gray, _roi.y, _roi.x, app_x, app_y);
    cv::Mat new_app_xf = multichafft(get_feature(patch_app, false), false);
    domini = gaussian_correlation(new_app_xf, new_app_xf, kernal_sigma);
    tmp.clear();
    cv::split(domini, tmp);
    tmp[0] += lambda;
    cv::merge(tmp, domini);
    cv::Mat new_app_alphaf = complexDivision(app_tar, domini);
    if(max_response.second > appearance_thresh)
    {
        app_xf = (1 - interp_factor)*app_xf + interp_factor*new_app_xf;
        app_alphaf = (1 - interp_factor)* app_alphaf + interp_factor*new_app_alphaf;
        det.train(im_gray, pos.first.y, pos.first.x, cv::Size(window_y, window_x), true);
    }

    float train_scale_factor[40];
    //std::cout<<currentscalefactor<<std::endl;
    for(int i = 0; i < nScale; i++)
        train_scale_factor[i] = scale_factor[i]*currentscalefactor;
    cv::Mat new_xs = get_scale_sample(im_gray, cv::Rect(_roi.x, _roi.y, app_y, app_x), train_scale_factor, scale_model_sz, true);
    cv::Mat new_xsf;
    cv::dft(new_xs, new_xsf, cv::DFT_COMPLEX_OUTPUT | cv::DFT_ROWS);

    cv::Mat new_sf_num;
    cv::mulSpectrums(scale_tar, xsf, new_sf_num, 0, true);
    sf_num = (1 - interp_factor)*sf_num + interp_factor*new_sf_num;


    cv::Mat sf_den_ori;
    cv::mulSpectrums(xsf, xsf, sf_den_ori, 0, true);
    tmp.clear();
    std::vector<cv::Mat>  ans;
    cv::split(sf_den_ori, tmp);
    cv::Mat sf_den_sp;
    for(int i = 0; i < tmp.size(); i++)
    {
        cv::reduce(tmp[i], sf_den_sp, 0, cv::REDUCE_SUM);
        ans.push_back(sf_den_sp.clone());
    }
    cv::Mat new_sf_den;
    cv::merge(ans, new_sf_den);
    sf_den = (1 - interp_factor)*sf_den + interp_factor*new_sf_den;

    if(resize_image)
        return pos.first*2;
    else return pos.first;
}

cv::Mat LCT2tracker::get_scale_sample(const cv::Mat &image, cv::Rect base_target, float *scale_factor,
                                      cv::Size model_sz, bool window = true) {
    std::vector<cv::Mat> tmp;
    cv::Mat ans(nScale, floor(model_sz.width/float(4)) * floor(model_sz.height/float(4))*31, CV_32F);
    for(int i = 0; i < nScale; i++)
    {
        int sh = floor(base_target.height * scale_factor[i]);
        int sw = floor(base_target.width * scale_factor[i]);
        cv::Mat patch = get_subwindow(image, base_target.y, base_target.x, sh, sw);
        cv::Mat resized;
        cv::resize(patch, resized, model_sz);
        cv::Mat feature = fhog(resized, 4);
        cv::split(feature, tmp);
        tmp[31].release();
        tmp.pop_back();
        //std::cout<<tmp.size()<<std::endl;
        cv::merge(tmp, feature);
        tmp.clear();
        cv::Mat feature_re = feature.reshape(1, 1);
        if(window)
        {
            if(nScale %2 == 0)
                feature_re *= 0.5 * (1 - std::cos(2*3.14159265358979323846*(i + 1)/nScale));
            else feature_re *= 0.5 * (1 - std::cos(2*3.14159265358979323846*(i)/(nScale - 1)));
        }
        feature_re.copyTo(ans.row(i));
    }
    return ans.clone().t();
}

cv::Mat LCT2tracker::get_feature(const cv::Mat &raw_image, bool hann)
{
    cv::Mat ans;
    std::vector<cv::Mat> features;

    //hog feature
    cv::Mat image = raw_image.clone();
    cv::Mat h1 = fhog(image, cell_size, hog_orientations);
    std::vector<cv::Mat> h1_features;
    cv::split(h1, h1_features);
    for(int i = 0; i < h1_features.size() - 1; i++)
        features.push_back(h1_features[i].clone());

    //hoi feature
    raw_image.convertTo(image, CV_32F);
    cv::Mat h2_all = hoi(image, nbins, window_size);
    std::vector<cv::Mat> split_h2;
    cv::split(h2_all, split_h2);
    for(int k = 0; k < split_h2.size(); k++)
    {
        cv::Mat &h2 = split_h2[k];
        cv::Mat h2_resized(h2.rows/cell_size, h2.cols/cell_size, CV_32F);
        for(int i = 0; i < h2_resized.rows; i++)
            for(int j = 0; j < h2_resized.cols;j++)
            {
                h2_resized.at<float>(i, j) = h2.at<float>((i + 1)*cell_size - 1, (j + 1)*cell_size -1);
            }
        features.push_back(h2_resized.clone());
    }

    //hoi iif feature
    cv::Mat iif_image = 255 - doWork(raw_image, cv::Size(cell_size, cell_size), 32);
    iif_image.convertTo(image, CV_32F);
    cv::Mat h3 = hoi(image, nbins, window_size);
    cv::Mat h3_all = hoi(h3, nbins, window_size);
    std::vector<cv::Mat> split_h3;
    cv::split(h3_all, split_h3);
    for(int k = 0; k < split_h3.size(); k++)
    {
        cv::Mat &h3 = split_h3[k];
        cv::Mat h3_resized(h3.rows/cell_size, h3.cols/cell_size, CV_32F);
        for(int i = 0; i < h3_resized.rows; i++)
            for(int j = 0; j < h3_resized.cols;j++)
            {
                h3_resized.at<float>(i, j) = h3.at<float>((i + 1)*cell_size - 1, (j + 1)*cell_size -1);
            }
        features.push_back(h3_resized.clone());
    }
    size_patch[0] = features[0].rows;
    size_patch[1] = features[0].cols;
    size_patch[2] = features.size();
    if(hann)
    {
        cv::Mat hann_map = create_hanning();
        for(int i = 0; i < features.size(); i++)
            features[i] = features[i].mul(hann_map);
    }
    //std::cout<<features.size()<<std::endl;
    cv::merge(features, ans);
    return ans.clone();

}

cv::Mat LCT2tracker::create_hanning() {
    cv::Mat hann1t = cv::Mat(cv::Size(size_patch[1],1), CV_32F, cv::Scalar(0));
    cv::Mat hann2t = cv::Mat(cv::Size(1,size_patch[0]), CV_32F, cv::Scalar(0));


    for (int i = 0; i < hann1t.cols; i++)
        hann1t.at<float > (0, i) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann1t.cols - 1)));
    for (int i = 0; i < hann2t.rows; i++)
        hann2t.at<float > (i, 0) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann2t.rows - 1)));

    cv::Mat hann2d = hann2t * hann1t;
    return hann2d;
}