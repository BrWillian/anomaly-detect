//
// Created by willian on 12/4/23.
//

#ifndef ANOMALYDETECTOR_ANOMALYDETECTOR_H
#define ANOMALYDETECTOR_ANOMALYDETECTOR_H

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "wavelet.h"

class AnomalyDetector {
private:
    cv::Mat prevPeaks;

    void plotFigures(const cv::Mat& image, const cv::Mat& hist, bool anomaly);
    bool calculateAnomaly(const cv::Mat& image, bool plot = true);
    bool detectOutliers(const cv::Mat& currentPeaks, double zscoreThreshold);
    void findPeaks(const cv::Mat& input, cv::Mat& peaks, double threshold, double distance, double prominence);
    cv::Mat vectorToMat(const std::vector<double>& vec, int type);
    std::vector<double> matToDoubleVector(const cv::Mat& mat);

    cv::Mat warpPerspective(const cv::Mat& image);
    cv::Mat preprocess(const std::string& imagePath);

public:
    void detectAnomaly(const std::string& source, bool plot = true);
};


#endif //ANOMALYDETECTOR_ANOMALYDETECTOR_H
