//
// Created by willian on 12/4/23.
//

#include "../include/AnomalyDetector.h"

void AnomalyDetector::plotFigures(const cv::Mat &image, const cv::Mat &hist, bool anomaly) {
    cv::Mat normalizedHist;
    cv::Mat histImage(image.rows, image.cols, CV_8UC3, cv::Scalar(255, 255, 255));
    normalizedHist = hist * histImage.rows;
    for (int i = 0; i < image.cols; i++) {
        cv::line(histImage, cv::Point(i, histImage.rows), cv::Point(i, histImage.rows - normalizedHist.at<float>(i)), cv::Scalar(0, 0, 0), 1);
    }
    std::string text = anomaly ? "ANOMALY" : "NORMAL";
    cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_DUPLEX, 1, 1, nullptr);

    cv::Point textPosition((histImage.rows - textSize.width) / 2, 50);
    cv::Scalar color = cv::Scalar(0, 0, 0);
    cv::putText(histImage, text, textPosition, cv::FONT_HERSHEY_DUPLEX, 1, color, 1);

    cv::imshow("Histograma", histImage);
    cv::imshow("Imagem", image);
    cv::waitKey(0);
}

cv::Mat AnomalyDetector::waveletSmoothing(const cv::Mat &signal, int level) {
    return cv::Mat();
}

bool AnomalyDetector::calculateAnomaly(const cv::Mat &image, bool plot) {
    // FELIPE PROGRAME AQUI!
}


cv::Mat AnomalyDetector::vectorToMat(const std::vector<double>& vec, int type) {
    cv::Mat mat(vec.size(), 1, type);
    for (int i = 0; i < vec.size(); ++i) {
        mat.at<float>(i, 0) = vec[i];
    }
    return mat.clone();
}

bool AnomalyDetector::detectOutliers(const cv::Mat &currentPeaks, double zscoreThreshold) {
    return false;
}

void AnomalyDetector::findPeaks(const cv::Mat& input, cv::Mat& peaks, double threshold, int distance, double prominence) {
    std::vector<int> peakIndices;

    for (int i = distance; i < input.cols - distance; ++i) {
        float currentValue = input.at<float>(0, i);
        float prevValue = input.at<float>(0, i - 1);
        float nextValue = input.at<float>(0, i + 1);

        if (currentValue > threshold && currentValue > prevValue && currentValue > nextValue) {
            bool isPeak = true;

            for (int j = i - distance; j <= i + distance; ++j) {
                if (j != i && currentValue <= input.at<float>(0, j)) {
                    isPeak = false;
                    break;
                }
            }

            if (isPeak && currentValue >= prominence) {
                peakIndices.push_back(i);

                std::cout << "Pico encontrado na posição " << i << " com valor " << currentValue << std::endl;
            }
        }
    }

    peaks = cv::Mat(peakIndices).clone();
}

cv::Mat AnomalyDetector::warpPerspective(const cv::Mat &image) {
    float rows = image.rows;
    float cols = image.cols;
    cv::Mat correctedImage;

    cv::Point2f srcPts[4] = {{100, 0}, {cols - 100, 0}, {0, rows}, {cols, rows}};
    cv::Point2f dstPts[4] = {{0, 0}, {cols, 0}, {0, rows}, {cols, rows}};
    cv::Mat perspectiveMatrix = cv::getPerspectiveTransform(srcPts, dstPts);

    cv::warpPerspective(image, correctedImage, perspectiveMatrix, image.size());

    return correctedImage;
}

cv::Mat AnomalyDetector::preprocess(const std::string &imagePath) {
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    cv::resize(image, image, cv::Size(640, 480));
    image = image(cv::Rect(0, 120, image.cols, 240));
    image = warpPerspective(image);

    return image;
}

void AnomalyDetector::detectAnomaly(const std::string &source, bool plot) {
    cv::Mat image;
    std::vector<cv::String> filenames;
    cv::glob(source, filenames, false);

    for (auto &file: filenames) {
        image = preprocess(file);
        calculateAnomaly(image);
    }
}
