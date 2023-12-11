//
// Created by willian on 12/4/23.
//

#include "../include/AnomalyDetector.h"

void AnomalyDetector::plotFigures(const cv::Mat &image, const cv::Mat &hist, bool anomaly) {
    cv::Mat normalizedHist;
    cv::Mat histImage(image.rows, image.cols, CV_8UC3, cv::Scalar(255, 255, 255));

    cv::normalize(hist, normalizedHist, 0, histImage.rows, cv::NORM_MINMAX);

    for (int i = 0; i < image.cols - 1; i++) {
        cv::line(histImage,
                 cv::Point(i, histImage.rows - normalizedHist.at<float>(i)),
                 cv::Point(i + 1, histImage.rows - normalizedHist.at<float>(i + 1)),
                 cv::Scalar(0, 0, 0),
                 1);
    }

    cv::line(histImage,
             cv::Point(image.cols - 1, histImage.rows - normalizedHist.at<float>(image.cols - 1)),
             cv::Point(image.cols - 1, histImage.rows),
             cv::Scalar(0, 0, 0),
             1);

    std::string text = anomaly ? "ANOMALY" : "NORMAL";
    cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_DUPLEX, 1, 1, nullptr);

    cv::Point textPosition((histImage.cols - textSize.width) / 2, 50);
    cv::Scalar color = cv::Scalar(0, 0, 0);
    cv::putText(histImage, text, textPosition, cv::FONT_HERSHEY_DUPLEX, 1, color, 1);

    cv::imshow("Histograma", histImage);
    cv::imshow("Imagem", image);
    cv::waitKey(0);
}

std::vector<double> AnomalyDetector::matToDoubleVector(const cv::Mat& mat) {
    CV_Assert(mat.type() == CV_32F);

    std::vector<double> result;
    result.reserve(mat.total());

    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            result.push_back(static_cast<double>(mat.at<float>(i, j)));
        }
    }

    return result;
}

bool AnomalyDetector::calculateAnomaly(const cv::Mat &image, bool plot) {
    try {
        cv::Mat hist;

        cv::reduce(image, hist, 0, cv::REDUCE_SUM, CV_32F);

        cv::Mat normalizedHist;
        cv::normalize(hist, normalizedHist, 0, 1,cv::NORM_MINMAX);

        const std::vector<double> vector = matToDoubleVector(normalizedHist);

        const std::vector<double> Lo_D = {0.06629126073623884,
                                          -0.19887378220871652,
                                          -0.15467960838455727,
                0.9943689110435825,
                0.9943689110435825,
                -0.15467960838455727,
                -0.19887378220871652,
                0.06629126073623884,

        };
        const std::vector<double> Hi_D = {0.0,
                0.0,
                -0.1767766952966369,
                0.5303300858899107,
                -0.5303300858899107,
                0.1767766952966369,
                0.0,
                0.0,
        };
        const std::vector<double> Lo_R = {0.0,
                0.0,
                0.1767766952966369,
                0.5303300858899107,
                0.5303300858899107,
                0.1767766952966369,
                0.0,
                0.0,

        };
        const std::vector<double> Hi_R = {0.06629126073623884,
                0.19887378220871652,
                -0.15467960838455727,
                -0.9943689110435825,
                0.9943689110435825,
                0.15467960838455727,
                -0.19887378220871652,
                -0.06629126073623884,
        };
        const Wavelet<double> bior34(Lo_D, Hi_D, Lo_R, Hi_R);

        Decomposition1D<double> dec1D = bior34.Wavedec(vector, 4);

        for (size_t level = 1; level < dec1D.NumLevels(); ++level) {
            dec1D.SetDetcoef(std::vector<double>(dec1D.GetDetcoef(level).size(), 0.0), level);
        }

        const std::vector<double> vector_rec = bior34.Waverec(dec1D, vector.size());

        cv::Mat vectorRecMat = vectorToMat(vector_rec, CV_32F);

        cv::Mat peaks;
        findPeaks(vectorRecMat, peaks, 0.0, 30, 0.2);

        bool anomaly = detectOutliers(peaks, 2.0);
        this->prevPeaks = peaks;

        plotFigures(image, vectorRecMat, anomaly);

        return true;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return false;
    }
}



cv::Mat AnomalyDetector::vectorToMat(const std::vector<double>& vec, int type) {
    cv::Mat mat(vec.size(), 1, type);
    for (int i = 0; i < vec.size(); ++i) {
        mat.at<float>(i, 0) = vec[i];
    }
    return mat.clone();
}

bool AnomalyDetector::detectOutliers(const cv::Mat &currentPeaks, double zscoreThreshold) {
    if(this->prevPeaks.empty() || currentPeaks.empty()){
        return false;
    }

    if (currentPeaks.rows != prevPeaks.rows || currentPeaks.cols != prevPeaks.cols) {
        return true;
    }

    cv::Scalar meanDiff, stddevDiff;

    cv::meanStdDev(currentPeaks, meanDiff, stddevDiff);

    double zscore = meanDiff[0] / stddevDiff[0];

    return zscore > zscoreThreshold;

}void AnomalyDetector::findPeaks(const cv::Mat& input, cv::Mat& peaks, double threshold, double distance, double prominence) {
    std::vector<int> peakIndices;

    for (int i = distance; i < input.rows - distance; ++i) {
        float currentValue = input.at<float>(i, 0);
        float prevValue = input.at<float>(i - 1, 0);
        float nextValue = input.at<float>(i + 1, 0);

        if (currentValue > threshold && currentValue > prevValue && currentValue > nextValue) {
            if (!peakIndices.empty() && i - peakIndices.back() < distance) {
                continue;
            }

            float minBaseline = std::min(prevValue, nextValue);
            for (int j = i - 1; j >= 0 && j >= i - distance; --j) {
                minBaseline = std::min(minBaseline, input.at<float>(j, 0));
            }
            for (int j = i + 1; j < input.rows && j <= i + distance; ++j) {
                minBaseline = std::min(minBaseline, input.at<float>(j, 0));
            }

            if (currentValue - minBaseline > prominence) {
                peakIndices.push_back(i);
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

    cv::warpPerspective(image, correctedImage, perspectiveMatrix, cv::Size(cols, rows));

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
