#include <iostream>
#include <opencv2/opencv.hpp>
#include "../include/AnomalyDetector.h"

int main() {
    auto *anomaly = new AnomalyDetector();
    anomaly->detectAnomaly("/home/solarbot/images/*.png");
}
