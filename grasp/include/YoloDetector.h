//
// Created by hustac on 8/9/19.
//

#ifndef GRASPC2_YOLOLIBTORCHDETECTOR_H
#define GRASPC2_YOLOLIBTORCHDETECTOR_H

#include "Darknet.h"

#include <iostream>
#include <chrono>
#include <memory>

// Libtorch
#include <torch/torch.h>

// Opencv
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class YoloDetector {
public:
    YoloDetector(const std::string& config_file, const std::string& weights_file);

    // Get the bounding boxes of objects
    std::pair<std::vector<cv::RotatedRect>, std::vector<int>> getRotRectsAndID(cv::Mat &image, int thresh, bool show);

    bool calRotatedRect(cv::Mat img_roi, cv::Mat mask, const cv::Rect& box,
                                                    std::vector<cv::RotatedRect> &rotRects, int juggleOrCube, bool show);

private:
    // init the network
    void init(const std::string& config_file, const std::string& weights_file);

    // Remove the bounding boxes with low confidence using non-maxima suppression
    std::pair<std::vector<cv::RotatedRect>, std::vector<int>> postprocess(cv::Mat& frame, int thresh, bool show);

    // Draw the predicted bounding box
        static void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame,
                  std::vector<std::string> classes);

    // yolo相关参数
    std::shared_ptr<Darknet> net_;
    torch::Device* device_{};
    std::vector<int> classIds_;
    std::vector<float> confidences_;
    std::vector<cv::Rect> boxes_;

    float confThreshold_ = 0.2; // Confidence threshold
    float nmsThreshold_ = 0.4;  // Non-maximum suppression threshold
    int image_size_ = 416;  // Size of network's input image 416 608
    float box_scale_ = 1.2; // 边框尺寸缩放
    static const int classes_num_ = 5;
    std::string classes_[classes_num_]={"cube", "cubiod", "sphere", "triangular", "cylinder"};

    std::vector<std::string> get_classes_vec() {
        std::vector<std::string> classes_vec(classes_, classes_ + classes_num_);
        return classes_vec;
    }
};

#endif //GRASPC2_YOLOLIBTORCHDETECTOR_H
