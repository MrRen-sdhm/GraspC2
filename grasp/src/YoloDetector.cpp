//
// Created by hustac on 8/9/19.
//

#include "YoloDetector.h"

YoloDetector::YoloDetector(const std::string& config_file, const std::string& weights_file) {
    init(config_file, weights_file);
}

void YoloDetector::init(const std::string& config_file, const std::string& weights_file) {
    torch::DeviceType device_type;

    if (torch::cuda::is_available() ) {
        device_type = torch::kCUDA;
    } else {
        device_type = torch::kCPU;
    }

    device_ = new torch::Device(device_type);

    net_ = std::make_shared<Darknet>(config_file.c_str(), device_);

    map<string, string> *info = net_->get_net_info();

    info->operator[]("height") = std::to_string(image_size_);

    net_->load_weights(weights_file.c_str());

    net_->to(*device_);

    torch::NoGradGuard no_grad;
    net_->eval();

    std::cout << "[INFO] Yolo detector initialize done." << endl;
}

void YoloDetector::detectObj(cv::Mat &image, std::vector<int> &classIds,
                    std::vector<float> &confidences, std::vector<cv::Rect> &boxes, cv::Rect rect, int thresh, int show) {

    cv::Mat resized_image, img_float;
    cv::Mat image_copy = image.clone();

    cv::cvtColor(image, resized_image,  cv::COLOR_RGB2BGR);
    cv::resize(resized_image, resized_image, cv::Size(image_size_, image_size_));

    resized_image.convertTo(img_float, CV_32F, 1.0/255);

    auto img_tensor = torch::CPU(torch::kFloat32).tensorFromBlob(img_float.data, {1, image_size_, image_size_, 3});

    img_tensor = img_tensor.permute({0,3,1,2});

    auto img_var = torch::autograd::make_variable(img_tensor, false).to(*device_);

    auto output = net_->forward(img_var);

    auto result = net_->write_results(output, classes_num_, confThreshold_, nmsThreshold_, box_scale_);

    cout << result << endl;

    if (result.dim() == 1)
    {
        std::cout << "no object found" << endl;
    }
    else
    {
        float w_scale = float(image.cols) / float(image_size_);
        float h_scale = float(image.rows) / float(image_size_);

        result.select(1,1).mul_(w_scale);
        result.select(1,2).mul_(h_scale);
        result.select(1,3).mul_(w_scale);
        result.select(1,4).mul_(h_scale);

        auto result_data = result.accessor<float, 2>();

        // 清空数据
        classIds.clear();
        confidences.clear();
        boxes.clear();

        // 保存结果, 以便后续处理
        for (size_t i = 0; i < result.size(0); i++)
        {
            int width = (int)(result_data[i][3]-result_data[i][1]);
            int height = (int)(result_data[i][4]-result_data[i][2]);
            int left = result_data[i][1];
            int top = result_data[i][2];

            // 滤除置信度较低的, Darknet中滤除的是最高置信度, 这里取最低置信度
            if (result_data[i][6] > confThreshold_ &&
                            top > rect.y && left > rect.x && left + width < rect.x + rect.width) { // 滤除不在工作范围内的
                classIds.push_back(result_data[i][7]);
                confidences.push_back(result_data[i][6]);
                boxes.emplace_back(left, top, width, height);
            }
        }

        std::cout << "[INFO] Detected " << boxes.size() << " objects by Yolo." << endl;

        cout << "[INFO] Class: " << classIds << endl;
        cout << "[INFO] Conf: " << confidences << endl;
        cout << "[INFO] Boxes: " << boxes << endl << endl << endl;

        if(show == 1 | show == 2) {
            // 保存图片
            for (size_t i = 0; i < result.size(0); i++) {
                if (result_data[i][6] > confThreshold_) {
                    cv::rectangle(image_copy, cv::Point(result_data[i][1], result_data[i][2]),
                                  cv::Point(result_data[i][3], result_data[i][4]), cv::Scalar(0, 0, 255), 1, 1, 0);
                }
            }

            cv::imshow("Yolo_Raw", image_copy);
             cv::waitKey(0);
        }
//        cv::imwrite("/home/hustac/out-det.jpg", image);
    }
}

// Draw the predicted bounding box
void YoloDetector::drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame)
{
    std::vector<std::string> classes = get_classes_vec();

    // Draw a rectangle displaying the bounding box
    rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(255, 178, 50), 2); // 边框

    // Get the label for the class name and its confidence
    std::string label = cv::format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)(classes.size()));
        label = classes[classId] + ":" + label;
    }

    // Display the label at the top of the bounding box
    int baseLine;
    cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = std::max(top, labelSize.height);
    rectangle(frame, cv::Point(left-1, top - round(1.2*labelSize.height)), // 标题背景
              cv::Point(left + round(1.2*labelSize.width), top + baseLine), cv::Scalar(255, 178, 50), -1);
    putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0),1,CV_AA); // 标题
}