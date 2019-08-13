//
// Created by hustac on 8/9/19.
//

#include "YoloLibtorchDetector.h"

YoloLibtorchDetector::YoloLibtorchDetector(const std::string& config_file, const std::string& weights_file) {
    init(config_file, weights_file);
}

void YoloLibtorchDetector::init(const std::string& config_file, const std::string& weights_file) {
    torch::DeviceType device_type;

    if (torch::cuda::is_available() ) {
        device_type = torch::kCUDA;
    } else {
        device_type = torch::kCPU;
    }

//    device_ = std::make_shared<torch::Device>(device_type);
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

std::pair<std::vector<cv::RotatedRect>, std::vector<int>> YoloLibtorchDetector::getRotRectsAndID(cv::Mat &image, int thresh, bool show) {

    cv::Mat resized_image, img_float;
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
        classIds_.clear();
        confidences_.clear();
        boxes_.clear();

        // 保存结果, 以便后续处理
        for (int i = 0; i < result.size(0); i++)
        {
            int width = (int)(result_data[i][3]-result_data[i][1]);
            int height = (int)(result_data[i][4]-result_data[i][2]);
            int left = result_data[i][1];
            int top = result_data[i][2];

            // 滤除置信度较低的, Darknet中滤除的是最高置信度, 这里取最低置信度
            if (result_data[i][6] > confThreshold_) {
                classIds_.push_back(result_data[i][7]);
                confidences_.push_back(result_data[i][6]);
                boxes_.emplace_back(left, top, width, height);
            }
        }

        std::cout << "[INFO] Detected " << boxes_.size() << " objects by Yolo." << endl;

        cout << "class: " << classIds_ << endl;
        cout << "conf: " << confidences_ << endl;
        cout << "boxes: " << boxes_ << endl;

        // 保存图片
//        for (size_t i = 0; i < result.size(0) ; i++)
//        {
//            if (result_data[i][6] > confThreshold_) {
//                cv::rectangle(image, cv::Point(result_data[i][1], result_data[i][2]),
//                              cv::Point(result_data[i][3], result_data[i][4]), cv::Scalar(0, 0, 255), 1, 1, 0);
//            }
//        }

//        if (show) cv::imshow("image", image);
//        if (show) cv::waitKey(0);
//
//        cv::imwrite("/home/hustac/out-det.jpg", image);
    }

    std::pair<std::vector<cv::RotatedRect>, std::vector<int>> RotRectsAndID;
    RotRectsAndID = postprocess(image, thresh, show); // Remove the bounding boxes with low confidence
    return RotRectsAndID;
}

std::pair<std::vector<cv::RotatedRect>, std::vector<int>> YoloLibtorchDetector::postprocess(cv::Mat& frame, int thresh, bool show)
{
    std::vector<cv::RotatedRect> RotatedRects;
    std::vector<int> RectsID;
    cv::Mat frame_copy = frame.clone();

    for (size_t i = 0; i < boxes_.size(); ++i) // 处理各检测到的物体
    {
        cv::Rect box = boxes_[i];

        if(box.height * box.width < 40*40 || box.height * box.width > 100*100) continue; // FIXME: 滤除过大/小的物体

        cv::Mat img_roi = frame.clone()(box);
        if(show) cv::imshow("roi", img_roi);

        cv::Mat img_hsv;
        cv::cvtColor(img_roi, img_hsv, CV_BGR2HSV);
        if(show) cv::imshow("hsv", img_hsv);

        /// HSV阈值分割获取掩码
        int thresh_v_high = thresh; // V通道阈值
        cv::Mat mask = cv::Mat::zeros(img_hsv.rows, img_hsv.cols, CV_8U); // 掩码
        for(int r = 0; r < img_hsv.rows; ++r)
        {
            auto *itM = mask.ptr<uint8_t>(r);
            const cv::Vec3b *itH = img_hsv.ptr<cv::Vec3b>(r);

            for(int c = 0; c < img_hsv.cols; ++c, ++itM, ++itH)
            {
                if (itH->val[0] < 35 && itH->val[2] > thresh_v_high) { /// HSV阈值分割 顶部>255 旁边>120
                    *itM = 255;
                }
            }
        }
        if(show) cv::imshow("mask", mask);

        /// 计算最小外接矩形
        cv::RotatedRect rotRect;
        if (calRotatedRect(img_roi, mask, box, rotRect, show)) {
            RotatedRects.push_back(rotRect); // 存储外接矩形
            RectsID.push_back(classIds_[i]); // 存储外接矩形对应的物体类别
        }

        if(show) std::cout << "minAreaRectOut: center:" << rotRect.center << " angle: " <<
                                                                rotRect.angle << " size: " << rotRect.size << std::endl;

        // 获取各目标位置
//        if (classes_[classIds_[idx]] == "bottle" && confidences_[idx] > 0.4) {
//            std::cout << box << std::endl;
//            obj_boxes.push_back(box);
//            bottleNum ++;
//        }

        if(show) drawPred(classIds_[i], confidences_[i], box.x, box.y, box.x + box.width, box.y + box.height,
                frame_copy, get_classes_vec()); // 画边框

        if(show) cv::imshow("result", frame_copy);
        if(show) cv::waitKey(0);
    }

    std::pair<std::vector<cv::RotatedRect>, std::vector<int>> RectsAndID {RotatedRects, RectsID};

    return RectsAndID;
}

bool YoloLibtorchDetector::calRotatedRect(cv::Mat img_roi, cv::Mat mask, cv::Rect box, cv::RotatedRect &rotRect, bool show){
    /// 轮廓查找
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    findContours(mask, contours, hierarchy, cv::RETR_TREE, cv:: CHAIN_APPROX_SIMPLE);

    if (contours.empty()) return false;

    int index = 0;
    for (; index >= 0; index = hierarchy[index][0]) {
        drawContours(img_roi, contours, index, cv::Scalar(0, 255, 255), 1, 8, hierarchy);
    }
    //        cv::imshow("roi_Contours", img_roi);

    /// 查找最大轮廓
    double max_area = 0;
    int maxAreaIdx = 0;
    for (int index = (int)contours.size() - 1; index >= 0; index--)
    {
        double tmp_area = fabs(contourArea(contours[index]));
    if (tmp_area > max_area)
    {
        max_area = tmp_area;
        maxAreaIdx = index; //记录最大轮廓的索引号
        }
    }

    std::vector<cv::Point> contourlist; // 轮廓列表
    contourlist = contours[maxAreaIdx]; // 最大轮廓

    /// 最大轮廓的最小外接矩形
    rotRect = minAreaRect(contourlist);
    if(show) std::cout << "minAreaRect: center:" << rotRect.center << " angle: " << rotRect.angle <<
                                                                                " size: " << rotRect.size << std::endl;

    // 外接矩形的四个角点
    cv::Point2f P[4];
    rotRect.points(P);
    for(int j=0; j <= 3; j++) {
        line(img_roi, P[j], P[(j + 1) % 4], cv::Scalar(0, 255 ,0), 2);
    }
    cv::circle(img_roi, P[0], 1, cv::Scalar(0, 255, 255), 2);
    cv::circle(img_roi, P[2], 1, cv::Scalar(0, 255, 255), 2);
    cv::circle(img_roi, P[1], 1, cv::Scalar(0, 0, 0), 2);
    cv::circle(img_roi, P[3], 1, cv::Scalar(0, 0, 0), 2);

    cv::circle(img_roi, rotRect.center, 1, cv::Scalar(0, 0, 255), 2);

    cout << "四个角点: " << P[0] << endl << P[1] << endl << P[2] << endl << P[3] << endl << endl;

    cv::Point2f P1;
    P1.x = P[0].x + (P[2].x - P[0].x)/4;
    P1.y = P[0].y + (P[2].y - P[0].y)/4;
    cv::circle(img_roi, P1, 1, cv::Scalar(255, 0, 0), 2);

    cv::Point2f P2;
    P2.x = P[0].x + (P[2].x - P[0].x)*3/4;
    P2.y = P[0].y + (P[2].y - P[0].y)*3/4;
    cv::circle(img_roi, P2, 1, cv::Scalar(0, 255, 0), 2);

    // 重新计算中心点
    cv::Point2f P0;
    P0.x = P1.x + (P2.x - P1.x)/2;
    P0.y = P1.y + (P2.y - P1.y)/2;
    cv::circle(img_roi, P0, 1, cv::Scalar(0, 0, 255), 2);

    if(show) cv::imshow("roi_minAreaRect", img_roi);

    // 获取整张图片下的中心位置及角度
    rotRect.center.x += box.x;
    rotRect.center.y += box.y;

    return true;
}

// Draw the predicted bounding box
void YoloLibtorchDetector::drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame, std::vector<std::string> classes)
{
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