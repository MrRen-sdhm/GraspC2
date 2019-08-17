//
// Created by hj on 18-12-29.
//

#include "GraphicsGrasp.h"
#include <QFile>
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <sstream>
#include <string>
#include <cobotsys/cobotsys_data_types.h>
#include <algorithm>

using namespace std;
using namespace cv;

namespace {
    std::string gpd_cfg_file = "../../../Grasp/Sources/pointnet_params.cfg";
    std::string yolo_config_filename = "../../../grasp/data/yolov3-voc.cfg";
    std::string yolo_weights_filename ="../../../grasp/data/yolov3-voc_900.weights";
}

GraphicsGrasp::GraphicsGrasp()
{
    /// gpd 初始化
//    grasp_detector_ = new gpd::GraspDetectorPointNet(gpd_cfg_file); // TODO

    /// yolo 初始化
    yoloDetector = new YoloDetector(yolo_config_filename, yolo_weights_filename);
}

// TODO
//std::vector<std::unique_ptr<gpd::candidate::Hand>> GraphicsGrasp::detectGraspPoses(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud)
//{
//    Eigen::Matrix3Xd view_points(3,1);
//    cloud_camera_ = new gpd::util::Cloud(cloud, 0, view_points);
//
//    // detect grasp poses
//    std::vector<std::unique_ptr<gpd::candidate::Hand>> grasps;
//
//    grasp_detector_->preprocessPointCloud(*cloud_camera_);
//
//    // detect grasps in the point cloud
//    grasps = grasp_detector_->detectGrasps(*cloud_camera_);
//
//    return grasps;
//}

std::pair<std::vector<cv::RotatedRect>, std::vector<int>> GraphicsGrasp::detectGraspYolo(cv::Mat &image, int thresh, int show)
{
    printf("\n[FUNC] Detect Grasp Yolo ...\n");
    cv::Mat resized;
    std::pair<std::vector<cv::RotatedRect>, std::vector<int>> RotatedRectsAndID;

    cv::resize(image, resized, cv::Size(960, 540)); // 缩小图片

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    cv::Rect rect(cv::Point(LU_[0], LU_[1]), cv::Size(RD_[0]-LU_[0], RD_[1]-LU_[1]));
    yoloDetector->detectObj(resized, classIds, confidences, boxes, rect, thresh, show);

    RotatedRectsAndID = getRotRectsAndID(resized, classIds, confidences, boxes, rect, thresh, show);

    printf("[INFO] Detected %zu rotated rects.\n", RotatedRectsAndID.first.size());

    return RotatedRectsAndID;
}

std::pair<std::vector<cv::RotatedRect>, std::vector<int>> GraphicsGrasp::detectBigObj(cv::Mat &image,
                                                  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud, int thresh, int show)
{
    cv::Mat frame;
    std::pair<std::vector<cv::RotatedRect>, std::vector<int>> RotatedRectsAndID;

    cv::resize(image, frame, cv::Size(960, 540)); // 缩小图片

    std::vector<cv::RotatedRect> RotatedRects;
    std::vector<int> RectsID;
    cv::Mat frame_copy = frame.clone();

    cv::Mat img_hsv;
    cv::cvtColor(frame, img_hsv, CV_BGR2HSV);
    if(show == 1 | show == 2) cv::imshow("hsv", img_hsv);

    /// HSV阈值分割获取掩码
    int thresh_v_high = thresh; // V通道阈值
    cv::Point LU(LU_[0], LU_[1]); // 桌面区域左上角点 (x, y)=(col, row)
    cv::Point RD(RD_[0], RD_[1]); // 桌面区域右下角点 (x, y)=(col, row)
    cv::Mat mask = cv::Mat::zeros(img_hsv.rows, img_hsv.cols, CV_8U); // 掩码
    cv::Mat maskBig = cv::Mat::zeros(img_hsv.rows, img_hsv.cols, CV_8U); // 掩码

    /// 获取目标区域掩码
    for(int r = 0; r < img_hsv.rows; ++r) {
        auto *itM = mask.ptr<uint8_t>(r);
        const cv::Vec3b *itH = img_hsv.ptr<cv::Vec3b>(r);

        for(int c = 0; c < img_hsv.cols; ++c, ++itM, ++itH) {
            if (r > LU.y && c > LU.x && c < RD.x) { /// 限定像素范围
                if (itH->val[0] < 40 && itH->val[2] > thresh_v_high) { /// HSV阈值分割 顶部>255 旁边>120
                    *itM = 255;
                }
            }
        }
    }

    for(int r = 0; r < img_hsv.rows; ++r) {
        auto *itM = maskBig.ptr<uint8_t>(r);
        const cv::Vec3b *itH = img_hsv.ptr<cv::Vec3b>(r);

        for(int c = 0; c < img_hsv.cols; ++c, ++itM, ++itH) {
            if (r > LU.y && c > LU.x && c < RD.x) { /// 限定像素范围
                if (itH->val[0] < 40 && itH->val[2] > thresh_v_high) { /// HSV阈值分割 顶部>255 旁边>120
                    /// 计算当前点在机器人坐标系下的坐标
                    float center_x, center_y, center_z;
                    int leftOrRight = 0; // 左臂: 0.66 右臂:-0.60
                    if (getPointLoc(r, c, center_x, center_y, center_z, cloud)) {
                        std::vector<float> coorRaw = {center_x, center_y, center_z};
                        std::vector<double> b2oXYZRPY = calcRealCoor(coorRaw, leftOrRight); // 计算基坐标到物体转换关系

//                        cout << "当前点在机器人坐标系下的坐标: " << b2oXYZRPY << endl;

                        if (b2oXYZRPY[0] < smallCubeThresh) { // 长方体积木立着<0.62 小立方体<0.6
                            cout << "阈值范围内的坐标: " << b2oXYZRPY << endl;
                            *itM = 255; // 高的物体为目标物体
                        }
                    }
                }
            }
        }
    }

    if(show == 1 | show == 2) cv::imshow("mask", mask);
    if(show == 1 | show == 2) cv::imshow("maskBig", maskBig);

    std::vector<cv::RotatedRect> rotRects;
    cv::Rect box(cv::Point(0, 0), cv::Size(0, 0));

    /// 计算最小外接矩形
    if (calRotatedRect(frame, mask, box, rotRects, 1, show)) {
        for (const auto & rotRect : rotRects) {
            RotatedRects.push_back(rotRect); // 存储外接矩形
            RectsID.push_back(6); // 存储外接矩形对应的物体类别, BigObj 类别号为6
        }

        printf("[INFO] Detected %zu big obj.\n", rotRects.size());
    }

    if(show == 1 | show == 2) {
        for (size_t i = 0; i < RotatedRects.size(); i++) {
            std::cout << "[INFO] minAreaRectOut" << i << " center:" << RotatedRects[i].center << " angle: " <<
                                                RotatedRects[i].angle << " size: " << RotatedRects[i].size << std::endl;
        }
    }

    std::pair<std::vector<cv::RotatedRect>, std::vector<int>> RectsAndID {RotatedRects, RectsID};

    return RectsAndID;
}

int GraphicsGrasp::detectBigBall(cv::Mat &image, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud,
                                                        std::pair<cv::RotatedRect, int> &BigRotRectsAndID, bool show) {
    printf("\n[FUNC] Detect BigBall ...\n");
    std::pair<std::vector<cv::RotatedRect>, std::vector<int>> RotRectsAndID;
    std::vector<float> area;
    std::vector<float> indices;

    RotRectsAndID = detectBigObj(image, cloud, 100, false); // 检测大球, 低阈值 NOTE:检测大球和大正方体使用的阈值不一样
    for (size_t  idx = 0; idx < RotRectsAndID.first.size(); idx++) {
        cout << "RotRect Size: " <<  RotRectsAndID.first[idx].size.area() << endl;
        if (RotRectsAndID.first[idx].size.area() > 10500) { // 球:12116、11030.4 立方体:8798.46、9968.54、9594.76
            area.push_back(RotRectsAndID.first[idx].size.area());
            indices.push_back(idx);
        }
    }

    if (!area.empty()) {
        auto max_area = std::max_element(area.begin(), area.end());
        auto distance = std::distance(area.begin(), max_area);
        BigRotRectsAndID.first = RotRectsAndID.first[indices[distance]]; // 在RotRectsAndID中的位置
        BigRotRectsAndID.second = RotRectsAndID.second[indices[distance]]; // 在RotRectsAndID中的ID
        std::cout << "Largest ball is " << *max_area << " at position " << distance << std::endl;

        return 1;
    } else return  -1;
}

int GraphicsGrasp::detectBigCube(cv::Mat &image, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud,
                                                        std::pair<cv::RotatedRect, int> &BigRotRectsAndID, bool show) {
    printf("\n[FUNC] Detect BigCube ...\n");

    std::pair<std::vector<cv::RotatedRect>, std::vector<int>> RotRectsAndID;
    std::vector<float> area;
    std::vector<float> indices;

    RotRectsAndID = detectBigObj(image, cloud, 200, show); // 检测大正方体, 高阈值 NOTE:检测大球和大正方体使用的阈值不一样
    for (size_t  idx = 0; idx < RotRectsAndID.first.size(); idx++) {
        cout << "RotRect Size: " <<  RotRectsAndID.first[idx].size.area() << endl;
        if (RotRectsAndID.first[idx].size.area() < 7000) { // 球:8713.38、7992.55 立方体:6000.45、5858.15
            area.push_back(RotRectsAndID.first[idx].size.area());
            indices.push_back(idx);
        }
    }

    if (!area.empty()) {
        auto max_area = std::max_element(area.begin(), area.end());
        auto distance = std::distance(area.begin(), max_area);
        BigRotRectsAndID.first = RotRectsAndID.first[indices[distance]]; // 在RotRectsAndID中的位置
        BigRotRectsAndID.second = RotRectsAndID.second[indices[distance]]; // 在RotRectsAndID中的ID
        std::cout << "Largest cube is " << *max_area << " at position " << distance << std::endl;

        return 1;
    } else return  -1;
}

std::pair<std::vector<cv::RotatedRect>, std::vector<int>> GraphicsGrasp::getRotRectsAndID(
                            cv::Mat &image, std::vector<int> &classIds, std::vector<float> &confidences,
                            std::vector<cv::Rect> &boxes, cv::Rect rect, int thresh, int show) {

    std::vector<cv::RotatedRect> RotatedRects;
    std::vector<int> RectsID;
    cv::Mat frame_copy = image.clone();

    for (size_t i = 0; i < boxes.size(); ++i) // 处理各检测到的物体
    {
        cv::Rect box = boxes[i];

//        if(box.height * box.width < 40*40 || box.height * box.width > 200*200) { // FIXME
//            printf("[WARN] RotRect’s size is not valid: %f", box.height * box.width);
//            continue; // NOTE: 滤除过大/小的物体
//        }

        cv::Mat img_roi = image.clone()(box);
        if(show == 1 | show == 2) cv::imshow("roi", img_roi);

        cv::Mat img_hsv;
        cv::cvtColor(img_roi, img_hsv, CV_BGR2HSV);
        if(show == 1 | show == 2) cv::imshow("hsv", img_hsv);

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
        if(show == 1 | show == 2) cv::imshow("mask", mask);

        /// 计算最小外接矩形
        std::vector<cv::RotatedRect> rotRects;
        if (calRotatedRect(img_roi, mask, box, rotRects, 0, show)) {
            RotatedRects.push_back(rotRects[0]); // 存储外接矩形, 每个积木仅有一个外接矩形
            RectsID.push_back(classIds[i]); // 存储外接矩形对应的物体类别

            if(show == 1 | show == 2) std::cout << "minAreaRectOut: center:" << rotRects[0].center << " angle: " <<
                                                rotRects[0].angle << " size: " << rotRects[0].size << std::endl;
        }

        if(show == 1 | show == 2) yoloDetector->drawPred(classIds[i], confidences[i], box.x, box.y, box.x + box.width,
                                           box.y + box.height, frame_copy); // 画边框

        if(show == 2) {
            cv::imshow("result", frame_copy);
            cv::waitKey(0);
        }
    }

    if(show == 1) {
        cv::imshow("result", frame_copy);
        cv::waitKey(0);
    }

    std::pair<std::vector<cv::RotatedRect>, std::vector<int>> RectsAndID {RotatedRects, RectsID};

    return RectsAndID;
}

bool GraphicsGrasp::calRotatedRect(cv::Mat img_roi, cv::Mat mask, const cv::Rect& box,
                                  std::vector<cv::RotatedRect> &rotRects, int juggleOrCube, int show){
    std::vector<int> bigAreaIdx;

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
    for (int index = (int)contours.size() - 1; index >= 0; index--) {
        double tmp_area = fabs(contourArea(contours[index]));

        // 查找面积很大的轮廓
        if (tmp_area > areaThresh) {
            bigAreaIdx.push_back(index);
        }
//        printf("[INFO] tmp_area: %f\n", tmp_area);

        if (tmp_area > max_area) {
            max_area = tmp_area;
            maxAreaIdx = index; // 记录最大轮廓的索引号
        }
    }

    printf("[INFO] Max area: %f\n", max_area);

    for (size_t i = 0; i < bigAreaIdx.size(); i++) {
        printf("[INFO] Big Area[%zu]: %f\n", i, fabs(contourArea(contours[bigAreaIdx[i]])));
    }

    std::vector<cv::Point> contourlist; // 轮廓列表
    /// 最大轮廓的最小外接矩形
    if (juggleOrCube == 0) { // 积木的外接矩形, 仅一个
        if (max_area > areaThresh) {
            printf("\033[0;32m[WARN] RotRect area is too large: %f at[x-%d, y-%d]\033[0m\n", max_area, box.x, box.y);

            return  false; // 面积过大, 不是积木, 积木面积最大值在 888 左右
        }

        printf("[INFO] Juggle area: %f\n", fabs(contourArea(contours[maxAreaIdx])));

        contourlist = contours[maxAreaIdx]; // 最大轮廓
        rotRects.push_back(minAreaRect(contourlist));
        // 获取整张图片下的中心位置
        rotRects[0].center.x += box.x;
        rotRects[0].center.y += box.y;

    } else if (juggleOrCube == 1) { // 所有立方体的外接矩形
        if (bigAreaIdx.empty())
            return false; // 未找到立方体

        for (size_t idx = 0; idx < bigAreaIdx.size(); idx++) {
            contourlist = contours[bigAreaIdx[idx]];
            printf("[INFO] Cube area: %f\n", fabs(contourArea(contourlist)));

            rotRects.push_back(minAreaRect(contourlist));

            // 外接矩形的四个角点
            cv::Point2f P[4];
            rotRects[idx].points(P);
            for (int j = 0; j <= 3; j++) {
                line(img_roi, P[j], P[(j + 1) % 4], cv::Scalar(0, 255, 0), 2);
            }
            cv::circle(img_roi, P[0], 1, cv::Scalar(0, 255, 255), 2);
            cv::circle(img_roi, P[2], 1, cv::Scalar(0, 255, 255), 2);
            cv::circle(img_roi, P[1], 1, cv::Scalar(0, 0, 0), 2);
            cv::circle(img_roi, P[3], 1, cv::Scalar(0, 0, 0), 2);

//            cout << "四个角点: " << P[0] << endl << P[1] << endl << P[2] << endl << P[3] << endl << endl;

            cv::circle(img_roi, rotRects[idx].center, 1, cv::Scalar(0, 0, 255), 2);

            cv::Point2f P1;
            P1.x = P[0].x + (P[2].x - P[0].x) / 8;
            P1.y = P[0].y + (P[2].y - P[0].y) / 8;
            cv::circle(img_roi, P1, 1, cv::Scalar(255, 0, 0), 2);

            cv::Point2f P2;
            P2.x = P[0].x + (P[2].x - P[0].x) * 7 / 8;
            P2.y = P[0].y + (P[2].y - P[0].y) * 7 / 8;
            cv::circle(img_roi, P2, 1, cv::Scalar(0, 255, 0), 2);

            // 重新计算中心点
            cv::Point2f P0;
            P0.x = P1.x + (P2.x - P1.x) / 2;
            P0.y = P1.y + (P2.y - P1.y) / 2;
            cv::circle(img_roi, P0, 1, cv::Scalar(0, 0, 255), 2);

            // 获取整张图片下的中心位置
            rotRects[idx].center.x += box.x;
            rotRects[idx].center.y += box.y;
        }
    }

    if(show == 1) cv::imshow("roi_minAreaRect", img_roi);

    return true;
}

std::vector<int> GraphicsGrasp::findAimObjLR(std::pair<std::vector<cv::RotatedRect>, std::vector<int>> RotRectsAndID, int RowOrCol) {
    printf("\n[FUNC] Detect objects in the left or right work area ...\n");

    std::vector<int> LIndices, RIndices;
    std::vector<float> LCenters, RCenters;
    for (size_t i = 0; i < RotRectsAndID.first.size(); i++) {
        float center_x = RotRectsAndID.first[i].center.x;
        float center_y = RotRectsAndID.first[i].center.y;

        /// 先区分左右工作区域
        if (center_x < LeftOrRightThresh && center_x > WorkAreaThreshL) { // 左侧
            if (RowOrCol == 1) { // 存储列值, 用于找横向最值
//                printf("LeftCenter[%zu]: %f\n", i, center_x);
                LCenters.push_back(center_x);
            } else if (RowOrCol == 0) { // 存储行值, 用于找纵向最值
//                printf("LeftCenter[%zu]: %f\n", i, center_y);
                LCenters.push_back(center_y);
            }
            LIndices.push_back(i);
        } else if (center_x > LeftOrRightThresh && center_x < WorkAreaThreshR){ // 右侧
            if (RowOrCol == 1) { // 存储列值, 用于找横向最值
//                printf("RightCenter[%zu]: %f\n", i, center_x);
                RCenters.push_back(center_x);
            } else if (RowOrCol == 0) { // 存储行值, 用于找纵向最值
//                printf("RightCenter[%zu]: %f\n", i, center_y);
                RCenters.push_back(center_y);
            }
            RIndices.push_back(i);
        }
    }

    printf("[INFO] Detect %zu objects in the left work area.\n", LIndices.size());
    cout << "[INFO] Left Object Centers: " << LCenters << endl;
    cout << "[INFO] Left Object Indices: " << LIndices << endl;

    printf("[INFO] Detect %zu objects in the right work area.\n", RIndices.size());
    cout << "[INFO] Right Object Centers: " << RCenters << endl;
    cout << "[INFO] Right Object Indices: " << RIndices << endl;

    // 左侧找最小的
    int positionRawL;
    if (!LCenters.empty()) { // 左侧有物体
        auto min_LCenter = std::min_element(LCenters.begin(), LCenters.end());
        auto distanceL = std::distance(LCenters.begin(), min_LCenter);
        positionRawL = LIndices[distanceL]; // 在RotRectsAndID中的位置
        std::cout << "[INFO] LeftCenter Min element is " << *min_LCenter << " at position "
                                                    << positionRawL << " in RotRectsAndID" << std::endl;
    } else {
        positionRawL = -1; // 未找到物体, 索引置-1
    }

    // 右侧找最大的
    int positionRawR;
    if (!RCenters.empty()) { // 左侧有物体
        auto max_RCenter = std::max_element(RCenters.begin(), RCenters.end());
        auto distanceR = std::distance(RCenters.begin(), max_RCenter);
        positionRawR = RIndices[distanceR]; // 在RotRectsAndID中的位置
        std::cout << "[INFO] RightCenter Max element is " << *max_RCenter << " at position "
                                                    << positionRawR << " in RotRectsAndID" << std::endl;
    } else {
        positionRawR = -1; // 未找到物体, 索引置-1
    }

    std::vector<int> AimObjIndicesLR = {positionRawL, positionRawR};
    return AimObjIndicesLR;
}

std::vector<double> GraphicsGrasp::calcRealCoor(std::vector<float> coorRaw, int leftOrRight)
{
    /// 物体姿态转换到机器人坐标系下
    Eigen::Vector3d c2o_ea(0, 0, 0); // YPR, 先绕z轴yaw, 再绕y轴pitch, 最后绕x轴roll
    Eigen::Quaterniond c2o_quat;
    c2o_quat = Eigen::AngleAxisd(c2o_ea[0], Eigen::Vector3d::UnitZ()) *
               Eigen::AngleAxisd(c2o_ea[1], Eigen::Vector3d::UnitY()) *
               Eigen::AngleAxisd(c2o_ea[2], Eigen::Vector3d::UnitX());

    Eigen::Matrix3d c2o_rot = c2o_quat.matrix(); // 相机到物体旋转矩阵
    Eigen::Vector3d c2o_trans (coorRaw[0], coorRaw[1], coorRaw[2]); // 相机到物体平移矩阵

    /// 相机->物体 齐次变换矩阵
    Eigen::Isometry3d c2o_T = Eigen::Isometry3d::Identity();
    c2o_T.rotate(c2o_rot);
    c2o_T.pretranslate(c2o_trans);

//    Eigen::Vector3d c2o_trans_ = c2o_T.translation();
//    Eigen::Matrix3d c2o_rot_ = c2o_T.rotation();
//    Eigen::Quaterniond c2o_quat_(c2o_rot_);

//    cout << "c2o_trans:\n" << c2o_trans << "\nc2o_euler:\n"<< c2o_rot.eulerAngles(2,1,0) << endl << endl;

    // 从外参中获取旋转向量
    std::vector<double> handEyeAxisAngle;
    if (leftOrRight == 0) { // 左臂
//        handEyeAxisAngle = {1.6493674939539318e+00,
//                            2.1497733612005401e-01,
//                            9.4954804224349210e-01,
//                            -2.2834898823134014e-01}; // 外参rotation_vector：[Angle, AxisX, AxisY, AxisZ] 单位rad
        handEyeAxisAngle = {1.64979,
                            0.226241,
                            0.949886,
                            -0.215711}; // 外参rotation_vector：[Angle, AxisX, AxisY, AxisZ] 单位rad
    } else if (leftOrRight == 1) { // 右臂
        handEyeAxisAngle = {1.5758424729195439e+00,
                            2.2768667381355578e-01,
                            -9.4690716445454615e-01,
                            2.2699251193017475e-01}; // 外参rotation_vector：[Angle, AxisX, AxisY, AxisZ] 单位rad
    }

    // 从外参中获取平移向量
    std::vector<double> handEyeTranslation;
    if (leftOrRight == 0) { // 左臂
//        handEyeTranslation = {-1.9141241908073422e+02, -9.3745023012161283e+01, -3.1072884798049927e+02}; // 外参translation_vector：[x, y ,z] 单位mm
        // 0815 新
        handEyeTranslation = {-111.763, -130.0, -285.0}; // 外参translation_vector：[x, y ,z] 单位mm
    } else if (leftOrRight == 1) { // 右臂
        handEyeTranslation = {1.8252700567245483e+02, -8.7103784084320054e+01, -1.2175752222537994e+02}; // 外参translation_vector：[x, y ,z] 单位mm
    }

    // 旋转向量
    Eigen::AngleAxisd b2c_rvec(handEyeAxisAngle[0], Eigen::Vector3d(handEyeAxisAngle[1], handEyeAxisAngle[2], handEyeAxisAngle[3]));
    Eigen::Matrix3d b2c_rot = b2c_rvec.matrix();
    // 平移向量
    Eigen::Vector3d b2c_trans { handEyeTranslation[0]/1000, handEyeTranslation[1]/1000, handEyeTranslation[2]/1000 };

    /// 基坐标系->相机 齐次变换矩阵
    Eigen::Isometry3d b2c_T = Eigen::Isometry3d::Identity();
    b2c_T.rotate(b2c_rot);
    b2c_T.pretranslate(b2c_trans);

//    cout << "b2c_trans:\n" << b2c_trans << "\nb2c_euler:\n"<< b2c_rot.eulerAngles(2,1,0) <<
//            "\nb2c_rot:\n" << b2c_rot << endl << endl;

    /// 基坐标系->物体 齐次变换矩阵
    Eigen::Isometry3d b2o_T;
    b2o_T = b2c_T * c2o_T;

    Eigen::Vector3d b2o_trans = b2o_T.translation();
    Eigen::Matrix3d b2o_rot = b2o_T.rotation();
    Eigen::Quaterniond b2o_quat(b2o_rot);

//    cout << "b2o_trans:\n" << b2o_trans << "\nb2o_euler:\n"<< b2o_rot.eulerAngles(2,1,0) <<
//            "\nb2o_rot:\n" << b2o_rot << endl << endl;

    std::vector<double> b2oXYZRPY = ConvertTransform2XYZRPY(b2o_T); // 获取笛卡尔空间姿态

    return b2oXYZRPY;
}

bool GraphicsGrasp::getObjPose(cv::RotatedRect& RotRect, std::vector<double> &b2oXYZRPY,
            const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud, int juggleOrCube, int longOrshort, int leftOrRight) {
    printf("\n[FUNC] Get Object Poses ...\n");
    int row = 0, col = 0;
    float center_x = 0, center_y = 0, center_z = 0;
    float center_angle = RotRect.angle;

    if (juggleOrCube == 0) { // 积木
        if (RotRect.size.area() < 25) { // 为空
            printf("\033[0;31m%s\033[0m\n", "[ERRO] RotRect invalid for getObjPose!\n");
            return false;
        }

        row = (int)RotRect.center.y;
        col = (int)RotRect.center.x;

        if(!getPointLoc(row, col, center_x, center_y, center_z, cloud)) return false;

    } else if (juggleOrCube == 1) { // 正方体
        if (RotRect.size.area() < 25) { // 为空
            printf("\033[0;31m%s\033[0m\n", "[ERRO] RotRect invalid for getObjPose!\n");
            return false;
        }

        float x, y, z; // 不准确的中心位置
        float x1, y1, z1; // 实际位置1
        float x2, y2, z2; // 实际位置2
        // 获取外接矩形四个角点
        cv::Point2f P[4];
        RotRect.points(P);

        cv::Point2f P1; // 对角线1/4处点
        P1.x = P[0].x + (P[2].x - P[0].x)/8;
        P1.y = P[0].y + (P[2].y - P[0].y)/8;

        cv::Point2f P2; // 对角线3/4处点
        P2.x = P[0].x + (P[2].x - P[0].x)*7/8;
        P2.y = P[0].y + (P[2].y - P[0].y)*7/8;

        // 获取实际位置1
        row = (int)P1.y;
        col = (int)P1.x;
        if (!getPointLoc(row, col, x1, y1, z1, cloud)) return false;
        // 获取实际位置2
        row = (int)P2.y;
        col = (int)P2.x;
        if (!getPointLoc(row, col, x2, y2, z2, cloud)) return false;

        // 获取不准确的中心位置
        row = (int)RotRect.center.y;
        col = (int)RotRect.center.x;
        if (!getPointLoc(row, col, x, y, z, cloud)) return false;

        printf("[INFO] Big cube raw Center[%f %f %f]\n", x, y, z);
        printf("[INFO] Big cube correct P1[%f %f %f] P2[%f %f %f]\n", x1, y1, z1, x2, y2, z2);

        center_x = x1 + (x2 - x1)/2;
        center_y = y1 + (y2 - y1)/2;
        center_z = (z1 + z2)/2;
    }

    printf("[INFO] Center angle raw: %f\n", center_angle);

    // angle为与width边的夹角, width > height 时angle为与长边夹角, 角度为顺时针方向, 一般为负, 即逆时针旋转, 基准轴为水平向右
    if (longOrshort == 0) { /// 转换为与长边夹角
        if (RotRect.size.width < RotRect.size.height) {
            center_angle -= 90; // 转换为长边的角度
        }
    } else if (longOrshort == 1) { /// 转换为与短边夹角
        if (RotRect.size.width > RotRect.size.height) {
            center_angle -= 90; // 转换为长边的角度
        }
    }

    printf("[INFO] Center angle correct: %f\n", center_angle);

    // 归一化到[-90, 90], 走最近的角度
    if (center_angle < -90) center_angle += 180;
    else if (center_angle > 90) center_angle -= 180;

    printf("[INFO] Center angle short: %f\n", center_angle);

    printf("[INFO] Center(相机坐标系) [row:%d col:%d] x:%f y:%f z:%f Angle:%f\n", row, col, center_x, center_y, center_z, center_angle);

    std::vector<float> coorRaw = {center_x, center_y, center_z};
    b2oXYZRPY = calcRealCoor(coorRaw, leftOrRight); // 计算基坐标到物体转换关系

    printf("[INFO] 基坐标系->物体: [%f,%f,%f,%f,%f,%f]\n", b2oXYZRPY[0], b2oXYZRPY[1],
           b2oXYZRPY[2], b2oXYZRPY[3], b2oXYZRPY[4], b2oXYZRPY[5]);

    /// 修改姿态
    float coeff = 1.0; // 角度校正系数
    if (center_angle > 90) coeff = 0.75;
    else if (center_angle < 90) coeff = 1.25;
    
    if (leftOrRight == 0) {
        b2oXYZRPY[3] = 1.54;
        b2oXYZRPY[4] = D2R(center_angle * coeff);  // 左臂正值容易到达, 加180度 FIXME:当前未处理
        b2oXYZRPY[5] = 1.54;
    } else if (leftOrRight == 1) {
        b2oXYZRPY[3] = 1.54;
        b2oXYZRPY[4] = D2R(center_angle * coeff);  // // 右臂负值更容易到达, 不处理
        b2oXYZRPY[5] = -1.54;
    }

    printf("[INFO] 基坐标系->物体(修改姿态): [%f,%f,%f,%f,%f,%f]\n", b2oXYZRPY[0], b2oXYZRPY[1],
           b2oXYZRPY[2], b2oXYZRPY[3], b2oXYZRPY[4], b2oXYZRPY[5]);

    return true;
}

bool GraphicsGrasp::getPointLoc(int row, int col, float &loc_x, float &loc_y, float &loc_z,
                                                        const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud) {

    // 获取外接矩形实际位置
    loc_x = cloud->points[row * cloud->width + col].x;
    loc_y = cloud->points[row * cloud->width + col].y;
    loc_z = cloud->points[row * cloud->width + col].z;

    /// 深度值不可用, 附近寻找可用深度
    if (loc_z < 0.1 || loc_z > 2.0 || qIsNaN(loc_z)) {
        const int size = 5; // 深度寻找范围
        // 右下侧
        for (int i = 0; i <= size; i++) {
            if (loc_z > 0.1 && loc_z < 2.0) break;
            for (int j = 0; j <= size; j++) {
                loc_z = cloud->points[(row + i) * cloud->width + (col + j)].z;
//                printf("Find loc z: %f\n", loc_z);
                if (loc_z > 0.1 && loc_z < 2.0) break;
            }
        }
        // 左下侧
        for (int i = 0; i <= size; i++) {
            if (loc_z > 0.1 && loc_z < 2.0) break;
            for (int j = 0; j >= -size; j--) {
                loc_z = cloud->points[(row + i) * cloud->width + (col + j)].z;
//                printf("Find loc z: %f\n", loc_z);
                if (loc_z > 0.1 && loc_z < 2.0) break;
            }
        }
        // 右上侧
        for (int i = 0; i >= -size; i--) {
            if (loc_z > 0.1 && loc_z < 2.0) break;
            for (int j = 0; j <= size; j++) {
                loc_z = cloud->points[(row + i) * cloud->width + (col + j)].z;
//                printf("Find loc z: %f\n", loc_z);
                if (loc_z > 0.1 && loc_z < 2.0) break;
            }
        }
        // 左上侧
        for (int i = 0; i >= -size; i--) {
            if (loc_z > 0.1 && loc_z < 2.0) break;
            for (int j = 0; j >= -size; j--) {
                loc_z = cloud->points[(row + i) * cloud->width + (col + j)].z;
//                printf("Find loc z: %f\n", loc_z);
                if (loc_z > 0.1 && loc_z < 2.0) break;
            }
        }
    }

    /// 未寻找到可用深度
    if (loc_z < 0.1 || loc_z > 2.0 || qIsNaN(loc_z)) {
        printf("[WARN] row: %d col: %d loc_z: %f\n", row, col, loc_z);
        printf("\033[0;31m[WARN] Center point's depth is not valid!\033[0m\n");
        return false;
//        throw std::runtime_error("\033[0;31mCenter point's depth is not valid!\033[0m\n");
    }

    return true;
}

void GraphicsGrasp::showWorkArea(cv::Mat &image) {
    cv::Mat frame;

    cv::resize(image, frame, cv::Size(960, 540)); // 缩小图片

    cv::Point LU(LU_[0], LU_[1]); // 桌面区域左上角点 (x, y)=(col, row)
    cv::Point RD(RD_[0], RD_[1]); // 桌面区域右下角点 (x, y)=(col, row)

    // 左
    cv::Point AreaLeftLU(LU_[0], LU_[1]); // 桌面区域左上角点 (x, y)=(col, row)
    cv::Point AreaLeftRD(WorkAreaThreshL, RD_[1]); // 桌面区域右下角点 (x, y)=(col, row)
    // 中
    cv::Point AreaMiddleLU(WorkAreaThreshL, LU_[1]); // 桌面区域左上角点 (x, y)=(col, row)
    cv::Point AreaMiddleRD(WorkAreaThreshR, RD_[1]); // 桌面区域右下角点 (x, y)=(col, row)
    // 右
    cv::Point AreaRightLU(WorkAreaThreshL, LU_[1]); // 桌面区域左上角点 (x, y)=(col, row)
    cv::Point AreaRightRD(RD_[0], RD_[1]); // 桌面区域右下角点 (x, y)=(col, row)

    // 左右分割框
    cv::Point AreaLeftRightLU(LU_[0], LU_[1]); // 桌面区域左上角点 (x, y)=(col, row)
    cv::Point AreaLeftRightRD(LeftOrRightThresh, RD_[1]); // 桌面区域右下角点 (x, y)=(col, row)

    rectangle(frame, LU, RD, cv::Scalar(255, 178, 50), 1);
    rectangle(frame, AreaLeftLU, AreaLeftRD, cv::Scalar(255, 178, 50), 1);
    rectangle(frame, AreaMiddleLU, AreaMiddleRD, cv::Scalar(255, 178, 50), 1);
    rectangle(frame, AreaRightLU, AreaRightRD, cv::Scalar(255, 178, 50), 1);
    rectangle(frame, AreaLeftRightLU, AreaLeftRightRD, cv::Scalar(255, 178, 50), 1);

    cv::Rect rect(cv::Point(LU_[0], LU_[1]), cv::Size(RD_[0]-LU_[0], RD_[1]-LU_[1]));
    rectangle(frame, rect, cv::Scalar(0, 0, 255), 1);

    cv::imshow("Work Area", frame);
    cv::waitKey(0);
}

void GraphicsGrasp::createPointCloud(cv::Mat &color, cv::Mat &depth, const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud) {

    const float badPoint = std::numeric_limits<float>::quiet_NaN();

#pragma omp parallel for
    for(int r = 0; r < depth.rows/2; ++r)
    {
        pcl::PointXYZRGBA *itP = &cloud->points[r * depth.cols/2];
        const auto *itD = depth.ptr<uint16_t>(r*2);
        const auto *itC = color.ptr<cv::Vec3b>(r*2);
        const float y = lookupY.at<float>(0, r*2);
        const float *itX = lookupX.ptr<float>();

        for(int c = 0; c < depth.cols/2; ++c, ++itP, itD+=2, itC+=2, itX+=2)
        {
            const float depthValue = (float)*itD/ 1000.0f;

            // Check for invalid measurements
            if(*itD == 0 || *itD > 1000)
            {
                // not valid
                itP->x = itP->y = itP->z = badPoint;
                itP->rgba = 0;
                continue;
            }
            itP->z = depthValue;
            itP->x = *itX * depthValue;
            itP->y = y * depthValue;
            itP->b = itC->val[0];
            itP->g = itC->val[1];
            itP->r = itC->val[2];
            itP->a = 255;
        }
    }
}

void GraphicsGrasp::createLookup(size_t width, size_t height)
{
    cv::Mat cameraMatrixColor;
    cameraMatrixColor = cv::Mat::zeros(3, 3, CV_64F);

    readCameraInfo(cameraMatrixColor);

    const float fx = 1.0f / cameraMatrixColor.at<double>(0, 0);
    const float fy = 1.0f / cameraMatrixColor.at<double>(1, 1);
    const float cx = cameraMatrixColor.at<double>(0, 2);
    const float cy = cameraMatrixColor.at<double>(1, 2);
    float *it;

    lookupY = cv::Mat(1, height, CV_32F);
    it = lookupY.ptr<float>();
    for(size_t r = 0; r < height; ++r, ++it)
    {
        *it = (r - cy) * fy;
    }

    lookupX = cv::Mat(1, width, CV_32F);
    it = lookupX.ptr<float>();
    for(size_t c = 0; c < width; ++c, ++it)
    {
        *it = (c - cx) * fx;
    }
}

void GraphicsGrasp::readCameraInfo(cv::Mat &cameraMatrix)
{
//    std::vector<double> cameraMatrixVec = {1.0751836750739103e+03, 0., 9.9204536064492709e+02,
//                                           0., 1.0787798824980591e+03, 5.5685612287788467e+02,
//                                           0., 0., 1.};
    std::vector<double> cameraMatrixVec = {1.0817351755988568e+03, 0., 9.4219086287825303e+02,
                                           0., 1.0823131063554199e+03, 5.7383408949778186e+02,
                                           0., 0., 1.};

    double *itC = cameraMatrix.ptr<double>(0, 0);
    for(size_t i = 0; i < 9; ++i, ++itC)
    {
        *itC = cameraMatrixVec[i];
    }
}
