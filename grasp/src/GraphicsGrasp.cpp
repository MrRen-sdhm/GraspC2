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
//    grasp_detector_ = new gpd::GraspDetectorPointNet(gpd_cfg_file); // FIXME

    /// yolo 初始化
    yoloDetector = new YoloDetector(yolo_config_filename, yolo_weights_filename);
}

// FIXME
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

std::pair<std::vector<cv::RotatedRect>, std::vector<int>> GraphicsGrasp::detectGraspYolo(cv::Mat &image, int thresh, bool show)
{
    cv::Mat resized;
    std::pair<std::vector<cv::RotatedRect>, std::vector<int>> RotatedRectsAndID;

    cv::resize(image, resized, cv::Size(960, 540)); // 缩小图片

    RotatedRectsAndID = yoloDetector->getRotRectsAndID(resized, thresh, show);

    printf("[INFO] Detected %zu rotated rects.\n", RotatedRectsAndID.first.size());

    return RotatedRectsAndID;
}

std::pair<std::vector<cv::RotatedRect>, std::vector<int>> GraphicsGrasp::detectBigCube(cv::Mat &image, int thresh, bool show)
{
    cv::Mat frame;
    std::pair<std::vector<cv::RotatedRect>, std::vector<int>> RotatedRectsAndID;

    cv::resize(image, frame, cv::Size(960, 540)); // 缩小图片

    std::vector<cv::RotatedRect> RotatedRects;
    std::vector<int> RectsID;
    cv::Mat frame_copy = frame.clone();

    cv::Mat img_hsv;
    cv::cvtColor(frame, img_hsv, CV_BGR2HSV);
    if(show) cv::imshow("hsv", img_hsv);

    /// HSV阈值分割获取掩码
    int thresh_v_high = thresh; // V通道阈值
    cv::Point LU(100, 240); // 桌面区域左上角点 (x, y)=(col, row)
    cv::Point RD(730, 540); // 桌面区域右下角点 (x, y)=(col, row)
    cv::Mat mask = cv::Mat::zeros(img_hsv.rows, img_hsv.cols, CV_8U); // 掩码
    for(int r = 0; r < img_hsv.rows; ++r)
    {
        auto *itM = mask.ptr<uint8_t>(r);
        const cv::Vec3b *itH = img_hsv.ptr<cv::Vec3b>(r);

        for(int c = 0; c < img_hsv.cols; ++c, ++itM, ++itH)
        {
            if (r > LU.y && c > LU.x && c < RD.x) { /// 限定像素范围
                if (itH->val[0] < 40 && itH->val[2] > thresh_v_high) { /// HSV阈值分割 顶部>255 旁边>120
                    *itM = 255;
                }
            }
        }
    }

    if(show) cv::imshow("mask", mask);

    std::vector<cv::RotatedRect> rotRects;
    cv::Rect box(cv::Point(0, 0), cv::Size(0, 0));

    /// 计算最小外接矩形
    if (yoloDetector->calRotatedRect(frame, mask, box, rotRects, 1, show)) {
        for (const auto & rotRect : rotRects) {
            RotatedRects.push_back(rotRect); // 存储外接矩形
            RectsID.push_back(6); // 存储外接矩形对应的物体类别
        }

        printf("[INFO] Detected %zu big cube.\n", rotRects.size());
    }

    if(show) {
        for (size_t i = 0; i < RotatedRects.size(); i++) {
            std::cout << "[INFO] minAreaRectOut" << i << " center:" << RotatedRects[i].center << " angle: " <<
                                                RotatedRects[i].angle << " size: " << RotatedRects[i].size << std::endl;
        }
    }

    rectangle(frame_copy, LU, RD, cv::Scalar(255, 178, 50), 1);

    if(show) cv::imshow("main area", frame_copy);
    if(show) cv::waitKey(0);

    std::pair<std::vector<cv::RotatedRect>, std::vector<int>> RectsAndID {RotatedRects, RectsID};

    return RectsAndID;
}

std::vector<int> GraphicsGrasp::findAimObjLR(std::pair<std::vector<cv::RotatedRect>, std::vector<int>> RotRectsAndID,
                                             int LeftOrRightThresh, int RowOrCol) {
    std::vector<int> LIndices, RIndices;
    std::vector<float> LCenters, RCenters;
    for (size_t i = 0; i < RotRectsAndID.first.size(); i++) {
        float center_x = RotRectsAndID.first[i].center.x;
        float center_y = RotRectsAndID.first[i].center.y;
        if (center_x < LeftOrRightThresh) { // 先区分左右
            if (RowOrCol == 1) { // 存储列值, 用于找横向最值
                printf("LeftCenter[%zu]: %f\n", i, center_x);
                LCenters.push_back(center_x);
            } else if (RowOrCol == 0) { // 存储行值, 用于找纵向最值
                printf("LeftCenter[%zu]: %f\n", i, center_y);
                LCenters.push_back(center_y);
            }
            LIndices.push_back(i);
        } else {
            if (RowOrCol == 1) { // 存储列值, 用于找横向最值
                printf("RightCenter[%zu]: %f\n", i, center_x);
                RCenters.push_back(center_x);
            } else if (RowOrCol == 0) { // 存储行值, 用于找纵向最值
                printf("RightCenter[%zu]: %f\n", i, center_y);
                RCenters.push_back(center_y);
            }
            RIndices.push_back(i);
        }
    }

    cout << "leftCenters: " << LCenters << endl;
    cout << "LIndices: " << LIndices << endl;
    cout << "rightCenters: " << RCenters << endl;
    cout << "RIndices: " << RIndices << endl;

    // 左侧找最小的
    auto min_LCenter = std::min_element(LCenters.begin(), LCenters.end());
    auto distanceL = std::distance(LCenters.begin(), min_LCenter);
    int positionRawL = LIndices[distanceL]; // 在RotRectsAndID中的位置
    std::cout << "LeftCenter Min element is " << *min_LCenter<< " at position " << positionRawL << std::endl;

    // 右侧找最大的
    auto max_RCenter = std::max_element(RCenters.begin(), RCenters.end());
    auto distanceR = std::distance(RCenters.begin(), max_RCenter);
    int positionRawR = RIndices[distanceR]; // 在RotRectsAndID中的位置
    std::cout << "RightCenter Max element is " << *max_RCenter<< " at position " << positionRawR << std::endl;

    std::vector<int> AimObjIndicesLR = {positionRawL, positionRawR};
    return AimObjIndicesLR;
}

std::vector<double> GraphicsGrasp::calcRealCoor(const Eigen::Matrix3d& rotMatrix,
                                                                    const Eigen::Vector3d& translation, int leftOrRight)
{
    /// 相机->物体 齐次变换矩阵
    Eigen::Isometry3d c2o_T = Eigen::Isometry3d::Identity();
    c2o_T.rotate(rotMatrix);
    c2o_T.pretranslate(translation);

    Eigen::Vector3d c2o_trans = c2o_T.translation();
    Eigen::Matrix3d c2o_rot = c2o_T.rotation();
    Eigen::Quaterniond c2o_quat(c2o_rot);

//    cout << "c2o_trans:\n" << c2o_trans << "\nc2o_euler:\n"<< c2o_rot.eulerAngles(2,1,0) << endl << endl;

    // 从外参中获取旋转向量
    std::vector<double> handEyeAxisAngle;
    if (leftOrRight == 0) { // 左臂
        handEyeAxisAngle = {1.6493674939539318e+00,
                            2.1497733612005401e-01,
                            9.4954804224349210e-01,
                            -2.2834898823134014e-01}; // 外参rotation_vector：[Angle, AxisX, AxisY, AxisZ] 单位rad

        // 0517 新
//        handEyeAxisAngle = {1.61536e+00,
//                            1.79356e-01,
//                            9.63044e-01,
//                            -2.00943e-01};
    } else if (leftOrRight == 1) { // 右臂
        handEyeAxisAngle = {1.5758424729195439e+00,
                            2.2768667381355578e-01,
                            -9.4690716445454615e-01,
                            2.2699251193017475e-01}; // 外参rotation_vector：[Angle, AxisX, AxisY, AxisZ] 单位rad
    }

    // 从外参中获取平移向量
    std::vector<double> handEyeTranslation;
    if (leftOrRight == 0) { // 左臂
        handEyeTranslation = {-1.9141241908073422e+02, -9.3745023012161283e+01, -3.1072884798049927e+02}; // 外参translation_vector：[x, y ,z] 单位mm
        // 0517 新
//        handEyeTranslation = {-1.779e+02, -17.829e+01, -3.22e+02};
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

std::vector<double> GraphicsGrasp::getObjPose(cv::RotatedRect& RotRect,
            const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud, int juggleOrCube, int longOrshort, int leftOrRight) {
    int row = 0, col = 0;
    float center_x = 0, center_y = 0, center_z = 0;
    float center_angle = RotRect.angle;

    if (juggleOrCube == 0) { // 积木
        row = (int)RotRect.center.y;
        col = (int)RotRect.center.x;
        getPointLoc(row, col, center_x, center_y, center_z, cloud);

    } else if (juggleOrCube == 1) { // 正方体
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
        getPointLoc(row, col, x1, y1, z1, cloud);
        // 获取实际位置2
        row = (int)P2.y;
        col = (int)P2.x;
        getPointLoc(row, col, x2, y2, z2, cloud);

        // 获取不准确的中心位置
        row = (int)RotRect.center.y;
        col = (int)RotRect.center.x;
        getPointLoc(row, col, x, y, z, cloud);

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

    /// 物体姿态转换到机器人坐标系下
    Eigen::Vector3d c2o_ea(0, 0, 0); // YPR, 先绕z轴yaw, 再绕y轴pitch, 最后绕x轴roll
    Eigen::Quaterniond c2o_quat;
    c2o_quat = Eigen::AngleAxisd(c2o_ea[0], Eigen::Vector3d::UnitZ()) *
               Eigen::AngleAxisd(c2o_ea[1], Eigen::Vector3d::UnitY()) *
               Eigen::AngleAxisd(c2o_ea[2], Eigen::Vector3d::UnitX());

    Eigen::Matrix3d c2o_rot = c2o_quat.matrix(); // 相机到物体旋转矩阵
    Eigen::Vector3d c2o_trans (center_x, center_y, center_z); // 相机到物体平移矩阵

    std::vector<double> b2oXYZRPY; // 基坐标到物体
    b2oXYZRPY = calcRealCoor(c2o_rot, c2o_trans, leftOrRight); // 计算基坐标到物体转换关系

    printf("[INFO] 基坐标系->物体: [%f,%f,%f,%f,%f,%f]\n", b2oXYZRPY[0], b2oXYZRPY[1],
           b2oXYZRPY[2], b2oXYZRPY[3], b2oXYZRPY[4], b2oXYZRPY[5]);

    /// 修改姿态
    if (leftOrRight == 0) {
        b2oXYZRPY[3] = -1.54;
        b2oXYZRPY[4] = D2R(center_angle + 0);  // 左臂正值容易到达, 加180度 FIXME:当前未处理
        b2oXYZRPY[5] = 1.54;
    } else if (leftOrRight == 1) {
        b2oXYZRPY[3] = 1.54;
        b2oXYZRPY[4] = D2R(center_angle);  // // 右臂负值更容易到达, 不处理
        b2oXYZRPY[5] = -1.54;
    }

    printf("[INFO] 基坐标系->物体(修改姿态): [%f,%f,%f,%f,%f,%f]\n", b2oXYZRPY[0], b2oXYZRPY[1],
           b2oXYZRPY[2], b2oXYZRPY[3], b2oXYZRPY[4], b2oXYZRPY[5]);

    return b2oXYZRPY;
}

void GraphicsGrasp::getPointLoc(int row, int col, float &loc_x, float &loc_y, float &loc_z,
                                                        const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud) {
    // 获取外接矩形实际位置
    loc_x = cloud->points[row * cloud->width + col].x;
    loc_y = cloud->points[row * cloud->width + col].y;
    loc_z = cloud->points[row * cloud->width + col].z;

    /// 深度值不可用, 附近寻找可用深度
    if (loc_z < 0.1 || loc_z > 2.0) {
        const int size = 5; // 深度寻找范围
        // 右下侧
        for (int i = 0; i <= size; i++) {
            if (loc_z > 0.1 && loc_z < 2.0) break;
            for (int j = 0; j <= size; j++) {
                loc_z = cloud->points[(row + i) * cloud->width + (col + j)].z;
                printf("Find loc z: %f\n", loc_z);
                if (loc_z > 0.1 && loc_z < 2.0) break;
            }
        }
        // 左下侧
        for (int i = 0; i <= size; i++) {
            if (loc_z > 0.1 && loc_z < 2.0) break;
            for (int j = 0; j >= -size; j--) {
                loc_z = cloud->points[(row + i) * cloud->width + (col + j)].z;
                printf("Find loc z: %f\n", loc_z);
                if (loc_z > 0.1 && loc_z < 2.0) break;
            }
        }
        // 右上侧
        for (int i = 0; i >= -size; i--) {
            if (loc_z > 0.1 && loc_z < 2.0) break;
            for (int j = 0; j <= size; j++) {
                loc_z = cloud->points[(row + i) * cloud->width + (col + j)].z;
                printf("Find loc z: %f\n", loc_z);
                if (loc_z > 0.1 && loc_z < 2.0) break;
            }
        }
        // 左上侧
        for (int i = 0; i >= -size; i--) {
            if (loc_z > 0.1 && loc_z < 2.0) break;
            for (int j = 0; j >= -size; j--) {
                loc_z = cloud->points[(row + i) * cloud->width + (col + j)].z;
                printf("Find loc z: %f\n", loc_z);
                if (loc_z > 0.1 && loc_z < 2.0) break;
            }
        }
    }

    /// 未寻找到可用深度
    if (loc_z < 0.1 || loc_z > 2.0) {
        printf("row: %d col: %d loc_z: %f\n", row, col, loc_z);
        throw std::runtime_error("\033[0;31mCenter point's depth is not valid!\033[0m\n");
    }
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
