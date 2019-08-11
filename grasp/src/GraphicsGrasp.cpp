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
    std::string yolo_weights_filename ="../../../grasp/data/yolov3-voc_23000.weights";
}

GraphicsGrasp::GraphicsGrasp()
{
    /// gpd 初始化
//    grasp_detector_ = new gpd::GraspDetectorPointNet(gpd_cfg_file); // FIXME

    /// yolo 初始化
    yoloDetector = new YoloLibtorchDetector(yolo_config_filename, yolo_weights_filename);
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

std::pair<std::vector<cv::RotatedRect>, std::vector<int>> GraphicsGrasp::detectGraspYolo(cv::Mat &image)
{
    cv::Mat resized;
    std::pair<std::vector<cv::RotatedRect>, std::vector<int>> RotatedRectsAndID;

    cv::resize(image, resized, cv::Size(960, 540)); // 缩小图片

    RotatedRectsAndID = yoloDetector->getRotRectsAndID(resized);

    printf("[INFO] Detected %zu rotated rects.\n", RotatedRectsAndID.first.size());

    return RotatedRectsAndID;
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

std::vector<double> GraphicsGrasp::getObjPose(const cv::RotatedRect& RotRect,
                                    const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud, int leftOrRight) {
    // 获取外接矩形实际位置
    int row = (int)RotRect.center.y;
    int col = (int)RotRect.center.x;
    float center_x = cloud->points[row * cloud->width + col].x;
    float center_y = cloud->points[row * cloud->width + col].y;
    float center_z = cloud->points[row * cloud->width + col].z;
    float center_angle = RotRect.angle;

    // 中心点附近寻找可用深度
    for (int i = -5; i <= 5; i++) {
        if (center_z > 0.1 && center_z < 2.0) break;
        for (int j = -5; j <= 5; j++) {
            center_z = cloud->points[(row+i) * cloud->width + (col+j)].z;
            printf("Find center z: %f\n", center_z);
            if (center_z > 0.1 && center_z < 2.0) break;
        }
    }

    if (center_z < 0.1 || center_z > 2.0) {
        printf("row: %d col: %d center_z: %f\n", row, col, center_z);
        throw std::runtime_error("\033[0;31mCenter point's depth is not valid!\033[0m\n");
    }

    // angle为与width边的夹角, width > height 时angle为与长边夹角
    if (RotRect.size.width < RotRect.size.height) {
        center_angle += 90; // 转换为长边的角度
    }

    printf("[INFO] Center [row:%d col:%d] x:%f y:%f z:%f Angle:%f\n", row, col, center_x, center_y, center_z, center_angle);

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

    return b2oXYZRPY;
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
    std::vector<double> cameraMatrixVec = {1.0751836750739103e+03, 0., 9.9204536064492709e+02,
                                           0., 1.0787798824980591e+03, 5.5685612287788467e+02,
                                           0., 0., 1.};

    double *itC = cameraMatrix.ptr<double>(0, 0);
    for(size_t i = 0; i < 9; ++i, ++itC)
    {
        *itC = cameraMatrixVec[i];
    }
}
