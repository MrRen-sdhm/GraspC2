//
// Created by hj on 18-12-29.
//

#ifndef MAGICCUBE_GRAPHICSCUBE_H
#define MAGICCUBE_GRAPHICSCUBE_H
#include <opencv2/opencv.hpp>
#include <QString>
#include <vision/Calibration/interface/CobotCalibrationData.h>
#include <vision/Calibration/interface/CalToolkit.h>

#include <pcl/io/pcd_io.h>

// GPD
//#include <gpd/util/cloud.h>
//#include <gpd/grasp_detector_pointnet.h>

// YOLO
#include "YoloDetector.h"

#define R2D(rad) ((rad) * 180.0f/(float)M_PI)
#define D2R(deg) ((deg) * (float)M_PI/180.0f)

namespace {
    //矩阵转化为笛卡尔坐标
    auto ConvertTransform2XYZRPY = [](const ::Eigen::Isometry3d &transform) {

        Eigen::Vector3d translation = transform.translation();
        Eigen::Matrix3d rotation = transform.rotation();
        Eigen::Vector3d rpy = rotation.eulerAngles(2, 1, 0);

        std::vector<double> xyzRpy = {translation[0], translation[1], translation[2],
                                      rpy[2], rpy[1], rpy[0]};

        return xyzRpy;
    };

    //笛卡尔坐标转化为矩阵
    auto ConvertXYZRPY2Transform = [](std::vector<double> xyzRpy) {

        Eigen::Vector3d translation(xyzRpy[0], xyzRpy[1], xyzRpy[2]);

        Eigen::Matrix3d rotation = Eigen::AngleAxisd(xyzRpy[5], Eigen::Vector3d::UnitZ()) *
                                   Eigen::AngleAxisd(xyzRpy[4], Eigen::Vector3d::UnitY()) *
                                   Eigen::AngleAxisd(xyzRpy[3], Eigen::Vector3d::UnitX()).matrix();

        Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
        transform.rotate(rotation);
        transform.pretranslate(translation);

        return transform;
    };
}

template <class Type>
Type stringToNum(const string& str)
{
    istringstream iss(str);
    Type num;
    iss >> num;
    return num;
}

class GraphicsGrasp {
public:
    GraphicsGrasp();

    // show: 1-全部显示 2-逐个显示 其他-不显示
    std::pair<std::vector<cv::RotatedRect>, std::vector<int>> detectGraspYolo(cv::Mat &image, int thresh, int show);

    // show: 1-全部显示 2-逐个显示 其他-不显示
    std::pair<std::vector<cv::RotatedRect>, std::vector<int>> detectGraspYoloPro(cv::Mat &image, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud, int thresh, int show);

    std::pair<std::vector<cv::RotatedRect>, std::vector<int>> getRotRectsAndID(cv::Mat &image,
                                                std::vector<int> &classIds, std::vector<float> &confidences,
                                                std::vector<cv::Rect> &boxes, const cv::Rect& rect, int thresh, int show);

    std::pair<std::vector<cv::RotatedRect>, std::vector<int>> detectJuggles(cv::Mat &image, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud,
                                               std::vector<int> &classIds, std::vector<float> &confidences,
                                               std::vector<cv::Rect> &boxes, const cv::Rect& rect, int thresh, int show);

    bool calRotatedRect(cv::Mat img_roi, cv::Mat mask, const cv::Rect& box,
                                       std::vector<cv::RotatedRect> &rotRects, int objLev, int show);

    /// objLev: 0-无高度检测 1-中等物体，有高度检测 2-超大物体，有高度检测 threshColor:颜色阈值 threshLo:高度阈值
    std::pair<std::vector<cv::RotatedRect>, std::vector<int>> detectBigObj(cv::Mat &image,
            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud, int objLev, int threshColor, float threshLoc, int show);

    bool detectBigBall(cv::Mat &image, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud,
                                std::pair<cv::RotatedRect, int> &BigRotRectsAndID, int show); // 1为检测到, -1 为未检测到

    bool detectBigCube(cv::Mat &image, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud,
                                std::pair<cv::RotatedRect, int> &BigRotRectsAndID, int show); // 1为检测到, -1 为未检测到
    /**
     * calcRealCoor  输入kinect点云图像抓取姿态，依据手眼标定结果，输出手臂基坐标系下抓取姿态
     * @param rotMatrix  相机坐标系下抓取姿态的旋转矩阵
     * @param position  相机坐标系下抓取姿态的平移向量
     * @param leftOrRight  0: 左臂 1：右臂
     * @return 手臂基坐标系下抓取姿态(Eigen::Isometry3d格式)
     */
    static std::vector<double> calcRealCoor(std::vector<float> coorRaw, int leftOrRight);

    /// 寻找左右侧的目标物体, 左侧找最左/上边的, 右侧找最右/下边的, RowOrCol: 1为左右最值 0为上下最值
    std::vector<int> findAimObjLR(std::pair<std::vector<cv::RotatedRect>, std::vector<int>> RotRectsAndID, int RowOrCol);

    /// 获取物体姿态和ID towPointOrNot: 0-单点计算坐标 1-两点计算平均坐标 longOrshort: 0为长边 leftOrRight: 0为左臂
    bool getObjPose(cv::RotatedRect& RotRect, std::vector<double> &b2oXYZRPY,
            const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud, int towPointOrNot, int longOrshort, int leftOrRight);

    bool getPointLoc (int row, int col, float &loc_x, float &loc_y, float &loc_z,
                             const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud);

    /**
     * \brief Detect grasp poses in a point cloud.
     * \return the list of grasp poses
    */
//    std::vector<std::unique_ptr<gpd::candidate::Hand>> detectGraspPoses(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud); //FIXME

    /// 显示工作区域
    void showWorkArea(cv::Mat &image);

    /// 创建点云 缩小一倍 color(1920*1080) depth(1920*1080) cloud(960*540)
    void createPointCloud(cv::Mat &color, cv::Mat &depth, const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud);

    void createLookup(size_t width, size_t height);

    static void readCameraInfo(cv::Mat &cameraMatrix);

public:
    /// 工作区域划分
    std::vector<int> LU_ = {140, 230}; // 桌面区域左上角点 (x, y)=(col, row)
    std::vector<int> RD_ = {680, 490}; // 桌面区域右下角点 (x, y)=(col, row)
    const float LeftOrRightThresh = 400.0; // 左右臂分工阈值, 列数小于阈值为左臂管辖
    const float WorkAreaThreshL = 280.0; // 左侧工作区域分割阈值
    const float WorkAreaThreshR = 525.0; // 右侧工作区域分割阈值

private:
    const double areaThresh = 1300.0; // 积木与立方体轮廓面积区分阈值
    /// NOTE：值越小离桌面越远
    const float smallCubeThresh = 0.6; // 在机器人坐标系下, 小立方体x方向坐标阈值 FIXME
    const float bigCubeThresh = 0.50; // 在机器人坐标系下, 大立方体x方向坐标阈值, 应小于立方体最高点x
    const float bigBallThresh = 0.47; // 在机器人坐标系下, 大球x方向坐标阈值, 应小于球最高点x

private:
    /// GPD FIXME
//    gpd::GraspDetectorPointNet* grasp_detector_; ///< used to run the GPD algorithm
//    gpd::util::Cloud* cloud_camera_; ///< stores point cloud with (optional) camera information and surface normals

    /// YOLO
    YoloDetector* yoloDetector;

    /// 点云生成相关参数
    cv::Mat lookupX, lookupY; // 像素坐标转换到相机坐标
};

#endif //MAGICCUBE_GRAPHICSCUBE_H
