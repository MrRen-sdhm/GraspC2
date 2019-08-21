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

// GPD FIXME
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

    // show: 1-全部显示 2-逐个显示 其他-不显示
    std::pair<std::vector<cv::RotatedRect>, std::vector<int>> detectGraspYoloProT2(cv::Mat &image, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud, int thresh, int show);

    std::pair<std::vector<cv::RotatedRect>, std::vector<int>> getRotRectsAndID(cv::Mat &image,
                                                std::vector<int> &classIds, std::vector<float> &confidences,
                                                std::vector<cv::Rect> &boxes, const cv::Rect& rect, int thresh, int show);

    std::pair<std::vector<cv::RotatedRect>, std::vector<int>> detectJuggles(cv::Mat &image, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud,
                                               std::vector<int> &classIds, std::vector<float> &confidences,
                                               std::vector<cv::Rect> &boxes, const cv::Rect& rect, int thresh, int show);

    std::pair<std::vector<cv::RotatedRect>, std::vector<int>> detectSmallCubeTask2(
            cv::Mat &image, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud, int threshColor, int show);

    std::pair<std::vector<cv::RotatedRect>, std::vector<int>> detectBigCubeTask3(cv::Mat &image,
                            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud, int thresh, int show);

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
    std::vector<double> calcRealCoor(std::vector<float> coorRaw, int leftOrRight);

    /// 寻找左右侧的目标物体, 左侧找最左/上边的, 右侧找最右/下边的, RowOrCol: 1为左右最值 0为上下最值
    std::vector<int> findAimObjLR(std::pair<std::vector<cv::RotatedRect>, std::vector<int>> RotRectsAndID, int RowOrCol);

    /// 获取物体姿态和ID towPointOrNot: 0-单点计算坐标 1-两点计算平均坐标 longOrshort: 0为长边 leftOrRight: 0为左臂
    bool getObjPose(cv::RotatedRect& RotRect, std::vector<double> &b2oXYZRPY,
            const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud, int towPointOrNot,
            int longOrshort, int leftOrRight, float scale = 8.0);

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

    void readCameraInfo(cv::Mat &cameraMatrix);

private:
    /// GPD FIXME
//    gpd::GraspDetectorPointNet* grasp_detector_; ///< used to run the GPD algorithm
//    gpd::util::Cloud* cloud_camera_; ///< stores point cloud with (optional) camera information and surface normals

    /// YOLO
    YoloDetector* yoloDetector;

    /// 点云生成相关参数
    cv::Mat lookupX, lookupY; //

private:
    /// 相及内外参
    // Ca190515
    const std::vector<double> cameraMatrixVec0515 = {1.0706240073964989e+03, 0., 9.3931044946522263e+02, 0.,
                                                     1.0725760043834784e+03, 5.8068060514352021e+02, 0., 0., 1.};
    const std::vector<double> handEyeAxisAngle0515L = {1.6279124026261493e+00, 1.9060879126994393e-01, 9.5844064358541980e-01, -2.1227298795225297e-01};
    const std::vector<double> handEyeAxisAngle0515R = {1.5727409322245274e+00, 1.9586364882379123e-01, -9.6331463492705915e-01, 1.8347300947217626e-01};
    const std::vector<double> handEyeTranslation0515L = {-9.9408057093317268e+01,-1.6319000041081046e+02, -3.0681161020943114e+02};
    const std::vector<double> handEyeTranslation0515R = {9.9050412480342914e+01, -1.6703976455603259e+02, -1.3477769591431235e+02};

    // Ca190517
    const std::vector<double> cameraMatrixVec0517 = {1.0817351755988568e+03, 0., 9.4219086287825303e+02, 0.,
                                                     1.0823131063554199e+03, 5.7383408949778186e+02, 0., 0., 1.};
    const std::vector<double> handEyeAxisAngle0517L = {1.5938399589494991e+00, 1.5111805336104953e-01, 9.7566277847803584e-01, -1.5888825205434856e-01};
    const std::vector<double> handEyeAxisAngle0517R = {1.5790132550603384e+00, 1.3340657539690240e-01, -9.7790841178594867e-01, 1.6092800812523028e-01};

    /// 左臂 想y+ 减小外参y  想z- 减外参z
    /// 左臂 想y+ 减小外参y  想z- 减外参z
    const std::vector<double> handEyeTranslation0517L = {-8.0998572336230325e+01,-1.2557347472865831e+02, -3.3062181423815463e+02};
    const std::vector<double> handEyeTranslation0517R = {1.2527476324483536e+02, -1.2075480479729293e+02, -1.4061563062093377e+02};

    // Ca190518
    const std::vector<double> cameraMatrixVec0518 = {1.0655523574354265e+03, 0., 9.5564423948507863e+02, 0.,
                                                     1.0686852421766414e+03, 5.7414567723487755e+02, 0., 0., 1.};
    const std::vector<double> handEyeAxisAngle0518L = {1.6188757482335823e+00, 1.8474791777310931e-01, -9.6038785750092137e-01, 2.0862255401391883e-01};
    const std::vector<double> handEyeAxisAngle0518R = {1.5790132550603384e+00, 1.3340657539690240e-01, -9.7790841178594867e-01, 1.6092800812523028e-01};
    const std::vector<double> handEyeTranslation0518L = {-1.0466845661427340e+02,-1.6080084092593262e+02, -3.2030724244113424e+02};
    const std::vector<double> handEyeTranslation0518R = {9.8680213952290671e+01, -1.4977331097017262e+02, -1.1243745393879215e+02};

    // Ca19xxxx
    const std::vector<double> cameraMatrixVecxxxx = {1.0918676765288562e+03, 0., 9.6563101765212969e+02, 0.,
                                                     1.0944019738976494e+03, 5.8725610308927321e+02, 0., 0., 1.};

    const std::vector<double> handEyeAxisAnglexxxxL = {1.5970140015765073e+00, 1.8733845897525306e-01, 9.6181578536328594e-01, -1.9953620427076141e-01};
    const std::vector<double> handEyeAxisAnglexxxxR = {1.5629580337468769e+00, 1.9902480439030915e-01, -9.6514342987439494e-01, 1.6996260473318289e-01};
    const std::vector<double> handEyeTranslationxxxxL = {-8.7732753608050814e+01, -1.5774499240041359e+02, -3.3291232010476563e+02};
    const std::vector<double> handEyeTranslationxxxxR = {8.9986226453914071e+01, -1.6187367033647485e+02, -1.3059369296089841e+02};

public:
    /// 工作区域划分
    std::vector<int> LU_ = {130, 195}; // 桌面区域左上角点 (x, y)=(col, row)
    std::vector<int> RD_ = {680, 460}; // 桌面区域右下角点 (x, y)=(col, row)
    const float LeftOrRightThresh = 400.0; // 左右臂分工阈值, 列数小于阈值为左臂管辖
    const float WorkAreaThreshL = 280.0; // 左侧工作区域分割阈值
    const float WorkAreaThreshR = 525.0; // 右侧工作区域分割阈值
    const float WorkAreaThreshSmallCube = 50; // 小立方体工作区域分割阈值, 在RD_[1] 基础上减此值

    const float lieThreshL = 0.61; // 积木躺着或立着x方向阈值, 左 // 大于此值则为躺着的
    const float lieThreshR = -0.54; // 积木躺着或立着x方向阈值, 右 // 小于此值即为立着的

    /// 积木抓取固定高度
    const float height_Lv1_L = 0.29; // 左臂抓取高积木所到深度
    const float height_Lv1_R = -0.29; // 右臂抓取高积木所到深度
    // 立着
//    const float height_Lv2_L = 0.30 - 0.03; // 左臂抓取低积木所到深度
//    const float height_Lv2_R = -0.30 + 0.03; // 右臂抓取低积木所到深度
    // 躺着
    const float height_Lv2_L = 0.345 - 0.03; // 左臂抓取低积木所到深度
    const float height_Lv2_R = -0.343 + 0.03; // 右臂抓取低积木所到深度

    /// 大长方体抓取固定高度
    const float height_bigCube_L = 0.33 - 0.03; // 右臂抓取低积木所到深度
    const float height_bigCube_R = -0.33 + 0.03; // 右臂抓取低积木所到深度

    const float BigCubeT3ThreshL = 0.233; // 左臂抓取大长方体所到深度
    const float BigCubeT3ThreshR = 0.233; // 右臂抓取大长方体所到深度

private:
    const double areaThresh = 1300.0; // 积木与立方体轮廓面积区分阈值
    /// NOTE：值越小离桌面越远
    const float smallCubeThresh = 0.67; // 在机器人坐标系下, 小立方体x方向坐标阈值  小立方体最高点 0.6475
    const float bigCubeThresh = 0.63; // 在机器人坐标系下, 大立方体x方向坐标阈值, 应小于立方体最高点x 立方体最高点 0.62
    const float bigBallThresh = 0.59; // 在机器人坐标系下, 大球x方向坐标阈值, 应小于球最高点x, NOTE 值越大分割出的掩码面积越大 球最高点 0.562
    const float bigCubeTask3Thresh = 0.71; // 在机器人坐标系下, 大长方体x方向坐标阈值, NOTE 越小限制越大
};

#endif //MAGICCUBE_GRAPHICSCUBE_H
