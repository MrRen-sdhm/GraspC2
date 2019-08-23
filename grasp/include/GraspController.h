
#ifndef GRASP_CONTROLLER_H
#define GRASP_CONTROLLER_H

#include <thread>
#include <iostream>
#include <future>
#include <stdlib.h>
#include <sstream>
#include <math.h>

#include <cobotsys/cobotsys_global_object_factory.h>
#include <cobotsys/cobotsys_file_finder.h>
#include <cobotsys/extra2.h>
#include <cobotsys_data_types.h>
#include <cobotsys_abstract_camera.h>
#include <cobotsys_abstract_arm_robot_link.h>
#include <cobotsys_abstract_kinematic_solver.h>
#include <cobotsys_abstract_dual_arm_robot_link.h>

//#include "../CustomDataType.h"
#include "GraphicsGrasp.h"

#include <Eigen/Dense>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>

template <typename T>
vector<size_t> sort_indexes_e(vector<T> &v)
{
    vector<size_t> idx(v.size());
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(),
         [&v](size_t i1, size_t i2) {return v[i1] < v[i2]; });
    return idx;
}

using namespace cobotsys;

class GraphicsGrasp;

class GraspController {

public:
    GraspController();

    ~GraspController();

    /**
     * 初始化设置
     * @param configFilePath 文件路径
     * @return true: 设置成功 false: 设置失败
     */
    bool setup(int taskNum, const std::string &configFilePath);

    /**
     * 启动程序
     */
    void start();

    /**
     * 暂停程序
     */
    void stop();

private:

    int taskNum_;

    //结束所有的工作线程
    void destoryTaskThreads();

    // 合并左右臂目标位置
    void mergeTargetLR(std::vector<double> &targetL, std::vector<double> &targetR, std::vector<double> &target);

    //获取机器人的关节角
    std::vector<double> getRobotJoints(int armID);

    //获取机器人的笛卡尔坐标
    std::vector<double> getRobotPose(int armID);

    //获取机器人是否空闲 isIdle 1: 空闲 0: 工作
    bool isIdle(uint16_t _Dev);

    //笛卡尔坐标转换成关节角, 输入size为12， 输出size为6
    std::vector<double> cart2Joints(const std::vector<double> &targetPose, int armId);

    //拍摄照片
    bool captureImage(int exposureTime = 500);

    //抓取
    bool graspControl();

    // 双臂同时抓取
    bool graspControlDual();

    bool graspControlTask2();

    // 双臂同时抓取大长方体
    bool graspControlBigCubeT3();

    bool graspControlTask4();

    // 移动指定位姿 实际使用的是MoveJ 输入笛卡尔 转换为关节角 armId: 0-左臂 1-右臂 2-双臂
    void Move(const std::vector<double> &targetPose, double vel, double acc, int armId);

    // 移动指定位姿 实际使用的是MoveL 输入笛卡尔 armId: 0-左臂 1-右臂 2-双臂
    void MovePose(const std::vector<double> &targetPose, double vel, double acc, int armId);

    // 移动指定关节角 armId: 0-左臂 1-右臂 2-双臂
    void MoveJoints(const std::vector<double> &targetJoints, double vel, double acc, int armId);

    // 移动指定路径 targetPose[xyzrpy] 单位[m/rad] armId[0为左臂 1为右臂]
    void MovePath(const std::vector<double> &targetPose, double vel, double acc, int armId);

    // 在当前位置基础上调整Joint6角度
    void MoveJoint6(double targetJoint6L, double targetJoint6R, int armId);

    void MoveInit();

    void HandOpen(int armId);

    void HandClose(int armId);

    // 抓取控制线程
    void graspControlThreadFunc();

    // 点云初始化
    void cloudInit();

    void saveCloudAndImages();

    static std::string getCurrentTimeStr() {
        time_t t = time(nullptr);
        char ch[64] = {0};
        strftime(ch, sizeof(ch) - 1, "%m%d", localtime(&t));     //年-月-日 时-分-秒
        return ch;
    }

/// ******************************* 抓取姿态生成 ******************************* ///
private:
    // gpd相关参数
    std::vector<Eigen::Vector3d> positions; // 抓取位置
    std::vector<Eigen::Matrix3d> rotMatrixs; // 抓取姿态
    std::vector<Eigen::Vector3d> samples; // 采样点
    Eigen::Vector3d approach, binormal, axis;
    std::vector<double> graspWidths; // 手抓开度
    std::vector<double> graspScores; // 姿态得分

//    void graspDetectGPD() {  //FIXME
//        //NOTE: gpd计算抓取姿态
//        std::vector<std::unique_ptr<gpd::candidate::Hand>> grasps = _graphicsGrasp->detectGraspPoses(cloud);
//        printf("Detect %zu grasp pose.\n\n", grasps.size());
//
//        for (size_t i = 0; i < grasps.size(); i++) {
//
//            rotMatrixs[i] = grasps[i]->getOrientation(); // 获取的姿态信息为旋转矩阵
//            positions[i] = grasps[i]->getPosition();
//            samples[i] = grasps[i]->getSample();
//            graspScores[i] = grasps[i]->getScore();
//            graspWidths[i] = grasps[i]->getGraspWidth();
//
//            approach = rotMatrixs[i].col(0);
//            binormal = rotMatrixs[i].col(1);
//            axis = rotMatrixs[i].col(2);
//        }
//    }

/// ******************************* YOLO ******************************* ///
private:
    std::pair<std::vector<cv::RotatedRect>, std::vector<int>> RotRectsAndID;

private:
    std::shared_ptr<GraphicsGrasp> _graphicsGrasp;

    //工作对象
    std::shared_ptr<DualArmRobotDriver> _cassemble2Driver;

    std::thread _graspControlThread;
    std::thread _handControlThread;

    //阶段任务执行开关变量
    volatile bool _graspControlStart;

    CameraFrame _captureImages; //存储相机拍摄的照片

    // 点云生成相关参数
    cv::Mat color, depth;
    pcl::PCDWriter writer;
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud;

    bool saveFlag = true; // 保存标志位

    void cloud_viewer() {
        pcl::visualization::PCLVisualizer::Ptr visualizer(new pcl::visualization::PCLVisualizer("Cloud Viewer"));
        const std::string cloudName = "rendered";
        visualizer->addPointCloud(cloud, cloudName);
        visualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, cloudName);
        visualizer->initCameraParameters();
        visualizer->setBackgroundColor(0, 0, 0);
        visualizer->setSize(cloud->width, cloud->height);
        visualizer->setShowFPS(true);
        visualizer->setCameraPosition(0, 0, 0, 0, -1, 0);

        while (!visualizer->wasStopped()) {
            visualizer->spinOnce(10);
        }
        visualizer->close();
    }

private:
    /// 抓取相关参数
    const double Vel_Lv1 = 0.5;
    const double Acc_Lv1 = 0.05;
    const double Vel_Lv2 = 1.5;
    const double Acc_Lv2 = 0.05;

    const double Vel_Lv2_ = 1.0;
    const double Acc_Lv2_ = 1.0;

    // 用于移动关节角
    const double Vel_Lv3 = 1.0;
    const double Acc_Lv3 = 1.0;

    // 用于转动末端姿态
    const double Vel_Lv4 = 4.0;
    const double Acc_Lv4 = 4.0;

    bool BigCubePicked = false;

    bool BallPicked = false;
    int PickedBigCubeCnt = 0; // 已抓取积木数
    int PickedSmallCubeCnt = 0; // 已抓取小立方体数
    int PlacedSmallCubeCntL = 0; // 左侧已放置小立方体数
    int PlacedSmallCubeCntR = 0; // 右侧已放置小立方体数

    std::vector<int> PlacedCubeIDL; // 存储四个位置对应的小立方体ID
    std::vector<int> PlacedCubeIDR; // 存储四个位置对应的小立方体ID

    /// 垂直抓取相关位置
    // 起始位置

    const std::vector<double> IniteJointsOld = {D2R(92.819), D2R(20.626), D2R(-110.007), D2R(-36.669), D2R(1.488),
                                                D2R(-0.001),
                                                D2R(-92.819), D2R(-20.626), D2R(110.007), D2R(36.669), D2R(-1.488),
                                                D2R(0.001)};

    const std::vector<double> IniteJoints = {D2R(91.238), D2R(64.396), D2R(-37.523), D2R(-15.631), D2R(-1.713),
                                             D2R(5.382),
                                             D2R(-91.238), D2R(-64.396), D2R(37.523), D2R(15.631), D2R(1.713),
                                             D2R(-5.382)};

    const std::vector<double> IniteJointsL = {D2R(91.238), D2R(64.396), D2R(-37.523), D2R(-15.631), D2R(-1.713),
                                              D2R(5.382)};

    const std::vector<double> IniteJointsR = {D2R(-91.238), D2R(-64.396), D2R(37.523), D2R(15.631), D2R(1.713),
                                              D2R(-5.382)};


    // 积木抓取初始位置
//    const std::vector<double> JuggleIniteJoints = {D2R(89.304), D2R(56.234), D2R(-110.007), D2R(-56.669), D2R(1.488), D2R(-20.0),
//                                                   D2R(-89.304), D2R(-56.234), D2R(110.007), D2R(56.669), D2R(-1.488), D2R(20.0)};

    const std::vector<double> JuggleIniteJoints = {D2R(89.948), D2R(65.239), D2R(-77.522), D2R(33.371), D2R(2.248),
                                                   D2R(-171.430),
                                                   D2R(-89.948), D2R(-65.239), D2R(77.522), D2R(-33.371), D2R(-2.248),
                                                   D2R(171.430)};

    // 球体抓取初始位置
    const std::vector<double> BallIniteJoints = {1.62, 0.360, -1.92, -0.64, D2R(-40), 0.00,
                                                 -1.62, -0.360, 1.92, 0.64, D2R(40), D2R(174)}; // FIXME

    // 立方体抓取初始位置
    const std::vector<double> CubeIniteJoints = {D2R(91.238), D2R(64.396), D2R(-37.523), D2R(-15.631), D2R(-40),
                                                 D2R(5.382),
                                                 D2R(-91.238), D2R(-64.396), D2R(37.523), D2R(15.631), D2R(40),
                                                 D2R(-5.382)}; // FIXME

    // 中间位置
    const std::vector<double> MiddlePose = {1.62, 0.920, -1.92, -0.64, 0.026, 0.00,
                                            -1.62, 0.920, 1.92, 0.64, -0.026, 0.00};

    /// 积木放置位置
    const std::vector<double> EndJoints = {D2R(97.402), D2R(93.306), D2R(-69.662), D2R(19.228), D2R(9.333),
                                           D2R(-169.197),
                                           D2R(-92.52), D2R(-104.37), D2R(12.86), D2R(-40.35), D2R(-4.63), D2R(157.63)};
    // 左侧放置位置1
    const std::vector<double> PlaceJointsL1 = {D2R(92.273), D2R(34.669), D2R(-121.205), D2R(19.228), D2R(2.178), D2R(-97.334)};

    // 左侧放置位置2
    const std::vector<double> PlaceJointsL2 = {D2R(91.706), D2R(44.431), D2R(-98.776), D2R(36.571), D2R(1.611), D2R(-97.276)};

    // 左侧放置位置3
    const std::vector<double> PlaceJointsL3 = {D2R(91.459), D2R(52.591), D2R(-81.059), D2R(46.086), D2R(1.364), D2R(-97.236)};

    // 右侧放置位置1
    const std::vector<double> PlaceJointsR1 = {D2R(-92.273), D2R(-34.699), D2R(121.205), D2R(-19.228), D2R(-2.178), D2R(97.334)};

    // 右侧放置位置2
    const std::vector<double> PlaceJointsR2 = {D2R(-91.706), D2R(-44.431), D2R(98.776), D2R(-36.571), D2R(-1.611), D2R(97.276)};

    // 右侧放置位置3
    const std::vector<double> PlaceJointsR3 = {D2R(-91.459), D2R(-52.591), D2R(81.059), D2R(-46.086), D2R(-1.364), D2R(97.236)};


    /// 任务2小立方体放置位置
    // 放置位置1-左臂
    const std::vector<double> PlaceJointsL1_T2 = {D2R(102.863), D2R(23.343), D2R(-102.095), D2R(54.561), D2R(12.863),
                                                  D2R(0.179)};

    // 放置位置2-左臂
    const std::vector<double> PlaceJointsL2_T2 = {D2R(98.871), D2R(48.680), D2R(-59.529), D2R(71.662), D2R(8.872),
                                                  D2R(0.126)};

    // 放置位置3-左臂
    const std::vector<double> PlaceJointsL3_T2 = {D2R(102.862), D2R(46.818), D2R(-105.216), D2R(27.888), D2R(12.863),
                                                  D2R(0.074)};

    // 放置位置4-左臂
    const std::vector<double> PlaceJointsL4_T2 = {D2R(98.871), D2R(64.909), D2R(-62.920), D2R(52.076), D2R(8.872),
                                                  D2R(0.092)};

    // 放置位置1-右臂
    const std::vector<double> PlaceJointsR1_T2 = {D2R(-102.863), D2R(-23.343), D2R(102.095), D2R(-54.561), D2R(-12.863),
                                                  D2R(-0.179)};

    // 放置位置2-右臂
    const std::vector<double> PlaceJointsR2_T2 = {D2R(-98.871), D2R(-48.680), D2R(59.529), D2R(-71.662), D2R(-8.872),
                                                  D2R(-0.126)};

    // 放置位置3-右臂
    const std::vector<double> PlaceJointsR3_T2 = {D2R(-102.862), D2R(-46.818), D2R(105.216), D2R(-27.888), D2R(-12.863),
                                                  D2R(-0.074)};

    // 放置位置4-右臂
    const std::vector<double> PlaceJointsR4_T2 = {D2R(-98.871), D2R(-64.909), D2R(62.920), D2R(-52.076), D2R(-8.872),
                                                  D2R(-0.092)};

    /// 笛卡尔

    // 放置位置1-左臂
    const std::vector<double> PlacePoseL1_T2 = {0.280, -0.280, 0.150, D2R(90), D2R(0), D2R(90)};

    // 放置位置2-左臂
    const std::vector<double> PlacePoseL2_T2 = {0.280, -0.420, 0.150, D2R(90), D2R(0), D2R(90)};

    // 放置位置3-左臂
    const std::vector<double> PlacePoseL3_T2 = {0.280, -0.280, 0.01, D2R(90), D2R(0), D2R(90)};

    // 放置位置4-左臂
    const std::vector<double> PlacePoseL4_T2 = {0.280, -0.420, 0.01, D2R(90), D2R(0), D2R(90)};

    // 放置位置1-右臂
    const std::vector<double> PlacePoseR1_T2 = {-0.280, 0.280, 0.150, D2R(90), D2R(0), D2R(-90)};

    // 放置位置2-右臂
    const std::vector<double> PlacePoseR2_T2 = {-0.280, 0.420, 0.150, D2R(90), D2R(0), D2R(-90)};

    // 放置位置3-右臂
    const std::vector<double> PlacePoseR3_T2 = {-0.280, 0.280, 0.01, D2R(90), D2R(0), D2R(-90)};

    // 放置位置4-右臂
    const std::vector<double> PlacePoseR4_T2 = {-0.280, 0.420, 0.01, D2R(90), D2R(0), D2R(-90)};

    /// 任务2装配位置
    // 装配位置-左臂 FIXME
    const std::vector<double> TogetherJointsL = {D2R(127.69), D2R(77.56), D2R(-27.28), D2R(-14.84), D2R(-89.91),
                                                 D2R(-142.22)};
    // 装配位置-右臂 FIXME 右臂垂直
    const std::vector<double> TogetherJointsRN = {D2R(-92.71), D2R(-96.69), D2R(41.42), D2R(-61.79), D2R(-4.41),
                                                  D2R(-71.21)};

    // 装配位置-左臂 FIXME 左臂垂直
    const std::vector<double> TogetherJointsLN = {D2R(92.61), D2R(100.37), D2R(-32.42), D2R(67.50), D2R(4.32),
                                                  D2R(70.79)};
    // 装配位置-右臂 FIXME
    const std::vector<double> TogetherJointsR = {D2R(-128.59), D2R(-82.95), D2R(16.25), D2R(9.20), D2R(89.90),
                                                 D2R(141.31)};

    /// 任务2装配初始置
    // 装配位置-左臂 FIXME
    const std::vector<double> TogetherInitPoseL = {0.4, -0.423, 0.1, D2R(180), D2R(-0.0), D2R(180)};
    // 装配位置-右臂 FIXME
    const std::vector<double> TogetherInitPoseR = {-0.407, -0.399, 0.1, D2R(-180), D2R(-0.0), D2R(-180)};

    /// 任务2立方体交换位置
    // 装配位置-左臂 FIXME
    const std::vector<double> ExchangeInitJointsL = {D2R(121.604), D2R(59.789), D2R(-52.394), D2R(-22.182),
                                                     D2R(-89.909), D2R(-148.307)};
    // 装配位置-右臂 FIXME
    const std::vector<double> ExchangeInitJointsR = {D2R(-123.516), D2R(-56.880), D2R(58.283), D2R(25.079), D2R(89.992),
                                                     D2R(146.484)};

};
#endif //GRASP_CONTROLLER_H
