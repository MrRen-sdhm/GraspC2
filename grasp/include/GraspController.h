
#ifndef GRASP_CONTROLLER_H
#define GRASP_CONTROLLER_H

#include <thread>
#include <iostream>
#include <future>
#include <stdlib.h>

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
    bool setup(const std::string& configFilePath);

    /**
     * 启动程序
     */
    void start();

    /**
     * 暂停程序
     */
    void stop();

private:

    //结束所有的工作线程
    void destoryTaskThreads();

    // 合并左右臂目标位置
    void mergeTargetLR(std::vector<double> &targetL, std::vector<double> &targetR, std::vector<double> &target);

    //获取机器人的关节角
    std::vector<double> getRobotJoints();

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

    // 移动指定位姿 实际使用的是MoveJ 输入笛卡尔 转换为关节角 armId: 0-左臂 1-右臂 2-双臂
    void Move(const std::vector<double>& targetPose, double vel, double acc, int armId);

    // 移动指定位姿 实际使用的是MoveL 输入笛卡尔 armId: 0-左臂 1-右臂 2-双臂
    void MovePose(const std::vector<double>& targetPose, double vel, double acc, int armId);

    // 移动指定关节角 armId: 0-左臂 1-右臂 2-双臂
    void MoveJoints(const std::vector<double>& targetJoints, double vel, double acc, int armId);

    // 移动指定路径 targetPose[xyzrpy] 单位[m/rad] armId[0为左臂 1为右臂]
    void MovePath(const std::vector<double>& targetPose, double vel, double acc, int armId);

    // 在当前位置基础上调整Joint6角度
    void MoveJoint6(double targetJoint6L, double targetJoint6R, int armId);

    void MoveInit();

    void HandOpen(int armId);

    void HandClose(int armId);

    // 抓取控制线程
    void graspControlThreadFunc();

    // 点云初始化
    void cloudInit();

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

        while(!visualizer->wasStopped()) {
            visualizer->spinOnce(10);
        }
        visualizer->close();
    }

private:
    /// 抓取相关参数
    const double Vel_Lv1 = 0.5;
    const double Acc_Lv1 = 0.05;
    const double Vel_Lv2 = 1.5;
    const double Acc_Lv2 = 0.5;

    /// 垂直抓取相关位置
    // 起始位置
    const std::vector<double> IniteJoints = {1.62, 0.360, -1.92, -0.64, 0.026, 0.00,
                                            -1.62, -0.360, 1.92, 0.64, -0.026, 0.00};
    // 中间位置
    const std::vector<double> MiddlePose = {1.62, 0.920, -1.92, -0.64, 0.026, 0.00,
                                            -1.62, 0.920, 1.92, 0.64, -0.026, 0.00};

    // 放置位置
    const std::vector<double> EndJoints = {D2R(97.402), D2R(93.306), D2R(-69.662), D2R(19.228), D2R(9.333), D2R(-169.197),
                                           D2R(-92.52), D2R(-104.37), D2R(12.86), D2R(-40.35), D2R(-4.63), D2R(157.63)};
    // 左侧放置位置1 FIXME
    const std::vector<double> PlaceJointsL1 = {D2R(97.402), D2R(93.306), D2R(-69.662), D2R(19.228), D2R(9.333), D2R(-169.197)};

    // 左侧放置位置2 FIXME
    const std::vector<double> PlaceJointsL2 = {D2R(97.402), D2R(93.306), D2R(-69.662), D2R(19.228), D2R(9.333), D2R(-169.197)};

    // 左侧放置位置3 FIXME
    const std::vector<double> PlaceJointsL3 = {D2R(97.402), D2R(93.306), D2R(-69.662), D2R(19.228), D2R(9.333), D2R(-169.197)};

    // 右侧放置位置1 FIXME
    const std::vector<double> PlaceJointsR1 = {D2R(97.402), D2R(93.306), D2R(-69.662), D2R(19.228), D2R(9.333), D2R(-169.197)};

    // 右侧放置位置2 FIXME
    const std::vector<double> PlaceJointsR2 = {D2R(97.402), D2R(93.306), D2R(-69.662), D2R(19.228), D2R(9.333), D2R(-169.197)};

    // 右侧放置位置3 FIXME
    const std::vector<double> PlaceJointsR3 = {D2R(97.402), D2R(93.306), D2R(-69.662), D2R(19.228), D2R(9.333), D2R(-169.197)};

    /// 水平抓取相关位置
    // 平着
    const std::vector<double> Pose1 = {0.10, -0.50, -0.10, -3.14, 0.00, 0.00,
                                       0.341, -0.354, 0.030, -1.54, -0.95, 1.54};

    // 平着初始位置
    const std::vector<double> Pose2 = {0.215, -0.369, 0.428, -2.009, -1.215, -1.09,
                                       0.341, -0.354, 0.030, -1.54, -0.95, 1.54};
};


#endif //GRASP_CONTROLLER_H
