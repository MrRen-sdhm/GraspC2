
#ifndef GRASP_CONTROLLER_H
#define GRASP_CONTROLLER_H

#include <thread>
#include <iostream>
#include <future>
#include <stdlib.h>

//#include <YinShiDriver.h> //FIXME
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

    //机器人复位
    void resetRobot();

    //获取机器人的关节角
    std::vector<double> getRobotJoints( uint16_t _Dev);

    //获取机器人是否空闲
    bool isIdle(uint16_t _Dev);

    //笛卡尔坐标转换成关节角
    int cart2Joints(const std::vector<double> &initJoints, const std::vector<double> & targetPos, std::vector<double> &targetJoints, uint16_t _Dev);

    //关节角转化成笛卡尔坐标
    int joint2Cart(const std::vector<double> &targetJoints, std::vector<double> &targetPos, uint16_t _Dev);

    //拍摄照片
    bool captureImage(int exposureTime = 500);

    //抓取
    bool graspControl();

//    void handControl(uint16_t handType, YinShiDriver::HandCMD _CMD); // FIXME

    // 移动指定路径 targetPose[xyzrpy] 单位[m/rad] armId[0为左臂 1为右臂]
    void movePath(const std::vector<double>& targetPose, double acc, double vel, int armId);

    //移动到指定点（关节角）
    void moveOnece(const std::vector<double>& targetJoints, ArmRobotMotionType motionType, double acc, double vel, int armId);

    //移动到指定点（欧拉角）
    void moveCart(std::vector<double> targetPose, double acc, double vel, int armId);

    //移动到指定点（旋转矩阵）
    void moveQuat(Eigen::Vector3d targetPosition, Eigen::Matrix3d targetQuat, ArmRobotMotionType motionType,
                                   double acc, double vel, int armId);

    //魔方控制线程
    void graspControlThreadFunc();

    //手爪控制线程
    void handControlThreadFunc();

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
    std::thread _viewControlThread;
    std::thread _handControlThread;

    //阶段任务执行开关变量
    volatile bool _graspControlStart;

    std::vector<std::vector<double> > _startHome;
    std::vector<std::vector<double> > _endHome;
    std::vector<std::vector<std::vector<double> > > _capturePoint;

    std::pair<int, int> _currentArmAndPointId;

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
};


#endif //GRASP_CONTROLLER_H
