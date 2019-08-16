#include "GraspController.h"

using namespace std;
using namespace cv;

GraspController::GraspController() {
    _graphicsGrasp = std::make_shared<GraphicsGrasp>(); // 创建视觉支持
    _graspControlStart = false;
}

GraspController::~GraspController() {
    destoryTaskThreads();
}

void GraspController::start() {
    _graspControlStart = true;

    ErrorInfo errorInfo;
    _cassemble2Driver->Start(errorInfo);
}

void GraspController::stop() {
    ErrorInfo errorInfo;
    _cassemble2Driver->Stop(errorInfo);
}

bool GraspController::setup(const std::string& configFilePath) {

    /// 1.创建设备
    _cassemble2Driver = std::dynamic_pointer_cast<cobotsys::DualArmRobotDriver>(
            GlobalObjectFactory::instance()->createObject("CAssemblyC2DriverFactory, Ver 1.0",
                                                          "CAssemblyC2Driver"));

    if (!_cassemble2Driver) {
        std::cout << "创建CAssembleC2驱动失败！！！" << std::endl;
        return false;
    }

    /// 3.启动多线程工作
    _graspControlThread = std::thread(&GraspController::graspControlThreadFunc, this);
//    _handControlThread = std::thread(&GraspController::handControlThreadFunc, this);

    return true;
}

void GraspController::graspControlThreadFunc() {
    start();
    while (true) {
        if (!_graspControlStart) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            continue;
        }

        auto startTime = std::chrono::high_resolution_clock::now();
        std::cout << "开始时间： " << std::chrono::system_clock::to_time_t(startTime) << std::endl;

        // 异步抓取
//        if(!graspControl()) return;

        // 同步抓取
        if(!graspControlDual()) return;

        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double , std::ratio<1,1> > duration_s(endTime - startTime);
        std::cout << "完成! 总体耗时：   " << duration_s.count() << "  秒 "<< std::endl;

        /// 退出
        destoryTaskThreads();
        stop();

        break;
    }
}

bool GraspController::graspControl() {
    bool ret = captureImage(-100); // 采集图像
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    /// 初始位置
    MoveInit(); // 同步移动到初始位置

    /// 手抓张开
    HandOpen(0);
    HandOpen(1);

    /// 点云初始化
    cloudInit();

    while (true) {
        captureImage(-100); // 采集图像

        if (!ret) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            std::cout << "graspGrasp wait capture!" << std::endl;
            continue;
        }

        if (_captureImages.frames.empty())
            return false;
        if (_captureImages.frames.at(0).data.empty())
            return false;

        printf("\033[0;31m%s\033[0m\n", "================================== 获取点云 =================================");

        // 创建点云使用1920*1080像素
        color = _captureImages.frames.at(0).data.clone();
        depth = _captureImages.frames.at(1).data.clone();

//        color = cv::imread("/home/hustac/rgb.jpg");
//        depth = cv::imread("/home/hustac/depth.png", -1);

        _graphicsGrasp->createPointCloud(color, depth, cloud); // 创建点云

        printf("\033[0;31m%s\033[0m\n", "=================================== 目标检测 ================================");

        bool BigBallFlag = false, BigCubeFlag = false;
        cv::RotatedRect BigBallRect, BigCubeRect;

//        BigBallFlag = _graphicsGrasp->detectBigBall(color, BigBallRect);
//        BigCubeFlag = _graphicsGrasp->detectBigCube(color, BigCubeRect);

        /// Yolo积木检测
        RotRectsAndID = _graphicsGrasp->detectGraspYolo(color, 200, false);
        /// 正方体检测
//        RotRectsAndID = _graphicsGrasp->detectBigCube(color, 200, false);

//        writer.writeBinary("/home/hustac/test.pcd", *cloud);
//        cv::imwrite("/home/hustac/test.jpg", color);
//        cv::imwrite("/home/hustac/test.png", depth);
//        exit(0);

//        graspDetectGPD(); // TODO:生成抓取姿态

        printf("\033[0;31m%s\033[0m\n", "================================= 生成移动任务 ===============================");

        if (RotRectsAndID.first.empty()) {
            printf("[ERROR] Did not get any rot rects!\n");
            continue;
        }

        if (!BigBallFlag && !BigCubeFlag) { // 没有大型物体
            printf("[INFO] Don't have big obj, pick up juggles.");

            int leftOrright = ((int) RotRectsAndID.first[0].center.x < 410) ? 0 : 1; // 简单处理左右手分工

            /// 获取积木位置
            std::vector<double> targetPose = _graphicsGrasp->getObjPose(RotRectsAndID.first[0], cloud, 0, 0, leftOrright);
            /// 获取正方体位置
    //            std::vector<double> targetPose = _graphicsGrasp->getObjPose(RotRectsAndID.first[i], cloud, 1, 0, leftOrright);

            /// 修改姿态, 移动到物体上方
            std::vector<double> targetPoseTmp = targetPose; // 目标位置副本
            if (leftOrright == 1) { // 右臂
                targetPoseTmp[0] += 0.25;
                targetPoseTmp[4] = 0; // 修改末端pitch为0
            } else if (leftOrright == 0) { // 左臂
                targetPoseTmp[0] -= 0.25; // 碰:0.23 不碰:0.25 0.3(大爪子)
                targetPoseTmp[4] = 0; // 修改末端pitch为0
            }
            MovePath(targetPoseTmp, 1.0, 0.1, leftOrright);

            /// 调整末端姿态
            targetPoseTmp[4] = targetPose[4]; // 重新获取末端姿态
            MoveJoint6(targetPoseTmp[4], 0, leftOrright); // 当前位置垂直向下, 末端目标pitch即为目标关节角

            printf("[INFO] 待抓取物体信息 ID:[%d] Angle:[%f] left/right[%d] Pose:[%f,%f,%f,%f,%f,%f]\n\n",
                   RotRectsAndID.second[0],
                   RotRectsAndID.first[0].angle, leftOrright, targetPose[0], targetPose[1],
                   targetPose[2], targetPose[3], targetPose[4], targetPose[5]);

    //            MoveMiddle(leftOrright); /// 移动到抓取中间点(待删除)
    //            MovePath(MiddlePose, 0.5, 0.05, leftOrright); /// 移动到抓取中间点(NEW)

            /// 修改姿态, 手臂垂直下降
            if (leftOrright == 1) { // 右臂
                targetPose[0] -= 0.05;
            } else if (leftOrright == 0) {
                targetPose[0] += 0.05;
            }
            MovePath(targetPose, 1.5, 0.5, leftOrright);

            /// 修改姿态, 手臂垂直抬起
            if (leftOrright == 1) { // 右臂
                targetPose[0] += 0.05;
            } else if (leftOrright == 0) {
                targetPose[0] -= 0.05;
            }
            MovePath(targetPose, 1.5, 0.5, leftOrright);

            HandClose(leftOrright); /// 抓取物体

    //            MovePath(Pose2, 0.05, 0.05, leftOrright); // 移动到平躺初始位置

            MoveInit(); // 移动到初始位置
        }
        else if (BigBallFlag) { // 有正方体
            int leftOrright = ((int) BigBallRect.center.x < 410) ? 0 : 1; // 简单处理左右手分工
            std::vector<double> targetPose = _graphicsGrasp->getObjPose(BigBallRect, cloud, 0, 0, leftOrright);
        }

        printf("\033[0;31m%s\033[0m\n\n\n", "================================== 抓取成功 =================================");

#if 0
        int leftOrright = (positions[0].x() <= 0) ? 1 : 0; //NOTE: 简单处理左右手分工

        /// 移动到物体附近
        // 按方向向量平移一定距离
        float distance = -0.15; // 平移的距离
        Eigen::Vector3d prePosition;
        prePosition.x() = positions[0].x() + approach.x()*distance;
        prePosition.y() = positions[0].y() + approach.y()*distance;
        prePosition.z() = positions[0].z() + approach.z()*distance;
        cout << "prePosition: \n" << prePosition << endl;

        /// 计算手臂基坐标系中抓取姿态(输入：旋转矩阵+位置 输出：变换后的旋转矩阵+位置)
        Eigen::Isometry3d transformAfter;
        if (1 != _graphicsGrasp->calcRealCoor(rotMatrixs[0], positions[0], transformAfter, leftOrright)) {
            cerr << "CAMERA_RESULT_READ_ERR\n" << endl;
            return false;
        }

        // 匹配抓取姿态与末端姿态
        Eigen::AngleAxisd rot_vector (M_PI/2, Eigen::Vector3d(0, 0, 1)); // 沿 Z 轴旋转 90 度
        transformAfter.rotate(rot_vector);

        std::vector<double> transformXYZRPY = ConvertTransform2XYZRPY(transformAfter); // 获取笛卡尔空间姿态

        /// 计算手臂基坐标系中预抓取姿态
        Eigen::Isometry3d transformPre(transformAfter);
        transformPre.pretranslate(approach * distance); // 平移
        std::vector<double> transformPreXYZRPY = ConvertTransform2XYZRPY(transformPre); // 获取笛卡尔空间姿态

        std::vector<double> targetJointQ;
        std::vector<double> initJoints;

//        exit(1); //NOTE: 调试用

        moveCart(transformPreXYZRPY, 0.2, 0.1, leftOrright == 1 ? LeftArm : RightArm);
//        handControl(leftOrright == 1 ? LeftHand : RightHand, YinShiDriver::OpenHand);
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));

        moveCart(transformXYZRPY, 0.2, 0.1, leftOrright == 1 ? LeftArm : RightArm);
//        handControl(leftOrright == 1 ? LeftHand : RightHand, YinShiDriver::CloseHand);
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));

        /// 移动到物体附近
        // 转换到关节角
        initJoints = getRobotJoints(leftOrright == 1 ? LeftArm : RightArm);
        cart2Joints(initJoints, transformPreXYZRPY, targetJointQ, leftOrright == 1 ? LeftArm : RightArm);
        // 移动
        moveOnece(targetJointQ, ArmRobotMotionType::MoveL, 0.5, 0.5, _currentArmAndPointId.first);
//        handControl(leftOrright == 1 ? LeftHand : RightHand, YinShiDriver::OpenHand);
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));

        /// 抓取物体
        // 转换到关节角
        initJoints = getRobotJoints(leftOrright == 1 ? LeftArm : RightArm);
        cart2Joints(initJoints, transformXYZRPY, targetJointQ, leftOrright == 1 ? LeftArm : RightArm);
        // 移动
        moveOnece(targetJointQ, ArmRobotMotionType::MoveL, 0.5, 0.5, leftOrright == 1 ? LeftArm : RightArm);
//        handControl(leftOrright == 1 ? LeftHand : RightHand, YinShiDriver::CloseHand);
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));

        return true; // 抓取完成 退出
#endif
    }
}

bool GraspController::graspControlDual() {
    bool ret = captureImage(-100); // 采集图像
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    /// 初始位置
    MoveInit(); // 同步移动到初始位置, 打开手抓

    /// 点云初始化
    cloudInit();

    while (true) {
        captureImage(-100); // 采集图像

        if (!ret) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            std::cout << "graspGrasp wait capture!" << std::endl;
            continue;
        }

        if (_captureImages.frames.empty())
            return false;
        if (_captureImages.frames.at(0).data.empty())
            return false;

        printf("\033[0;31m%s\033[0m\n", "================================== 获取点云 =================================");

        // 创建点云使用1920*1080像素
        color = _captureImages.frames.at(0).data.clone();
        depth = _captureImages.frames.at(1).data.clone();

//        color = cv::imread("/home/hustac/rgb.jpg");
//        depth = cv::imread("/home/hustac/depth.png", -1);

//        color = cv::imread("../../../grasp/data/images/cube1.jpg");
//        depth = cv::imread("../../../grasp/data/images/cube1.png", -1);

        _graphicsGrasp->createPointCloud(color, depth, cloud); // 创建点云

        printf("\033[0;31m%s\033[0m\n", "=================================== 目标检测 ================================");

        const int juggleOrCube = 0; /// 0为积木, 1为立方体

        if (juggleOrCube == 0) {
            /// Yolo积木检测
            RotRectsAndID = _graphicsGrasp->detectGraspYolo(color, 200, false);
        } else if (juggleOrCube == 1) {
            /// 带孔正方体检测
            RotRectsAndID = _graphicsGrasp->detectBigObj(color, 200, false);
        }

//        writer.writeBinary("/home/hustac/test.pcd", *cloud);
//        cv::imwrite("/home/hustac/test.jpg", color);
//        cv::imwrite("/home/hustac/test.png", depth);
//        exit(0);

        printf("\033[0;31m%s\033[0m\n", "================================= 生成移动任务 ===============================");

        if (RotRectsAndID.first.empty()) {
            printf("[ERROR] Did not get any rot rects!\n");
            continue;
        }

        // 确定两侧待抓取物体索引
        std::vector<int> AimObjIndicesLR = _graphicsGrasp->findAimObjLR(RotRectsAndID, 0);

        int moveDevice = -1;
        int LeftObjID = -1, RightObjID = -1;

        vector<double> targetPose; // 双臂目标位置
        std::vector<double> targetPoseL, targetPoseR, targetPoseTmpL, targetPoseTmpR;
        bool HaveObjL = (AimObjIndicesLR[0] != -1), HaveObjR = (AimObjIndicesLR[1] != -1);

        printf("[INFO] Have Object Left[%d] Right[%d]\n", HaveObjL, HaveObjR);

        if ((AimObjIndicesLR[0] != -1) && (AimObjIndicesLR[1] == -1)) { // 左侧有物体
            moveDevice = 0;
        } else if ((AimObjIndicesLR[0] == -1) && (AimObjIndicesLR[1] != -1)) { // 右侧有物体
            moveDevice = 1;
        } else if ((AimObjIndicesLR[0] != -1) && (AimObjIndicesLR[1] != -1)) { // 两侧有物体
            moveDevice = 2;
        }

        printf("\033[0;31m%s\033[0m\n", "================================= 移动到正上方 ===============================");

        /// 获取目标位置(左臂)
        if (AimObjIndicesLR[0] != -1) { // 左侧有物体
            cv::RotatedRect RotatedRectL = RotRectsAndID.first[AimObjIndicesLR[0]];
            LeftObjID = RotRectsAndID.second[AimObjIndicesLR[0]];
            targetPoseL = _graphicsGrasp->getObjPose(RotatedRectL, cloud, juggleOrCube, 0, 0);
            targetPoseTmpL = targetPoseL; // 目标位置副本

            // 移动到物体上方
            targetPoseTmpL[0] -= 0.30; // 碰:0.23 不碰:0.25 0.3(大爪子)
            targetPoseTmpL[4] = 0; // 修改末端pitch为0

            cout << "[INFO] targetPoseL: " << targetPoseL << endl;

            printf("[INFO] 左臂待抓取物体信息 ID:[%d] Angle:[%f] Pose:[%f,%f,%f,%f,%f,%f]\n\n",
                   RotRectsAndID.second[AimObjIndicesLR[0]], R2D(targetPoseTmpL[4]), targetPoseTmpL[0], targetPoseTmpL[1],
                   targetPoseTmpL[2], targetPoseTmpL[3], targetPoseTmpL[4], targetPoseTmpL[5]);
        }

        /// 获取目标位置(右臂)
        if (AimObjIndicesLR[1] != -1) { // 右侧有物体
            cv::RotatedRect RotatedRectR = RotRectsAndID.first[AimObjIndicesLR[1]];
            RightObjID = RotRectsAndID.second[AimObjIndicesLR[1]];
            targetPoseR = _graphicsGrasp->getObjPose(RotatedRectR, cloud, juggleOrCube, 0, 1);
            targetPoseTmpR = targetPoseR; // 目标位置副本

            // 移动到物体上方
            targetPoseTmpR[0] += 0.20; // 碰:0.23 不碰:0.25 0.3(大爪子) NOTE：受摄像头角度影响
            targetPoseTmpR[4] = 0; // 修改末端pitch为0

            cout << "[INFO] targetPoseR: " << targetPoseR << endl;

            printf("[INFO] 右臂待抓取物体信息 ID:[%d] Angle:[%f] Pose:[%f,%f,%f,%f,%f,%f]\n\n",
                   RotRectsAndID.second[AimObjIndicesLR[1]], R2D(targetPoseTmpR[4]), targetPoseTmpR[0], targetPoseTmpR[1],
                   targetPoseTmpR[2], targetPoseTmpR[3], targetPoseTmpR[4], targetPoseTmpR[5]);
        }

        mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);

        cout << "[INFO] targetPose: " << targetPose << endl;

        MovePose(targetPose, Vel_Lv1, Acc_Lv1, moveDevice);

        printf("\033[0;31m%s\033[0m\n", "================================= 调整末端姿态 ===============================");

        if(HaveObjL) targetPoseTmpL[4] = targetPoseL[4]; // 获取原始末端姿态
        if(HaveObjR) targetPoseTmpR[4] = targetPoseR[4]; // 获取原始末端姿态
        MoveJoint6(targetPoseTmpL[4], targetPoseTmpR[4], moveDevice); // 当前位置基础上调整末端姿态(当前位置垂直向下, 末端目标pitch即为目标关节角)

        printf("\033[0;31m%s\033[0m\n", "================================= 手抓下降并抓取 ==============================");

        /// 修改姿态, 手臂垂直下降
        targetPoseTmpL[0] += 0.02; // 左臂
        targetPoseTmpR[0] -= 0.02; // 右臂
//        if(HaveObjL) targetPoseTmpL[4] -= M_PI; // 获取原始末端姿态
//        if(HaveObjR) targetPoseTmpR[4] -= M_PI; // 获取原始末端姿态 FIXME

        mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);

//        MovePose(targetPose, Vel_Lv1, Acc_Lv1, moveDevice);

        /// 手抓闭合
//        HandClose(0);
//        HandClose(1);

        /// 修改姿态, 手臂垂直抬起
        targetPoseTmpR[0] += 0.02; // 右臂
        targetPoseTmpL[0] -= 0.02; // 左臂

        mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);

//        MovePose(targetPose, Vel_Lv1, Acc_Lv1, moveDevice);

        printf("\033[0;31m%s\033[0m\n", "================================= 放置到指定位置 ==============================");

        std::vector<double> PlaceJointsL, PlaceJointsR, PlaceJoints;
        if (LeftObjID == 0) PlaceJointsL  = PlaceJointsL1;
        else if (LeftObjID == 3) PlaceJointsL  = PlaceJointsL2;
        else if (LeftObjID == 4) PlaceJointsL  = PlaceJointsL3;

        if (LeftObjID == 0) PlaceJointsR  = PlaceJointsR1;
        else if (LeftObjID == 3) PlaceJointsR  = PlaceJointsR2;
        else if (LeftObjID == 4) PlaceJointsR  = PlaceJointsR3;

        mergeTargetLR(PlaceJointsL, PlaceJointsR, PlaceJoints);

        MoveJoints(PlaceJoints, Vel_Lv2, Acc_Lv2, moveDevice); // 放置物体

        printf("\033[0;31m%s\033[0m\n", "=================================== 回初始位置 ===============================");

        MoveInit(); // 同步移动到初始位置, 打开手抓

        printf("\033[0;31m%s\033[0m\n\n\n", "================================== 抓取成功 =================================");
    }
}

bool GraspController::captureImage(int exposureTime) {
    ErrorInfo errorInfo;
    _captureImages.frames.clear();
    _captureImages = _cassemble2Driver->CaptureImage(errorInfo, HeadCamera, exposureTime);
    _captureImages = _cassemble2Driver->GetImage(errorInfo, HeadCamera);

    return !_captureImages.frames.empty();
}

bool GraspController::isIdle(uint16_t _Dev) {
    ErrorInfo errorInfo;
    DeviceStatus deviceStatus;
    deviceStatus._ID = _Dev;

    _cassemble2Driver->GetRobotStatus(deviceStatus, errorInfo);

    if (_Dev == LeftArm) {
        return deviceStatus._Status.at(0) == 1;
    } else if (_Dev == RightArm) {
        return deviceStatus._Status.at(1) == 1;
    } else {
        return false;
    }
}

void GraspController::Move(const std::vector<double>& targetPose, double vel, double acc, int armId) { // TODO 测试
    std::cout << "[INFO] Move ..." << std::endl;

    std::vector<double> targetJointsL;
    std::vector<double> targetJointsR;

    ErrorInfo _ErrorInfo;
    DeviceStatus _DeviceStatus{};
    std::vector<DeviceStatus> _vDeviceStatus;

    _DeviceStatus._Joints.insert(_DeviceStatus._Joints.begin(), 12, 0.0); // 赋初值

    uint16_t actionArmId = LeftArm | RightArm;
    if (armId == 0) {
        actionArmId = LeftArm;
        targetJointsL = cart2Joints(targetPose, armId); // 笛卡尔转关节角, size为6

        for (int i = 0; i < 6; i++) {
            _DeviceStatus._Joints[i] = targetJointsL[i]; // 指定pose
        }
    } else if (armId == 1) {
        actionArmId = RightArm;
        targetJointsR = cart2Joints(targetPose, armId); // 笛卡尔转关节角, size为6

        for (int i = 0; i < 6; i++) {
            _DeviceStatus._Joints[i] = targetJointsR[i]; // 指定pose
        }
    }
    else if (armId == 2) {
        actionArmId = LeftArm | RightArm;
        std::vector<double> targetJoints;
        targetJointsL = cart2Joints(targetPose, armId); // 笛卡尔转关节角, size为6
        targetJointsR = cart2Joints(targetPose, armId); // 笛卡尔转关节角, size为6
        mergeTargetLR(targetJointsL, targetJointsR, targetJoints);

        for (int i = 0; i < 12; i++) {
            _DeviceStatus._Joints[i] = targetJoints[i]; // 指定pose
        }
    }

    _DeviceStatus._Vel = {vel, vel};
    _DeviceStatus._Acc = {acc, acc};
    _DeviceStatus._ID = actionArmId;
    _vDeviceStatus.push_back(_DeviceStatus);

    std::cout << "[Move] MoveTargetJoints:" << targetPose << " acc:" << acc << " vel:" << vel << std::endl;
    std::cout << "[Move] DeviceStatus.Joints: " << _DeviceStatus._Joints << std::endl << "  ArmId: " << armId << std::endl;
    std::cout << "[Move] DeviceStatus.Vel: " <<  _DeviceStatus._Vel << "  DeviceStatus.Acc: " <<  _DeviceStatus._Acc << std::endl;

    _cassemble2Driver->MovePath(_vDeviceStatus, _ErrorInfo);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000)); /// 很关键
//    sleep(1); /// 很关键

    while (true) {
        if (armId == 0) {
            if (isIdle(LeftArm)) {
                std::cout << "[INFO] Move LeftArm 移动完成." << std::endl;
                break;
            }
        } else if (armId == 1) {
            if (isIdle(RightArm)) {
                std::cout << "[INFO] Move RightArm 移动完成." << std::endl;
                break;
            }
        }
        else if (armId == 2) {
            if (isIdle(LeftArm) && isIdle(RightArm)) {
                std::cout << "[INFO] Move LeftArm and RightArm 移动完成." << std::endl;
                break;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(200)); /// 很关键
    }
}

void GraspController::MovePose(const std::vector<double>& targetPose, double vel, double acc, int armId){
    std::cout << "[MovePose] MovePose ..." << std::endl;

    ErrorInfo _ErrorInfo;
    DeviceStatus _DeviceStatus{};
    std::vector<DeviceStatus> _vDeviceStatus;

    _DeviceStatus._Pos.insert(_DeviceStatus._Pos.begin(), 12, 0.0); // 赋初值

    uint16_t actionArmId = -1;
    if (armId == 0) {
        actionArmId = LeftArm;
        for (int i = 0; i < 6; i++) {
            _DeviceStatus._Pos[i] = targetPose[i]; // 指定pose
        }
    } else if (armId == 1) {
        actionArmId = RightArm;
        for (int i = 0; i < 6; i++) {
            _DeviceStatus._Pos[i+6] = targetPose[i+6]; // 指定pose
        }
    }
    else if (armId == 2) {
        actionArmId = LeftArm | RightArm;
        for (int i = 0; i < 12; i++) {
            _DeviceStatus._Pos[i] = targetPose[i]; // 指定pose
        }
    }

    _DeviceStatus._Vel = {vel, vel};
    _DeviceStatus._Acc = {acc, acc};
    _DeviceStatus._ID = actionArmId;
    _vDeviceStatus.push_back(_DeviceStatus);

    std::cout << "[MovePose] MoveTargetPose:" << targetPose << " acc:" << acc << " vel:" << vel << std::endl;
    std::cout << "[MovePose] DeviceStatus.Pos: " << _DeviceStatus._Pos << "  ArmId: " << armId << std::endl;
    std::cout << "[MovePose] DeviceStatus.Vel: " <<  _DeviceStatus._Vel << "  DeviceStatus.Acc: " <<  _DeviceStatus._Acc << std::endl;

    _cassemble2Driver->MovePath(_vDeviceStatus, _ErrorInfo);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000)); /// 很关键
//    sleep(1); /// 很关键

    while (true) {
        if (armId == 0) {
            if (isIdle(LeftArm)) {
                std::cout << "[MovePose] MovePose LeftArm 移动完成." << std::endl;
                break;
            }
        } else if (armId == 1) {
            if (isIdle(RightArm)) {
                std::cout << "[MovePose] MovePose RightArm 移动完成." << std::endl;
                break;
            }
        }
        else if (armId == 2) {
            if (isIdle(LeftArm) && isIdle(RightArm)) {
                std::cout << "[MovePose] MovePose LeftArm and RightArm 移动完成." << std::endl;
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200)); /// 很关键
    }
}

void GraspController::MoveJoints(const std::vector<double>& targetJoints, double vel, double acc, int armId) {
    std::cout << "[INFO] MoveJoints ..." << std::endl;

    ErrorInfo _ErrorInfo;
    DeviceStatus _DeviceStatus{};
    std::vector<DeviceStatus> _vDeviceStatus;

    _DeviceStatus._Joints.insert(_DeviceStatus._Joints.begin(), 12, 0.0); // 赋初值

    uint16_t actionArmId = LeftArm | RightArm;
    if (armId == 0) {
        actionArmId = LeftArm;
        for (int i = 0; i < 6; i++) {
            _DeviceStatus._Joints[i] = targetJoints[i]; // 指定pose
        }
    } else if (armId == 1) {
        actionArmId = RightArm;
        for (int i = 0; i < 6; i++) {
            _DeviceStatus._Joints[i+6] = targetJoints[i+6]; // 指定pose
        }
    }
    else if (armId == 2) {
        actionArmId = LeftArm | RightArm;
        for (int i = 0; i < 12; i++) {
            _DeviceStatus._Joints[i] = targetJoints[i]; // 指定pose
        }
    }

    _DeviceStatus._Vel = {vel, vel};
    _DeviceStatus._Acc = {acc, acc};
    _DeviceStatus._ID = actionArmId;
    _vDeviceStatus.push_back(_DeviceStatus);

    std::cout << "[MoveJoints] MoveTargetJoints:" << targetJoints << " acc:" << acc << " vel:" << vel << std::endl;
    std::cout << "[MoveJoints] DeviceStatus.Joints: " << _DeviceStatus._Joints << std::endl << "  ArmId: " << armId << std::endl;
    std::cout << "[MoveJoints] DeviceStatus.Vel: " <<  _DeviceStatus._Vel << "  DeviceStatus.Acc: " <<  _DeviceStatus._Acc << std::endl;

    _cassemble2Driver->MovePath(_vDeviceStatus, _ErrorInfo);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000)); /// 很关键
//    sleep(1); /// 很关键

    while (true) {
        if (armId == 0) {
            if (isIdle(LeftArm)) {
                std::cout << "[INFO] MoveJoints LeftArm 移动完成." << std::endl;
                break;
            }
        } else if (armId == 1) {
            if (isIdle(RightArm)) {
                std::cout << "[INFO] MoveJoints RightArm 移动完成." << std::endl;
                break;
            }
        }
        else if (armId == 2) {
            if (isIdle(LeftArm) && isIdle(RightArm)) {
                std::cout << "[INFO] MoveJoints LeftArm and RightArm 移动完成." << std::endl;
                break;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(200)); /// 很关键
    }
}

void GraspController::MovePath(const std::vector<double>& targetPose, double vel, double acc, int armId){
    std::cout << "[INFO] MovePath ..." << std::endl;

    ErrorInfo _ErrorInfo;
    DeviceStatus _DeviceStatus{};
    std::vector<DeviceStatus> _vDeviceStatus;
    uint16_t actionArmId = armId == 0 ? LeftArm : RightArm;

    _DeviceStatus._Pos.insert(_DeviceStatus._Pos.begin(), 12, 0.0); // 赋初值

    for (int i = 0; i < 6; i++) {
        if (armId == 0) { // 左臂
            _DeviceStatus._Pos[i] = targetPose[i]; // 指定pose
        } else if (armId == 1) { // 右臂
            _DeviceStatus._Pos[i+6] = targetPose[i]; // 指定pose
        }
    }

    _DeviceStatus._Vel = {vel, vel};
    _DeviceStatus._Acc = {acc, acc};
    _DeviceStatus._ID = actionArmId;
    _vDeviceStatus.push_back(_DeviceStatus);

    std::cout << "[MovePath] targetPose:" << targetPose << " acc:" << acc << " vel:" << vel << " id:" << armId << std::endl;
    std::cout << "[MovePath] DeviceStatus.Pos: " << _DeviceStatus._Pos << std::endl;
    std::cout << "[MovePath] DeviceStatus.Vel: " <<  _DeviceStatus._Vel << "  DeviceStatus.Acc: " <<  _DeviceStatus._Acc << std::endl;

    _cassemble2Driver->MovePath(_vDeviceStatus, _ErrorInfo);

    while (true) {
        if (isIdle(armId == 0 ? LeftArm : RightArm)) {
            std::cout << "[INFO] MovePath 移动完成." << std::endl;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

void GraspController::MoveJoint6(double targetJoint6L, double targetJoint6R, int armId) {
    std::vector<double> targetJoints = getRobotJoints();

    if (armId == 0) {
        targetJoints[5] = targetJoint6L;
    } else if (armId == 1) {
        targetJoints[11] = targetJoint6R;
    } else if (armId == 2) {
        targetJoints[5] = targetJoint6L;
        targetJoints[11] = targetJoint6R;
    }
    
    MoveJoints(targetJoints, Vel_Lv2, Acc_Lv2, armId);
}

void GraspController::MoveInit() {
    ErrorInfo _ErrorInfo;
    DeviceStatus _DeviceStatus{};
    std::vector<DeviceStatus> _vDeviceStatus;

//    uint16_t actionArmId = armId == 0 ? LeftArm : RightArm;

    // 起始位置
    _DeviceStatus._Joints = {1.62, 0.360, -1.92, -0.64, 0.026, 0.00,
                             -1.62, -0.360, 1.92, 0.64, -0.026, 0.00};
    _DeviceStatus._Vel = {1.5, 1.5};
    _DeviceStatus._Acc = {0.5, 0.5};
    _DeviceStatus._ID = LeftArm | RightArm;
    _vDeviceStatus.push_back(_DeviceStatus);

    _cassemble2Driver->MovePath(_vDeviceStatus, _ErrorInfo);

    while (true) {
        if (isIdle(LeftArm) && isIdle(RightArm)) { // isIdle 1: 空闲 0: 工作
            std::cout << "[INFO] 初始位置移动完成." << std::endl;
            break;
        }
    }

    /// 手抓张开
//    HandOpen(0);
//    HandOpen(1);
}

void GraspController::HandOpen(int armId) {
    ErrorInfo _ErrorInfo;
    DeviceStatus _DeviceStatus{};
    uint16_t actionArmId = armId == 0 ? LeftHand : RightHand;

    if(_cassemble2Driver) {
        _DeviceStatus._Pos = {0, 0}; // 开
        _DeviceStatus._Vel = {1000, 1000};
        _DeviceStatus._Acc = {1000, 1000};
        _DeviceStatus._ID = actionArmId;
        _cassemble2Driver->GraspB(_DeviceStatus, _ErrorInfo);
//        sleep(1);
    }

    _cassemble2Driver->GraspB(_DeviceStatus, _ErrorInfo);
}

void GraspController::HandClose(int armId) {
    ErrorInfo _ErrorInfo;
    DeviceStatus _DeviceStatus{};
    uint16_t actionArmId = armId == 0 ? LeftHand : RightHand;

    if(_cassemble2Driver) {
        _DeviceStatus._Pos = {1000, 1000}; // 关
        _DeviceStatus._Vel = {1000, 1000};
        _DeviceStatus._Acc = {1000, 1000};
        _DeviceStatus._ID = actionArmId;
        _cassemble2Driver->GraspB(_DeviceStatus, _ErrorInfo);
//        sleep(1);
    }

    _cassemble2Driver->GraspB(_DeviceStatus, _ErrorInfo);
}

std::vector<double> GraspController::cart2Joints(const std::vector<double> &targetPose, int armId) {

    uint16_t _Dev = armId == 0 ? LeftArm : RightArm;

    // 先读取当前关节角
    std::vector<double> initJoints = getRobotJoints(); // 获取当前关节角 size为12

    ErrorInfo errorInfo;
    DeviceStatus deviceStatus;
    deviceStatus._ID = _Dev;

    if (_Dev == LeftArm) {
        deviceStatus._Joints = initJoints;
        deviceStatus._Joints.insert(deviceStatus._Joints.begin() + 6, 6, 0);

        deviceStatus._Pos = targetPose;
        deviceStatus._Pos.insert(deviceStatus._Pos.begin() + 6, 6, 0);
    } else if (_Dev == RightArm) {
        deviceStatus._Joints.insert(deviceStatus._Joints.begin(), 6, 0);
        deviceStatus._Joints.insert(deviceStatus._Joints.end(), initJoints.begin() + 6, initJoints.end());

        deviceStatus._Pos.insert(deviceStatus._Pos.begin(), 6, 0);
        deviceStatus._Pos.insert(deviceStatus._Pos.end(), targetPose.begin() + 6, targetPose.end());
    }

    _cassemble2Driver->CartToJoint(deviceStatus, errorInfo);

    std::vector<double> targetJoints;
    if (_Dev == LeftArm) {
        targetJoints = std::vector<double>(deviceStatus._Joints.begin(), deviceStatus._Joints.begin() + 6);
    } else if (_Dev == RightArm) {
        targetJoints = std::vector<double>(deviceStatus._Joints.begin() + 6, deviceStatus._Joints.begin() + 12);
    }

    return targetJoints;
}

void GraspController::mergeTargetLR(std::vector<double> &targetL, std::vector<double> &targetR,
                                                                                    std::vector<double> &target) {
    if (targetL.empty()) targetL.insert(targetL.begin(), 6, 0.0); // 确保有值
    if (targetR.empty()) targetR.insert(targetR.begin(), 6, 0.0); // 确保有值

    target.clear();
    target.insert(target.end(), targetL.begin(), targetL.end());
    target.insert(target.end(), targetR.begin(), targetR.end());
}

std::vector<double> GraspController::getRobotJoints() { // FIXME
    ErrorInfo errorInfo;
    DeviceStatus deviceStatus;
    _cassemble2Driver->GetRobotStatus(deviceStatus, errorInfo);

    cout << "[INFO] Robot Joints: " << deviceStatus._Joints << endl;

    return deviceStatus._Joints;
}

void GraspController::cloudInit() {
    int height = _captureImages.frames.at(0).data.rows/2; // 点云高度
    int width = _captureImages.frames.at(0).data.cols/2; // 点云宽度
    printf("[INFO] image height:%d width:%d\n", height, width);

    cloud = pcl::PointCloud<pcl::PointXYZRGBA>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBA>());
    cloud->height = height;
    cloud->width = width;
    cloud->is_dense = false;
    cloud->points.resize(cloud->height * cloud->width);

    _graphicsGrasp->createLookup(cloud->width*2, cloud->height*2); // 创建坐标映射
}

void GraspController::destoryTaskThreads() {
    if (_graspControlThread.joinable()) {
        _graspControlThread.join();
        std::puts("\n结束第一阶段线程");
    }

    if (_handControlThread.joinable()) {
        _handControlThread.join();
        std::puts("\n结束第三阶段线程");
    }
}