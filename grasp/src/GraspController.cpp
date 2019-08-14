#include "GraspController.h"

using namespace std;
using namespace cv;

GraspController::GraspController() {
    _graphicsGrasp = std::make_shared<GraphicsGrasp>(); // 创建视觉支持
    _graspControlStart = false;
    _currentArmAndPointId = {0, 1};
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

    /// 2.读取配置文件 FIXME
//    std::string file_path = cobotsys::FileFinder::find(configFilePath);
//    QJsonObject jsonObject;
//
//    if (!loadJson(jsonObject, file_path)) {
//        return false;
//    }
//
//    if (!jsonObject.contains("obj_config") || !jsonObject.contains("armActionPoint_config")) {
//        return false;
//    }
//
//    std::string objConfigRelativePath = jsonObject["obj_config"].toString().toStdString();
//    std::string objAbsoluPath = cobotsys::FileFinder::find(objConfigRelativePath);
//
//    std::string armActionConfigRelativePath = jsonObject["armActionPoint_config"].toString().toStdString();
//    std::string armActionAbsoluPath = cobotsys::FileFinder::find(armActionConfigRelativePath);
//
//    if (!setupArmPath(armActionAbsoluPath)) { // 固定位置点读取
//        std::puts("\n读取配置点失败！");
//        return false;
//    }

    /// 3.启动多线程工作
    _graspControlThread = std::thread(&GraspController::graspControlThreadFunc, this);
//    _viewControlThread = std::thread(&GraspController::viewControlThreadFunc, this);
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
//    MoveInit(0); // 移动到初始位置 FIXME:待删除
//    MoveInit(1); // 移动到初始位置
    moveSync(InitPose, 2.0, 0.5);// 同步移动到初始位置

    /// 手抓张开
    HandOpen(0);
//    HandOpen(1);

    /// 点云初始化
    cloudInit();

    while (true) {
        captureImage(-100); // 采集图像

        if (!ret) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            std::cout << "graspGrasp wait capture!" << std::endl;
            continue;
        }

        if(_captureImages.frames.empty())
            return  false;
        if(_captureImages.frames.at(0).data.empty())
            return  false;

        printf("\033[0;31m%s\033[0m\n", "================================== 获取点云 =================================");

        // 创建点云使用1920*1080像素
        color = _captureImages.frames.at(0).data.clone();
        depth = _captureImages.frames.at(1).data.clone();

//        color = cv::imread("/home/hustac/rgb.jpg");
//        depth = cv::imread("/home/hustac/depth.png", -1);

        _graphicsGrasp->createPointCloud(color, depth, cloud); // 创建点云

        printf("\033[0;31m%s\033[0m\n", "=================================== 目标检测 ================================");

        /// Yolo积木检测
        RotRectsAndID = _graphicsGrasp->detectGraspYolo(color, 200, false);
        /// 正方体检测
//        RotRectsAndID = _graphicsGrasp->detectBigCube(color, 200, false);

//        writer.writeBinary("/home/hustac/test.pcd", *cloud);
//        cv::imwrite("/home/hustac/test.jpg", color);
//        cv::imwrite("/home/hustac/test.png", depth);
//        exit(0);

//        cloud_viewer(); // 显示点云 FIXME

//        graspDetectGPD(); // TODO:生成抓取姿态

        printf("\033[0;31m%s\033[0m\n", "================================= 生成移动任务 ===============================");

        if (RotRectsAndID.first.empty()) {
            printf("[ERROR] Did not get any rot rects!\n");
            continue;
        }
//        for (size_t i = 0; i < RotRectsAndID.first.size(); i++) {
        for (size_t i = 0; i < 1; i++) {

            int leftOrright = ((int)RotRectsAndID.first[i].center.x < 410) ? 0 : 1; // 简单处理左右手分工

            /// 获取积木位置
            std::vector<double> targetPose = _graphicsGrasp->getObjPose(RotRectsAndID.first[i], cloud, 0, 0, leftOrright);
            /// 获取正方体位置
//            std::vector<double> targetPose = _graphicsGrasp->getObjPose(RotRectsAndID.first[i], cloud, 1, 0, leftOrright);

            /// 修改姿态, 移动到物体上方
            if (leftOrright == 1) { // 右臂
                targetPose[0] += 0.30;
                targetPose[3] = 1.54;
                targetPose[4] = RotRectsAndID.first[i].angle * M_PI / 180.0;
                targetPose[5] = -1.54;
            } else if (leftOrright == 0) { // 左臂
                targetPose[0] -= 0.30; // 碰:0.23 不碰:0.25 0.3(大爪子)
                targetPose[3] = -1.54;
                targetPose[4] = RotRectsAndID.first[i].angle * M_PI / 180.0;
                targetPose[5] = -1.54;
            }
            movePath(targetPose, 0.25, 0.05, leftOrright);

            printf("[INFO] 待抓取物体信息 ID:[%d] Angle:[%f] left/right[%d] Pose:[%f,%f,%f,%f,%f,%f]\n\n",
                   RotRectsAndID.second[i],
                   RotRectsAndID.first[i].angle, leftOrright, targetPose[0], targetPose[1],
                   targetPose[2], targetPose[3], targetPose[4], targetPose[5]);

//            MoveMiddle(leftOrright); /// 移动到抓取中间点(待删除)
//            movePath(MiddlePose, 0.5, 0.05, leftOrright); /// 移动到抓取中间点(NEW)


            /// 修改姿态, 手臂垂直下降
            if (leftOrright == 1) { // 右臂
                targetPose[0] -= 0.05;
            } else if (leftOrright == 0) {
                targetPose[0] += 0.05;
            }
            movePath(targetPose, 0.25, 0.05, leftOrright);

            /// 修改姿态, 手臂垂直抬起
            if (leftOrright == 1) { // 右臂
                targetPose[0] += 0.05;
            } else if (leftOrright == 0) {
                targetPose[0] -= 0.05;
            }
            movePath(targetPose, 0.25, 0.05, leftOrright);

            HandClose(leftOrright); /// 抓取物体

//            movePath(Pose2, 0.05, 0.05, leftOrright); // 移动到平躺初始位置

            MoveInit(leftOrright); // 移动到初始位置
        }

        printf("\033[0;31m%s\033[0m\n\n\n", "================================== 抓取成功 =================================");
//        exit(666);

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
//    moveSync(InitPose, 2.0, 0.5); // 同步移动到初始位置

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

        color = cv::imread("../../../grasp/data/images/cube1.jpg");
        depth = cv::imread("../../../grasp/data/images/cube1.png", -1);

        _graphicsGrasp->createPointCloud(color, depth, cloud); // 创建点云

        printf("\033[0;31m%s\033[0m\n", "=================================== 目标检测 ================================");

        const int juggleOrCube = 0; /// 0为积木, 1为立方体

        if (juggleOrCube == 0) {
            /// Yolo积木检测
            RotRectsAndID = _graphicsGrasp->detectGraspYolo(color, 200, false);
        } else if (juggleOrCube == 1) {
            /// 正方体检测
            RotRectsAndID = _graphicsGrasp->detectBigCube(color, 200, false);
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
        std::vector<int> AimObjIndicesLR = _graphicsGrasp->findAimObjLR(RotRectsAndID, LeftOrRightThresh, 0);

        int leftOrRight;

        vector<double> targetPose; // 双臂目标位置

        /// 获取目标位置(左臂)
        leftOrRight = 0;
        cv::RotatedRect RotatedRectL = RotRectsAndID.first[AimObjIndicesLR[0]];
        std::vector<double> targetPoseL = _graphicsGrasp->getObjPose(RotatedRectL, cloud, juggleOrCube, 0, leftOrRight);

        // 移动到物体上方
        targetPoseL[0] -= 0.30; // 碰:0.23 不碰:0.25 0.3(大爪子)

        /// 获取目标位置(右臂)
        leftOrRight = 1;
        cv::RotatedRect RotatedRectR = RotRectsAndID.first[AimObjIndicesLR[1]];
        std::vector<double> targetPoseR = _graphicsGrasp->getObjPose(RotatedRectR, cloud, juggleOrCube, 0, leftOrRight);

        // 移动到物体上方
        targetPoseR[0] += 0.30; // 碰:0.23 不碰:0.25 0.3(大爪子)

        mergeTargetPoseLR(targetPoseL, targetPoseR, targetPose);

        cout << "[INFO] targetPoseL: " << targetPoseL << endl;
        cout << "[INFO] targetPoseR: " << targetPoseR << endl;
        cout << "[INFO] targetPose: " << targetPose << endl;

//        moveSync(targetPose, 0.25, 0.05); /// 双臂同时移动到物体上方

        printf("[INFO] 左臂待抓取物体信息 ID:[%d] Angle:[%f] Pose:[%f,%f,%f,%f,%f,%f]\n\n",
               RotRectsAndID.second[AimObjIndicesLR[0]], R2D(targetPoseL[4]), targetPoseL[0], targetPoseL[1],
               targetPoseL[2], targetPoseL[3], targetPoseL[4], targetPoseL[5]);

        printf("[INFO] 右臂待抓取物体信息 ID:[%d] Angle:[%f] Pose:[%f,%f,%f,%f,%f,%f]\n\n",
               RotRectsAndID.second[AimObjIndicesLR[1]], R2D(targetPoseR[4]), targetPoseR[0], targetPoseR[1],
               targetPoseR[2], targetPoseR[3], targetPoseR[4], targetPoseR[5]);

//            moveSync(MiddlePose, 0.5, 0.05); /// 移动到抓取中间点(NEW)

        /// 修改姿态, 手臂垂直下降
        targetPoseR[0] -= 0.05; // 右臂
        targetPoseL[0] += 0.05; // 左臂

        mergeTargetPoseLR(targetPoseL, targetPoseR, targetPose);

        moveSync(targetPose, 0.25, 0.05);

        /// 修改姿态, 手臂垂直抬起
        targetPoseR[0] += 0.05; // 右臂
        targetPoseL[0] -= 0.05; // 左臂

        mergeTargetPoseLR(targetPoseL, targetPoseR, targetPose);

        moveSync(targetPose, 0.25, 0.05);

        /// 手抓闭合
        HandClose(0); /// 抓取物体
        HandClose(1); /// 抓取物体

//            movePath(Pose2, 0.05, 0.05, leftOrRight); // 移动到平躺初始位置

        moveSync(InitPose, 2.0, 0.5); // 同步移动到初始位置


        printf("\033[0;31m%s\033[0m\n\n\n", "================================== 抓取成功 =================================");
//        exit(666);
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

void GraspController::movePath(const std::vector<double>& targetPose, double vel, double acc, int armId){
    ErrorInfo _ErrorInfo;
    DeviceStatus _DeviceStatus = DeviceStatus();
    std::vector<DeviceStatus> _vDeviceStatus;
    uint16_t actionArmId = armId == 0 ? LeftArm : RightArm;

    _DeviceStatus._Pos = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // FIXME 待删除
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    _DeviceStatus._Pos.insert(_DeviceStatus._Pos.begin(), 12, 0.0);// 清空

    if (armId == 0) { // 左臂
        for (int i = 0; i < 6; i++) {
            _DeviceStatus._Pos[i] = targetPose[i]; // 指定pose
        }
    } else if (armId == 1){ // 右臂
        for (int i = 0; i < 6; i++) {
            _DeviceStatus._Pos[i+6] = targetPose[i]; // 指定pose
        }
    }

    _DeviceStatus._Vel = {vel, vel};
    _DeviceStatus._Acc = {acc, acc};
    _DeviceStatus._ID = actionArmId;
    _vDeviceStatus.push_back(_DeviceStatus);

    _cassemble2Driver->MovePath(_vDeviceStatus, _ErrorInfo);

    std::cout << "[MovePath] targetPose:" << targetPose << " acc:" << acc << " vel:" << vel << " id:" << armId << std::endl;
    std::cout << "[MovePath] DeviceStatus.Pos: " << _DeviceStatus._Pos << std::endl;
    std::cout << "[MovePath] DeviceStatus.Vel: " <<  _DeviceStatus._Vel << "  DeviceStatus.Acc: " <<  _DeviceStatus._Acc << std::endl;

    while (true) {
        if (isIdle(armId == 0 ? LeftArm : RightArm)) {
            std::cout << "[INFO] MovePath 移动完成." << std::endl;
            break;
        }
//        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

void GraspController::moveSync(const std::vector<double>& targetPose, double vel, double acc){
    ErrorInfo _ErrorInfo;
    DeviceStatus _DeviceStatus = DeviceStatus();
    std::vector<DeviceStatus> _vDeviceStatus;

    _DeviceStatus._Pos.insert(_DeviceStatus._Pos.begin(), 12, 0.0); // 赋初值

    for (int i = 0; i < 12; i++) {
        _DeviceStatus._Pos[i] = targetPose[i]; // 指定pose
    }

    _DeviceStatus._Vel = {vel, vel};
    _DeviceStatus._Acc = {acc, acc};
    _DeviceStatus._ID = LeftArm | RightArm;
    _vDeviceStatus.push_back(_DeviceStatus);

    _cassemble2Driver->MovePath(_vDeviceStatus, _ErrorInfo);

    std::cout << "[MoveSync] targetPose:" << targetPose << " acc:" << acc << " vel:" << vel << std::endl;
    std::cout << "[MoveSync] DeviceStatus.Pos: " << _DeviceStatus._Pos << std::endl;
    std::cout << "[MoveSync] DeviceStatus.Vel: " <<  _DeviceStatus._Vel << "  DeviceStatus.Acc: " <<  _DeviceStatus._Acc << std::endl;

    while (true) {
        if (isIdle(LeftArm) && isIdle(RightArm)) {
            std::cout << "[INFO] MoveSync 移动完成." << std::endl;
            break;
        }
//        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

void GraspController::MoveInit(int armId) {

    ErrorInfo _ErrorInfo;
    DeviceStatus _DeviceStatus{};
    std::vector<DeviceStatus> _vDeviceStatus;

    uint16_t actionArmId = armId == 0 ? LeftArm : RightArm;

    // 起始位置
    _DeviceStatus._Joints = {1.62, 0.360, -1.92, -0.64, 0.026, 0.00,
                             -1.62, -0.360, 1.92, 0.64, -0.026, 0.00};
    _DeviceStatus._Vel = {2.0, 2.0};
    _DeviceStatus._Acc = {0.5, 0.5};
    _DeviceStatus._ID = actionArmId;
    _vDeviceStatus.push_back(_DeviceStatus);

    _cassemble2Driver->MovePath(_vDeviceStatus, _ErrorInfo);

    while (true) {
        if (isIdle(armId == 0 ? LeftArm : RightArm)) {
            std::cout << "[INFO] 初始位置移动完成." << std::endl;
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    HandOpen(actionArmId);
}

// FIXME (待删除)
void GraspController::MoveMiddle(int armId) {

    ErrorInfo _ErrorInfo;
    DeviceStatus _DeviceStatus{};
    std::vector<DeviceStatus> _vDeviceStatus;

    uint16_t actionArmId = armId == 0 ? LeftArm : RightArm;

    // 起始位置
    _DeviceStatus._Joints = {1.62, 0.920, -1.92, -0.64, 0.026, 0.00,
                             -1.62, 0.920, 1.92, 0.64, -0.026, 0.00};
    _DeviceStatus._Vel = {0.5, 0.5};
    _DeviceStatus._Acc = {0.15, 0.15};
    _DeviceStatus._ID = actionArmId;
    _vDeviceStatus.push_back(_DeviceStatus);

    _cassemble2Driver->MovePath(_vDeviceStatus, _ErrorInfo);

    while (true) {
        if (isIdle(armId == 0 ? LeftArm : RightArm)) {
            std::cout << "[INFO] Middle 位置移动完成" << std::endl;
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
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

int GraspController::cart2Joints(const std::vector<double> &initJoints, const std::vector<double> &targetPos,
                                std::vector<double> &targetJoints, uint16_t _Dev) {
    ErrorInfo errorInfo;
    DeviceStatus deviceStatus;
    deviceStatus._ID = _Dev;

    if (_Dev == LeftArm) {
        deviceStatus._Joints = initJoints;
        deviceStatus._Pos = targetPos;

        deviceStatus._Joints.insert(deviceStatus._Joints.begin() + 6, 6, 0);
        deviceStatus._Pos.insert(deviceStatus._Pos.begin() + 6, 6, 0);
    } else if (_Dev == RightArm) {
        deviceStatus._Joints.insert(deviceStatus._Joints.begin(), 6, 0);
        deviceStatus._Joints.insert(deviceStatus._Joints.end(), initJoints.begin(), initJoints.end());
        deviceStatus._Pos.insert(deviceStatus._Pos.begin(), 6, 0);
        deviceStatus._Pos.insert(deviceStatus._Pos.end(), targetPos.begin(), targetPos.end());
    }

    _cassemble2Driver->CartToJoint(deviceStatus, errorInfo);

    if (_Dev == LeftArm) {
        targetJoints = std::vector<double>(deviceStatus._Joints.begin(), deviceStatus._Joints.begin() + 6);
    } else if (_Dev == RightArm) {
        targetJoints = std::vector<double>(deviceStatus._Joints.begin() + 6, deviceStatus._Joints.begin() + 12);
    } else {
        targetJoints = deviceStatus._Joints;
    }
    return 0;
}

int GraspController::joint2Cart(const std::vector<double> &targetJoints, std::vector<double> &targetPos, uint16_t _Dev) {
    ErrorInfo errorInfo;
    DeviceStatus deviceStatus = DeviceStatus();
    deviceStatus._Pos.insert(deviceStatus._Pos.begin(), 12, 0); // 清空pos
    deviceStatus._ID = _Dev;
    if (_Dev == LeftArm) {
        deviceStatus._Joints = targetJoints;
        deviceStatus._Joints.insert(deviceStatus._Joints.end(), 6, 0);
    } else if (_Dev == RightArm) {
        deviceStatus._Joints.insert(deviceStatus._Joints.begin(), 6, 0);
        deviceStatus._Joints.insert(deviceStatus._Joints.end(), targetJoints.begin(), targetJoints.end());
    }
    _cassemble2Driver->JointToCart(deviceStatus, errorInfo);

    targetPos.clear();
    if (_Dev == LeftArm) {
        std::copy(deviceStatus._Pos.begin(), deviceStatus._Pos.begin() + 6, std::back_inserter(targetPos));
    } else if (_Dev == RightArm) {
        std::copy(deviceStatus._Pos.begin() + 6, deviceStatus._Pos.begin() + 12, std::back_inserter(targetPos));
    } else {
        targetPos.resize(12);
        targetPos = deviceStatus._Pos;
    }

    std::cout << "关节角" << targetJoints << std::endl;
    std::cout << "目标点的笛卡尔坐标" << targetPos << std::endl;

    return 0;

}

void GraspController::mergeTargetPoseLR(std::vector<double> &targetPoseL, std::vector<double> &targetPoseR,
                                                                                    std::vector<double> &targetPose) {
    targetPose.clear();
    targetPose.insert(targetPose.end(),targetPoseL.begin(),targetPoseL.end());
    targetPose.insert(targetPose.end(), targetPoseR.begin(), targetPoseR.end());
}

std::vector<double> GraspController::getRobotJoints(uint16_t _Dev) {
    ErrorInfo errorInfo;
    DeviceStatus deviceStatus;
    deviceStatus._ID = _Dev;
    _cassemble2Driver->GetRobotStatus(deviceStatus, errorInfo);

    if (_Dev == LeftArm) {
        return std::vector<double>(deviceStatus._Joints.begin(), deviceStatus._Joints.begin() + 6);
    } else if (_Dev == RightArm) {
        return std::vector<double>(deviceStatus._Joints.begin() + 6, deviceStatus._Joints.begin() + 12);
    } else {
        return deviceStatus._Joints;
    }
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

    if (_viewControlThread.joinable()) {
        _viewControlThread.join();
        std::puts("\n结束第二阶段线程");
    }

    if (_handControlThread.joinable()) {
        _handControlThread.join();
        std::puts("\n结束第三阶段线程");
    }
}

std::vector<float> GraspController::RVec2DVec(const std::vector<float> &RVec) {
    std::vector<float> DVec;
    for (size_t i = 0; i < RVec.size(); i++) {
        DVec[i] = RVec[i] * 180.0f/M_PI;
    }
    cout << "Vector in Rad to Vector in Deg:" << endl;
    cout << "Vector in Rad: " << RVec << endl;
    cout << "Vector in Deg: " << DVec << endl;
     return DVec;
}

std::vector<float> GraspController::DVec2RVec(const std::vector<float> &DVec) {
    std::vector<float> RVec;
    for (size_t i = 0; i < DVec.size(); i++) {
        RVec[i] = DVec[i] * M_PI/180.0f;
    }
    cout << "Vector in Deg to Vector in Rad:" << endl;
    cout << "Vector in Deg: " << DVec << endl;
    cout << "Vector in Rad: " << RVec << endl;
    return RVec;
}





