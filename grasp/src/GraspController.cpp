#include "GraspController.h"

using namespace std;
using namespace cv;

GraspController::GraspController() {
    _graphicsGrasp = std::make_shared<GraphicsGrasp>(); // 创建视觉支持
    // FIXME：获取相机参数
//    _graphicsGrasp->Setup("../../../data/caliresult/3DCameraInCailResult.xml",
//               "../../../data/caliresult/3DCameraExCailResult_left.xml",
//               "../../../data/caliresult/3DCameraExCailResult_right.xml");

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

        //机器人复位 FIXME：线程问题
//        resetRobot();

        //TODO 抓取
        if(!graspControl()) return;

        //机器人复位到初始位置
        resetRobot();
        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double , std::ratio<1,1> > duration_s(endTime - startTime);
        std::cout << "解魔方完成! 总体耗时：   " << duration_s.count() << "  秒 "<< std::endl;
        break;
    }
}


bool GraspController::graspControl() {
    bool ret = captureImage(-100); // 采集图像
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    int height = _captureImages.frames.at(0).data.rows/2; // 点云高度
    int width = _captureImages.frames.at(0).data.cols/2; // 点云宽度
    printf("[INFO] image height:%d width:%d\n", height, width);

    cloud = pcl::PointCloud<pcl::PointXYZRGBA>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBA>());
    cloud->height = height;
    cloud->width = width;
    cloud->is_dense = false;
    cloud->points.resize(cloud->height * cloud->width);

    _graphicsGrasp->createLookup(cloud->width*2, cloud->height*2); // 创建坐标映射

    while (true) {
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

        color = cv::imread("/home/hustac/rgb.jpg");
        depth = cv::imread("/home/hustac/depth.png", -1);

        _graphicsGrasp->createPointCloud(color, depth, cloud); // 创建点云

        RotRectsAndID = _graphicsGrasp->detectGraspYolo(color); // Yolo积木检测

//        writer.writeBinary("/home/hustac/test.pcd", *cloud);
//        cloud_viewer(); // 显示点云 FIXME

//        graspDetectGPD(); // TODO:生成抓取姿态

        printf("\033[0;31m%s\033[0m\n", "================================= 生成移动任务 ===============================");

        int leftOrright = ((int)RotRectsAndID.first[0].center.x <= width/2) ? 0 : 1; // 简单处理左右手分工

        std::vector<double> targetPose = _graphicsGrasp->getObjPose(RotRectsAndID.first[0], cloud, leftOrright); // 获取积木中心点

        // 修改姿态
        if (leftOrright == 1) { // 右臂
            targetPose[0] -= 0.2;
            targetPose[3] = 1.54;
            targetPose[4] = RotRectsAndID.first[0].angle * M_PI / 180.0;
            targetPose[5] = -1.54;
        } else if (leftOrright == 0) {
            targetPose[0] += 0.2;
            targetPose[3] = -1.54;
            targetPose[4] = RotRectsAndID.first[0].angle * M_PI / 180.0;
            targetPose[5] = 1.54;
        }

        printf("[INFO] 带抓取物体信息 ID:[%d] Angle:[%f] left/right[%d] Pose:[%f,%f,%f,%f,%f,%f]\n", RotRectsAndID.second[0],
               RotRectsAndID.first[0].angle, leftOrright, targetPose[0], targetPose[1],
               targetPose[2], targetPose[3], targetPose[4], targetPose[5]);


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

// FIXME:待测试
void GraspController::movePath(const std::vector<double>& targetPose, double acc, double vel, int armId){
    ErrorInfo _ErrorInfo;
    DeviceStatus _DeviceStatus;
    std::vector<DeviceStatus> _vDeviceStatus;
    uint16_t actionArmId = armId == 0 ? LeftArm : RightArm;

    if (armId == 0) { // 左臂
        for (int i = 0; i < 7; i++) {
            _DeviceStatus._Pos[i] = targetPose[i]; // 指定pose
        }
    } else if (armId == 1){ // 右臂
        for (int i = 0; i < 7; i++) {
            _DeviceStatus._Pos[i+6] = targetPose[i]; // 指定pose
        }
    }

    _DeviceStatus._Vel = {vel, vel};
    _DeviceStatus._Acc = {acc, acc};
    _DeviceStatus._ID = actionArmId;
    _vDeviceStatus.push_back(_DeviceStatus);

    //        _DeviceStatus._Pos = {0.0, -0.213, 0.782, 1.57, 0, 0,
    //                              0.0, -0.213, 0.782, 1.57, 0, 0};
    //        _vDeviceStatus.push_back(_DeviceStatus);

    _cassemble2Driver->MovePath(_vDeviceStatus, _ErrorInfo);

    std::cout << "[MovePath] targetPose:" << targetPose << " acc:" << acc << " vel:" << vel << " id:" << armId << std::endl;
}

void GraspController::moveOnece(const std::vector<double>& targetJoints, ArmRobotMotionType motionType, double acc, double vel,
                                int armId) {
    std::cout << "移动任务： " << targetJoints << "  ," << acc << "  ," << vel << "  , " << armId << std::endl;
    DeviceStatus deviceStatus;
    uint16_t actionArmId = armId == 0 ? LeftArm : RightArm;
    deviceStatus._ID = actionArmId;
    deviceStatus._Acc = {acc};
    deviceStatus._Vel = {vel};
//        deviceStatus._Acc = {0.1};
//        deviceStatus._Vel = {0.2};
    deviceStatus._Joints = targetJoints;
    ErrorInfo errorInfo;

    if (motionType == ArmRobotMotionType::MoveL) {
        joint2Cart(deviceStatus._Joints, deviceStatus._Pos, actionArmId);
        deviceStatus._Joints.clear();
    }

    std::vector<DeviceStatus> movePath{deviceStatus};
    _cassemble2Driver->MovePath(movePath, errorInfo);

    while (true) {
        if (isIdle(armId == 0 ? LeftArm : RightArm)) {
            std::cout << "移动完成" << std::endl;
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

//TODO: 当前不可用
void GraspController::moveCart(std::vector<double> targetPose, double acc, double vel, int armId) {
    std::cout << "移动任务（欧拉角）： " << targetPose << "  ," << acc << "  ," << vel << "  , " << armId << std::endl;
    ErrorInfo errorInfo;
    DeviceStatus deviceStatus;
    std::vector<DeviceStatus> vDeviceStatus;
    uint16_t actionArmId = armId == 0 ? LeftArm : RightArm;
    deviceStatus._ID = actionArmId;
    deviceStatus._Acc = {acc};
    deviceStatus._Vel = {vel};
    if (armId == 0) {
        for (int i = 0; i < 7; i++) {
            deviceStatus._Pos[i] = targetPose[i]; // 指定pose
        }
    } else if (armId == 1){
        for (int i = 0; i < 7; i++) {
            deviceStatus._Pos[i+6] = targetPose[i]; // 指定pose
        }
    }

    vDeviceStatus.push_back(deviceStatus);
    _cassemble2Driver->MovePath(vDeviceStatus, errorInfo);

    while (true) {
        if (isIdle(armId == 0 ? LeftArm : RightArm)) {
            std::cout << "移动完成(moveCart)" << std::endl;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

void GraspController::moveQuat(Eigen::Vector3d targetPosition, Eigen::Matrix3d targetQuat, ArmRobotMotionType motionType,
                               double acc, double vel, int armId) {
    std::cout << "移动任务： " << targetQuat << "  ," << acc << "  ," << vel << "  , " << armId << std::endl;

    // 旋转矩阵转欧拉角
    Eigen::Vector3d eulerAngle = targetQuat.matrix().eulerAngles(2,1,0);
    cout << "yaw(z) pitch(y) roll(x) = " << eulerAngle.transpose() << endl;
    cout << "eulerAngle: " << eulerAngle.x() << " " << eulerAngle.y() << " " << eulerAngle.z() << endl;

    std::vector<double> targetPose;
    targetPose[0] = targetPosition.x();
    targetPose[1] = targetPosition.y();
    targetPose[2] = targetPosition.z();
    targetPose[3] = eulerAngle.x();
    targetPose[4] = eulerAngle.y();
    targetPose[5] = eulerAngle.z();

    DeviceStatus deviceStatus;
    uint16_t actionArmId = armId == 0 ? LeftArm : RightArm;
    deviceStatus._ID = actionArmId;
    deviceStatus._Acc = {acc};
    deviceStatus._Vel = {vel};
    deviceStatus._Pos = targetPose; // 指定pose
    ErrorInfo errorInfo;

    std::vector<DeviceStatus> movePath{deviceStatus};
    _cassemble2Driver->MovePath(movePath, errorInfo);

    while (true) {
        if (isIdle(armId == 0 ? LeftArm : RightArm)) {
            std::cout << "移动完成(moveQuat)" << std::endl;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
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

void GraspController::resetRobot() {

    std::cout << "机器人开始复位！！" << std::endl;

    auto thread1 = std::async(std::launch::async, &GraspController::moveOnece, this, _startHome.at(0),
                              ArmRobotMotionType::MoveJ, 1.5, 1.5, 0);
//    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    auto thread2 = std::async(std::launch::async, &GraspController::moveOnece, this, _startHome.at(1),
                              ArmRobotMotionType::MoveJ, 1.5, 1.5, 1);

    thread1.wait();
    thread2.wait();

//    handControl(LeftHand, YinShiDriver::OpenHand);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
//    handControl(RightHand, YinShiDriver::OpenHand);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
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





