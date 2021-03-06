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
    sleep(10);
}

void GraspController::stop() {
    ErrorInfo errorInfo;
    _cassemble2Driver->Stop(errorInfo);
}

bool GraspController::setup(int taskNum, const std::string& configFilePath) {

    taskNum_ = taskNum;

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

        printf("\033[0;33m[INFO] <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Do Task %d >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\033[0m\n\n", taskNum_);
        // 异步抓取
//        if(!graspControl()) return;

        // 同步抓取
        if (taskNum_ == 1) {
//            if(!graspControlDual()) return;
            if(!graspControlJuggleFirst()) return;
        } else if (taskNum_ == 2) {
            if (!graspControlTask2()) return;
        } else if (taskNum_ == 3) {
            if (!graspControlTask3()) return;
        } else if (taskNum_ == 4) {
            if (!graspControlTask4()) return;
        }

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
            std::vector<double> targetPose;
            if (!_graphicsGrasp->getObjPose(RotRectsAndID.first[0], targetPose, cloud, 0, 0, leftOrright)) continue;

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
            std::vector<double> targetPose;
            _graphicsGrasp->getObjPose(BigBallRect, targetPose, cloud, 0, 0, leftOrright);
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

    printf("\033[0;31m%s\033[0m\n", "================================== 初始化成功 =================================");

    while (true) {
        printf("\033[0;31m%s\033[0m\n", "================================== 采集图像 =================================");
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

        if (saveFlag) { // 每次运行程序保存一次数据
            saveCloudAndImages();
            saveFlag = false;
//            exit(1);
        }

        printf("\033[0;31m%s\033[0m\n", "=================================== 目标检测 ================================");

        const int juggleOrCube = 0; /// 0为积木, 1为立方体

        bool BigBallFlag = false, BigCubeFlag = false;
        std::pair<cv::RotatedRect, int> BigBallRect, BigCubeRect;

        // 球体检测
        if (!BallPicked) BigBallFlag = _graphicsGrasp->detectBigBall(color, cloud, BigBallRect, 0);

        if (!BigBallFlag) { // 无球再检测立方体
            if (!BigCubePicked) {
                BigCubeFlag = _graphicsGrasp->detectBigCube(color, cloud, BigCubeRect, 0);
            }
        }

        if (juggleOrCube == 0 && !BigBallFlag) {
            /// Yolo积木检测
//            RotRectsAndID = _graphicsGrasp->detectGraspYolo(color, 200, 0);
            RotRectsAndID = _graphicsGrasp->detectGraspYoloPro(color, cloud, 120, 0);
        } else if (juggleOrCube == 1) {
            /// 带孔正方体检测
//            RotRectsAndID = _graphicsGrasp->detectBigObj(color, cloud, 0, 200, false);
        }

        printf("\033[0;31m%s\033[0m\n", "=================================== 回初始位置 ===============================");
        MoveJoints(IniteJoints, Vel_Lv3, Acc_Lv3, 2);
//        MoveJoints(IniteJointsOld, Vel_Lv3, Acc_Lv3, 2);

        printf("\033[0;31m%s\033[0m\n", "================================= 生成移动任务 ===============================");

        if (!BigBallFlag && !BigCubeFlag) {
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
            } else if ((AimObjIndicesLR[0] = -1) && (AimObjIndicesLR[1] = -1)) continue;

            printf("\033[0;31m%s\033[0m\n", "============================= 移动到积木抓取初始位置 ===========================");
            MoveJoints(JuggleIniteJoints, Vel_Lv3, Acc_Lv3, 2);

            printf("\033[0;31m%s\033[0m\n", "================================= 移动到正上方 ===============================");

            bool lieFlagR = true, lieFlagL = true;

            /// 获取目标位置(左臂)
            if (AimObjIndicesLR[0] != -1) { // 左侧有物体
                cv::RotatedRect RotatedRectL = RotRectsAndID.first[AimObjIndicesLR[0]];
                LeftObjID = RotRectsAndID.second[AimObjIndicesLR[0]];
                // 获取机器人坐标系下坐标
                if (!_graphicsGrasp->getObjPose(RotatedRectL, targetPoseL, cloud, juggleOrCube, 0, 0)) continue;

                targetPoseTmpL = targetPoseL; // 目标位置副本

                /// 高低判别
                if (targetPoseL[0] > _graphicsGrasp->lieThreshL) { // 躺着的
                    targetPoseTmpL[0] = _graphicsGrasp->height_Lv2_L;
                    printf("\033[0;31m%s\033[0m\n", "[INFO] 左臂待抓取物体为躺着的");
                    lieFlagL = true;
                } else { // 立着的
                    targetPoseTmpL[0] = _graphicsGrasp->height_Lv1_L;
                    printf("\033[0;31m%s\033[0m\n", "[INFO] 左臂待抓取物体为立着的");
                    lieFlagL = false;
                }
//                targetPoseTmpL[0] = _graphicsGrasp->height_Lv2_L;

                // 移动到物体上方
                targetPoseTmpL[4] = 0;

                cout << "[INFO] targetPoseL: " << targetPoseL << endl;

                printf("[INFO] 左臂待抓取物体信息 ID:[%d] Angle:[%f(%f)] Pose:[%f,%f,%f,%f,%f,%f]\n\n",
                       RotRectsAndID.second[AimObjIndicesLR[0]], targetPoseL[4], R2D(targetPoseL[4]), targetPoseTmpL[0],
                       targetPoseTmpL[1], targetPoseTmpL[2], targetPoseTmpL[3], targetPoseTmpL[4], targetPoseTmpL[5]);
            }

            /// 获取目标位置(右臂)
            if (AimObjIndicesLR[1] != -1) { // 右侧有物体
                cv::RotatedRect RotatedRectR = RotRectsAndID.first[AimObjIndicesLR[1]];
                RightObjID = RotRectsAndID.second[AimObjIndicesLR[1]];
                // 获取机器人坐标系下坐标
                if (!_graphicsGrasp->getObjPose(RotatedRectR, targetPoseR, cloud, juggleOrCube, 0, 1)) continue;
                targetPoseTmpR = targetPoseR; // 目标位置副本

                /// 高低判别
                if (targetPoseR[0] < _graphicsGrasp->lieThreshR) { // 躺着的
                    targetPoseTmpR[0] = _graphicsGrasp->height_Lv2_R;
                    printf("\033[0;31m%s\033[0m\n", "[INFO] 右臂待抓取物体为躺着的");
                    lieFlagR = true;
                } else { // 立着的
                    targetPoseTmpR[0] = _graphicsGrasp->height_Lv1_R;
                    printf("\033[0;31m%s\033[0m\n", "[INFO] 右臂待抓取物体为立着的");
                    lieFlagR = false;
                }
//                targetPoseTmpR[0] = _graphicsGrasp->height_Lv2_R;

                // 移动到物体上方
                targetPoseTmpR[4] = 0;

                cout << "[INFO] targetPoseR: " << targetPoseR << endl;

                printf("[INFO] 右臂待抓取物体信息 ID:[%d] Angle:[%f(%f)] Pose:[%f,%f,%f,%f,%f,%f]\n\n",
                       RotRectsAndID.second[AimObjIndicesLR[1]], targetPoseR[4], R2D(targetPoseR[4]), targetPoseTmpR[0],
                       targetPoseTmpR[1],
                       targetPoseTmpR[2], targetPoseTmpR[3], targetPoseTmpR[4], targetPoseTmpR[5]);
            }

            /// 处理待抓取的两个物体靠的太近
            if (AimObjIndicesLR[0] != -1 && AimObjIndicesLR[1] != -1) {
                float distance = RotRectsAndID.first[AimObjIndicesLR[1]].center.x -
                                 RotRectsAndID.first[AimObjIndicesLR[0]].center.x;
                printf("[INFO] Distance between Obj in pix: %f\n", distance);
                if (distance < 100) { // FIXME
                    printf("\033[0;32m%s\033[0m\n",
                           "[WARN] Distance is too short between the Objects, left arm first!\n");
                    moveDevice = 0; // 物体靠的太近, 左臂优先
                }
            }

            mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);

            cout << "[INFO] targetPose: " << targetPose << endl;

            MovePose(targetPose, Vel_Lv2, Acc_Lv2, moveDevice);

            printf("\033[0;31m%s\033[0m\n", "================================= 调整末端姿态 ===============================");

            targetPose = getRobotJoints(2); // 获取当前关节角
            if (HaveObjL) targetPose[5] += D2R(180 + R2D(targetPoseL[4])); // 获取原始末端关节角, 加补角
            if (HaveObjR) targetPose[11] += targetPoseR[4]; // 获取原始末端关节角

            MoveJoints(targetPose, Vel_Lv4, Acc_Lv4, 2);

            printf("\033[0;31m%s\033[0m\n", "================================= 手抓下降并抓取 ==============================");

            /// 修改姿态, 手臂垂直下降
            targetPose = getRobotPose(2); // 获取双臂当前位置
            if (lieFlagL) targetPose[0] += 0.03; // 左臂
            else targetPose[0] += 0.09; // 左臂

            if (lieFlagR) targetPose[6] -= 0.03; // 右臂
            else targetPose[6] -= 0.09; // 右臂

            MovePose(targetPose, Vel_Lv2_, Acc_Lv2_, moveDevice); // FIXME

            /// 手抓闭合
            HandClose(0);
            HandClose(1);

            /// 修改姿态, 手臂垂直抬起
            targetPoseTmpR[0] += 0.01; // 右臂
            targetPoseTmpL[0] -= 0.01; // 左臂

            mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);

            MovePose(targetPose, Vel_Lv2_, Acc_Lv2_, moveDevice);

            printf("\033[0;31m%s\033[0m\n", "================================= 放置到指定位置 ==============================");

            std::vector<double> PlaceJointsL, PlaceJointsR, PlaceJoints;
            bool placeFlagL = true, placeFlagR = true;
            if (LeftObjID == 0) PlaceJointsL = PlaceJointsL1;
            else if (LeftObjID == 1) PlaceJointsL = PlaceJointsL2;
            else if (LeftObjID == 2) PlaceJointsL = PlaceJointsL3;
            else { placeFlagL = false; PlaceJointsL = IniteJointsL;} // FIXME 若没有目标物体，回初始位置

            if (RightObjID == 0) PlaceJointsR = PlaceJointsR1;
            else if (RightObjID == 1) PlaceJointsR = PlaceJointsR2;
            else if (RightObjID == 2) PlaceJointsR = PlaceJointsR3;
            else { placeFlagR = false; PlaceJointsR = IniteJointsR;}

            mergeTargetLR(PlaceJointsL, PlaceJointsR, PlaceJoints);

            MoveJoints(PlaceJoints, Vel_Lv3, Acc_Lv3, moveDevice); // 放置物体

            /// 手抓张开
            HandOpen(0);
            HandOpen(1);
            std::this_thread::sleep_for(std::chrono::milliseconds(1000)); /// 很关键

            printf("\033[0;31m%s\033[0m\n\n\n", "================================== 抓取成功 =================================");
        }

        if (BigBallFlag) { /// 有球体
            printf("\033[0;33m[INFO] Detected big ball, begin to pick it up...\033[0m\n");

            std::vector<double> targetPoseL, targetPoseR, targetPose;
            if(_graphicsGrasp->getObjPose(BigBallRect.first, targetPoseL, cloud, 1, 0, 0, 3.0) && // 两点法计算位置
               _graphicsGrasp->getObjPose(BigBallRect.first, targetPoseR, cloud, 1, 0, 1, 3.0)) {

                MoveJoints(BallIniteJoints, Vel_Lv3, Acc_Lv3, 2); /// 移动到球抓取初始位置
                HandClose(0);
                HandClose(1);

                printf("\033[0;31m%s\033[0m\n\n\n", "================================= 准备抓取球体 ================================");
                /// 左臂准备抓取球体
                std::vector<double> targetPoseTmpL = targetPoseL; // 目标位置副本

                printf("[INFO] 待抓取球体信息 ID:[%d] LeftOrRight[%d] Pose:[%f,%f,%f,%f,%f,%f]\n\n",
                       BigBallRect.second, 0, targetPoseTmpL[0], targetPoseTmpL[1], targetPoseTmpL[2],
                       targetPoseTmpL[3], targetPoseTmpL[4], targetPoseTmpL[5]);

                targetPoseTmpL = getRobotPose(0); // 获取当前位置
//                targetPoseTmpL[0] += 0.12; // x 移动到球下半部 FIXME 改成固定高度 0.329
                targetPoseTmpL[0] = 0.36; // x 移动到球下半部 FIXME 改成固定高度 0.329
                targetPoseTmpL[1] = targetPoseL[1] - 0.02; // y移动到球所在位置, 向前微调
                targetPoseTmpL[2] = targetPoseL[2] + 0.25; // z 移动到球附近

                cout << "[INFO] targetPose[BigBall] Left: " << targetPoseTmpL << endl << endl;

                /// 右臂准备抓取球体
                std::vector<double> targetPoseTmpR = targetPoseR; // 目标位置副本

                printf("[INFO] 待抓取球体信息 ID:[%d] LeftOrRight[%d] Pose:[%f,%f,%f,%f,%f,%f]\n\n",
                       BigBallRect.second, 1, targetPoseTmpR[0], targetPoseTmpR[1], targetPoseTmpR[2],
                       targetPoseTmpR[3], targetPoseTmpR[4], targetPoseTmpR[5]);

                targetPoseTmpR = getRobotPose(1); // 获取当前位置
//                targetPoseTmpR[0] -= 0.12; // x移动到球下半部 FIXME 改成固定高度 -0.329
                targetPoseTmpR[0] = -0.36; // x移动到球下半部 FIXME 改成固定高度 -0.329
                targetPoseTmpR[1] = targetPoseR[1] - 0.02; // y 移动到球所在位置, 向前微调
                targetPoseTmpR[2] = targetPoseR[2] + 0.25; // z 移动到球附近

                cout << "[INFO] targetPose[BigBall] Right: " << targetPoseTmpR << endl << endl;

                mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);

                MovePose(targetPose, Vel_Lv2, Acc_Lv2, 2);

                printf("\033[0;31m%s\033[0m\n\n\n", "================================== 抬起球体 =================================");

                /// 左臂抬起
                targetPoseTmpL = getRobotPose(0); // 获取当前位置
                targetPoseTmpL[0] -= 0.1;

                /// 右臂抬起
                targetPoseTmpR = getRobotPose(1); // 获取当前位置
                targetPoseTmpR[0] += 0.1;

                mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);
                MovePose(targetPose, Vel_Lv1, Acc_Lv1, 2);

                printf("\033[0;31m%s\033[0m\n\n\n", "================================== 拖回球体 =================================");

                /// 左臂平移
                targetPoseTmpL = getRobotPose(0); // 获取当前位置 -0.246
//                if (targetPoseTmpL[1] < -0.35) targetPoseTmpL[1] += 0.1;
                targetPoseTmpL[1]  = -0.37;

                /// 右臂平移
                targetPoseTmpR = getRobotPose(1); // 获取当前位置
//                if (targetPoseTmpR[1] > 0.35) targetPoseTmpR[1] -= 0.1;
                targetPoseTmpR[1] = -0.37;

                mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);
                MovePose(targetPose, Vel_Lv1, Acc_Lv1, 2);

                printf("\033[0;31m%s\033[0m\n\n\n", "================================== 平移球体 =================================");

                /// 右臂平移
                targetPoseTmpR = getRobotPose(1); // 获取当前位置
                const double BallPlaceR_z = 0.23; // 右臂z方向绝对放置位置, 末端坐标系, 不是TCP
                double distance = BallPlaceR_z - targetPoseTmpR[2];
                targetPoseTmpR[2] = BallPlaceR_z;

                /// 左臂平移
                targetPoseTmpL = getRobotPose(0); // 获取当前位置
                targetPoseTmpL[2] -= distance; // 相对运, 移动距离为右臂所走距离

                mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);
                MovePose(targetPose, Vel_Lv1, Acc_Lv1, 2);

                printf("\033[0;31m%s\033[0m\n\n\n", "================================== 放下球体 =================================");

                /// 左臂抬起
                targetPoseTmpL = getRobotPose(0); // 获取当前位置
                targetPoseTmpL[0] += 0.1;

                /// 右臂抬起
                targetPoseTmpR = getRobotPose(1); // 获取当前位置
                targetPoseTmpR[0] -= 0.1;

                mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);
                MovePose(targetPose, Vel_Lv1, Acc_Lv1, 1);

                printf("\033[0;31m%s\033[0m\n\n\n", "================================== 抬起手臂 =================================");

                /// 左臂抬起 NOTE: 此时为关节角
                targetPoseTmpL = getRobotJoints(0); // 获取当前关节角
                targetPoseTmpL[0] -= D2R(10);
                targetPoseTmpL[1] -= D2R(10);

                /// 右臂抬起 NOTE: 此时为关节角
                targetPoseTmpR = getRobotJoints(1); // 获取当前关节角
                targetPoseTmpR[0] += D2R(10);
                targetPoseTmpR[1] += D2R(10);

                mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);
                MoveJoints(targetPose, Vel_Lv1, Acc_Lv1, 2);

                /// 左臂抬起
                targetPoseTmpL = getRobotPose(0); // 获取当前位置
                targetPoseTmpL[0] -= 0.1;

                /// 右臂抬起
                targetPoseTmpR = getRobotPose(1); // 获取当前位置
                targetPoseTmpR[0] += 0.1;

                mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);
//                MovePose(targetPose, Vel_Lv1, Acc_Lv1, 2);

                printf("\033[0;31m%s\033[0m\n\n\n", "================================= 球体抓取成功 ================================");
                BigCubePicked = true; // 球体抓取完成

                printf("\033[0;31m%s\033[0m\n\n\n", "================================== 回初始位置 ================================");
                MoveInit();

//                exit(1);
            }
        }

        if (BigCubeFlag) { /// 有立方体
            printf("\033[0;33m[INFO] Detected big cube, begin to pick it up...\033[0m\n");

            std::vector<double> targetPoseL, targetPoseR, targetPose;
            std::vector<double> targetPoseTmpL, targetPoseTmpR;
            if (_graphicsGrasp->getObjPose(BigCubeRect.first, targetPoseL, cloud, 1, 0, 0) && // 两点法计算位置
                _graphicsGrasp->getObjPose(BigCubeRect.first, targetPoseR, cloud, 1, 0, 1)) {

                printf("\033[0;31m%s\033[0m\n", "=============================== 立方体抓取初始位置 ==============================");
                MoveJoints(CubeIniteJoints, Vel_Lv3, Acc_Lv3, 2); /// 移动到立方体抓取初始位置

                printf("\033[0;31m%s\033[0m\n", "================================ 准备抓取立方体 ===============================");
                /// 左臂准备抓取球体
                targetPoseTmpL = targetPoseL; // 目标位置副本

                printf("[INFO] 待抓取球体信息 ID:[%d] LeftOrRight[%d] Pose:[%f,%f,%f,%f,%f,%f]\n\n",
                       BigCubeRect.second, 0, targetPoseTmpL[0], targetPoseTmpL[1], targetPoseTmpL[2],
                       targetPoseTmpL[3], targetPoseTmpL[4], targetPoseTmpL[5]);

                targetPoseTmpL = getRobotPose(0); // 获取当前位置
//                targetPoseTmpL[0] += 0.07; // x移动到立方体下半部
                targetPoseTmpL[0] = 0.34; // x移动到立方体下半部 FIXME 改成固定高度 -0.329
                targetPoseTmpL[1] = targetPoseL[1] - 0.00; // y移动到立方体所在位置, 向前微调
                targetPoseTmpL[2] = targetPoseL[2] + 0.275; // z 移动到立方体附近

                cout << "[INFO] targetPose[BigCube] Left: " << targetPoseTmpL << endl << endl;

                /// 右臂准备抓取立方体
                targetPoseTmpR = targetPoseR; // 目标位置副本

                printf("[INFO] 待抓取立方体体信息 ID:[%d] LeftOrRight[%d] Pose:[%f,%f,%f,%f,%f,%f]\n\n",
                       BigBallRect.second, 1, targetPoseTmpR[0], targetPoseTmpR[1], targetPoseTmpR[2],
                       targetPoseTmpR[3], targetPoseTmpR[4], targetPoseTmpR[5]);

                targetPoseTmpR = getRobotPose(1); // 获取当前位置
//                targetPoseTmpR[0] -= 0.07; // x移动到立方体下半部
                targetPoseTmpR[0] = -0.34; // x移动到立方体下半部 FIXME 改成固定高度 -0.329
                targetPoseTmpR[1] = targetPoseR[1] - 0.00; // y移动到立方体所在位置, 向前微调
                targetPoseTmpR[2] = targetPoseR[2] + 0.275; // z 移动到立方体附近

                cout << "[INFO] targetPose[BigCube] Right: " << targetPoseTmpR << endl << endl;

                mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);

                MovePose(targetPose, Vel_Lv2, Acc_Lv2, 2);

                printf("\033[0;31m%s\033[0m\n\n\n", "================================== 抓住立方体 ===============================");

                targetPoseTmpL = getRobotPose(0); // 获取当前位置
                targetPoseTmpL[2] -= 0.03;

                targetPoseTmpR = getRobotPose(1); // 获取当前位置
                targetPoseTmpR[2] -= 0.03;

                mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);
                MovePose(targetPose, Vel_Lv1, Acc_Lv1, 2);

                HandClose(0);
                HandClose(1);

                printf("\033[0;31m%s\033[0m\n\n\n", "================================== 抬起立方体 ===============================");

                /// 左臂抬起
                targetPoseTmpL = getRobotPose(0); // 获取当前位置
                targetPoseTmpL[0] -= 0.1;

                /// 右臂抬起
                targetPoseTmpR = getRobotPose(1); // 获取当前位置
                targetPoseTmpR[0] += 0.1;

                mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);
                MovePose(targetPose, Vel_Lv1, Acc_Lv1, 2);

                printf("\033[0;31m%s\033[0m\n\n\n", "================================== 拖回立方体 =================================");

                /// 左臂平移
                targetPoseTmpL = getRobotPose(0); // 获取当前位置 -0.246
//                if (targetPoseTmpL[1] < -0.35) targetPoseTmpL[1] += 0.1;
                targetPoseTmpL[1]  = -0.37;

                /// 右臂平移
                targetPoseTmpR = getRobotPose(1); // 获取当前位置
//                if (targetPoseTmpR[1] > 0.35) targetPoseTmpR[1] -= 0.1;
                targetPoseTmpR[1] = -0.37;

                mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);
                MovePose(targetPose, Vel_Lv1, Acc_Lv1, 2);

                printf("\033[0;31m%s\033[0m\n\n\n", "================================== 平移立方体 =================================");

                /// 左臂平移
                targetPoseTmpL = getRobotPose(0); // 获取当前位置
                const double BallCubePlaceL_z = 0.23; // 左臂z方向绝对放置位置, 末端坐标系, 不是TCP
                double distance = BallCubePlaceL_z - targetPoseTmpL[2];
                targetPoseTmpL[2] = BallCubePlaceL_z;

                /// 右臂平移
                targetPoseTmpR = getRobotPose(1); // 获取当前位置
                targetPoseTmpR[2] -= distance; // 相对运, 移动距离为右臂所走距离

                mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);
                MovePose(targetPose, Vel_Lv1, Acc_Lv1, 2);

                printf("\033[0;31m%s\033[0m\n\n\n", "================================== 放下立方体 =================================");

                /// 左臂抬起
                targetPoseTmpL = getRobotPose(0); // 获取当前位置
                targetPoseTmpL[0] += 0.1;

                /// 右臂抬起
                targetPoseTmpR = getRobotPose(1); // 获取当前位置
                targetPoseTmpR[0] -= 0.1;

                mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);
                MovePose(targetPose, Vel_Lv1, Acc_Lv1, 2);

                HandOpen(0);
                HandOpen(1);

                printf("\033[0;31m%s\033[0m\n\n\n", "================================== 退回 =================================");

                /// 左臂抬起
                targetPoseTmpL = getRobotPose(0); // 获取当前位置
                targetPoseTmpL[2] += 0.05;

                /// 右臂抬起
                targetPoseTmpR = getRobotPose(1); // 获取当前位置
                targetPoseTmpR[2] += 0.05;

                mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);
                MovePose(targetPose, Vel_Lv1, Acc_Lv1, 2);

                printf("\033[0;31m%s\033[0m\n\n\n", "================================== 抬起手臂 =================================");

                /// 左臂抬起 NOTE: 此时为关节角
                targetPoseTmpL = getRobotJoints(0); // 获取当前关节角
                targetPoseTmpL[0] -= D2R(10);
                targetPoseTmpL[1] -= D2R(10);

                /// 右臂抬起 NOTE: 此时为关节角
                targetPoseTmpR = getRobotJoints(1); // 获取当前关节角
                targetPoseTmpR[0] += D2R(10);
                targetPoseTmpR[1] += D2R(10);

                mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);
                MoveJoints(targetPose, Vel_Lv1, Acc_Lv1, 2);

                /// 左臂抬起
                targetPoseTmpL = getRobotPose(0); // 获取当前位置
                targetPoseTmpL[0] -= 0.1;

                /// 右臂抬起
                targetPoseTmpR = getRobotPose(1); // 获取当前位置
                targetPoseTmpR[0] += 0.1;

                mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);

                BigCubeFlag = true;
                MoveInit();
            }
        }
    }
}

bool GraspController::graspControlJuggleFirst() {
    bool ret = captureImage(-100); // 采集图像
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    /// 初始位置
    MoveInit(); // 同步移动到初始位置, 打开手抓

    /// 点云初始化
    cloudInit();

    printf("\033[0;31m%s\033[0m\n", "================================== 初始化成功 =================================");

    while (true) {
        printf("\033[0;31m%s\033[0m\n", "================================== 采集图像 =================================");
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

        if (saveFlag) { // 每次运行程序保存一次数据
            saveCloudAndImages();
            saveFlag = false;
//            exit(1);
        }

        printf("\033[0;31m%s\033[0m\n", "=================================== 目标检测 ================================");

        const int juggleOrCube = 0; /// 0为积木, 1为立方体

        bool BigBallFlag = false, BigCubeFlag = false;
        std::pair<cv::RotatedRect, int> BigBallRect, BigCubeRect;

        /// Yolo积木检测
        RotRectsAndID = _graphicsGrasp->detectGraspYoloPro(color, cloud, 120, 0);

        // 球体检测
        if (!BallPicked && (JugglesPicked || RotRectsAndID.first.empty())) BigBallFlag = _graphicsGrasp->detectBigBall(color, cloud, BigBallRect, 0);

        if (BallPicked) { // 无球再检测立方体
            BigCubeFlag = _graphicsGrasp->detectBigCube(color, cloud, BigCubeRect, 0);
        }

        printf("\033[0;31m%s\033[0m\n", "=================================== 回初始位置 ===============================");
        MoveJoints(IniteJoints, Vel_Lv3, Acc_Lv3, 2);

        printf("\033[0;31m%s\033[0m\n", "================================= 生成移动任务 ===============================");

        std::vector<int> AimObjIndicesLR;

        //////////////////////////////////////////////  积木抓取   //////////////////////////////////////////////
        if (!JugglesPicked) {
            pickAttemptCnt++; // 积木抓取尝试次数
            if (pickAttemptCnt > 15) JugglesPicked = true;

            if (RotRectsAndID.first.empty()) {
                printf("[ERROR] Did not get any rot rects!\n");
                continue;
            }

            // 确定两侧待抓取物体索引
            AimObjIndicesLR = _graphicsGrasp->findAimObjLR(RotRectsAndID, 0);

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
            } else if ((AimObjIndicesLR[0] == -1) && (AimObjIndicesLR[1] == -1)) continue;

            printf("\033[0;31m%s\033[0m\n", "============================= 移动到积木抓取初始位置 ===========================");
            MoveJoints(JuggleIniteJoints, Vel_Lv3, Acc_Lv3, 2);

            printf("\033[0;31m%s\033[0m\n", "================================= 移动到正上方 ===============================");

            bool lieFlagR = true, lieFlagL = true; /// 积木躺着标志位

            /// 获取目标位置(左臂)
            if (AimObjIndicesLR[0] != -1) { // 左侧有物体
                cv::RotatedRect RotatedRectL = RotRectsAndID.first[AimObjIndicesLR[0]];
                LeftObjID = RotRectsAndID.second[AimObjIndicesLR[0]];
                // 获取机器人坐标系下坐标
                if (!_graphicsGrasp->getObjPose(RotatedRectL, targetPoseL, cloud, juggleOrCube, 0, 0)) continue;

                targetPoseTmpL = targetPoseL; // 目标位置副本

                targetPoseTmpL[0] = _graphicsGrasp->height_Lv2_L;
                /// 高低判别
                if (targetPoseL[0] > _graphicsGrasp->lieThreshL) { // 躺着的
//                    targetPoseTmpL[0] = _graphicsGrasp->height_Lv2_L;
                    printf("\033[0;31m%s\033[0m\n", "[INFO] 左臂待抓取物体为躺着的");
                    lieFlagL = true;
                } else { // 立着的
                    targetPoseTmpL[1] -= 0.001;
                    printf("\033[0;31m%s\033[0m\n", "[INFO] 左臂待抓取物体为立着的");
                    lieFlagL = false;
                }

                // 移动到物体上方
                targetPoseTmpL[4] = 0;

                cout << "[INFO] targetPoseL: " << targetPoseL << endl;

                printf("[INFO] 左臂待抓取物体信息 ID:[%d] Angle:[%f(%f)] Pose:[%f,%f,%f,%f,%f,%f]\n\n",
                       RotRectsAndID.second[AimObjIndicesLR[0]], targetPoseL[4], R2D(targetPoseL[4]), targetPoseTmpL[0],
                       targetPoseTmpL[1], targetPoseTmpL[2], targetPoseTmpL[3], targetPoseTmpL[4], targetPoseTmpL[5]);
            }

            /// 获取目标位置(右臂)
            if (AimObjIndicesLR[1] != -1) { // 右侧有物体
                cv::RotatedRect RotatedRectR = RotRectsAndID.first[AimObjIndicesLR[1]];
                RightObjID = RotRectsAndID.second[AimObjIndicesLR[1]];
                // 获取机器人坐标系下坐标
                if (!_graphicsGrasp->getObjPose(RotatedRectR, targetPoseR, cloud, juggleOrCube, 0, 1)) continue;
                targetPoseTmpR = targetPoseR; // 目标位置副本

                targetPoseTmpR[0] = _graphicsGrasp->height_Lv2_R;
                /// 高低判别
                if (targetPoseR[0] < _graphicsGrasp->lieThreshR) { // 躺着的
//                    targetPoseTmpR[0] = _graphicsGrasp->height_Lv2_R;
                    printf("\033[0;31m%s\033[0m\n", "[INFO] 右臂待抓取物体为躺着的");
                    lieFlagR = true;
                } else { // 立着的
                    targetPoseTmpR[1] -= 0.001;
                    printf("\033[0;31m%s\033[0m\n", "[INFO] 右臂待抓取物体为立着的");
                    lieFlagR = false;
                }

                // 移动到物体上方
                targetPoseTmpR[4] = 0;

                cout << "[INFO] targetPoseR: " << targetPoseR << endl;

                printf("[INFO] 右臂待抓取物体信息 ID:[%d] Angle:[%f(%f)] Pose:[%f,%f,%f,%f,%f,%f]\n\n",
                       RotRectsAndID.second[AimObjIndicesLR[1]], targetPoseR[4], R2D(targetPoseR[4]), targetPoseTmpR[0],
                       targetPoseTmpR[1],
                       targetPoseTmpR[2], targetPoseTmpR[3], targetPoseTmpR[4], targetPoseTmpR[5]);
            }

            /// 处理待抓取的两个物体靠的太近
            if (AimObjIndicesLR[0] != -1 && AimObjIndicesLR[1] != -1) {
                float distance = RotRectsAndID.first[AimObjIndicesLR[1]].center.x -
                                 RotRectsAndID.first[AimObjIndicesLR[0]].center.x;
                printf("[INFO] Distance between Obj in pix: %f\n", distance);
                if (distance < 100) {
                    printf("\033[0;32m%s\033[0m\n",
                           "[WARN] Distance is too short between the Objects, left arm first!\n");
                    moveDevice = 0; // 物体靠的太近, 左臂优先
                }
            }

            //////////////////////////////////////////////  推倒积木   //////////////////////////////////////////////
            /// 处理立着的积木
//            if (AimObjIndicesLR[0] != -1 && !lieFlagL) // 左侧立着
//            {
//                printf("\033[0;33m%s\033[0m\n", "================= 左臂待抓取积木是立着的, 推倒并进入下一次抓取 ================");
//                // 移动到积木左侧
//                targetPoseTmpL[0] = _graphicsGrasp->height_Lv2_L - 0.04; // 调整高度 FIXME !!!!!!!!!!!!!!!
//                targetPoseTmpL[2] += 0.05;
//                mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);
//                MovePose(targetPose, Vel_Lv2, Acc_Lv2, moveDevice);
//
//                HandClose(0); // 手抓闭合
//                /// 向右侧推到
//                targetPose = getRobotPose(2); // 获取双臂当前位置
//                targetPose[2] -= 0.1; // 左臂
//                MovePose(targetPose, Vel_Lv3_, Acc_Lv3_, moveDevice);
//                continue; // 跳过此次循环, 进入下一次抓取
//            }
//
//            if (AimObjIndicesLR[1] != -1 && !lieFlagL) // 右侧立着
//            {
//                printf("\033[0;33m%s\033[0m\n", "================= 右臂待抓取积木是立着的, 推倒并进入下一次抓取 ================");
//                // 移动到积木右侧
//                targetPoseTmpR[0] = _graphicsGrasp->height_Lv2_R + 0.04; // 调整高度 FIXME !!!!!!!!!!!!!!!
//                targetPoseTmpR[2] += 0.05;
//                mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);
//                MovePose(targetPose, Vel_Lv2, Acc_Lv2, moveDevice);
//
//                HandClose(1); // 手抓闭合
//                /// 向左侧推到
//                targetPose = getRobotPose(2); // 获取双臂当前位置
//                targetPose[2] -= 0.1; // 左臂
//                MovePose(targetPose, Vel_Lv3_, Acc_Lv3_, moveDevice);
//                continue; // 跳过此次循环, 进入下一次抓取
//            }

            mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);

            cout << "[INFO] targetPose: " << targetPose << endl;

            MovePose(targetPose, Vel_Lv2, Acc_Lv2, moveDevice);

            printf("\033[0;31m%s\033[0m\n", "================================= 调整末端姿态 ===============================");

            targetPose = getRobotJoints(2); // 获取当前关节角
            if (HaveObjL) targetPose[5] += D2R(180 + R2D(targetPoseL[4])); // 获取原始末端关节角, 加补角
            if (HaveObjR) targetPose[11] += targetPoseR[4]; // 获取原始末端关节角

            MoveJoints(targetPose, Vel_Lv4, Acc_Lv4, 2);

            printf("\033[0;31m%s\033[0m\n", "================================= 手抓下降并抓取 ==============================");

            /// 手臂垂直下降
            targetPose = getRobotPose(2); // 获取双臂当前位置
            targetPose[0] += 0.07; // 左臂
            targetPose[6] -= 0.07; // 右臂

            MovePose(targetPose, Vel_Lv2_, Acc_Lv2_, moveDevice);

            /// 手抓闭合
            HandClose(0);
            HandClose(1);

            /// 手臂垂直抬起
            targetPose = getRobotPose(2); // 获取双臂当前位置
            targetPose[0] -= 0.07; // 左臂
            targetPose[6] += 0.07; // 右臂

            MovePose(targetPose, Vel_Lv2_, Acc_Lv2_, moveDevice); // FIXME !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

//            /// 修改姿态, 手臂垂直抬起
//            targetPoseTmpR[0] += 0.03; // 右臂
//            targetPoseTmpL[0] -= 0.03; // 左臂
//
//            mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);
//
//            MovePose(targetPose, Vel_Lv2_, Acc_Lv2_, moveDevice);

            printf("\033[0;31m%s\033[0m\n", "================================= 放置到指定位置 ==============================");

            std::vector<double> PlaceJointsL, PlaceJointsR, PlaceJoints;
            bool placeFlagL = true, placeFlagR = true;
            if (LeftObjID == 0) PlaceJointsL = PlaceJointsL1;
            else if (LeftObjID == 1) PlaceJointsL = PlaceJointsL1;
            else if (LeftObjID == 2) PlaceJointsL = PlaceJointsL1;
            else { placeFlagL = false; PlaceJointsL = IniteJointsL;}

            if (RightObjID == 0) PlaceJointsR = PlaceJointsR1;
            else if (RightObjID == 1) PlaceJointsR = PlaceJointsR1;
            else if (RightObjID == 2) PlaceJointsR = PlaceJointsR1;
            else { placeFlagR = false; PlaceJointsR = IniteJointsR;}

            mergeTargetLR(PlaceJointsL, PlaceJointsR, PlaceJoints);

            MoveJoints(PlaceJoints, Vel_Lv3, Acc_Lv3, moveDevice); // 放置物体

//            /// 手臂垂直下降
//            targetPose = getRobotPose(2); // 获取双臂当前位置
//            targetPose[0] += 0.05; // 左臂
//            targetPose[6] -= 0.05; // 右臂
//
//            MovePose(targetPose, Vel_Lv2_, Acc_Lv2_, moveDevice);

            /// 手抓张开
            HandOpen(0);
            HandOpen(1);
            std::this_thread::sleep_for(std::chrono::milliseconds(1000)); /// 很关键

            printf("\033[0;31m%s\033[0m\n\n\n", "================================== 抓取成功 =================================");

            JugglesPickedNum++; /// 已抓取积木计数
            if (JugglesPickedNum > 10) {
                printf("\033[0;33m%s\033[0m\n", "================================= 所有积木抓取完成 ==============================");
                JugglesPicked = true; /// 积木抓取超过7次, 球抓取
            }
        }

        ///////////////////////////////////////////////  球抓取   //////////////////////////////////////////////
        if (BigBallFlag) { /// 有球体
            printf("\033[0;33m%s\033[0m\n", "=============================== 积木抓取完成 开始抓取球 ===========================");

            std::vector<double> targetPoseL, targetPoseR, targetPose;
            if(_graphicsGrasp->getObjPose(BigBallRect.first, targetPoseL, cloud, 1, 0, 0, 3.0) && // 两点法计算位置
               _graphicsGrasp->getObjPose(BigBallRect.first, targetPoseR, cloud, 1, 0, 1, 3.0)) {

                MoveJoints(BallIniteJoints, Vel_Lv3, Acc_Lv3, 2); /// 移动到球抓取初始位置
//                HandClose(0);
//                HandClose(1);

                printf("\033[0;31m%s\033[0m\n\n\n", "================================= 准备抓取球体 ================================");
                /// 左臂准备抓取球体
                std::vector<double> targetPoseTmpL = targetPoseL; // 目标位置副本

                printf("[INFO] 待抓取球体信息 ID:[%d] LeftOrRight[%d] Pose:[%f,%f,%f,%f,%f,%f]\n\n",
                       BigBallRect.second, 0, targetPoseTmpL[0], targetPoseTmpL[1], targetPoseTmpL[2],
                       targetPoseTmpL[3], targetPoseTmpL[4], targetPoseTmpL[5]);

                targetPoseTmpL = getRobotPose(0); // 获取当前位置
//                targetPoseTmpL[0] += 0.12; // x 移动到球下半部 FIXME 改成固定高度 0.329
                targetPoseTmpL[0] = 0.37; // x 移动到球下半部 FIXME 改成固定高度 0.329
                targetPoseTmpL[1] = targetPoseL[1] - 0.02; // y移动到球所在位置, 向前微调
                targetPoseTmpL[2] = targetPoseL[2] + 0.24; // z 移动到球附近

                cout << "[INFO] targetPose[BigBall] Left: " << targetPoseTmpL << endl << endl;

                /// 右臂准备抓取球体
                std::vector<double> targetPoseTmpR = targetPoseR; // 目标位置副本

                printf("[INFO] 待抓取球体信息 ID:[%d] LeftOrRight[%d] Pose:[%f,%f,%f,%f,%f,%f]\n\n",
                       BigBallRect.second, 1, targetPoseTmpR[0], targetPoseTmpR[1], targetPoseTmpR[2],
                       targetPoseTmpR[3], targetPoseTmpR[4], targetPoseTmpR[5]);

                targetPoseTmpR = getRobotPose(1); // 获取当前位置
//                targetPoseTmpR[0] -= 0.12; // x移动到球下半部 FIXME 改成固定高度 -0.329
                targetPoseTmpR[0] = -0.37; // x移动到球下半部 FIXME 改成固定高度 -0.329
                targetPoseTmpR[1] = targetPoseR[1] - 0.02; // y 移动到球所在位置, 向前微调
                targetPoseTmpR[2] = targetPoseR[2] + 0.24; // z 移动到球附近

                cout << "[INFO] targetPose[BigBall] Right: " << targetPoseTmpR << endl << endl;

                mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);

                MovePose(targetPose, Vel_Lv2, Acc_Lv2, 2);

                printf("\033[0;31m%s\033[0m\n\n\n", "================================== 抬起球体 =================================");

                /// 左臂抬起
                targetPoseTmpL = getRobotPose(0); // 获取当前位置
                targetPoseTmpL[0] -= 0.1;

                /// 右臂抬起
                targetPoseTmpR = getRobotPose(1); // 获取当前位置
                targetPoseTmpR[0] += 0.1;

                mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);
                MovePose(targetPose, Vel_Lv1, Acc_Lv1, 2);

                printf("\033[0;31m%s\033[0m\n\n\n", "================================== 拖回球体 =================================");

                /// 左臂平移
                targetPoseTmpL = getRobotPose(0); // 获取当前位置 -0.246
//                if (targetPoseTmpL[1] < -0.35) targetPoseTmpL[1] += 0.1;
                targetPoseTmpL[1]  = -0.37;

                /// 右臂平移
                targetPoseTmpR = getRobotPose(1); // 获取当前位置
//                if (targetPoseTmpR[1] > 0.35) targetPoseTmpR[1] -= 0.1;
                targetPoseTmpR[1] = -0.37;

                mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);
                MovePose(targetPose, Vel_Lv1, Acc_Lv1, 2);

                printf("\033[0;31m%s\033[0m\n\n\n", "================================== 平移球体 =================================");

                /// 右臂平移
                targetPoseTmpR = getRobotPose(1); // 获取当前位置
                const double BallPlaceR_z = 0.23; // 右臂z方向绝对放置位置, 末端坐标系, 不是TCP
                double distance = BallPlaceR_z - targetPoseTmpR[2];
                targetPoseTmpR[2] = BallPlaceR_z;

                /// 左臂平移
                targetPoseTmpL = getRobotPose(0); // 获取当前位置
                targetPoseTmpL[2] -= distance; // 相对运, 移动距离为右臂所走距离

                mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);
                MovePose(targetPose, Vel_Lv1, Acc_Lv1, 2);

                printf("\033[0;31m%s\033[0m\n\n\n", "================================== 放下球体 =================================");

                /// 左臂抬起
                targetPoseTmpL = getRobotPose(0); // 获取当前位置
                targetPoseTmpL[0] += 0.1;

                /// 右臂抬起
                targetPoseTmpR = getRobotPose(1); // 获取当前位置
                targetPoseTmpR[0] -= 0.1;

                mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);
                MovePose(targetPose, Vel_Lv1, Acc_Lv1, 1);

                printf("\033[0;31m%s\033[0m\n\n\n", "================================== 抬起手臂 =================================");

                /// 左臂抬起 NOTE: 此时为关节角
                targetPoseTmpL = getRobotJoints(0); // 获取当前关节角
                targetPoseTmpL[0] -= D2R(10);
                targetPoseTmpL[1] -= D2R(10);

                /// 右臂抬起 NOTE: 此时为关节角
                targetPoseTmpR = getRobotJoints(1); // 获取当前关节角
                targetPoseTmpR[0] += D2R(10);
                targetPoseTmpR[1] += D2R(10);

                mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);
                MoveJoints(targetPose, Vel_Lv1, Acc_Lv1, 2);

                /// 左臂抬起
                targetPoseTmpL = getRobotPose(0); // 获取当前位置
                targetPoseTmpL[0] -= 0.1;

                /// 右臂抬起
                targetPoseTmpR = getRobotPose(1); // 获取当前位置
                targetPoseTmpR[0] += 0.1;

                mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);
//                MovePose(targetPose, Vel_Lv1, Acc_Lv1, 2);

                printf("\033[0;31m%s\033[0m\n\n\n", "================================= 球体抓取成功 ================================");
                BallPicked = true; // 球体抓取完成

                printf("\033[0;31m%s\033[0m\n\n\n", "================================== 回初始位置 ================================");
                MoveInit();
            }
        }

        if (BigCubeFlag) { /// 有立方体
            printf("\033[0;33m[INFO] Detected big cube, begin to pick it up...\033[0m\n");

            std::vector<double> targetPoseL, targetPoseR, targetPose;
            std::vector<double> targetPoseTmpL, targetPoseTmpR;
            if (_graphicsGrasp->getObjPose(BigCubeRect.first, targetPoseL, cloud, 1, 0, 0) && // 两点法计算位置
                _graphicsGrasp->getObjPose(BigCubeRect.first, targetPoseR, cloud, 1, 0, 1)) {

                printf("\033[0;31m%s\033[0m\n", "=============================== 立方体抓取初始位置 ==============================");
                MoveJoints(CubeIniteJoints, Vel_Lv3, Acc_Lv3, 2); /// 移动到立方体抓取初始位置

                printf("\033[0;31m%s\033[0m\n", "================================ 准备抓取立方体 ===============================");
                /// 左臂准备抓取球体
                targetPoseTmpL = targetPoseL; // 目标位置副本

                printf("[INFO] 待抓取球体信息 ID:[%d] LeftOrRight[%d] Pose:[%f,%f,%f,%f,%f,%f]\n\n",
                       BigCubeRect.second, 0, targetPoseTmpL[0], targetPoseTmpL[1], targetPoseTmpL[2],
                       targetPoseTmpL[3], targetPoseTmpL[4], targetPoseTmpL[5]);

                targetPoseTmpL = getRobotPose(0); // 获取当前位置
//                targetPoseTmpL[0] += 0.07; // x移动到立方体下半部
                targetPoseTmpL[0] = 0.34; // x移动到立方体下半部 FIXME 改成固定高度 -0.329
                targetPoseTmpL[1] = targetPoseL[1] - 0.00; // y移动到立方体所在位置, 向前微调
                targetPoseTmpL[2] = targetPoseL[2] + 0.23; // z 移动到立方体附近

                cout << "[INFO] targetPose[BigCube] Left: " << targetPoseTmpL << endl << endl;

                /// 右臂准备抓取立方体
                targetPoseTmpR = targetPoseR; // 目标位置副本

                printf("[INFO] 待抓取立方体体信息 ID:[%d] LeftOrRight[%d] Pose:[%f,%f,%f,%f,%f,%f]\n\n",
                       BigBallRect.second, 1, targetPoseTmpR[0], targetPoseTmpR[1], targetPoseTmpR[2],
                       targetPoseTmpR[3], targetPoseTmpR[4], targetPoseTmpR[5]);

                targetPoseTmpR = getRobotPose(1); // 获取当前位置
//                targetPoseTmpR[0] -= 0.07; // x移动到立方体下半部
                targetPoseTmpR[0] = -0.34; // x移动到立方体下半部 FIXME 改成固定高度 -0.329
                targetPoseTmpR[1] = targetPoseR[1] - 0.00; // y移动到立方体所在位置, 向前微调
                targetPoseTmpR[2] = targetPoseR[2] + 0.23; // z 移动到立方体附近

                cout << "[INFO] targetPose[BigCube] Right: " << targetPoseTmpR << endl << endl;

                mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);

                MovePose(targetPose, Vel_Lv2, Acc_Lv2, 2);

                printf("\033[0;31m%s\033[0m\n\n\n", "================================== 抓住立方体 ===============================");

                targetPoseTmpL = getRobotPose(0); // 获取当前位置
                targetPoseTmpL[2] -= 0.01;

                targetPoseTmpR = getRobotPose(1); // 获取当前位置
                targetPoseTmpR[2] -= 0.01;

                mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);
                MovePose(targetPose, Vel_Lv1, Acc_Lv1, 2);

                HandClose(0);
                HandClose(1);

                printf("\033[0;31m%s\033[0m\n\n\n", "================================== 抬起立方体 ===============================");

                /// 左臂抬起
                targetPoseTmpL = getRobotPose(0); // 获取当前位置
                targetPoseTmpL[0] -= 0.1;

                /// 右臂抬起
                targetPoseTmpR = getRobotPose(1); // 获取当前位置
                targetPoseTmpR[0] += 0.1;

                mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);
                MovePose(targetPose, Vel_Lv1, Acc_Lv1, 2);

                printf("\033[0;31m%s\033[0m\n\n\n", "================================== 拖回立方体 =================================");

                /// 左臂平移
                targetPoseTmpL = getRobotPose(0); // 获取当前位置 -0.246
//                if (targetPoseTmpL[1] < -0.35) targetPoseTmpL[1] += 0.1;
                targetPoseTmpL[1]  = -0.37;

                /// 右臂平移
                targetPoseTmpR = getRobotPose(1); // 获取当前位置
//                if (targetPoseTmpR[1] > 0.35) targetPoseTmpR[1] -= 0.1;
                targetPoseTmpR[1] = -0.37;

                mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);
                MovePose(targetPose, Vel_Lv1, Acc_Lv1, 2);

                printf("\033[0;31m%s\033[0m\n\n\n", "================================== 平移立方体 =================================");

                /// 左臂平移
                targetPoseTmpL = getRobotPose(0); // 获取当前位置
                const double BallCubePlaceL_z = 0.23; // 左臂z方向绝对放置位置, 末端坐标系, 不是TCP
                double distance = BallCubePlaceL_z - targetPoseTmpL[2];
                targetPoseTmpL[2] = BallCubePlaceL_z;

                /// 右臂平移
                targetPoseTmpR = getRobotPose(1); // 获取当前位置
                targetPoseTmpR[2] -= distance; // 相对运, 移动距离为右臂所走距离

                mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);
                MovePose(targetPose, Vel_Lv1, Acc_Lv1, 2);

                printf("\033[0;31m%s\033[0m\n\n\n", "================================== 放下立方体 =================================");

                /// 左臂抬起
                targetPoseTmpL = getRobotPose(0); // 获取当前位置
                targetPoseTmpL[0] += 0.1;

                /// 右臂抬起
                targetPoseTmpR = getRobotPose(1); // 获取当前位置
                targetPoseTmpR[0] -= 0.1;

                mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);
                MovePose(targetPose, Vel_Lv1, Acc_Lv1, 2);

                HandOpen(0);
                HandOpen(1);

                printf("\033[0;31m%s\033[0m\n\n\n", "================================== 退回 =================================");

                /// 左臂抬起
                targetPoseTmpL = getRobotPose(0); // 获取当前位置
                targetPoseTmpL[2] += 0.05;

                /// 右臂抬起
                targetPoseTmpR = getRobotPose(1); // 获取当前位置
                targetPoseTmpR[2] += 0.05;

                mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);
                MovePose(targetPose, Vel_Lv1, Acc_Lv1, 2);

                printf("\033[0;31m%s\033[0m\n\n\n", "================================== 抬起手臂 =================================");

                /// 左臂抬起 NOTE: 此时为关节角
                targetPoseTmpL = getRobotJoints(0); // 获取当前关节角
                targetPoseTmpL[0] -= D2R(10);
                targetPoseTmpL[1] -= D2R(10);

                /// 右臂抬起 NOTE: 此时为关节角
                targetPoseTmpR = getRobotJoints(1); // 获取当前关节角
                targetPoseTmpR[0] += D2R(10);
                targetPoseTmpR[1] += D2R(10);

                mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);
                MoveJoints(targetPose, Vel_Lv1, Acc_Lv1, 2);

                /// 左臂抬起
                targetPoseTmpL = getRobotPose(0); // 获取当前位置
                targetPoseTmpL[0] -= 0.1;

                /// 右臂抬起
                targetPoseTmpR = getRobotPose(1); // 获取当前位置
                targetPoseTmpR[0] += 0.1;

                mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);

                BigCubeFlag = true;
                MoveInit();
            }
        }
    }
}

bool GraspController::graspControlTask2() {
    bool ret = captureImage(-100); // 采集图像
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    /// 初始位置
    MoveInit(); // 同步移动到初始位置, 打开手抓

    /// 点云初始化
    cloudInit();

    printf("\033[0;31m%s\033[0m\n", "================================== 初始化成功 =================================");

    while (true) {
        printf("\033[0;31m%s\033[0m\n", "================================== 采集图像 =================================");
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

        if (saveFlag) { // 每次运行程序保存一次数据
            saveCloudAndImages();
            saveFlag = false;
//            exit(1);
        }

        printf("\033[0;31m%s\033[0m\n", "=================================== 目标检测 ================================");

        bool SmallCubeFlag = false;

        /// 小立方体检测
        if (PickedSmallCubeCnt < 4) { // FIXME 小立方体总数！！！！！！！！！！！！！！！！！！！！！！！！！！！
            RotRectsAndID = _graphicsGrasp->detectSmallCubeTask2(color, cloud, 120, 0);
            if (RotRectsAndID.first.empty()) {
                printf("[ERROR] Did not get any rot rects!\n");
                continue;
            } else SmallCubeFlag = true;
        }

        /// Yolo积木检测
        if (!SmallCubeFlag) {
            RotRectsAndID = _graphicsGrasp->detectGraspYoloPro(color, cloud, 120, 0);
            if (RotRectsAndID.first.empty()) {
                printf("[ERROR] Did not get any rot rects!\n");
                continue;
            }
        }

        printf("\033[0;31m%s\033[0m\n", "=================================== 回初始位置 ===============================");
        MoveJoints(IniteJoints, Vel_Lv3, Acc_Lv3, 2);

        printf("\033[0;31m%s\033[0m\n", "================================= 生成移动任务 ===============================");

        if (!SmallCubeFlag) {
            // 确定两侧待抓取物体索引
            std::vector<int> AimObjIndicesLR = _graphicsGrasp->findAimObjLR(RotRectsAndID, 0);

            int moveDevice = -1;
            int LeftObjID = -1, RightObjID = -1;

            vector<double> targetPose; // 双臂目标位置
            std::vector<double> targetPoseL, targetPoseR, targetPoseTmpL, targetPoseTmpR;
            bool HaveObjL = (AimObjIndicesLR[0] != -1), HaveObjR = (AimObjIndicesLR[1] != -1);

            printf("[INFO] Have Object Left[%d] Right[%d]\n", HaveObjL, HaveObjR);

            if (!HaveObjL && !HaveObjR) continue; // 左右均无积木, 不动

            printf("\033[0;31m%s\033[0m\n", "============================= 移动到积木抓取初始位置 ===========================");
            MoveJoints(JuggleIniteJoints, Vel_Lv3, Acc_Lv3, 2);

            printf("\033[0;31m%s\033[0m\n", "=============================== 获取左右臂积木位置 =============================");

            /// 获取目标位置(左臂)
            if (AimObjIndicesLR[0] != -1) { // 左侧有物体
                cv::RotatedRect RotatedRectL = RotRectsAndID.first[AimObjIndicesLR[0]];
                LeftObjID = RotRectsAndID.second[AimObjIndicesLR[0]];
                // 获取机器人坐标系下坐标
                if (!_graphicsGrasp->getObjPose(RotatedRectL, targetPoseL, cloud, 0, 1, 0)) continue; // 短边

                targetPoseTmpL = targetPoseL; // 目标位置副本

                targetPoseTmpL[0] = _graphicsGrasp->height_Lv2_L;

                // 移动到物体上方
                targetPoseTmpL[4] = 0;

                cout << "[INFO] targetPoseL: " << targetPoseL << endl;

                printf("[INFO] 左臂待抓取物体信息 ID:[%d] Angle:[%f(%f)] Pose:[%f,%f,%f,%f,%f,%f]\n\n",
                       RotRectsAndID.second[AimObjIndicesLR[0]], targetPoseL[4], R2D(targetPoseL[4]), targetPoseTmpL[0],
                       targetPoseTmpL[1], targetPoseTmpL[2], targetPoseTmpL[3], targetPoseTmpL[4], targetPoseTmpL[5]);
            }

            /// 获取目标位置(右臂)
            if (AimObjIndicesLR[1] != -1) { // 右侧有物体
                cv::RotatedRect RotatedRectR = RotRectsAndID.first[AimObjIndicesLR[1]];
                RightObjID = RotRectsAndID.second[AimObjIndicesLR[1]];
                // 获取机器人坐标系下坐标
                if (!_graphicsGrasp->getObjPose(RotatedRectR, targetPoseR, cloud, 0, 1, 1)) continue;
                targetPoseTmpR = targetPoseR; // 目标位置副本

                targetPoseTmpR[0] = _graphicsGrasp->height_Lv2_R;

                // 移动到物体上方
                targetPoseTmpR[4] = 0;

                cout << "[INFO] targetPoseR: " << targetPoseR << endl;

                printf("[INFO] 右臂待抓取物体信息 ID:[%d] Angle:[%f(%f)] Pose:[%f,%f,%f,%f,%f,%f]\n\n",
                       RotRectsAndID.second[AimObjIndicesLR[1]], targetPoseR[4], R2D(targetPoseR[4]), targetPoseTmpR[0],
                       targetPoseTmpR[1],
                       targetPoseTmpR[2], targetPoseTmpR[3], targetPoseTmpR[4], targetPoseTmpR[5]);
            }

            std::vector<double> PickJointsL, PickJointsR, PickJoints;
            std::vector<std::vector<double>> PickJointsL_T2, PickJointsR_T2; // 抓取位置列表(关节角) NOTE 小立方体抓取位置即为放置位置
            PickJointsL_T2.push_back(PlaceJointsL1_T2);
            PickJointsL_T2.push_back(PlaceJointsL2_T2);
            PickJointsL_T2.push_back(PlaceJointsL3_T2);
            PickJointsL_T2.push_back(PlaceJointsL4_T2);

            PickJointsR_T2.push_back(PlaceJointsR1_T2);
            PickJointsR_T2.push_back(PlaceJointsR2_T2);
            PickJointsR_T2.push_back(PlaceJointsR3_T2);
            PickJointsR_T2.push_back(PlaceJointsR4_T2);

            // 确保 PickJointsL PickJointsR 都有值
            PickJointsL = IniteJointsL;
            PickJointsR = IniteJointsR;

            printf("\033[0;31m%s\033[0m\n", "=========================== 寻找右侧是否有对应的小立方体 =========================");

            /// 寻找右侧是否有对应的小立方体 NOTE 不需换立方体 左臂：积木 右臂：立方体
            if (AimObjIndicesLR[0] != -1) { // 左臂有待抓取物
                for (size_t id = 0; id < PlacedCubeIDR.size(); id++) {
                    if (PlacedCubeIDR[id] == LeftObjID) { // 右侧找到对应小立方体
                        printf("\033[0;32m%s\033[0m\n", "[INFO] 左臂有待抓取物, 右侧找到对应小立方体");
                        moveDevice = 0; // 左臂抓取积木
                        // NOTE 左臂抓取积木
                        printf("\033[0;32m%s\033[0m\n", "[INFO] 左臂先抓取积木");
                        MoveJoints(JuggleIniteJoints, Vel_Lv4, Acc_Lv4, 0);
                        // 移动到固定高度
                        if (HaveObjL) targetPoseTmpL[0] = _graphicsGrasp->height_smallCube_L; // FIXME
                        if (HaveObjR) targetPoseTmpR[0] = _graphicsGrasp->height_smallCube_R;
                        mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);
                        MovePose(targetPose, Vel_Lv2, Acc_Lv2, moveDevice);

                        // 调整末端姿态
                        targetPose = getRobotJoints(2); // 获取当前关节角
                        if (HaveObjL) targetPose[5] += D2R(180 + R2D(targetPoseL[4])); // 获取原始末端关节角, 加补角
                        if (HaveObjR) targetPose[11] += targetPoseR[4]; // 获取原始末端关节角
                        MoveJoints(targetPose, Vel_Lv4, Acc_Lv4, 2);

                        // 左臂下降
                        targetPose = getRobotPose(2); // 获取双臂当前位置
                        targetPose[0] += 0.03; // 左臂
                        MovePose(targetPose, Vel_Lv2, Acc_Lv2, moveDevice);
                        HandClose(0);
                        // 左臂抬起
                        targetPose = getRobotPose(2); // 获取双臂当前位置
                        targetPose[0] -= 0.03; // 左臂
                        MovePose(targetPose, Vel_Lv2, Acc_Lv2, moveDevice);

                        printf("\033[0;32m%s\033[0m\n", "[INFO] 左臂积木抓取完成, 右臂抓取立方体");
                        // id号位是对应小立方体 右臂抓取
                        PickJointsR = PickJointsR_T2[id];
                        printf("[INFO] id号位是对应小立方体 左臂抓取 左臂抓取立方体 id: %zu\n", id);
                        mergeTargetLR(PickJointsL, PickJointsR, PickJoints);
                        MoveJoints(PickJoints, Vel_Lv3, Acc_Lv3, 1);

                        // 右臂下降
                        targetPose = getRobotPose(2); // 获取双臂当前位置
                        targetPose[6] -= 0.055; // 右臂 FIXME 右臂下降不够深
                        MovePose(targetPose, Vel_Lv2, Acc_Lv2, 1);
                        HandClose(1);

                        // 右臂抬起
                        targetPose = getRobotPose(2); // 获取双臂当前位置
                        targetPose[6] += 0.1; // 右臂 NOTE 多抬一点
                        MovePose(targetPose, Vel_Lv2, Acc_Lv2, 1);

                        /// 小立方体抓取成功, 开始装配 右臂为立方体
                        printf("\033[0;32m%s\033[0m\n", "[INFO] 积木及立方体准备就绪, 双臂移动到装配出事位置, 准备开始装配");
                        // 双臂移动到装配初始位置
                        PickJointsL = TogetherJointsLN;
                        PickJointsR = TogetherJointsR;
                        mergeTargetLR(PickJointsL, PickJointsR, PickJoints);
                        MoveJoints(PickJoints, Vel_Lv2_, Vel_Lv2_, 2);

                        // 左臂向中间移动
                        targetPose = getRobotPose(2); // 获取双臂当前位置
                        targetPose[8] -= 0.01; // 右臂
                        MovePose(targetPose, Vel_Lv2, Acc_Lv2, 0);

                        HandOpen(0);

                        MoveJoints(JuggleIniteJoints, Vel_Lv2_, Vel_Lv2_, 0);

                        printf("\033[0;32m%s\033[0m\n", "[INFO] 装配成功, 右臂放置装配体");
                        // 右臂放置装配体
                        targetPose = getRobotPose(2); // 获取双臂当前位置
                        targetPose[6] -= 0.1; // 右臂
                        targetPose[8] += 0.2; // 右臂
                        MovePose(targetPose, Vel_Lv2, Acc_Lv2, 1);
                    }
                }
            }

            printf("\033[0;31m%s\033[0m\n", "=========================== 寻找左侧是否有对应的小立方体 =========================");

            /// 寻找左侧是否有对应的小立方体 NOTE 不需换立方体 左臂：立方体 右臂：积木
            if (AimObjIndicesLR[1] != -1  && moveDevice == -1) { // 右臂有待抓取物, 左臂空闲
                for (size_t id = 0; id < PlacedCubeIDL.size(); id++) {
                    if (PlacedCubeIDL[id] == RightObjID) { // 左侧找到对应小立方体
                        printf("\033[0;32m%s\033[0m\n", "[INFO] 右臂有积木待抓取, 左臂空闲, 左侧找到对应小立方体");
                        moveDevice = 1; // 右臂抓取积木
                        // NOTE 右臂抓取积木
                        printf("\033[0;32m%s\033[0m\n", "[INFO] 右臂先抓取积木");
                        MoveJoints(JuggleIniteJoints, Vel_Lv4, Acc_Lv4, 1);
                        // 移动到固定高度
                        if (HaveObjL) targetPoseTmpL[0] = _graphicsGrasp->height_smallCube_L; // FIXME
                        if (HaveObjR) targetPoseTmpR[0] = _graphicsGrasp->height_smallCube_R;
                        mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);
                        MovePose(targetPose, Vel_Lv2, Acc_Lv2, moveDevice);

                        // 调整末端姿态
                        targetPose = getRobotJoints(2); // 获取当前关节角
                        if (HaveObjL) targetPose[5] += D2R(180 + R2D(targetPoseL[4])); // 获取原始末端关节角, 加补角
                        if (HaveObjR) targetPose[11] += targetPoseR[4]; // 获取原始末端关节角
                        MoveJoints(targetPose, Vel_Lv4, Acc_Lv4, 2);

                        printf("[INFO] 右臂下降\n");
                        // 右臂下降
                        targetPose = getRobotPose(2); // 获取双臂当前位置
                        targetPose[6] -= 0.03; // 右臂
                        MovePose(targetPose, Vel_Lv2, Acc_Lv2, moveDevice);
                        HandClose(1);

                        printf("[INFO] 右臂抬起\n");
                        // 右臂抬起
                        targetPose = getRobotPose(2); // 获取双臂当前位置
                        targetPose[6] += 0.03; // 右臂
                        MovePose(targetPose, Vel_Lv2, Acc_Lv2, moveDevice);

                        printf("\033[0;32m%s\033[0m\n", "[INFO] 右臂积木抓取完成, 左臂抓取立方体");
                        // id号位是对应小立方体 左臂抓取
                        PickJointsL = PickJointsL_T2[id];
                        mergeTargetLR(PickJointsL, PickJointsR, PickJoints);
                        printf("[INFO] id号位是对应小立方体 左臂抓取 左臂抓取立方体 id: %zu\n", id);
                        MoveJoints(PickJoints, Vel_Lv2_, Vel_Lv2_, 0);

                        // 左臂下降
                        targetPose = getRobotPose(2); // 获取双臂当前位置
                        targetPose[0] += 0.055; // 左臂 FIXME 左臂下降不够深
                        MovePose(targetPose, Vel_Lv2, Acc_Lv2, 0);
                        HandClose(0);

                        // 左臂抬起
                        targetPose = getRobotPose(2); // 获取双臂当前位置
                        targetPose[0] -= 0.1; // 左臂 NOTE 多抬一点
                        MovePose(targetPose, Vel_Lv2, Acc_Lv2, 0);

                        /// 小立方体抓取成功, 开始装配 左臂为立方体
                        printf("\033[0;32m%s\033[0m\n", "[INFO] 积木及立方体准备就绪, 双臂移动到装配出事位置, 准备开始装配");
                        // 双臂移动到装配初始位置
                        PickJointsL = TogetherJointsL;
                        PickJointsR = TogetherJointsRN;
                        mergeTargetLR(PickJointsL, PickJointsR, PickJoints);
                        MoveJoints(PickJoints, Vel_Lv3, Acc_Lv3, 2);

                        // 右臂向中间移动
                        targetPose = getRobotPose(2); // 获取双臂当前位置
                        targetPose[8] -= 0.01; // 右臂
                        MovePose(targetPose, Vel_Lv2, Acc_Lv2, 1);

                        HandOpen(1);

                        MoveJoints(JuggleIniteJoints, Vel_Lv2_, Vel_Lv2_, 1);

                        printf("\033[0;32m%s\033[0m\n", "[INFO] 装配成功, 左臂放置装配体");
                        // 左臂放置装配体
                        targetPose = getRobotPose(2); // 获取双臂当前位置
                        targetPose[0] += 0.1; // 左臂
                        targetPose[2] += 0.2; // 左臂
                        MovePose(targetPose, Vel_Lv2, Acc_Lv2, 0);

                    }
                }
            }

            printf("\033[0;31m%s\033[0m\n", "================== 左右臂均空闲(积木与待装配小立方体在同一侧), 左臂有积木, 从左侧抓取小立方体到右侧 ===============");

            /// 左右臂均空闲(积木与待装配小立方体在同一侧), 左臂有积木, 从左侧抓取小立方体到右侧  NOTE 需换立方体！ 左臂先抓立方体后抓积木
            if (moveDevice == -1 && AimObjIndicesLR[0] != -1) {
                for (size_t id = 0; id < PlacedCubeIDL.size(); id++) {
                    if (PlacedCubeIDL[id] == LeftObjID) { // 左侧找到对应小立方体
                        printf("\033[0;32m%s\033[0m\n", "\"[INFO] 左右臂均空闲(积木与待装配小立方体在同一侧), 左侧找到对应小立方体, 立方体左->右\"");
                        moveDevice = 0; // 左臂抓取积木, 然后抓取立方体

                        printf("\033[0;32m%s\033[0m\n", "[INFO] 左臂抓取需要的小立方体");
                        // id号位是对应小立方体 左臂抓取
                        PickJointsL = PickJointsL_T2[id];
                        printf("[INFO] id号位是对应小立方体 左臂抓取 左臂抓取立方体 id: %zu\n", id);
                        mergeTargetLR(PickJointsL, PickJointsR, PickJoints);
                        MoveJoints(PickJoints, Vel_Lv3, Acc_Lv3, moveDevice);
                        // 下降
                        targetPose = getRobotPose(2); // 获取双臂当前位置
                        targetPose[0] += 0.03; // 左臂
                        MovePose(targetPose, Vel_Lv2, Acc_Lv2, moveDevice);
                        HandClose(moveDevice);
                        // 抬起
                        targetPose = getRobotPose(2); // 获取双臂当前位置
                        targetPose[0] -= 0.06; // 左臂  FIXME 抬高
                        MovePose(targetPose, Vel_Lv2, Acc_Lv2, moveDevice);

                        printf("\033[0;32m%s\033[0m\n", "[INFO] 立方体抓取完成, 左右臂移动到交换初始位置, 立方体在左手");
                        // 双臂移动到交换初始位置
                        PickJointsL = TogetherJointsL;
                        PickJointsR = TogetherJointsR;
                        mergeTargetLR(PickJointsL, PickJointsR, PickJoints);
                        MoveJoints(PickJoints, Vel_Lv3, Acc_Lv3, 2);

                        printf("\033[0;32m%s\033[0m\n", "[INFO] 右臂转动90度");
                        // 调整末端姿态
                        targetPose = getRobotJoints(2); // 获取当前关节角
                        targetPose[11] -= D2R(90); // 转动90度  FIXME
                        MoveJoints(targetPose, Vel_Lv4, Acc_Lv4, 1);

                        printf("\033[0;32m%s\033[0m\n", "[INFO] 左右臂向中间移动");
                        targetPose = getRobotPose(2); // 获取双臂当前位置
                        targetPose[2] -= 0.02; // 左臂
                        targetPose[8] -= 0.02; // 右臂
                        MovePose(targetPose, Vel_Lv2, Acc_Lv2, 2);

                        // 右臂抓住小立方体
                        HandClose(1);

                        printf("\033[0;32m%s\033[0m\n", "[INFO] 右臂已拿到小立方体, 左手打开, 左臂外移");
                        HandOpen(moveDevice);
                        // 左臂离开
                        targetPose = getRobotPose(2); // 获取双臂当前位置
                        targetPose[2] += 0.06; // 左臂
                        MovePose(targetPose, Vel_Lv2, Acc_Lv2, moveDevice);

                        /// 右臂拿到小立方体, NOTE 左臂抓取积木
                        printf("\033[0;32m%s\033[0m\n", "[INFO] 左臂开始抓取积木");

                        MoveJoints(JuggleIniteJoints, Vel_Lv3, Acc_Lv3, 0); // 左臂回积木抓取初始位置  FIXME!!!!

                        /// 需记录当前右臂位置 后返回

                        // 移动到固定高度
                        if (HaveObjL) targetPoseTmpL[0] = _graphicsGrasp->height_smallCube_L; // FIXME
                        mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);
                        MovePose(targetPose, Vel_Lv2, Acc_Lv2, moveDevice);

                        // 调整末端姿态
                        targetPose = getRobotJoints(2); // 获取当前关节角
                        if (HaveObjL) targetPose[5] += D2R(180 + R2D(targetPoseL[4])); // 获取原始末端关节角, 加补角
                        MoveJoints(targetPose, Vel_Lv4, Acc_Lv4, 2);

                        // 左臂下降
                        targetPose = getRobotPose(2); // 获取双臂当前位置
                        targetPose[0] += 0.03; // 左臂
                        MovePose(targetPose, Vel_Lv2, Acc_Lv2, moveDevice);
                        HandClose(0);
                        // 左臂抬起
                        targetPose = getRobotPose(2); // 获取双臂当前位置
                        targetPose[0] -= 0.03; // 左臂
                        MovePose(targetPose, Vel_Lv2, Acc_Lv2, moveDevice);

                        printf("\033[0;32m%s\033[0m\n", "[INFO] 左臂积木抓取成功, 移动到装配位置");
                        // 左臂移动到交换初始位置 FIXME 应该是垂直位置
                        PickJointsL = TogetherJointsLN;
                        mergeTargetLR(PickJointsL, PickJointsR, PickJoints);
                        MoveJoints(PickJoints, Vel_Lv3, Acc_Lv3, moveDevice);

                        printf("\033[0;32m%s\033[0m\n", "[INFO] 左臂向中间靠拢");
                        // 左臂向中间移动
                        targetPose = getRobotPose(2); // 获取双臂当前位置
                        targetPose[2] -= 0.03; // 左臂
                        MovePose(targetPose, Vel_Lv2, Acc_Lv2, moveDevice);
                        HandOpen(0);

                        // 调整末端姿态
                        targetPose = getRobotJoints(2); // 获取当前关节角
                        targetPose[11] += D2R(90); // 转动90度  FIXME
                        MoveJoints(targetPose, Vel_Lv4, Acc_Lv4, 1);

                        printf("\033[0;32m%s\033[0m\n", "[INFO] 装配完成 右臂放置装配体");
                        // 右臂放置装配体
                        // y方向平移
                        targetPose = getRobotPose(2); // 获取双臂当前位置
                        targetPose[7] = -0.2; // 右臂
                        MovePose(targetPose, Vel_Lv2, Acc_Lv2, 1);

                        // z方向平移
                        targetPose = getRobotPose(2); // 获取双臂当前位置
                        targetPose[8] += 0.2; // 右臂
                        MovePose(targetPose, Vel_Lv2, Acc_Lv2, 1);

                        HandOpen(1);

                        MoveInit();
                    }
                }
            }

            printf("\033[0;31m%s\033[0m\n", "================== 左右臂均空闲(积木与待装配小立方体在同一侧), 左臂无积木, 从右侧抓取小立方体到左侧 ===============");

            /// 左右臂均空闲(积木与待装配小立方体在同一侧), 左臂无积木, 从右侧抓取小立方体到左侧 NOTE 需换立方体！ 右臂先抓立方体后抓积木
            if (moveDevice == -1 && AimObjIndicesLR[0] == -1 && AimObjIndicesLR[1] != -1) {
                for (size_t id = 0; id < PlacedCubeIDR.size(); id++) {
                    if (PlacedCubeIDR[id] == RightObjID) { // 右侧找到对应小立方体
                        printf("\033[0;32m%s\033[0m\n", "\"[INFO] 左右臂均空闲(积木与待装配小立方体在同一侧), 右侧找到对应小立方体, 立方体右->左");
                        moveDevice = 1; // 右臂抓取

                        printf("\033[0;32m%s\033[0m\n", "[INFO] 右臂抓取需要的小立方体");
                        // id号位是对应小立方体 右臂抓取
                        PickJointsR = PickJointsR_T2[id];
                        printf("[INFO] id号位是对应小立方体 左臂抓取 左臂抓取立方体 id: %zu\n", id);
                        mergeTargetLR(PickJointsL, PickJointsR, PickJoints);
                        MoveJoints(PickJoints, Vel_Lv3, Acc_Lv3, moveDevice);
                        // 下降
                        targetPose = getRobotPose(2); // 获取双臂当前位置
                        targetPose[6] -= 0.03; // 右臂
                        MovePose(targetPose, Vel_Lv2, Acc_Lv2, moveDevice);
                        HandClose(moveDevice);
                        // 抬起
                        targetPose = getRobotPose(2); // 获取双臂当前位置
                        targetPose[6] += 0.06; // 右臂
                        MovePose(targetPose, Vel_Lv2, Acc_Lv2, moveDevice);

                        printf("\033[0;32m%s\033[0m\n", "[INFO] 立方体抓取完成, 左右臂移动到交换初始位置, 立方体在右手");
                        // 双臂移动到交换初始位置
                        PickJointsL = TogetherJointsL;
                        PickJointsR = TogetherJointsR;
                        mergeTargetLR(PickJointsL, PickJointsR, PickJoints);
                        MoveJoints(PickJoints, Vel_Lv3, Acc_Lv3, 2);

                        printf("\033[0;32m%s\033[0m\n", "[INFO] 左臂转动90度");
                        // 调整末端姿态
                        targetPose = getRobotJoints(2); // 获取当前关节角
                        targetPose[5] -= D2R(90); // 转动90度
                        MoveJoints(targetPose, Vel_Lv4, Acc_Lv4, 1);

                        // 左右臂向中间移动
                        targetPose = getRobotPose(2); // 获取双臂当前位置
                        targetPose[2] -= 0.02; // 左臂
                        targetPose[8] -= 0.02; // 右臂
                        MovePose(targetPose, Vel_Lv2, Acc_Lv2, 2);

                        // 左臂抓住小立方体
                        HandClose(0);

                        /// 左臂拿到小立方体, 右臂抓取积木
                        printf("\033[0;32m%s\033[0m\n", "[INFO] 左臂已拿到小立方体, 右手打开, 右臂外移");
                        HandOpen(moveDevice);
                        // 右臂离开
                        targetPose = getRobotPose(2); // 获取双臂当前位置
                        targetPose[8] += 0.03; // 右臂
                        MovePose(targetPose, Vel_Lv2, Acc_Lv2, moveDevice);

                        /// 左臂拿到小立方体, NOTE 右臂抓取积木
                        printf("\033[0;32m%s\033[0m\n", "[INFO] 右臂开始抓取积木");
                        MoveJoints(JuggleIniteJoints, Vel_Lv3, Acc_Lv3, 1); // 左臂回积木抓取初始位置  FIXME!!!!
                        // 移动到固定高度
                        if (HaveObjR) targetPoseTmpR[0] = _graphicsGrasp->height_smallCube_R;
                        mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);
                        MovePose(targetPose, Vel_Lv2, Acc_Lv2, moveDevice);

                        // 调整末端姿态
                        targetPose = getRobotJoints(2); // 获取当前关节角
                        if (HaveObjR) targetPose[11] += targetPoseR[4]; // 获取原始末端关节角
                        MoveJoints(targetPose, Vel_Lv4, Acc_Lv4, 2);

                        // 右臂下降
                        targetPose = getRobotPose(2); // 获取双臂当前位置
                        targetPose[6] -= 0.03; // 右臂
                        MovePose(targetPose, Vel_Lv2, Acc_Lv2, moveDevice);
                        HandClose(moveDevice);

                        // 右臂抬起
                        targetPose = getRobotPose(2); // 获取双臂当前位置
                        targetPose[6] += 0.03; // 右臂
                        MovePose(targetPose, Vel_Lv2, Acc_Lv2, moveDevice);

                        printf("\033[0;32m%s\033[0m\n", "[INFO] 右臂积木抓取成功, 移动到装配位置");
                        // 右臂移动到装配位置 FIXME 应该是垂直位置
                        PickJointsR = TogetherJointsRN;
                        mergeTargetLR(PickJointsL, PickJointsR, PickJoints);
                        MoveJoints(PickJoints, Vel_Lv3, Acc_Lv3, moveDevice);

                        printf("\033[0;32m%s\033[0m\n", "[INFO] 右臂向中间靠拢");
                        targetPose = getRobotPose(2); // 获取双臂当前位置
                        targetPose[8] -= 0.03; // 右臂
                        MovePose(targetPose, Vel_Lv2, Acc_Lv2, moveDevice);
                        HandOpen(moveDevice);

                        printf("\033[0;32m%s\033[0m\n", "[INFO] 装配完成 左臂放置装配体");
                        targetPose = getRobotPose(2); // 获取双臂当前位置
                        targetPose[0] += 0.1; // 左臂
                        targetPose[2] += 0.2; // 左臂
                        MovePose(targetPose, Vel_Lv2, Acc_Lv2, 0);

                        // 调整末端姿态
                        targetPose = getRobotJoints(2); // 获取当前关节角
                        targetPose[5] += D2R(90); // 转动90度  FIXME
                        MoveJoints(targetPose, Vel_Lv4, Acc_Lv4, 0);

                        printf("\033[0;32m%s\033[0m\n", "[INFO] 装配完成 左臂放置装配体");
                        // 左臂放置装配体
                        // y方向平移
                        targetPose = getRobotPose(2); // 获取双臂当前位置
                        targetPose[1] = -0.2; // 左臂
                        MovePose(targetPose, Vel_Lv2, Acc_Lv2, 0);

                        // z方向平移
                        targetPose = getRobotPose(2); // 获取双臂当前位置
                        targetPose[2] += 0.2; // 左臂
                        MovePose(targetPose, Vel_Lv2, Acc_Lv2, 0);

                        HandOpen(0);

                        MoveInit();
                    }
                }
            }

            MoveInit();
        }

        //////////////////////////////////////////  小立方体  ///////////////////////////////////////////////
        if (SmallCubeFlag) { /// 有小立方体
            printf("\033[0;33m[INFO] Detected small cube, begin to pick it up...\033[0m\n");

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
            } else if ((AimObjIndicesLR[0] = -1) && (AimObjIndicesLR[1] = -1)) continue;

            printf("\033[0;31m%s\033[0m\n", "============================ 移动到小立方体抓取初始位置 =========================");
            MoveJoints(JuggleIniteJoints, Vel_Lv3, Acc_Lv3, 2);

            printf("\033[0;31m%s\033[0m\n", "================================== 移动到正上方 ==============================");

            /// 获取目标位置(左臂)
            if (AimObjIndicesLR[0] != -1) { // 左侧有物体
                cv::RotatedRect RotatedRectL = RotRectsAndID.first[AimObjIndicesLR[0]];
                LeftObjID = RotRectsAndID.second[AimObjIndicesLR[0]];
                // 获取机器人坐标系下坐标
                if (!_graphicsGrasp->getObjPose(RotatedRectL, targetPoseL, cloud, 1, 1, 0, 8.0)) continue; // 短边 两点法

                targetPoseTmpL = targetPoseL; // 目标位置副本

                targetPoseTmpL[0] = _graphicsGrasp->height_Lv2_L;

                // 移动到物体上方
                targetPoseTmpL[4] = 0;

                cout << "[INFO] targetPoseL: " << targetPoseL << endl;

                printf("[INFO] 左臂待抓取物体信息 ID:[%d] Angle:[%f(%f)] Pose:[%f,%f,%f,%f,%f,%f]\n\n",
                       RotRectsAndID.second[AimObjIndicesLR[0]], targetPoseL[4], R2D(targetPoseL[4]), targetPoseTmpL[0],
                       targetPoseTmpL[1], targetPoseTmpL[2], targetPoseTmpL[3], targetPoseTmpL[4], targetPoseTmpL[5]);
            }

            /// 获取目标位置(右臂)
            if (AimObjIndicesLR[1] != -1) { // 右侧有物体
                cv::RotatedRect RotatedRectR = RotRectsAndID.first[AimObjIndicesLR[1]];
                RightObjID = RotRectsAndID.second[AimObjIndicesLR[1]];
                // 获取机器人坐标系下坐标
                if (!_graphicsGrasp->getObjPose(RotatedRectR, targetPoseR, cloud, 1, 1, 1, 8.0)) continue; // 短边 两点法
                targetPoseTmpR = targetPoseR; // 目标位置副本

                targetPoseTmpR[0] = _graphicsGrasp->height_Lv2_R;

                // 移动到物体上方
                targetPoseTmpR[4] = 0;

                cout << "[INFO] targetPoseR: " << targetPoseR << endl;

                printf("[INFO] 右臂待抓取物体信息 ID:[%d] Angle:[%f(%f)] Pose:[%f,%f,%f,%f,%f,%f]\n\n",
                       RotRectsAndID.second[AimObjIndicesLR[1]], targetPoseR[4], R2D(targetPoseR[4]), targetPoseTmpR[0],
                       targetPoseTmpR[1],
                       targetPoseTmpR[2], targetPoseTmpR[3], targetPoseTmpR[4], targetPoseTmpR[5]);
            }

            /// 处理待抓取的两个物体靠的太近
            if (AimObjIndicesLR[0] != -1 && AimObjIndicesLR[1] != -1) {
                float distance = RotRectsAndID.first[AimObjIndicesLR[1]].center.x -
                                 RotRectsAndID.first[AimObjIndicesLR[0]].center.x;
                printf("[INFO] Distance between Obj in pix: %f\n", distance);
                if (distance < 100) { // FIXME
                    printf("\033[0;32m%s\033[0m\n",
                           "[WARN] Distance is too short between the Objects, left arm first!\n");
                    moveDevice = 0; // 物体靠的太近, 左臂优先
                }
            }

            mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);

            cout << "[INFO] targetPose: " << targetPose << endl;

            MovePose(targetPose, Vel_Lv2, Acc_Lv2, moveDevice);

            printf("\033[0;31m%s\033[0m\n", "================================= 调整末端姿态 ===============================");

            targetPose = getRobotJoints(2); // 获取当前关节角
            if (HaveObjL) targetPose[5] += D2R(180 + R2D(targetPoseL[4])); // 获取原始末端关节角, 加补角
            if (HaveObjR) targetPose[11] += targetPoseR[4]; // 获取原始末端关节角

            MoveJoints(targetPose, Vel_Lv4, Acc_Lv4, 2);

            printf("\033[0;31m%s\033[0m\n", "================================= 手抓下降并抓取 ==============================");

            /// 修改姿态, 手臂垂直下降
            targetPose = getRobotPose(2); // 获取双臂当前位置
            targetPose[0] += 0.04; // 左臂
            targetPose[6] -= 0.04; // 右臂

            MovePose(targetPose, Vel_Lv2_, Acc_Lv2_, moveDevice);

            /// 手抓闭合
            HandClose(0);
            HandClose(1);

            /// 修改姿态, 手臂垂直抬起
            targetPoseTmpR[0] += 0.04; // 右臂
            targetPoseTmpL[0] -= 0.04; // 左臂

            mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);

            MovePose(targetPose, Vel_Lv2_, Acc_Lv2_, moveDevice);

            printf("\033[0;31m%s\033[0m\n", "================================= 放置到指定位置 ==============================");

            std::vector<double> PlaceJointsL, PlaceJointsR, PlaceJoints; // 左右臂目标关节角

            std::vector<std::vector<double>> PlaceJointsL_T2, PlaceJointsR_T2; // 放置位置列表(关节角)

            PlaceJointsL_T2.push_back(PlaceJointsL1_T2);
            PlaceJointsL_T2.push_back(PlaceJointsL2_T2);
            PlaceJointsL_T2.push_back(PlaceJointsL3_T2);
            PlaceJointsL_T2.push_back(PlaceJointsL4_T2);

            PlaceJointsR_T2.push_back(PlaceJointsR1_T2);
            PlaceJointsR_T2.push_back(PlaceJointsR2_T2);
            PlaceJointsR_T2.push_back(PlaceJointsR3_T2);
            PlaceJointsR_T2.push_back(PlaceJointsR4_T2);

            // 确保 PlaceJointsL PlaceJointsR 都有值
            PlaceJointsL = IniteJointsL;
            PlaceJointsR = IniteJointsR;

            // 放置小立方体
            PlaceJointsL = PlaceJointsL_T2[PlacedSmallCubeCntL]; // 按顺序放置到指定位置
            PlaceJointsR = PlaceJointsR_T2[PlacedSmallCubeCntR]; // 按顺序放置到指定位置
            mergeTargetLR(PlaceJointsL, PlaceJointsR, PlaceJoints);
            MoveJoints(PlaceJoints, Vel_Lv3, Acc_Lv3, moveDevice); // 放置物体

            /// 手臂垂直下降
            targetPose = getRobotPose(2); // 获取双臂当前位置
            targetPose[0] += 0.04; // 左臂
            targetPose[6] -= 0.04; // 右臂

            MovePose(targetPose, Vel_Lv2_, Acc_Lv2_, moveDevice);

            /// 手抓张开
            HandOpen(0);
            HandOpen(1);

            /// 手臂垂直抬起
            targetPose = getRobotPose(2); // 获取双臂当前位置
            targetPose[0] -= 0.04; // 左臂
            targetPose[6] += 0.04; // 右臂

            MovePose(targetPose, Vel_Lv2_, Acc_Lv2_, moveDevice);

            /// 已抓取小立方体数统计
            if (moveDevice == 2) {
                PickedSmallCubeCnt += 2;
                PlacedSmallCubeCntL++;
                PlacedSmallCubeCntR++;
                PlacedCubeIDL.push_back(LeftObjID);
                PlacedCubeIDR.push_back(RightObjID);
            } else if (moveDevice == 0) {
                PickedSmallCubeCnt++;
                PlacedSmallCubeCntL++;
                PlacedCubeIDL.push_back(LeftObjID);
            } else if (moveDevice == 1) {
                PickedSmallCubeCnt++;
                PlacedSmallCubeCntR++;
                PlacedCubeIDR.push_back(RightObjID);
            }

            MoveInit();

            printf("\033[0;31m%s\033[0m\n\n\n", "=============================== 小立方体抓取放置成功 ==============================");
        }
    }
}

bool GraspController::graspControlTask3() {
    bool ret = captureImage(-100); // 采集图像
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    /// 初始位置
    MoveInit(); // 同步移动到初始位置, 打开手抓

    /// 点云初始化
    cloudInit();

    printf("\033[0;31m%s\033[0m\n", "================================== 初始化成功 =================================");

    while (true) {
        printf("\033[0;31m%s\033[0m\n", "================================== 采集图像 =================================");
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

//        color = cv::imread("../../../grasp/data/images/cube1.jpg");
//        depth = cv::imread("../../../grasp/data/images/cube1.png", -1);

        _graphicsGrasp->createPointCloud(color, depth, cloud); // 创建点云

        if (saveFlag) { // 每次运行程序保存一次数据
            saveCloudAndImages();
            saveFlag = false;
//            exit(-100);
        }

        printf("\033[0;31m%s\033[0m\n", "=================================== 目标检测 ================================");

        if (PickedCubeCntTask3 < 0) { // NOTE 大长方体体总数！！！！！！！！！！！！！！！！！！！！！！！！！！！
            RotRectsAndID = _graphicsGrasp->detectBigCubeTask3(color, cloud, 120, 0);
            if (RotRectsAndID.first.empty()) {
                printf("[ERROR] Did not get any rot rects for Big Cube!\n");
                continue;
            }

            // 确定两侧待抓取物体索引
            std::vector<int> AimObjIndicesLR = _graphicsGrasp->findAimObjLR(RotRectsAndID, 1); /// 寻找col最值

            int moveDevice = -1;

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
            } else if ((AimObjIndicesLR[0] = -1) && (AimObjIndicesLR[1] = -1)) continue;

            printf("\033[0;31m%s\033[0m\n", "============================= 移动到积木抓取初始位置 ===========================");
            MoveJoints(JuggleIniteJoints, Vel_Lv3, Acc_Lv3, 2);

            printf("\033[0;31m%s\033[0m\n", "================================= 移动到正上方 ===============================");

            /// 获取目标位置(左臂)
            if (AimObjIndicesLR[0] != -1) { // 左侧有物体
                cv::RotatedRect RotatedRectL = RotRectsAndID.first[AimObjIndicesLR[0]];
                // 获取机器人坐标系下坐标
                if (!_graphicsGrasp->getObjPose(RotatedRectL, targetPoseL, cloud, 0, 1, 0)) continue;

                targetPoseTmpL = targetPoseL; // 目标位置副本

                targetPoseTmpL[0] = _graphicsGrasp->height_Lv2_L;

                // 移动到物体上方
                targetPoseTmpL[4] = 0;

                cout << "[INFO] targetPoseL: " << targetPoseL << endl;

                printf("[INFO] 左臂待抓取物体信息 ID:[%d] Angle:[%f(%f)] Pose:[%f,%f,%f,%f,%f,%f]\n\n",
                       RotRectsAndID.second[AimObjIndicesLR[0]], targetPoseL[4], R2D(targetPoseL[4]), targetPoseTmpL[0],
                       targetPoseTmpL[1], targetPoseTmpL[2], targetPoseTmpL[3], targetPoseTmpL[4], targetPoseTmpL[5]);
            }

            /// 获取目标位置(右臂)
            if (AimObjIndicesLR[1] != -1) { // 右侧有物体
                cv::RotatedRect RotatedRectR = RotRectsAndID.first[AimObjIndicesLR[1]];
                // 获取机器人坐标系下坐标
                if (!_graphicsGrasp->getObjPose(RotatedRectR, targetPoseR, cloud, 0, 1, 1)) continue;
                targetPoseTmpR = targetPoseR; // 目标位置副本

                targetPoseTmpR[0] = _graphicsGrasp->height_Lv2_R;

                // 移动到物体上方
                targetPoseTmpR[4] = 0;

                cout << "[INFO] targetPoseR: " << targetPoseR << endl;

                printf("[INFO] 右臂待抓取物体信息 ID:[%d] Angle:[%f(%f)] Pose:[%f,%f,%f,%f,%f,%f]\n\n",
                       RotRectsAndID.second[AimObjIndicesLR[1]], targetPoseR[4], R2D(targetPoseR[4]), targetPoseTmpR[0],
                       targetPoseTmpR[1],
                       targetPoseTmpR[2], targetPoseTmpR[3], targetPoseTmpR[4], targetPoseTmpR[5]);
            }

            mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);

            cout << "[INFO] targetPose: " << targetPose << endl;

            MovePose(targetPose, Vel_Lv2, Acc_Lv2, moveDevice);

            printf("\033[0;31m%s\033[0m\n", "================================= 调整末端姿态 ===============================");

            targetPose = getRobotJoints(2); // 获取当前关节角
            if (HaveObjL) targetPose[5] += D2R(180 + R2D(targetPoseL[4])); // 获取原始末端关节角, 加补角
            if (HaveObjR) targetPose[11] += targetPoseR[4]; // 获取原始末端关节角

            MoveJoints(targetPose, Vel_Lv4, Acc_Lv4, 2);

            printf("\033[0;31m%s\033[0m\n", "================================= 手抓下降并抓取 ==============================");

            /// 手臂垂直下降
            targetPose = getRobotPose(2); // 获取双臂当前位置
            targetPose[0] += 0.07; // 左臂
            targetPose[6] -= 0.07; // 右臂

            MovePose(targetPose, Vel_Lv2_, Acc_Lv2_, moveDevice);

            /// 手抓闭合
            HandClose(0);
            HandClose(1);

            /// 手臂垂直抬起
            targetPose = getRobotPose(2); // 获取双臂当前位置
            targetPose[0] -= 0.07; // 左臂
            targetPose[6] += 0.07; // 右臂

            MovePose(targetPose, Vel_Lv2_, Acc_Lv2_, moveDevice); // FIXME !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            printf("\033[0;31m%s\033[0m\n", "================================= 放置到指定位置 ==============================");

            std::vector<double> PlaceJointsL, PlaceJointsR, PlaceJoints;
            if (BigCubePickedNumL == 0) PlaceJointsL = PlaceJointsL1_T3;
            if (BigCubePickedNumL == 1) PlaceJointsL = PlaceJointsL2_T3;
            if (BigCubePickedNumR == 0) PlaceJointsR = PlaceJointsR1_T3;
            if (BigCubePickedNumR == 1) PlaceJointsR = PlaceJointsR2_T3;

            mergeTargetLR(PlaceJointsL, PlaceJointsR, PlaceJoints);

            MoveJoints(PlaceJoints, Vel_Lv3, Acc_Lv3, moveDevice); // 放置物体

            /// 手臂垂直下降
            targetPose = getRobotPose(2); // 获取双臂当前位置
            targetPose[0] += 0.04; // 左臂
            targetPose[6] -= 0.04; // 右臂

            MovePose(targetPose, Vel_Lv2_, Acc_Lv2_, moveDevice);

            /// 手抓张开
            HandOpen(0);
            HandOpen(1);

            /// 手臂垂直抬起
            targetPose = getRobotPose(2); // 获取双臂当前位置
            targetPose[0] -= 0.04; // 左臂
            targetPose[6] += 0.04; // 右臂

            MovePose(targetPose, Vel_Lv2_, Acc_Lv2_, moveDevice);

            MoveInit(); /// 回初始位置拍照

            printf("\033[0;31m%s\033[0m\n\n\n",
                   "================================== 抓取成功 =================================");
            if (moveDevice == 0) {
                BigCubePickedNumL++;
                PickedCubeCntTask3++;
            }
            else if (moveDevice == 1) {
                BigCubePickedNumR++;
                PickedCubeCntTask3++;
            }
            else if (moveDevice == 2) {
                BigCubePickedNumL++;
                BigCubePickedNumR++;
                PickedCubeCntTask3 += 2;
            }
        }

        PickedCubeCntTask3 = 4;
        /// 已有四个大木块
        if (PickedCubeCntTask3 == 4) {
            std::vector<double> PickJointsL, PickJointsR, PickJoints;
            std::vector<double> targetPose;
            // 左臂抓取第一个木块
            PickJointsL = PlaceJointsL2_T3;
            mergeTargetLR(PickJointsL, PickJointsR, PickJoints);
            MoveJoints(PickJoints, Vel_Lv2_, Vel_Lv2_, 0);

            // 左臂下降
            targetPose = getRobotPose(2); // 获取双臂当前位置
            targetPose[0] += 0.04; // 左臂
            MovePose(targetPose, Vel_Lv2, Acc_Lv2, 0);
            HandClose(0);

            // 左臂抬起
            targetPose = getRobotPose(2); // 获取双臂当前位置
            targetPose[0] -= 0.04; // 左臂
            MovePose(targetPose, Vel_Lv2, Acc_Lv2, 0);

            /// 移动到中间固定位置1
            targetPose = getRobotPose(2); // 获取双臂当前位置
            targetPose[2] = -0.22; // Z
            targetPose[1] = -0.26; // Y FIXME ！！！！！！！！
            MovePose(targetPose, Vel_Lv2, Acc_Lv2, 0);

            // 调节角度
            double deg = 60;
            printf("\033[0;32m%s\033[0m\n", "[INFO] 左臂转动60度");
            // 调整末端姿态
            targetPose = getRobotJoints(2); // 获取当前关节角
            targetPose[5] += D2R(deg); // 转动60度
            MoveJoints(targetPose, Vel_Lv4, Acc_Lv4, 0);

            // 左臂下降
            targetPose = getRobotPose(2); // 获取双臂当前位置
            targetPose[0] += 0.04; // 左臂
            MovePose(targetPose, Vel_Lv2, Acc_Lv2, 0);

            // 左臂移动到积木一侧
            double radiu = 0.08;
            HandOpen(0);
            targetPose = getRobotPose(2); // 获取双臂当前位置
            targetPose[2] -= radiu * cos(deg); // Z
            targetPose[1] -= radiu * sin(deg); // Y FIXME ！！！！！！！！
            MovePose(targetPose, Vel_Lv2, Acc_Lv2, 0);
            HandClose(0);

            /// 右臂移动关节角到积木另一侧
            std::vector<double> JointsR = {D2R(-105.46), D2R(-86.96), D2R(81.26), D2R(14.44), D2R(-11.41), D2R(35.74),
                                           D2R(-105.46), D2R(-86.96), D2R(81.26), D2R(14.44), D2R(-11.41), D2R(35.74)}; // FIXME!!!
            MoveJoints(JointsR, Vel_Lv2, Acc_Lv2, 1);

            // 右臂下降
            targetPose = getRobotPose(2); // 获取双臂当前位置
            targetPose[6] -= 0.04; // 右臂 FIXME 下降到与左臂相同高度
            MovePose(targetPose, Vel_Lv2, Acc_Lv2, 1);
            HandClose(0);
            HandClose(1);

            // 双臂同时平移并抬起
            targetPose = getRobotPose(2); // 获取双臂当前位置
            // 平移
            targetPose[1] += 0.01; // 左臂 y
            targetPose[7] -= 0.01; // 右臂 y
            MovePose(targetPose, Vel_Lv2, Acc_Lv2, 2);

            // 抬起
            targetPose[1] -= 0.01; // 左臂 x
            targetPose[7] += 0.01; // 右臂 x
            MovePose(targetPose, Vel_Lv2, Acc_Lv2, 2);

            // 双臂同时下降
            targetPose = getRobotPose(2); // 获取双臂当前位置
            // 下降
            targetPose[1] += 0.01; // 左臂 x
            targetPose[7] -= 0.01; // 右臂 x
            MovePose(targetPose, Vel_Lv2, Acc_Lv2, 2);
            HandOpen(0);
            HandOpen(1);

            // 双臂张开爪子后抬起
            targetPose[1] -= 0.04; // 左臂 x
            targetPose[7] += 0.04; // 右臂 x
            MovePose(targetPose, Vel_Lv2, Acc_Lv2, 2);

            /// 左臂抓取第二个木块
            PickJointsL = PlaceJointsL1_T3;
            mergeTargetLR(PickJointsL, PickJointsR, PickJoints);
            MoveJoints(PickJoints, Vel_Lv2_, Vel_Lv2_, 0);

            // 左臂下降
            targetPose = getRobotPose(2); // 获取双臂当前位置
            targetPose[0] += 0.04; // 左臂
            MovePose(targetPose, Vel_Lv2, Acc_Lv2, 0);
            HandClose(0);

            // 左臂抬起
            targetPose = getRobotPose(2); // 获取双臂当前位置
            targetPose[0] -= 0.04; // 左臂
            MovePose(targetPose, Vel_Lv2, Acc_Lv2, 0);

            /// 移动到中间固定位置2 FIXME ！！！！！！！！
            targetPose = getRobotPose(2); // 获取双臂当前位置
            targetPose[2] = -0.00; // Z
            targetPose[1] = -0.40; // Y
            MovePose(targetPose, Vel_Lv2, Acc_Lv2, 0);

            // 调节角度
            deg = -30;
            printf("\033[0;32m%s\033[0m\n", "[INFO] 左臂转动60度");
            // 调整末端姿态
            targetPose = getRobotJoints(2); // 获取当前关节角
            targetPose[5] += D2R(deg); // 转动60度
            MoveJoints(targetPose, Vel_Lv4, Acc_Lv4, 0);

            // 左臂移动到积木一侧
            radiu = 0.08;
            HandOpen(0);
            targetPose = getRobotPose(2); // 获取双臂当前位置
            targetPose[2] += radiu * cos(deg); // Z
            targetPose[1] += radiu * sin(deg); // Y FIXME ！！！！！！！！
            MovePose(targetPose, Vel_Lv2, Acc_Lv2, 0);
            HandClose(0);

            /// 右臂移动关节角到积木另一侧
            std::vector<double> JointsL = {D2R(100.065), D2R(35.986), D2R(-76.171), D2R(67.843), D2R(10.065), D2R(0.000)}; // FIXME!!!
            MoveJoints(JointsR, Vel_Lv2, Acc_Lv2, 1);

            // 右臂下降
            targetPose = getRobotPose(2); // 获取双臂当前位置
            targetPose[6] -= 0.04; // 右臂 FIXME 下降到与左臂相同高度
            MovePose(targetPose, Vel_Lv2, Acc_Lv2, 0);
            HandClose(1);

            // 抬起
            targetPose = getRobotPose(2); // 获取双臂当前位置
            targetPose[1] -= 0.05; // 左臂 x
            targetPose[7] += 0.05; // 右臂 x
            MovePose(targetPose, Vel_Lv2, Acc_Lv2, 2);

            // 双臂同时平移
            targetPose = getRobotPose(2); // 获取双臂当前位置
            // 向后平移
            targetPose[1] += 0.01; // 左臂 y  FIXME
            targetPose[7] -= 0.01; // 右臂 y

            MovePose(targetPose, Vel_Lv2, Acc_Lv2, 2);

            // 双臂同时下降
            targetPose = getRobotPose(2); // 获取双臂当前位置
            // 下降
            targetPose[1] += 0.01; // 左臂 x
            targetPose[7] -= 0.01; // 右臂 x
            MovePose(targetPose, Vel_Lv2, Acc_Lv2, 2);
            HandOpen(0);
            HandOpen(1);

            // 双臂张开爪子后抬起
            targetPose[1] -= 0.04; // 左臂 x
            targetPose[7] += 0.04; // 右臂 x
            MovePose(targetPose, Vel_Lv2, Acc_Lv2, 2);

            MoveInit();
        }
    }
}

bool GraspController::graspControlTask4() {
    bool ret = captureImage(-100); // 采集图像
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    /// 初始位置
    MoveInit(); // 同步移动到初始位置, 打开手抓

    /// 点云初始化
    cloudInit();

    printf("\033[0;31m%s\033[0m\n", "================================== 初始化成功 =================================");

    while (true) {
        printf("\033[0;31m%s\033[0m\n", "================================== 采集图像 =================================");
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

        if (saveFlag) { // 每次运行程序保存一次数据
            saveCloudAndImages();
            saveFlag = false;
        }

        printf("\033[0;31m%s\033[0m\n", "=================================== 回初始位置 ===============================");
        MoveJoints(IniteJoints, Vel_Lv3, Acc_Lv3, 2);

        printf("\033[0;31m%s\033[0m\n", "=================================== 到水平位置 ===============================");

        std::vector<double> targetPoseTmpL, targetPoseTmpR, targetPose;
        targetPoseTmpL = TogetherJointsL;
        targetPoseTmpR = TogetherJointsR;
        mergeTargetLR(targetPoseTmpL, targetPoseTmpR, targetPose);
        MoveJoints(targetPoseTmpL, Vel_Lv3, Acc_Lv3, 2);

        printf("\033[0;31m%s\033[0m\n", "================================ 手抓移动到盘子位置 ============================");

        /// 手臂垂直下降
        targetPose = getRobotPose(2); // 获取双臂当前位置
        targetPose[0] += 0.05; // 左臂
        targetPose[6] -= 0.05; // 右臂

        MovePose(targetPose, Vel_Lv2_, Acc_Lv2_, 2);

        /// 手臂水平向前移动
        targetPose = getRobotPose(2); // 获取双臂当前位置
        targetPose[1] -= 0.05; // 左臂
        targetPose[7] -= 0.05; // 右臂

        MovePose(targetPose, Vel_Lv2_, Acc_Lv2_, 2);

        printf("\033[0;31m%s\033[0m\n", "=================================== 端起盘子 ================================");

        /// 手臂向中间靠拢
        targetPose = getRobotPose(2); // 获取双臂当前位置
        targetPose[2] -= 0.05; // 左臂
        targetPose[8] -= 0.05; // 右臂

        MovePose(targetPose, Vel_Lv2, Acc_Lv2, 2);

        /// 手臂垂直上升
        targetPose = getRobotPose(2); // 获取双臂当前位置
        targetPose[0] -= 0.05; // 左臂
        targetPose[6] += 0.05; // 右臂

        MovePose(targetPose, Vel_Lv2, Acc_Lv2, 2);

        printf("\033[0;31m%s\033[0m\n", "=================================== 移动盘子 ================================");

        /// 手臂水平向前移动
        targetPose = getRobotPose(2); // 获取双臂当前位置
        targetPose[1] += 0.3; // 左臂
        targetPose[7] += 0.3; // 右臂

        MovePose(targetPose, Vel_Lv2, Acc_Lv2, 2);

        printf("\033[0;31m%s\033[0m\n", "=================================== 放下盘子 ================================");
        /// 手臂垂直下降
        targetPose = getRobotPose(2); // 获取双臂当前位置
        targetPose[0] -= 0.05; // 左臂
        targetPose[6] += 0.05; // 右臂

        MovePose(targetPose, Vel_Lv2_, Acc_Lv2_, 2);

        /// 手臂向两侧移动
        targetPose = getRobotPose(2); // 获取双臂当前位置
        targetPose[2] += 0.05; // 左臂
        targetPose[8] += 0.05; // 右臂

        MovePose(targetPose, Vel_Lv2_, Acc_Lv2_, 2);


        printf("\033[0;31m%s\033[0m\n\n\n", "================================== 端取成功 =================================");
        MoveInit();
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

#if 0
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

    std::cout << "[Move] MoveTargetPose:" << targetPose << " acc:" << acc << " vel:" << vel << std::endl;
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
#endif

void GraspController::Move(const std::vector<double>& targetPose, double vel, double acc, int armId) {
    std::cout << "[INFO] Move ..." << std::endl;

    uint16_t actionArmId = -1;
    if (armId == 0) {
        actionArmId = LeftArm;
    } else if (armId == 1) {
        actionArmId = RightArm;
    }
    else if (armId == 2) {
        actionArmId = LeftArm | RightArm;
    }

    ErrorInfo _ErrorInfo;
    DeviceStatus _DeviceStatus{};
    std::vector<DeviceStatus> _vDeviceStatus;

    ErrorInfo _ErrorInfo_Move;
    DeviceStatus _DeviceStatus_Move{};
    std::vector<DeviceStatus> _vDeviceStatus_Move;

    std::vector<double> AimJoints;

    std::vector<double> initJoints;
//    initJoints = getRobotJoints(_cassemble2Driver);

    initJoints = {D2R(91.238), D2R(64.396), D2R(-37.523), D2R(-15.631), D2R(-1.713), D2R(5.382),
                                          D2R(-91.238), D2R(-64.396), D2R(37.523), D2R(15.631), D2R(1.713), D2R(-5.382)};

    _DeviceStatus._ID = actionArmId;
    _DeviceStatus._Status = {0,0};

    _DeviceStatus._Joints = initJoints;
    _DeviceStatus._Pos = targetPose;

    _cassemble2Driver->CartToJoint(_DeviceStatus, _ErrorInfo);

    for (size_t i = 0; i < _DeviceStatus._Joints.size(); i++) {
        printf("[Move] AimJoint[%zu]: %f ", i, R2D(_DeviceStatus._Joints[i]));
    }

    cout << endl;

    std::cout << "[Move] MoveTargetPose:" << targetPose << " acc:" << acc << " vel:" << vel << std::endl;
    std::cout << "[Move] CalcJoints: " << _DeviceStatus._Joints << std::endl;
    std::cout << "[Move] DeviceStatus.Pos: " << _DeviceStatus._Pos << "  ArmId: " << armId << std::endl;
    std::cout << "[Move] DeviceStatus.Vel: " <<  _DeviceStatus._Vel << "  DeviceStatus.Acc: " <<  _DeviceStatus._Acc << std::endl;

    AimJoints = _DeviceStatus._Joints;

    // 移动
    _DeviceStatus_Move._Vel = {vel, vel};
    _DeviceStatus_Move._Acc = {acc, acc};
    _DeviceStatus_Move._ID =  actionArmId;
    _DeviceStatus_Move._Joints = AimJoints;
    _vDeviceStatus_Move.push_back(_DeviceStatus_Move);

    _cassemble2Driver->MovePath(_vDeviceStatus_Move, _ErrorInfo_Move);

    std::cout << "[Move] ErrorInfo: " << _ErrorInfo._ErrIndex << "  " << _ErrorInfo._SubErrIndex  << std::endl;

    while (true) {
        if (armId == 0) {
            if (isIdle(LeftArm)) {
                std::cout << "[Move] Move LeftArm 移动完成." << std::endl;
                break;
            }
        } else if (armId == 1) {
            if (isIdle(RightArm)) {
                std::cout << "[Move] Move RightArm 移动完成." << std::endl;
                break;
            }
        }
        else if (armId == 2) {
            if (isIdle(LeftArm) && isIdle(RightArm)) {
                std::cout << "[Move] Move LeftArm and RightArm 移动完成." << std::endl;
                break;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(200)); /// 很关键
    }

}

void GraspController::MovePose(const std::vector<double>& targetPose, double vel, double acc, int armId){
    std::cout << "\n[MovePose] MovePose ..." << std::endl;

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
        if (targetPose.size() == 12) {
            for (int i = 0; i < 6; i++) {
                _DeviceStatus._Pos[i + 6] = targetPose[i + 6]; // 指定pose
            }
        } else if (targetPose.size() == 6) {
            for (int i = 0; i < 6; i++) {
                _DeviceStatus._Pos[i + 6] = targetPose[i]; // 指定pose
            }
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
//    std::this_thread::sleep_for(std::chrono::milliseconds(1000)); /// 很关键

    std::cout << "[MovePose] ErrorInfo: " << _ErrorInfo._ErrIndex << "  " << _ErrorInfo._SubErrIndex  << std::endl;

    while (true) {
        if (armId == 0) {
            if (isIdle(LeftArm)) {
                std::cout << "[MovePose] MovePose LeftArm 移动完成.\n" << std::endl;
                break;
            }
        } else if (armId == 1) {
            if (isIdle(RightArm)) {
                std::cout << "[MovePose] MovePose RightArm 移动完成.\n" << std::endl;
                break;
            }
        }
        else if (armId == 2) {
            if (isIdle(LeftArm) && isIdle(RightArm)) {
                std::cout << "[MovePose] MovePose LeftArm and RightArm 移动完成.\n" << std::endl;
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200)); /// 很关键
    }
}

void GraspController::MoveJoints(const std::vector<double>& targetJoints, double vel, double acc, int armId) {
    std::cout << "\n[INFO] MoveJoints ..." << std::endl;

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
//    std::this_thread::sleep_for(std::chrono::milliseconds(1000)); /// 很关键
//    sleep(1); /// 很关键

    std::cout << "[MoveJoints] ErrorInfo: " << _ErrorInfo._ErrIndex << "  " << _ErrorInfo._SubErrIndex  << std::endl;

    while (true) {
        if (armId == 0) {
            if (isIdle(LeftArm)) {
                std::cout << "[INFO] MoveJoints LeftArm 移动完成.\n" << std::endl;
                break;
            }
        } else if (armId == 1) {
            if (isIdle(RightArm)) {
                std::cout << "[INFO] MoveJoints RightArm 移动完成.\n" << std::endl;
                break;
            }
        }
        else if (armId == 2) {
            if (isIdle(LeftArm) && isIdle(RightArm)) {
                std::cout << "[INFO] MoveJoints LeftArm and RightArm 移动完成.\n" << std::endl;
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

    std::cout << "[MovePath] ErrorInfo: " << _ErrorInfo._ErrIndex << "  " << _ErrorInfo._SubErrIndex  << std::endl;

    while (true) {
        if (isIdle(armId == 0 ? LeftArm : RightArm)) {
            std::cout << "[INFO] MovePath 移动完成." << std::endl;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

void GraspController::MoveJoint6(double targetJoint6L, double targetJoint6R, int armId) {
    std::vector<double> targetJoints = getRobotJoints(2);

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

    // 起始位置
//    _DeviceStatus._Joints = {1.62, 0.360, -1.92, -0.64, 0.026, 0.00,
//                             -1.62, -0.360, 1.92, 0.64, -0.026, 0.00};

    _DeviceStatus._Joints = {D2R(91.238), D2R(64.396), D2R(-37.523), D2R(-15.631), D2R(-1.713), D2R(5.382),
                             D2R(-91.238), D2R(-64.396), D2R(37.523), D2R(15.631), D2R(1.713), D2R(-5.382)};

    _DeviceStatus._Vel = {Vel_Lv3, Vel_Lv3};
    _DeviceStatus._Acc = {Acc_Lv3, Acc_Lv3};
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
    HandOpen(0);
    HandOpen(1);
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
        sleep(1);
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
        sleep(1);
    }

    _cassemble2Driver->GraspB(_DeviceStatus, _ErrorInfo);
}

std::vector<double> GraspController::cart2Joints(const std::vector<double> &targetPose, int armId) {

    uint16_t _Dev = armId == 0 ? LeftArm : RightArm;

    // 先读取当前关节角
    std::vector<double> initJoints = getRobotJoints(2); // 获取当前左右臂关节角 size为12

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

std::vector<double> GraspController::getRobotJoints(int armID) { // FIXME
    ErrorInfo errorInfo;
    DeviceStatus deviceStatus;
    deviceStatus._ID = LeftArm | RightArm;
    _cassemble2Driver->GetRobotStatus(deviceStatus, errorInfo);

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    cout << "[INFO] Robot Joints: " << deviceStatus._Joints << endl;

    if (armID == 0) return std::vector<double> (deviceStatus._Joints.begin(), deviceStatus._Joints.begin() + 6);
    else if (armID == 1) return std::vector<double> (deviceStatus._Joints.begin() + 6, deviceStatus._Joints.begin() + 12);
    else if (armID == 2) return deviceStatus._Joints;
}

std::vector<double> GraspController::getRobotPose(int armID) {
    ErrorInfo errorInfo;
    DeviceStatus deviceStatus;
    deviceStatus._ID = LeftArm | RightArm;

    _cassemble2Driver->GetRobotStatus(deviceStatus, errorInfo);

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    cout << "[INFO] Robot Pose: " << deviceStatus._Tcp << endl;

    if (armID == 0) return std::vector<double> (deviceStatus._Tcp.begin(), deviceStatus._Tcp.begin() + 6);
    else if (armID == 1) return std::vector<double> (deviceStatus._Tcp.begin() + 6, deviceStatus._Tcp.begin() + 12);
    else if (armID == 2) return deviceStatus._Tcp;
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

void GraspController::saveCloudAndImages()
{
    std::string baseName, cloudName, colorName, resizeName, depthName;
    std::ostringstream oss;
    int frame = 0;

    while (true) {
        oss.str("");
        oss << std::setfill('0') << std::setw(2) << frame;
        baseName = oss.str();
        cloudName = "../../../grasp/data/images/" + baseName + "_cloud_" + getCurrentTimeStr() + ".pcd";
        colorName = "../../../grasp/data/images/" + baseName + "_color_" + getCurrentTimeStr() + ".jpg";
        resizeName = "../../../grasp/data/images/" + baseName + "_resize_" + getCurrentTimeStr() + ".jpg";
        depthName = "../../../grasp/data/images/" + baseName + "_depth_" + getCurrentTimeStr() + ".png";

        if ((access(colorName.c_str(), 0)) == 0) { // 0已存在,-1不存在
            frame++;
        }
        else {
            break;
        }
    }

    cv::Mat resize;
    cv::Point2f P[4];
    cv::resize(color, resize, cv::Size(960, 540));

    printf("%s\n", ("[INFO] Saving cloud: " + cloudName).c_str());
    writer.writeBinary(cloudName, *cloud);
    printf("%s\n", ("[INFO] Saving color: " + colorName).c_str());
    cv::imwrite(colorName, color);
    printf("%s\n", ("[INFO] Saving resize: " + resizeName).c_str());
    cv::imwrite(resizeName, resize);
    printf("%s\n", ("[INFO] Saving depth: " + depthName).c_str());
    cv::imwrite(depthName, depth);

    printf("[INFO] Saving complete!\n");
}