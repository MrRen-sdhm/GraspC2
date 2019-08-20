//
// Created by cobot on 18-5-23.
//

#include <vector>
#include <cobotsys.h>
#include <cobotsys_global_object_factory.h>
#include <cobotsys_file_finder.h>
#include <cobotsys_abstract_controller.h>
#include <QApplication>
#include <cobotsys_abstract_dual_arm_robot_link.h>

#include "GraspController.h"
#include "GraphicsGrasp.h"

#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"

using namespace cobotsys;


void image_process(std::shared_ptr<GraphicsGrasp> _graphicsGrasp, cv::Mat color, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud) {
    std::pair<std::vector<cv::RotatedRect>, std::vector<int>> RotRectsAndID, RotRectsAndIDTop;
    std::vector<double> Pose;

    const int juggleOrCube = 0; /// 0为积木, 1为立方体

    if (juggleOrCube == 0) {
        /// Yolo积木检测
        RotRectsAndID = _graphicsGrasp->detectGraspYolo(color, 200, 1);
    } else if (juggleOrCube == 1) {
        /// 正方体/球检测
        std::pair<cv::RotatedRect, int> BigBallRect, BigCubeRect;

//        if(_graphicsGrasp->detectBigBall(color, BigBallRect)) {
//            RotRectsAndID.first.push_back(BigBallRect.first);
//        }

        if(_graphicsGrasp->detectBigCube(color, cloud, BigCubeRect, true)) {
            RotRectsAndID.first.push_back(BigCubeRect.first);
        }
    }

#if 1  /// 左右臂目标物体确定
    std::vector<int> AimObjIndicesLR = _graphicsGrasp->findAimObjLR(RotRectsAndID, 0);

    if (AimObjIndicesLR[0] != -1 && AimObjIndicesLR[1] != -1) {
        printf("[INFO] Distance between Obj in pix: %f\n",
               RotRectsAndID.first[AimObjIndicesLR[1]].center.y - RotRectsAndID.first[AimObjIndicesLR[0]].center.y);
    }

    if (!AimObjIndicesLR.empty()) {
        for (int indicesLr : AimObjIndicesLR) {
            if (indicesLr == -1) continue; // 无效

            int leftOrRight = ((int) RotRectsAndID.first[indicesLr].center.x < 410) ? 0 : 1;

            int ret = _graphicsGrasp->getObjPose(RotRectsAndID.first[indicesLr], Pose, cloud, juggleOrCube, 0, leftOrRight);

            printf("[INFO] 待抓取物体信息 ID:[%d] Angle:[%f] left/right[%d] Pose:[%f,%f,%f,%f,%f,%f]\n",
                   RotRectsAndID.second[indicesLr],
                   RotRectsAndID.first[indicesLr].angle, leftOrRight, Pose[0], Pose[1],
                   Pose[2], Pose[3], Pose[4], Pose[5]);

            /// 显示目标物体外接矩形
            cv::Mat resized;
            cv::Point2f P[4];
            cv::resize(color, resized, cv::Size(960, 540));
            RotRectsAndID.first[indicesLr].points(P);
            for (int j = 0; j <= 3; j++) {
                line(resized, P[j], P[(j + 1) % 4], cv::Scalar(0, 255, 0), 2);
            }
            cv::circle(resized, RotRectsAndID.first[indicesLr].center, 1, cv::Scalar(0, 0, 255), 2);

            cv::imshow("roi_minAreaRect", resized);
            cv::waitKey(0);
        }
    }
#endif

#if 1 /// 显示所有目标框
    // 显示所有目标框
    for (size_t i = 0; i < RotRectsAndID.first.size(); i++) {
        int leftOrright = ((int) RotRectsAndID.first[i].center.x < 410) ? 0 : 1;

        _graphicsGrasp->getObjPose(RotRectsAndID.first[i], Pose, cloud, juggleOrCube, 0, leftOrright);

//        printf("[INFO] 待抓取物体信息 ID:[%d] Angle:[%f] left/right[%d] Pose:[%f,%f,%f,%f,%f,%f]\n\n",
//               RotRectsAndID.second[i],
//               RotRectsAndID.first[i].angle, leftOrright, Pose[0], Pose[1],
//               Pose[2], Pose[3], Pose[4], Pose[5]);

        /// 显示目标物体外接矩形
        cv::Mat resized;
        cv::Point2f P[4];
        cv::resize(color, resized, cv::Size(960, 540));
        RotRectsAndID.first[i].points(P);
        for (int j = 0; j <= 3; j++) {
            line(resized, P[j], P[(j + 1) % 4], cv::Scalar(0, 255, 0), 2);
        }
        cv::circle(resized, RotRectsAndID.first[i].center, 1, cv::Scalar(0, 0, 255), 2);

        cv::imshow("roi_minAreaRect", resized);
        cv::waitKey(0);
    }
#endif
}

int main(int argc, char** argv)
{
    QApplication a(argc, argv);
    cobotsys::init_library(argc, argv);
    cobotsys::FileFinder::addSearchPath("/opt/cobotsys/data");

    QString libPath = "/opt/cobotsys/plugins";
    GlobalObjectFactory globalObjectFactory;
    globalObjectFactory.loadLibrarys(libPath.toStdString());
    auto pFactory =GlobalObjectFactory::instance();

    if (pFactory) {

        QString Factory = "CAssemblyC2DriverFactory, Ver 1.0";
        QString Type = "CAssemblyC2Driver";
        auto obj = globalObjectFactory.createObject(Factory, Type);
        std::shared_ptr<DualArmRobotDriver> m_robot = std::dynamic_pointer_cast<DualArmRobotDriver>(obj);
        /*以上内容复制即可*/

        std::shared_ptr<GraphicsGrasp> _graphicsGrasp = std::make_shared<GraphicsGrasp>();
        pcl::PCDWriter writer;

        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBA>());
        cloud->height = 540;
        cloud->width = 960;
        cloud->is_dense = false;
        cloud->points.resize(cloud->height * cloud->width);

        _graphicsGrasp->createLookup(cloud->width*2, cloud->height*2); // 创建坐标映射

        cv::Mat color, depth;
        cobotsys::CameraFrame frame;
        
        ErrorInfo _Err;
        DeviceStatus _mDeviceStatus;
        if(m_robot) {
            m_robot->Start(_Err);

            frame = m_robot->CaptureImage(_Err, HeadCamera, -100);
            frame = m_robot->GetImage(_Err, HeadCamera);

            color = frame.frames.at(0).data.clone();
            depth = frame.frames.at(1).data.clone();

            _graphicsGrasp->createPointCloud(color, depth, cloud); // 创建点云

            _graphicsGrasp->showWorkArea(color); // 显示工作区域q

            image_process(_graphicsGrasp, color, cloud);

            cv::Mat resized;
            std::pair<std::vector<cv::RotatedRect>, std::vector<int>> RotatedRectsAndID;
            cv::resize(color, resized, cv::Size(960, 540)); // 缩小图片

            cv::imshow("color", resized);
            cv::waitKey(0);

            // 保存
            cv::imwrite("/home/hustac/test.jpg", color);
            cv::imwrite("/home/hustac/resize.jpg", resized);
            cv::imwrite("/home/hustac/test.png", depth);
            writer.writeBinary("/home/hustac/test.pcd", *cloud);

            return 0;
        }
    }
}