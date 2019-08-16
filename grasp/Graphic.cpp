//
// Created by sdhm on 19-5-26.
//
#include "include/GraphicsGrasp.h"

int main(int argc, char** argv)
{
    std::shared_ptr<GraphicsGrasp> _graphicsGrasp = std::make_shared<GraphicsGrasp>();
    pcl::PCDWriter writer;

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBA>());
    cloud->height = 540;
    cloud->width = 960;
    cloud->is_dense = false;
    cloud->points.resize(cloud->height * cloud->width);

    _graphicsGrasp->createLookup(cloud->width*2, cloud->height*2); // 创建坐标映射

    cv::Mat color, depth;
    std::pair<std::vector<cv::RotatedRect>, std::vector<int>> RotRectsAndID, RotRectsAndIDTop;
    std::vector<double> Pose;

//    color = cv::imread("../../../grasp/data/images/ball5.jpg");
//    depth = cv::imread("../../../grasp/data/images/cube1.png", -1);

    color = cv::imread("/home/hustac/test.jpg");
    depth = cv::imread("/home/hustac/test.png", -1);

    _graphicsGrasp->showWorkArea(color); // 显示工作区域

    _graphicsGrasp->createPointCloud(color, depth, cloud); // 创建点云

    const int juggleOrCube = 0; /// 0为积木, 1为立方体

    if (juggleOrCube == 0) {
        /// Yolo积木检测
        RotRectsAndID = _graphicsGrasp->detectGraspYolo(color, 200, 1);
    } else if (juggleOrCube == 1) {
        /// 正方体检测
//        RotRectsAndID = _graphicsGrasp->detectBigObj(color, 200, true); // 检测立方体, 高阈值

//        RotRectsAndID = _graphicsGrasp->detectBigObj(color, 100, true); // 检测大球, 低阈值

        cv::RotatedRect BigBallRect, BigCubeRect;

        if(_graphicsGrasp->detectBigBall(color, BigBallRect)) {
            RotRectsAndID.first.push_back(BigBallRect);
        }

//        if(_graphicsGrasp->detectBigCube(color, BigCubeRect)) {
//            RotRectsAndID.first.push_back(BigCubeRect);
//        }
    }

#if 1  /// 左右臂目标物体确定
    std::vector<int> AimObjIndicesLR = _graphicsGrasp->findAimObjLR(RotRectsAndID, 0);

    printf("[INFO] Distance between Obj in pix: %f\n",
           RotRectsAndID.first[AimObjIndicesLR[1]].center.y - RotRectsAndID.first[AimObjIndicesLR[0]].center.y);

    if (!AimObjIndicesLR.empty()) {
        for (int indicesLr : AimObjIndicesLR) {
            int leftOrRight = ((int) RotRectsAndID.first[indicesLr].center.x < 410) ? 0 : 1;

            Pose = _graphicsGrasp->getObjPose(RotRectsAndID.first[indicesLr], cloud, juggleOrCube, 0, leftOrRight);

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

#if 0 /// 显示所有目标框
    // 显示所有目标框
    for (size_t i = 0; i < RotRectsAndID.first.size(); i++) {
        int leftOrright = ((int) RotRectsAndID.first[i].center.x < 410) ? 0 : 1;

        Pose = _graphicsGrasp->getObjPose(RotRectsAndID.first[i], cloud, juggleOrCube, 0, leftOrright);

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

    writer.writeBinary("/home/hustac/test.pcd", *cloud);
    return 0;
}