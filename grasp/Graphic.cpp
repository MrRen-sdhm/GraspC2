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

    color = cv::imread("../../../grasp/data/images/test7.jpg");
    depth = cv::imread("../../../grasp/data/images/test7.png", -1);

//    color = cv::imread("/home/hustac/test.jpg");
//    depth = cv::imread("/home/hustac/test.png", -1);

    _graphicsGrasp->createPointCloud(color, depth, cloud); // 创建点云

    RotRectsAndID = _graphicsGrasp->detectGraspYolo(color, 100, false); // Yolo积木检测
    RotRectsAndIDTop = _graphicsGrasp->detectGraspYolo(color, 200, false); // Yolo积木检测

    RotRectsAndID = _graphicsGrasp->detectBigCube(color, 200, true); // Yolo积木检测

    for (size_t i = 0; i < RotRectsAndID.first.size(); i++) {
        int leftOrright = ((int) RotRectsAndID.first[i].center.x < 410) ? 0 : 1;

        Pose = _graphicsGrasp->getObjPose(RotRectsAndID.first[i], cloud, 1, 0, leftOrright);

        printf("[INFO] 待抓取物体信息 ID:[%d] Angle:[%f] left/right[%d] Pose:[%f,%f,%f,%f,%f,%f]\n\n",
               RotRectsAndID.second[i],
               RotRectsAndID.first[i].angle, leftOrright, Pose[0], Pose[1],
               Pose[2], Pose[3], Pose[4], Pose[5]);

        /// 显示目标物体外界矩形
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

//    writer.writeBinary("/home/hustac/test.pcd", *cloud);
    return 0;
}