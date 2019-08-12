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
    std::pair<std::vector<cv::RotatedRect>, std::vector<int>> RotRectsAndID;
    std::vector<double> Pose;

    color = cv::imread("../../../grasp/data/images/lizhe.jpg");
    depth = cv::imread("../../../grasp/data/images/lizhe.png", -1);

//    color = cv::imread("/home/hustac/test.jpg");
//    depth = cv::imread("/home/hustac/test.png", -1);

    _graphicsGrasp->createPointCloud(color, depth, cloud); // 创建点云

    RotRectsAndID = _graphicsGrasp->detectGraspYolo(color, true); // Yolo积木检测

    int leftOrright = ((int)RotRectsAndID.first[0].center.x < 410) ? 0 : 1;

    Pose = _graphicsGrasp->getObjPose(RotRectsAndID.first[0], cloud, leftOrright);

    printf("[INFO] 待抓取物体信息 ID:[%d] Angle:[%f] left/right[%d] Pose:[%f,%f,%f,%f,%f,%f]\n\n",
           RotRectsAndID.second[0],
           RotRectsAndID.first[0].angle, leftOrright, Pose[0], Pose[1],
           Pose[2], Pose[3], Pose[4], Pose[5]);

//    writer.writeBinary("/home/hustac/test.pcd", *cloud);
    return 0;
}