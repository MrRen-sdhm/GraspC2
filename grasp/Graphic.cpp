//
// Created by sdhm on 19-5-26.
//
#include "include/GraphicsGrasp.h"

int main(int argc, char** argv)
{
    std::shared_ptr<GraphicsGrasp> _graphicsGrasp = std::make_shared<GraphicsGrasp>();

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBA>());
    cloud->height = 540;
    cloud->width = 960;
    cloud->is_dense = false;
    cloud->points.resize(cloud->height * cloud->width);

    _graphicsGrasp->createLookup(cloud->width*2, cloud->height*2); // 创建坐标映射

    cv::Mat color, depth;
    std::pair<std::vector<cv::RotatedRect>, std::vector<int>> RotRectsAndID;
    std::pair<std::vector<std::vector<double>>, std::vector<int>> PosesAndID;

    color = cv::imread("../../../grasp/data/images/rgb.jpg");
    depth = cv::imread("../../../grasp/data/images/depth.png", -1);

//    _graphicsGrasp->createPointCloud(color, depth, cloud); // 创建点云

    RotRectsAndID = _graphicsGrasp->detectGraspYolo(color); // Yolo积木检测

//    PosesAndID = _graphicsGrasp->getObjPosesAndID(RotRectsAndID, cloud, 0); // 获取积木中心点及ID


    return 0;
}