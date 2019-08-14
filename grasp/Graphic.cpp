//
// Created by sdhm on 19-5-26.
//
#include "include/GraphicsGrasp.h"

std::vector<int> findAimObjLR(std::pair<std::vector<cv::RotatedRect>, std::vector<int>> RotRectsAndID, int RowOrCol) { // 寻找左右侧的目标物体, 左侧找最左边的, 右侧找最右边的
    std::vector<int> LIndices, RIndices;
    std::vector<float> LCenters, RCenters;
    for (size_t i = 0; i < RotRectsAndID.first.size(); i++) {
        float center_x = RotRectsAndID.first[i].center.x;
        float center_y = RotRectsAndID.first[i].center.y;
        if (center_x < 470) { // 先区分左右
            if (RowOrCol == 1) { // 存储列值, 用于找横向最值
                printf("LeftCenter[%zu]: %f\n", i, center_x);
                LCenters.push_back(center_x);
            } else if (RowOrCol == 0) { // 存储行值, 用于找纵向最值
                printf("LeftCenter[%zu]: %f\n", i, center_y);
                LCenters.push_back(center_y);
            }
            LIndices.push_back(i);
        } else {
            if (RowOrCol == 1) { // 存储列值, 用于找横向最值
                printf("RightCenter[%zu]: %f\n", i, center_x);
                RCenters.push_back(center_x);
            } else if (RowOrCol == 0) { // 存储行值, 用于找纵向最值
                printf("RightCenter[%zu]: %f\n", i, center_y);
                RCenters.push_back(center_y);
            }
            RIndices.push_back(i);
        }
    }

    cout << "leftCnters: " << LCenters << endl;
    cout << "LIndices: " << LIndices << endl;
    cout << "rightCnters: " << RCenters << endl;
    cout << "RIndices: " << RIndices << endl;

    // 左侧找最小的
    auto min_LCenter = std::min_element(LCenters.begin(), LCenters.end());
    auto distanceL = std::distance(LCenters.begin(), min_LCenter);
    int positionRawL = LIndices[distanceL]; // 在RotRectsAndID中的位置
    std::cout << "LeftCenter Min element is " << *min_LCenter<< " at position " << positionRawL << std::endl;

    // 右侧找最大的
    auto max_RCenter = std::max_element(RCenters.begin(), RCenters.end());
    auto distanceR = std::distance(RCenters.begin(), max_RCenter);
    int positionRawR = RIndices[distanceR]; // 在RotRectsAndID中的位置
    std::cout << "RightCenter Max element is " << *max_RCenter<< " at position " << positionRawR << std::endl;

    std::vector<int> AimObjIndicesLR = {positionRawL, positionRawR};
    return AimObjIndicesLR;
}

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

    color = cv::imread("../../../grasp/data/images/cube1.jpg");
    depth = cv::imread("../../../grasp/data/images/cube1.png", -1);

//    color = cv::imread("/home/hustac/test.jpg");
//    depth = cv::imread("/home/hustac/test.png", -1);

    _graphicsGrasp->createPointCloud(color, depth, cloud); // 创建点云

    const int juggleOrCube = 1; /// 0为积木, 1为立方体

    if (juggleOrCube == 0) {
        /// Yolo积木检测
        RotRectsAndID = _graphicsGrasp->detectGraspYolo(color, 200, false);
    } else if (juggleOrCube == 1) {
        /// 正方体检测
        RotRectsAndID = _graphicsGrasp->detectBigCube(color, 200, true);
    }

    // 显示所有目标框
    for (size_t i = 0; i < RotRectsAndID.first.size(); i++) {
        int leftOrright = ((int) RotRectsAndID.first[i].center.x < 410) ? 0 : 1;

        Pose = _graphicsGrasp->getObjPose(RotRectsAndID.first[i], cloud, juggleOrCube, 0, leftOrright);

        printf("[INFO] 待抓取物体信息 ID:[%d] Angle:[%f] left/right[%d] Pose:[%f,%f,%f,%f,%f,%f]\n\n",
               RotRectsAndID.second[i],
               RotRectsAndID.first[i].angle, leftOrright, Pose[0], Pose[1],
               Pose[2], Pose[3], Pose[4], Pose[5]);

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

#if 0  /// 左右臂目标物体确定
    std::vector<int> AimObjIndicesLR = findAimObjLR(RotRectsAndID, 0);

    printf("[INFO] Distance between Obj in pix: %f\n",
           RotRectsAndID.first[AimObjIndicesLR[1]].center.y - RotRectsAndID.first[AimObjIndicesLR[0]].center.y);
    for (int indicesLr : AimObjIndicesLR) {
        int leftOrRight = ((int) RotRectsAndID.first[indicesLr].center.x < 410) ? 0 : 1;

        Pose = _graphicsGrasp->getObjPose(RotRectsAndID.first[indicesLr], cloud, juggleOrCube, 0, leftOrRight);

        printf("[INFO] 待抓取物体信息 ID:[%d] Angle:[%f] left/right[%d] Pose:[%f,%f,%f,%f,%f,%f]\n\n",
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
#endif

//    writer.writeBinary("/home/hustac/test.pcd", *cloud);
    return 0;
}