//
// Created by sdhm on 19-5-26.
//
#include "include/GraphicsGrasp.h"

template <typename T>
vector<size_t> sort_indexes_e(vector<T> &v)
{
    vector<size_t> idx(v.size());
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(),
    [&v](size_t i1, size_t i2) {return v[i1] < v[i2]; });
    return idx;
}


void image_process(const std::shared_ptr<GraphicsGrasp>& _graphicsGrasp, cv::Mat color, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud) {
    std::pair<std::vector<cv::RotatedRect>, std::vector<int>> RotRectsAndID, RotRectsAndIDTop;
    std::vector<double> PoseL, PoseR;

    const int juggleOrCube = 0; /// 0为积木, 1为大型物体

    if (juggleOrCube == 0) {
        /// Yolo积木检测
//        RotRectsAndID = _graphicsGrasp->detectGraspYolo(color, 200, 1); // 利用HSV阈值分割顶面
        RotRectsAndID = _graphicsGrasp->detectGraspYoloPro(color, cloud, 120, 1); // 利用深度分割获取顶面
//        RotRectsAndID = _graphicsGrasp->detectGraspYoloProT2(color, cloud, 120, 1); // 不检测三棱柱
    } else if (juggleOrCube == 1) {
        /// 大型物体检测
//        RotRectsAndID = _graphicsGrasp->detectBigObj(color, cloud, 2, 200, 0.7, 2); // 检测大正方体, 高阈值 NOTE:检测大球和大正方体使用的阈值不一样

        /// 正方体/球检测
        std::pair<cv::RotatedRect, int> BigBallRect, BigCubeRect;

//        if(_graphicsGrasp->detectBigBall(color, cloud, BigBallRect, 1)) {
//            RotRectsAndID.first.push_back(BigBallRect.first);
//            RotRectsAndID.second.push_back(BigBallRect.second);
//        }

//        if(_graphicsGrasp->detectBigCube(color, cloud, BigCubeRect, true)) {
//            RotRectsAndID.first.push_back(BigCubeRect.first);
//            RotRectsAndID.second.push_back(BigCubeRect.second);
//        }

# if 1 /// 大长方体检测
        RotRectsAndID = _graphicsGrasp->detectBigCubeTask3(color, cloud, 120, 1);

//        vector<int> vec = {5,31,9,11,8,21,9,7,4};
//        vector<size_t> idx;
//        idx = sort_indexes_e(vec);//注意vec中的内容不变，不是返回排序后的向量
//
//        //sort_indexes(idx,vec);//注意vec中的内容不变，不是返回排序后的向量
//
//        vector<int> vecs{};
//
//        for (int i = 0; i < vec.size(); i++)
//        {
//            vecs.push_back(vec[idx[i]]);
//        }
//
//        cout << vecs << endl;
//        cout << idx << endl;
//
//        exit(-10);

        /// 计算中心点及抓取点像素位置
        std::vector<std::pair<float, float>> PointListL, PointListR; // 存储所有点对
        std::vector<float> BigCubeAngle;
        std::vector<cv::Point2f> BigCubeCenter;
        std::vector<float> BigCubeCentery;

        for (size_t i = 0; i < RotRectsAndID.first.size(); i++) {
            std::pair<float, float> PointL, PointR;
            cv::Point2f P[4];
            RotRectsAndID.first[i].points(P);

            if (RotRectsAndID.first[i].size.width < RotRectsAndID.first[i].size.height) {
                // 计算width边上的中心点
                cv::Point2f Pwidth1;
                Pwidth1.x = P[0].x + (P[3].x - P[0].x) / 2;
                Pwidth1.y = P[0].y + (P[3].y - P[0].y) / 2;

                cv::Point2f Pwidth2;
                Pwidth2.x = P[2].x + (P[1].x - P[2].x) / 2;
                Pwidth2.y = P[2].y + (P[1].y - P[2].y) / 2;

                cv::Point2f Pwidth_1;
                Pwidth_1.x = Pwidth1.x + (Pwidth2.x - Pwidth1.x) / 6;
                Pwidth_1.y = Pwidth1.y + (Pwidth2.y - Pwidth1.y) / 6;

                cv::Point2f Pwidth_2;
                Pwidth_2.x = Pwidth1.x + (Pwidth2.x - Pwidth1.x) * 5 / 6;
                Pwidth_2.y = Pwidth1.y + (Pwidth2.y - Pwidth1.y) * 5 / 6;

                if (Pwidth_1.x < Pwidth_2.x) {
                    PointL.first = Pwidth_1.x;
                    PointL.second = Pwidth_1.y;
                    PointR.first = Pwidth_2.x;
                    PointR.second = Pwidth_2.y;
                } else {
                    PointL.first = Pwidth_2.x;
                    PointL.second = Pwidth_2.y;
                    PointR.first = Pwidth_1.x;
                    PointR.second = Pwidth_1.y;
                }

                BigCubeAngle.push_back(RotRectsAndID.first[i].angle); // 存储短边角度

            } else {
                // 计算height边上的中心点
                cv::Point2f Pheight1;
                Pheight1.x = P[0].x + (P[1].x - P[0].x) / 2;
                Pheight1.y = P[0].y + (P[1].y - P[0].y) / 2;

                cv::Point2f Pheight2;
                Pheight2.x = P[2].x + (P[3].x - P[2].x) / 2;
                Pheight2.y = P[2].y + (P[3].y - P[2].y) / 2;

                cv::Point2f Pheight_1;
                Pheight_1.x = Pheight1.x + (Pheight2.x - Pheight1.x) / 6;
                Pheight_1.y = Pheight1.y + (Pheight2.y - Pheight1.y) / 6;

                cv::Point2f Pheight_2;
                Pheight_2.x = Pheight1.x + (Pheight2.x - Pheight1.x) * 5 / 6;
                Pheight_2.y = Pheight1.y + (Pheight2.y - Pheight1.y) * 5 / 6;

                if (Pheight_1.x < Pheight_2.x) {
                    PointL.first = Pheight_1.x;
                    PointL.second = Pheight_1.y;
                    PointR.first = Pheight_2.x;
                    PointR.second = Pheight_2.y;
                } else {
                    PointL.first = Pheight_2.x;
                    PointL.second = Pheight_2.y;
                    PointR.first = Pheight_1.x;
                    PointR.second = Pheight_1.y;
                }

                BigCubeAngle.push_back(RotRectsAndID.first[i].angle - 90); // 存储短边角度
            }

            PointListL.push_back(PointL);
            PointListR.push_back(PointR);
            BigCubeCenter.push_back(RotRectsAndID.first[i].center);
            BigCubeCentery.push_back(RotRectsAndID.first[i].center.y);
        }

        /// 查找距离机器人最近的大木块, 即中心点行数最大
        auto max_Centery = std::max_element(BigCubeCentery.begin(), BigCubeCentery.end());
        auto distance = std::distance(BigCubeCentery.begin(), max_Centery);
//        positionRawL = LIndices[distanceL]; // 在RotRectsAndID中的位置
        std::cout << "[INFO] BigCubeCenter Max element is " << *max_Centery << " at position "
                  << distance << " in RotRectsAndID" << std::endl;


        // 从小到大排序
        vector<size_t> idx;
        idx = sort_indexes_e(BigCubeCentery);//注意vec中的内容不变，不是返回排序后的向量

        vector<float> vecs{};

        for (size_t i = 0; i < BigCubeCentery.size(); i++)
        {
            vecs.push_back(BigCubeCentery[idx[i]]);
        }

        cout << BigCubeCentery << endl;
        cout << vecs << endl;
        cout << idx << endl;

        std::vector<int> pickIndices; // 待抓取木块顺序列表索引, 先y大的后y小的
        for (size_t i = 0; i < BigCubeCentery.size()/2; i++) {
            pickIndices.push_back(idx[idx.size()-1]);
            pickIndices.push_back(idx[i]);
        }
        cout << "pickIndices: " << pickIndices << endl;

        exit(-10);
#endif

#if 0 /// 小立方体检测
        RotRectsAndID = _graphicsGrasp->detectSmallCubeTask2(color, cloud, 120, 1);
#endif
    }

#if 0  /// 左右臂目标物体确定
    std::vector<int> AimObjIndicesLR = _graphicsGrasp->findAimObjLR(RotRectsAndID, 0);

    if (AimObjIndicesLR[0] != -1 && AimObjIndicesLR[1] != -1) {
        printf("[INFO] Distance between Obj in pix: %f\n",
               RotRectsAndID.first[AimObjIndicesLR[1]].center.y - RotRectsAndID.first[AimObjIndicesLR[0]].center.y);
    }

    if (!AimObjIndicesLR.empty()) {
        for (int indicesLr : AimObjIndicesLR) {
            if (indicesLr == -1) continue; // 无效

            int leftOrRight = ((int) RotRectsAndID.first[indicesLr].center.x < 410) ? 0 : 1;

            if(_graphicsGrasp->getObjPose(RotRectsAndID.first[indicesLr], Pose, cloud, juggleOrCube, 0, leftOrRight)) {

                printf("[INFO] 待抓取物体信息 ID:[%d] Angle:[%f] left/right[%d] Pose:[%f,%f,%f,%f,%f,%f]\n",
                       RotRectsAndID.second[indicesLr],
                       RotRectsAndID.first[indicesLr].angle, leftOrRight, Pose[0], Pose[1],
                       Pose[2], Pose[3], Pose[4], Pose[5]);
            }

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
//            cv::waitKey(0);
        }
    }
#endif

#if 1 /// 显示所有目标框
    // 显示所有目标框
    cv::Mat resized;
    cv::resize(color, resized, cv::Size(960, 540));

    for (size_t i = 0; i < RotRectsAndID.first.size(); i++) {

        _graphicsGrasp->getObjPose(RotRectsAndID.first[i], PoseL, cloud, juggleOrCube, 0, 0, 3.0);

        if (PoseL[0] > _graphicsGrasp->lieThreshL) { // 躺着的
            printf("[INFO] 左臂待抓取物体为躺着的\n");
        } else { // 立着的
            printf("[INFO] 左臂待抓取物体为立着的");
        }

        _graphicsGrasp->getObjPose(RotRectsAndID.first[i], PoseR, cloud, juggleOrCube, 0, 1, 3.0);

        if (PoseR[0] < _graphicsGrasp->lieThreshR) { // 躺着的
            printf("[INFO] 右臂待抓取物体为躺着的");
        } else { // 立着的
            printf("[INFO] 右臂待抓取物体为立着的");
        }

        /// 显示目标物体外接矩形
        cv::Point2f P[4];
        RotRectsAndID.first[i].points(P);
        for (int j = 0; j <= 3; j++) {
            line(resized, P[j], P[(j + 1) % 4], cv::Scalar(0, 255, 0), 2);
        }
        cv::circle(resized, RotRectsAndID.first[i].center, 1, cv::Scalar(0, 0, 255), 2);
//        cv::circle(resized, RotRectsAndID.first[i].center, 30, cv::Scalar(0, 0, 255), 2);

        cout << "[INFO] RotRects ID: " << RotRectsAndID.second[i] << endl;

        cv::imshow("RotRectsAndID Result ", resized);
        cv::waitKey(0);
    }

    cv::imwrite("/home/hustac/result.png", resized);
//    cv::waitKey(0);

    /// 计算大球和大立方体的高度
    std::vector<float> coorRaw;
    std::vector<double> coorReal;

    // 球体 0.441943
    coorRaw = {-0.114937, 0.009031, 0.690000};
    coorReal = _graphicsGrasp->calcRealCoor(coorRaw, 0);
    cout << "coorRealBall: " << coorReal << endl;

    // 立方体 0.475721
    coorRaw = {0.019815, 0.046960, 0.634000};
    coorReal = _graphicsGrasp->calcRealCoor(coorRaw, 0);
    cout << "coorRealCube: " << coorReal << endl;

    // 桌面 0.561311
    coorRaw = {0.167363, 0.191053, 0.671000};
    coorReal = _graphicsGrasp->calcRealCoor(coorRaw, 0);
    cout << "coorRealTable: " << coorReal << endl;

#endif
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

    // 球
//    color = cv::imread("../../../grasp/data/images/old/27_color_0820.jpg");
//    depth = cv::imread("../../../grasp/data/images/old/27_depth_0820.png", -1);
//    color = cv::imread("../../../grasp/data/images/47_color_0821.jpg");
//    depth = cv::imread("../../../grasp/data/images/47_depth_0821.png", -1);

    // 小立方体
//    color = cv::imread("../../../grasp/data/images/49_color_0821.jpg");
//    depth = cv::imread("../../../grasp/data/images/49_depth_0821.png", -1);
    // 堆叠
//    color = cv::imread("../../../grasp/data/images/50_color_0821.jpg");
//    depth = cv::imread("../../../grasp/data/images/50_depth_0821.png", -1);

    // 大立方体
//    color = cv::imread("../../../grasp/data/images/13_color_0822.jpg");
//    depth = cv::imread("../../../grasp/data/images/13_depth_0822.png", -1);

    // 大长方体
//    color = cv::imread("../../../grasp/data/images/old/20_color_0818.jpg");
//    depth = cv::imread("../../../grasp/data/images/old/20_depth_0818.png", -1);

//    color = cv::imread("../../../grasp/data/images/35_color_0822.jpg");
//    depth = cv::imread("../../../grasp/data/images/35_depth_0822.png", -1);

    // 积木
//    color = cv::imread("../../../grasp/data/images/old/04_color_0817.jpg");
//    depth = cv::imread("../../../grasp/data/images/old/04_depth_0817.png", -1);

    // 立着
    color = cv::imread("../../../grasp/data/images/27_color_0821.jpg");
    depth = cv::imread("../../../grasp/data/images/27_depth_0821.png", -1);

    // 躺着
//    color = cv::imread("../../../grasp/data/images/37_color_0821.jpg");
//    depth = cv::imread("../../../grasp/data/images/37_depth_0821.png", -1);

//    color = cv::imread("../../../grasp/data/images/34_color_0821.jpg");
//    depth = cv::imread("../../../grasp/data/images/34_depth_0821.png", -1);


//    color = cv::imread("/home/hustac/test1.jpg");
//    depth = cv::imread("/home/hustac/test1.png", -1);

//    color = cv::imread("/home/hustac/图片/现场采集/test1.jpg");
//    depth = cv::imread("/home/hustac/图片/现场采集/test1.png", -1);

    _graphicsGrasp->showWorkArea(color); // 显示工作区域

    _graphicsGrasp->createPointCloud(color, depth, cloud); // 创建点云

    image_process(_graphicsGrasp, color, cloud); // 图像处理

    writer.writeBinary("/home/hustac/test.pcd", *cloud);

    return 0;
}