//
// Created by cobot on 18-5-23.
//

#include <vector>
#include <cobotsys.h>
#include <extra2.h>
#include <cobotsys_global_object_factory.h>
#include <cobotsys_file_finder.h>
#include <cobotsys_abstract_controller.h>
#include <QApplication>
#include <cobotsys_abstract_dual_arm_robot_link.h>

#include "GraspController.h"

#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"

#define PI2DE   M_PI/180

using namespace cobotsys;

int main(int argc, char** argv)
{
    int taskNum = 1;

    if (atoi(argv[1]) == 1) {
        taskNum = 1;
    } else if (atoi(argv[1]) == 2) {
        taskNum = 2;
    } else if (atoi(argv[1]) == 3) {
        taskNum = 3;
    } else if (atoi(argv[1]) == 4) {
        taskNum = 4;
    } else {
        printf("\033[0;33m%s\033[0m\n", "[WARN] Did not give property task num, defualt task 1 !");
    }

    QApplication a(argc, argv);
    cobotsys::init_library(argc, argv);
    cobotsys::FileFinder::addSearchPath("/opt/cobotsys/data");

    QString libPath = "/opt/cobotsys/plugins";
    GlobalObjectFactory globalObjectFactory;
    globalObjectFactory.loadLibrarys(libPath.toStdString());
    auto pFactory =GlobalObjectFactory::instance();

    if (pFactory) {

        std::shared_ptr<GraspController> graspController = std::make_shared<GraspController>(); // 抓取控制器

        std::string configFile = "../../../grasp/grasp_config.json";
        if(!graspController->setup(taskNum, configFile)) {
            //初始化失败直接退出
            std::cout << "初始化失败！" << std::endl;
            return 1;
        }

        ///////  相机测试
//        QString Factory = "CAssemblyC2DriverFactory, Ver 1.0";
//        QString Type = "CAssemblyC2Driver";
//        auto obj = globalObjectFactory.createObject(Factory, Type);
//        std::shared_ptr<DualArmRobotDriver> m_robot = std::dynamic_pointer_cast<DualArmRobotDriver>(obj);
//
//        ErrorInfo _Err;
//        DeviceStatus _mDeviceStatus;
//        if(m_robot) {
////            m_robot->Init(argc, argv, _Err, 1);//最后一个参数为0时,则为联机模式,为1时连接仿真平台进行仿真操作
//            m_robot->Start(_Err);
//
//            cobotsys::CameraFrame _frame;
//            while (true) {
//                std::cout << "CaptureImage" << std::endl;
//                _frame = m_robot->CaptureImage(_Err, HeadCamera, -100);
//                _frame = m_robot->GetImage(_Err, HeadCamera);
//            }
//        }

        return a.exec();
    }

    return 0;
}




