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
#include "ArmMoveTo.h"

#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"

#define PI2DE   M_PI/180

using namespace cobotsys;

std::shared_ptr<DualArmRobotDriver> m_robot;

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

        ErrorInfo _Err;
        DeviceStatus _mDeviceStatus;
        if(m_robot) {
//            m_robot->Init(argc, argv, _Err, 1);//最后一个参数为0时,则为联机模式,为1时连接仿真平台进行仿真操作
            m_robot->Start(_Err);

            cobotsys::CameraFrame _frame;
            while (true) {
                std::cout << "CaptureImage" << std::endl;
                _frame = m_robot->CaptureImage(_Err, HeadCamera, -100);
                _frame = m_robot->GetImage(_Err, HeadCamera);
            }
        }

        return a.exec();
    }

    return 0;
}




