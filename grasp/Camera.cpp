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

#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"

using namespace cobotsys;

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
        
        cv::Mat color, depth;
        cobotsys::CameraFrame frame;
        
        ErrorInfo _Err;
        DeviceStatus _mDeviceStatus;
        if(m_robot) {
            m_robot->Start(_Err);

            while (true) {
                frame = m_robot->CaptureImage(_Err, HeadCamera, -100);
                frame = m_robot->GetImage(_Err, HeadCamera);

                color = frame.frames.at(0).data.clone();
                depth = frame.frames.at(1).data.clone();
                
                cv::imshow("color", color);
                cv::imshow("depth", depth);
                
                cv::waitKey(0);

                // 保存
                cv::imwrite("/home/hustac/rgb.jpg", color);
                cv::imwrite("/home/hustac/depth.png", depth);
            }
        }

        return a.exec();
    }

    return 0;
}