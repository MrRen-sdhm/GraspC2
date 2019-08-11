//
// Created by cobot on 19-3-30.
//

#ifndef COBOTSYS_ARMMOVETO_H
#define COBOTSYS_ARMMOVETO_H

#include <vector>
#include <thread>
#include <chrono>
#include <cmath>
#include "cobotsys_abstract_dual_arm_robot_link.h"

class ArmMoveTo {
public:
    ArmMoveTo(std::shared_ptr<cobotsys::DualArmRobotDriver> _pArm) {_Arm = _pArm;};
    ~ArmMoveTo() {};

    bool MoveTo(cobotsys::DeviceStatus _DeviceStatus, cobotsys::ErrorInfo _ErrorInfo);

private:

    std::vector<double> InterpCubic(double t, double T,
                                        std::vector<double> p0_pos,
                                        std::vector<double> p1_pos,
                                        std::vector<double> p0_vel,
                                        std::vector<double> p1_vel);

    bool EnableMotion = false;

    std::shared_ptr<cobotsys::DualArmRobotDriver> _Arm;
};


#endif //COBOTSYS_ARMMOVETO_H
