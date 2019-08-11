//
// Created by cobot on 19-3-30.
//

#include "ArmMoveTo.h"

bool ArmMoveTo::MoveTo(cobotsys::DeviceStatus _DeviceStatus, cobotsys::ErrorInfo _ErrorInfo)
{
    if (!_Arm) return false;
    if(_DeviceStatus._Pos.size() != 6) return false;
    EnableMotion = true;

    std::vector<double> curQ;
    curQ = _DeviceStatus._Joints;

    std::vector<double> curVel(6);
    std::vector<double> tarVel(6);
    for (int ii = 0; ii < 6; ++ii) {
        curVel[ii] = 0;
        tarVel[ii] = 0;

        curQ[ii] = curQ[ii+6];
    }
    curQ.resize(6);

    std::vector<double> TargetQ = _DeviceStatus._Pos;

    std::chrono::high_resolution_clock::time_point t0, t; double _dt = _DeviceStatus._TimeMS[0];

    cobotsys::ErrorInfo _eErrorInfo;
    cobotsys::DeviceStatus _mDeviceStatus;
    _mDeviceStatus._Vel = _DeviceStatus._Vel;
    _mDeviceStatus._Acc = _DeviceStatus._Acc;
    _mDeviceStatus._ID = _DeviceStatus._ID;

    t0 = std::chrono::high_resolution_clock::now();
    t = t0;
    while (_dt >= std::chrono::duration_cast < std::chrono::duration < double >> (t - t0).count()) {
        std::vector<double> _Joints = InterpCubic(std::chrono::duration_cast < std::chrono::duration < double >> (t - t0).count(),
                                     _dt, curQ, TargetQ, curVel, tarVel);
        _mDeviceStatus._Joints.resize(12);
        for(int ii=0;ii<6;ii++) {
            _mDeviceStatus._Joints[ii] = _Joints[ii];
            _mDeviceStatus._Joints[ii+6] = _Joints[ii];
        }

        if (EnableMotion) {
            _Arm->Move(_mDeviceStatus, _eErrorInfo); //此函数可进行仿真
        } else {
            return false;
        }

        // oversample with 2 ms
        std::this_thread::sleep_for( std::chrono::milliseconds(8));
        t = std::chrono::high_resolution_clock::now();
    }
    return true;
}

std::vector<double> ArmMoveTo::InterpCubic(double t, double T,
                                                           std::vector<double> p0_pos,
                                                           std::vector<double> p1_pos,
                                                           std::vector<double> p0_vel,
                                                           std::vector<double> p1_vel) {
    /*Returns positions of the joints at time 't' */
    std::vector<double> positions;
    for (unsigned int i = 0; i < p0_pos.size(); i++) {
        double a = p0_pos[i];
        double b = p0_vel[i];
        double c = (-3 * p0_pos[i] + 3 * p1_pos[i] - 2 * T * p0_vel[i]
                    - T * p1_vel[i]) / pow(T, 2);
        double d = (2 * p0_pos[i] - 2 * p1_pos[i] + T * p0_vel[i]
                    + T * p1_vel[i]) / pow(T, 3);
        positions.push_back(a + b * t + c * pow(t, 2) + d * pow(t, 3));
    }
    return positions;
}
