#include <iostream>
#include <cmath>
using namespace std;

#include <Eigen/Core>
#include <Eigen/Geometry>
#include "sophus/so3.hpp"
#include "sophus/se3.hpp"

int main(int argc, char** argv)
{
    // 设置输出精度，方便观察矩阵中的细微变化
    cout.precision(3);

    // ==========================================
    // 第一阶段：SO(3) 旋转与坐标变换回顾
    // ==========================================
    
    // 世界坐标系下的点 P_w = [1, 0, 0]^T
    Eigen::Vector3d p_w(1, 0, 0); 

    // 机器人相对于世界坐标系的旋转：绕 Z 轴转 45 度 (向左转)
    Eigen::AngleAxisd angle_axis(M_PI / 4, Eigen::Vector3d(0, 0, 1));
    Eigen::Matrix3d R_rw = angle_axis.toRotationMatrix();
    Eigen::Quaterniond Q_rw(angle_axis);
    Sophus::SO3d SO3_rw(R_rw);

    // 坐标变换公式：P_robot = R_rw^-1 * P_world
    // 物理直觉：机器人左转 45 度，点在机器人看来就像是向右转了 45 度 (Y 轴变为负)
    Eigen::Vector3d p_r_mat = R_rw.inverse() * p_w;
    cout << "Point in Robot frame (Matrix inverse):    " << p_r_mat.transpose() << endl;

    // ==========================================
    // 第二阶段：SE(3) 位姿合成与李代数
    // ==========================================

    // 1. 构造位姿：沿世界系 X 轴前进 2 米，并保持之前的旋转
    Eigen::Vector3d t(2, 0, 0);
    Sophus::SE3d T_wa(R_rw, t); // T_wa 表示从 A 系到世界系 W 的变换

    // 2. 提取李代数 se(3)
    // 数学上 se(3) 是 4x4 矩阵，但工程上用 6 维向量存储，通常记为 xi (ξ)
    // Sophus 顺序：前三维为平移相关(rho)，后三维为旋转轴角(phi)
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    Vector6d se3_T = T_wa.log(); 
    cout << "se(3) vector (xi): " << se3_T.transpose() << endl;

    // 3. 位姿合成：重复“走2米，转45度”的动作
    Sophus::SE3d T_ab(R_rw, t); 
    Sophus::SE3d T_wb = T_wa * T_ab; // 连续变换：W <- A <- B

    // 4. 验证 B 点在世界系下的坐标
    Eigen::Vector3d p_wb = T_wb * Eigen::Vector3d(0, 0, 0);
    cout << "Final position p_wb: " << p_wb.transpose() << endl;

    // ==========================================
    // 第三阶段：扰动模型 (左乘 vs 右乘)
    // ==========================================

    // 定义微小扰动向量 xi (10cm 平移, 0.01rad 旋转)
    Vector6d update_se3;
    update_se3.setZero();
    update_se3(0, 0) = 0.1;  // dx
    update_se3(5, 0) = 0.01; // d_theta (绕Z轴)

    // 左乘更新：在“世界坐标系”下观察到的误差
    // T_new = exp(hat(xi)) * T_old
    Sophus::SE3d T_wb_updated_left = Sophus::SE3d::exp(update_se3) * T_wb;

    // 右乘更新：在“机体(局部)坐标系”下观察到的误差 (传感器误差常用)
    // T_new = T_old * exp(hat(xi))
    Sophus::SE3d T_wb_updated_right = T_wb * Sophus::SE3d::exp(update_se3);

    cout << "Original T_wb:\n" << T_wb.matrix() << endl;
    cout << "Left updated T_wb (World-fixed):\n" << T_wb_updated_left.matrix() << endl;
    cout << "Right updated T_wb (Body-fixed):\n" << T_wb_updated_right.matrix() << endl;

    // ==========================================
    // 深度验证：为什么机体 Y 轴平移会改变世界系 X 轴？
    // ==========================================
    Vector6d update_se3_2;
    update_se3_2.setZero();
    update_se3_2(1, 0) = 0.05; // 仅在机体 Y 方向移动 5cm
    
    // 右乘更新：位移方向由当前 T_wb 的旋转 R 决定
    // 此时机器人朝向已经偏转，机体 Y 轴投影到了世界系的 X 轴上
    Sophus::SE3d T_wb_updated_right_2 = T_wb * Sophus::SE3d::exp(update_se3_2);
    cout << "Right updated (Body Y-shift 0.05m):\n" << T_wb_updated_right_2.matrix() << endl;

    return 0;
}