#include<iostream>
#include<Eigen/Geometry>
using namespace std;
int main(int argc, char **argv)
{
    Eigen::Matrix3d rotation_matrix=Eigen::Matrix3d::Identity();//3x3单位矩阵,无旋转
    Eigen::AngleAxisd rotation_vector(M_PI/4,Eigen::Vector3d(0,0,1));//绕Z轴旋转45度,角轴
    cout.precision(3);//设置输出精度
    cout<<"rotation matrix = \n"<<rotation_vector.matrix()<<"\n";//将角轴转换为旋转矩阵
    rotation_matrix=rotation_vector.toRotationMatrix();//也可以直接转换为旋转矩阵

    //旋转向量(angle axis/rotation matrix)
    Eigen::Vector3d v(1,0,0);//定义一个初始向量
    Eigen::Vector3d v_rotated=rotation_matrix*v;//旋转后的向量(rotation matrix)
    cout<<"(1,0,0) after rotation_matrix = "<<v_rotated.transpose()<<"\n";
    v_rotated=rotation_vector*v;//旋转后的向量(angle axis)
    cout<<"(1,0,0) after rotation_vector = "<<v_rotated.transpose()<<"\n";

    //欧拉角
    Eigen::Vector3d euler_angles=rotation_matrix.eulerAngles(2,1,0);//将rotation_matrix转为欧拉角形式,按照 ZYX顺序，即yaw pitch roll
    cout<<"yaw pitch roll = "<<euler_angles.transpose()<<"\n";//yaw=Z轴旋转角=pai/4，pitch=Y轴旋转角，roll=X轴旋转角

    //欧式变换矩阵 Eigen::Isometry3d
    Eigen::Vector3d t(1,3,4);//自定义一个平移向量
    Eigen::Isometry3d T=Eigen::Isometry3d::Identity();//虽然称为3d,但是实际是4x4单位矩阵
    T.rotate(rotation_vector);//将T进行旋转
    T.pretranslate(t);//将T进行平移
    cout<<"Transform matrix = \n"<<T.matrix()<<"\n"; //输出变换矩阵

    //用变换矩阵进行坐标变换
    Eigen::Vector3d v_transformed=T*v;//使用欧式变换矩阵进行坐标变换
    cout << "(1,0,0) after transform = " << v_transformed.transpose() << endl;

    //四元数
    Eigen::Quaterniond q=Eigen::Quaterniond(rotation_vector);//直接将角轴转换为四元数
    cout<<"quaternion from angle axis = \n"<<q.coeffs()<<"\n";//coeffs的顺序是(x,y,z,w),w为实部,xyz为虚部
    q=Eigen::Quaterniond(rotation_matrix);//也可以直接将旋转矩阵转换为四元数
    cout<<"quaternion from rotation matrix = \n"<<q.coeffs()<<"\n";
    v_rotated=q*v;//用四元数旋转一个向量
    cout << "(1,0,0) after rotation(quaternion) = " << v_rotated.transpose() << endl;

    return 0;
}