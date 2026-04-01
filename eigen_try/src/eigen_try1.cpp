#include <iostream>
#include <ctime>
#include <chrono>
#include <Eigen/Dense>
#include <Eigen/Core>
using namespace std;

int main(int argc, char **argv)
{
    //Eigen矩阵和向量基本操作
    Eigen::Matrix<double, 3, 3> R;
    R << 1, 2, 3, 4, 5, 6, 7, 8, 9;
    Eigen::Matrix<double, 3, 1> t;
    t << 4, 5, 6;
    cout<<t<<endl;
    Eigen::Matrix<double, 3, 1> result = R * t;  //赋值前必须初始化矩阵大小
    t = Eigen::Vector3d::Random();//vector3d是Matrix<double,3,1>的typedef,而matrix3d是Matrix<double,3,3>的typedef
    cout << t << endl;
    cout<<R.transpose()<<endl;                  //矩阵转置
    cout<<R.sum()<<endl;                        //矩阵元素之和
    cout<<R.trace()<<endl;                      //矩阵的迹
    cout<<R.inverse()<<endl;                    //矩阵求逆
    cout<<R.determinant()<<endl;                //矩阵的行列式
    //求特征值和特征向量
    // EigenSolver<Matrix3d> 是一个模板类，用于计算非对称矩阵的特征值和特征向量
    // EigenSolver 会返回复数特征值，因为非对称矩阵可能有复特征值
    // 构造函数 EigenSolver<Matrix3d> es(R) 自动计算特征值和特征向量
    // 其中 eigenvalues() 返回复数形式的特征值（即使虚部为0）
    // eigenvectors() 返回特征向量矩阵，列向量为对应的特征向量
    Eigen::EigenSolver<Eigen::Matrix3d> es(R);
    cout << "The eigenvalues of R are:\n" << es.eigenvalues().real() << endl;//求实部特征值
    cout << "The matrix of eigenvectors of R is:\n" << es.eigenvectors().real() << endl;//求实部特征向量
    
    //解线性方程组：R*x = b
    Eigen::Matrix<double, 3, 1> b;
    b << 3, 2, 1;
    
    cout << "\n========== 求解线性方程组 R*x = b ==========" << endl;
    cout << "Matrix R:\n" << R << endl;
    cout << "Vector b:\n" << b << endl;
    
    // ==================== 方法1: QR分解 ====================
    // 原理：
    // 1. 将矩阵R分解为 R = Q*R_upper，其中：
    //    - Q 是列正交矩阵（Q^T * Q = I）
    //    - R_upper 是上三角矩阵
    // 2. 原方程 R*x = b 变为：
    //    (Q*R_upper)*x = b
    //    Q*R_upper*x = b
    // 3. 两边左乘 Q^T：
    //    Q^T*Q*R_upper*x = Q^T*b
    //    R_upper*x = Q^T*b
    // 4. 由于R_upper是上三角矩阵，可以用回代法(back substitution)快速求解
    // 
    // colPivHouseholderQr() 是带列主元的Householder QR分解，优势：
    // - 对数值不稳定的矩阵（接近奇异）也能处理
    // - 通过列主元选择提高数值稳定性
    // - 自动排列列，使得最大元素在主对角线上
    //
    // 时间复杂度：O(m²n) 其中m=n=3，所以是 O(n³)
    // 但QR分解只需一次，之后可以多次快速求解
    
    // ========== 计时代码详解 ==========
    // chrono::high_resolution_clock 是C++标准库中精度最高的时钟（通常是纳秒级）
    // .now() 返回当前时刻的时间点，类型是 time_point
    // auto 会自动推导类型为 std::chrono::time_point<std::chrono::high_resolution_clock>
    // start1 记录开始时刻
    auto start1 = chrono::high_resolution_clock::now();
    
    // 循环100000次调用QR分解+求解，用来测量总耗时
    // 注意：每次循环都会重新进行QR分解，实际上R没有变化
    // 这样可以测量"总体"性能（包括分解+求解的开销）
    // 如果要单独测量求解速度，应该在循环外先做一次分解
    for(int i = 0; i < 100000; i++) {
        Eigen::Matrix<double, 3, 1> x1 = R.colPivHouseholderQr().solve(b); 
    }
    
    // .now() 再次被调用，记录结束时刻
    auto end1 = chrono::high_resolution_clock::now();
    
    // 计算时间差
    // end1 - start1 返回一个 duration 对象（时间段），类型是 std::chrono::nanoseconds（纳秒）
    // chrono::duration<double> 构造函数隐式转换纳秒为秒
    // 所以 elapsed1 是以秒为单位的 double 类型时间差
    // 相当于：(end1 - start1).count() / 1e9
    chrono::duration<double> elapsed1 = end1 - start1;
    
    Eigen::Vector3d x1 =R.colPivHouseholderQr().solve(b);
    cout << "\n【方法1】QR分解 (colPivHouseholderQr)" << endl;
    cout << "Solution x:\n" << x1 << endl;
    cout << "验证 R*x = b: \n" << R * x1 << endl;
    cout << "Time for 100000 iterations: " << elapsed1.count() << " seconds" << endl;
    
    // ==================== 方法2: 直接求逆 ====================
    // 原理：
    // 方程 R*x = b，两边左乘 R^(-1)：
    // R^(-1)*R*x = R^(-1)*b
    // I*x = R^(-1)*b
    // x = R^(-1)*b
    //
    // 缺点：
    // 1. 数值不稳定：R.inverse() 容易因为舍入误差产生大的误差
    // 2. 性能差：
    //    - 求逆需要完整的矩阵分解，时间复杂度是O(n³)
    //    - 之后还要乘以向量b，又是O(n²)
    //    - 总共O(n³)，比QR分解多了矩阵乘法的开销
    // 3. 当R接近奇异时（det接近0），R^(-1)的元素会很大，导致舍入误差放大
    // 4. 当R完全奇异时，无法计算逆矩阵，得到nan或inf
    //
    // 在这个例子中，R的行列式为0（奇异矩阵），所以R^(-1)不存在
    // 结果全是nan（Not a Number）
    
    auto start2 = chrono::high_resolution_clock::now();
    for(int i = 0; i < 100000; i++) {
        Eigen::Matrix<double, 3, 1> x2 = R.inverse() * b;
    }
    auto end2 = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed2 = end2 - start2;
    
    Eigen::Matrix<double, 3, 1> x2 = R.inverse() * b;
    cout << "\n【方法2】直接求逆 (R.inverse() * b)" << endl;
    cout << "R的行列式: " << R.determinant() << " (0表示奇异矩阵)" << endl;
    cout << "Solution x (错误):\n" << x2 << endl;
    cout << "Time for 100000 iterations: " << elapsed2.count() << " seconds" << endl;
    
    cout << "\n========== 总结 ==========" << endl;
    cout << "QR分解速度: " << elapsed1.count() << "s" << endl;
    cout << "直接求逆速度: " << elapsed2.count() << "s" << endl;
    cout << "虽然直接求逆看似快，但结果错误，且数值不稳定" << endl;
    cout << "建议:永远使用QR/LU分解来求解线性方程组,不要直接求逆!" << endl;
    return 0;
}