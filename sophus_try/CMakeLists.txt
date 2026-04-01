cmake_minimum_required( VERSION 2.8 )
project( sophus_study )

# 设置 C++ 标准（Sophus 模板版需要 C++11 或更高）
set(CMAKE_CXX_STANDARD 11)

# 1. 寻找 Eigen3（这个你应该已经通过 apt 安装好了，可以用 find_package）
find_package( Eigen3 REQUIRED )

# 2. 手动指定 Sophus 和 Eigen 的头文件目录
# 我们不再使用 find_package(Sophus)，因为你手动拷贝了头文件
include_directories( 
    ${EIGEN3_INCLUDE_DIRS}
    "/usr/local/include"    # 这是你拷贝 sophus 文件夹到的地方
)

# 3. 生成可执行文件
# 注意：如果 useSophus.cpp 和 sophus_study.cpp 都有 main 函数，
# 你不能把它们放在同一个 add_executable 里，否则会报“main 函数重复定义”。
# 建议分开写：
#add_executable( useSophus useSophus.cpp )
add_executable( sophus_study sophus_study.cpp )

# 4. 链接库
# 模板版 Sophus 不需要链接库文件，但 Eigen 有时需要
#target_link_libraries( useSophus ${EIGEN3_LIBRARIES} )
target_link_libraries( sophus_study ${EIGEN3_LIBRARIES} )