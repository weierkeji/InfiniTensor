#include "../exercise.h"
#include <vector>

// READ: std::vector <https://zh.cppreference.com/w/cpp/container/vector_bool>
// READ: 模板特化 <https://zh.cppreference.com/w/cpp/language/template_specialization>

// TODO: 将下列 `?` 替换为正确的代码
int main(int argc, char **argv) {
    std::vector<bool> vec(100, true);  // 创建100个元素，全部初始化为true
    ASSERT(vec[0], "Make this assertion pass.");
    ASSERT(vec[99], "Make this assertion pass.");
    ASSERT(vec.size() == 100, "Make this assertion pass.");
    // NOTICE: 平台相关！注意 CI:Ubuntu 上的值。
    std::cout << "sizeof(std::vector<bool>) = " << sizeof(std::vector<bool>) << std::endl;
    ASSERT(sizeof(vec) == 40, "Fill in the correct value.");  // vector<bool>在Ubuntu上是40字节
    {
        vec[20] = false;  // 设置某个位为false
        ASSERT(!vec[20], "Fill in `vec[20]` or `!vec[20]`.");
    }
    {
        vec.push_back(false);  // 添加一个false
        ASSERT(vec.size() == 101, "Fill in the correct value.");
        ASSERT(!vec[100], "Fill in `vec[100]` or `!vec[100]`.");
    }
    {
        auto ref = vec[30];  // 获取引用
        ASSERT(ref, "Fill in `ref` or `!ref`");  // 初始值为true
        ref = false;  // 通过引用修改值
        ASSERT(!ref, "Fill in `ref` or `!ref`");  // 修改后为false
        // THINK: WHAT and WHY?
        ASSERT(!vec[30], "Fill in `vec[30]` or `!vec[30]`.");  // 原始值也被修改
    }
    return 0;
}
