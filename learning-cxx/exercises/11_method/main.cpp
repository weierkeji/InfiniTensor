#include "../exercise.h"

struct Fibonacci {
    unsigned long long cache[128];
    int cached;

    // TODO: 实现正确的缓存优化斐波那契计算
    unsigned long long get(int i) {
        // 检查输入范围
        if (i < 0 || i >= 128) {
            throw std::out_of_range("fibonacci index out of range");
        }
        
        // 初始化前两个数
        if (cached < 2) {
            cache[0] = 0;
            cache[1] = 1;
            cached = 2;
        }
        
        // 计算到需要的位置
        for (; cached <= i; ++cached) {
            cache[cached] = cache[cached - 1] + cache[cached - 2];
        }
        
        return cache[i];
    }
};

int main(int argc, char **argv) {
    // TODO: 初始化缓存结构体，使计算正确
    Fibonacci fib = {
        {0, 1},  // 初始化数组的前两个元素
        2        // 已缓存的数量
    };
    
    ASSERT(fib.get(10) == 55, "fibonacci(10) should be 55");
    std::cout << "fibonacci(10) = " << fib.get(10) << std::endl;
    return 0;
}
