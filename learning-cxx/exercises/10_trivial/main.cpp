#include "../exercise.h"

// READ: Trivial type <https://learn.microsoft.com/zh-cn/cpp/cpp/trivial-standard-layout-and-pod-types?view=msvc-170>

struct FibonacciCache {
    unsigned long long cache[16];
    int cached;
};

// TODO: 实现正确的缓存优化斐波那契计算
static unsigned long long fibonacci(FibonacciCache &cache, int i) {
    // 检查输入范围
    if (i < 0 || i >= 16) {
        throw std::out_of_range("fibonacci index out of range");
    }
    
    // 初始化前两个数
    if (cache.cached < 2) {
        cache.cache[0] = 0;
        cache.cache[1] = 1;
        cache.cached = 2;
    }
    
    // 计算到需要的位置
    for (; cache.cached <= i; ++cache.cached) {
        cache.cache[cache.cached] = cache.cache[cache.cached - 1] + cache.cache[cache.cached - 2];
    }
    
    return cache.cache[i];
}

int main(int argc, char **argv) {
    // TODO: 初始化缓存结构体，使计算正确
    // NOTICE: C/C++ 中，读取未初始化的变量（包括结构体变量）是未定义行为
    // READ: 初始化的各种写法 <https://zh.cppreference.com/w/cpp/language/initialization>
    FibonacciCache fib = {
        {0, 1},  // 初始化数组的前两个元素
        2        // 已缓存的数量
    };
    
    ASSERT(fibonacci(fib, 10) == 55, "fibonacci(10) should be 55");
    std::cout << "fibonacci(10) = " << fibonacci(fib, 10) << std::endl;
    return 0;
}
