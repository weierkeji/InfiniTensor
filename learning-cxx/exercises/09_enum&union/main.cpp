#include "../exercise.h"

// READ: 枚举 <https://zh.cppreference.com/w/cpp/language/enum>
// READ: 联合体 <https://zh.cppreference.com/w/cpp/language/union>

// TODO: 补充枚举类型定义
enum class DataType {
    Int8,    // 8位整数
    Int32,   // 32位整数
    Float32, // 32位浮点数
    Float64  // 64位浮点数
};

// TODO: 补充联合体定义
union Data {
    int8_t i8;    // 8位整数
    int32_t i32;  // 32位整数
    float f32;    // 32位浮点数
    double f64;   // 64位浮点数
};

// TODO: 实现数据类型大小查询
size_t size_of(DataType type) {
    switch (type) {
        case DataType::Int8:    return sizeof(int8_t);
        case DataType::Int32:   return sizeof(int32_t);
        case DataType::Float32: return sizeof(float);
        case DataType::Float64: return sizeof(double);
        default:               return 0;
    }
}

int main(int argc, char **argv) {
    // 验证联合体大小
    ASSERT(sizeof(Data) == 8, "sizeof union is size of largest member");
    
    // 验证类型大小查询
    ASSERT(size_of(DataType::Int8) == 1, "sizeof int8_t is 1");
    ASSERT(size_of(DataType::Int32) == 4, "sizeof int32_t is 4");
    ASSERT(size_of(DataType::Float32) == 4, "sizeof float is 4");
    ASSERT(size_of(DataType::Float64) == 8, "sizeof double is 8");
    
    // 验证联合体成员共享内存
    Data data;
    data.f64 = 3.14159;
    std::cout << "double: " << data.f64 << std::endl;
    data.f32 = 2.71828f;
    std::cout << "float: " << data.f32 << std::endl;
    data.i32 = 42;
    std::cout << "int32: " << data.i32 << std::endl;
    data.i8 = 7;
    std::cout << "int8: " << static_cast<int>(data.i8) << std::endl;
    
    return 0;
}
