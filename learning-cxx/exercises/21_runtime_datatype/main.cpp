#include "../exercise.h"
#include <cmath>
#include <cstring>

// 这个练习是一个小型项目，目标是实现一个用于编译器的运行时类型系统。
// 运行时类型系统需要支持：
// 1. 基本数据类型（至少包括 int8, int32, float32, float64）
// 2. 类型大小查询
// 3. 类型名称查询
// 4. 类型相等判断

// 提示：可以使用前面练习中的知识：
// 1. enum class 表示类型枚举
// 2. union 节省内存空间
// 3. virtual 实现运行时多态
// 4. static 实现类型注册

enum class DataType {
    Int8,    // 8位整数
    Int32,   // 32位整数
    Float32, // 32位浮点数
    Float64  // 64位浮点数
};

/// @brief Tagged union 即标签化联合体，是联合体的一种常见应用。
///        Rust enum 在实现上就是标签化联合体。
struct TaggedUnion {
    DataType type;
    // NOTICE: struct/union 可以相互任意嵌套。
    union {
        float f;
        double d;
    };
};

// TODO: 将这个函数模板化用于 sigmoid_dyn
float sigmoid(float x) {
    return 1 / (1 + std::exp(-x));
}

TaggedUnion sigmoid_dyn(TaggedUnion x) {
    TaggedUnion ans{x.type};
    // TODO: 根据 type 调用 sigmoid
    return ans;
}

// 基类：类型描述器
class TypeDescriptor {
public:
    virtual ~TypeDescriptor() = default;
    virtual size_t size() const = 0;
    virtual const char* name() const = 0;
    virtual bool equals(const TypeDescriptor& other) const = 0;
};

// 模板类：具体类型的描述器
template<DataType type>
class ConcreteType : public TypeDescriptor {
public:
    size_t size() const override;
    const char* name() const override;
    bool equals(const TypeDescriptor& other) const override {
        // 使用dynamic_cast检查是否是相同类型
        return dynamic_cast<const ConcreteType<type>*>(&other) != nullptr;
    }
};

// 特化size()方法
template<> size_t ConcreteType<DataType::Int8>::size() const { return 1; }
template<> size_t ConcreteType<DataType::Int32>::size() const { return 4; }
template<> size_t ConcreteType<DataType::Float32>::size() const { return 4; }
template<> size_t ConcreteType<DataType::Float64>::size() const { return 8; }

// 特化name()方法
template<> const char* ConcreteType<DataType::Int8>::name() const { return "int8"; }
template<> const char* ConcreteType<DataType::Int32>::name() const { return "int32"; }
template<> const char* ConcreteType<DataType::Float32>::name() const { return "float32"; }
template<> const char* ConcreteType<DataType::Float64>::name() const { return "float64"; }

// 全局类型实例
static const ConcreteType<DataType::Int8> Int8Type;
static const ConcreteType<DataType::Int32> Int32Type;
static const ConcreteType<DataType::Float32> Float32Type;
static const ConcreteType<DataType::Float64> Float64Type;

int main(int argc, char **argv) {
    // 测试类型大小
    ASSERT(Int8Type.size() == 1, "Int8 size should be 1");
    ASSERT(Int32Type.size() == 4, "Int32 size should be 4");
    ASSERT(Float32Type.size() == 4, "Float32 size should be 4");
    ASSERT(Float64Type.size() == 8, "Float64 size should be 8");

    // 测试类型名称
    ASSERT(strcmp(Int8Type.name(), "int8") == 0, "Int8 name should be 'int8'");
    ASSERT(strcmp(Int32Type.name(), "int32") == 0, "Int32 name should be 'int32'");
    ASSERT(strcmp(Float32Type.name(), "float32") == 0, "Float32 name should be 'float32'");
    ASSERT(strcmp(Float64Type.name(), "float64") == 0, "Float64 name should be 'float64'");

    // 测试类型相等性
    ASSERT(Int8Type.equals(Int8Type), "Same type should be equal");
    ASSERT(!Int8Type.equals(Int32Type), "Different types should not be equal");
    ASSERT(!Float32Type.equals(Float64Type), "Different types should not be equal");

    return 0;
}
