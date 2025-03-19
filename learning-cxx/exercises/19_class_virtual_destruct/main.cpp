#include "../exercise.h"

// READ: 静态字段 <https://zh.cppreference.com/w/cpp/language/static>
// READ: 虚析构函数 <https://zh.cppreference.com/w/cpp/language/destructor>

struct A {
    // TODO: 正确初始化静态字段
    static int num_a;  // 静态成员变量声明

    A() {
        ++num_a;
    }
    virtual ~A() {  // 虚析构函数
        --num_a;
    }

    virtual char name() const {
        return 'A';
    }
};

// 静态成员变量定义
int A::num_a = 0;

struct B final : public A {
    // TODO: 正确初始化静态字段
    static int num_b;  // 静态成员变量声明

    B() {
        ++num_b;
    }
    ~B() override {  // 覆盖基类的虚析构函数
        --num_b;
    }

    char name() const final {
        return 'B';
    }
};

// 静态成员变量定义
int B::num_b = 0;

int main(int argc, char **argv) {
    auto a = new A;
    auto b = new B;
    ASSERT(A::num_a == 2, "Fill in the correct value for A::num_a");  // A和B各创建一个
    ASSERT(B::num_b == 1, "Fill in the correct value for B::num_b");  // 只创建了一个B
    ASSERT(a->name() == 'A', "Fill in the correct value for a->name()");
    ASSERT(b->name() == 'B', "Fill in the correct value for b->name()");

    delete a;
    delete b;
    ASSERT(A::num_a == 0, "Every A was destroyed");
    ASSERT(B::num_b == 0, "Every B was destroyed");

    A *ab = new B;  // 派生类指针可以转换为基类指针
    ASSERT(A::num_a == 1, "Fill in the correct value for A::num_a");  // 创建了一个B（也是一个A）
    ASSERT(B::num_b == 1, "Fill in the correct value for B::num_b");  // 创建了一个B
    ASSERT(ab->name() == 'B', "Fill in the correct value for ab->name()");  // 虚函数调用

    // TODO: 基类指针无法随意转换为派生类指针，补全正确的转换语句
    B &bb = dynamic_cast<B&>(*ab);  // 使用dynamic_cast进行安全的向下转换
    ASSERT(bb.name() == 'B', "Fill in the correct value for bb->name()");

    delete ab;  // 通过基类指针删除派生类对象，需要虚析构函数
    ASSERT(A::num_a == 0, "Every A was destroyed");
    ASSERT(B::num_b == 0, "Every B was destroyed");

    return 0;
}
