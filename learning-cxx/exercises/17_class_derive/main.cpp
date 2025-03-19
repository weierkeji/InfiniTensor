#include "../exercise.h"

// READ: 继承 <https://zh.cppreference.com/w/cpp/language/derived_class>

struct X {
    int x;
    X(int x) : x(x) {}
};

struct A : public X {
    int a;
    A(int a) : X(a), a(a) {}
};

struct B : public A {
    int b;
    B(int b) : A(b), b(b) {}
};

int main(int argc, char **argv) {
    X x = X(1);
    A a = A(2);
    B b = B(3);

    // TODO: 补全三个类型的大小
    ASSERT(sizeof(X) == 4, "Size of X should be 4 (one int)");
    ASSERT(sizeof(A) == 8, "Size of A should be 8 (two ints)");
    ASSERT(sizeof(B) == 12, "Size of B should be 12 (three ints)");

    // 打印继承关系
    std::cout << "Inheritance:" << std::endl;
    std::cout << "X: " << x.x << std::endl;
    std::cout << "A: " << a.x << ", " << a.a << std::endl;
    std::cout << "B: " << b.x << ", " << b.a << ", " << b.b << std::endl;

    // 打印对象内存布局
    int i = 0;
    std::cout << std::endl
              << "Memory layout:" << std::endl;
    std::cout << "X[" << i++ << "]: " << x.x << std::endl;
    std::cout << "A[" << i++ << "]: " << a.x << std::endl;
    std::cout << "A[" << i++ << "]: " << a.a << std::endl;
    std::cout << "B[" << i++ << "]: " << b.x << std::endl;
    std::cout << "B[" << i++ << "]: " << b.a << std::endl;
    std::cout << "B[" << i++ << "]: " << b.b << std::endl;

    i = 0;
    std::cout << std::endl
              << "-------------------------" << std::endl
              << std::endl;

    return 0;
}
