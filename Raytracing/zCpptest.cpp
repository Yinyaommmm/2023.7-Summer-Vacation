#include <iostream>
#include <cstdlib>
#include "rtweek.h"
using namespace std;
class A
{
public:
    A() // 构造函数
    {
        cout << "construct A" << endl;
    }

    ~A() // 析构函数
    {
        cout << "Destructor A" << endl;
    }
};

// 派生类
class B : public A
{
public:
    B() // 构造函数
    {
        cout << "construct B" << endl;
    }

    ~B() // 析构函数
    {
        cout << "Destructor B" << endl;
    }
};
class X
{
public:
    void a()
    {
    }
};
int main()
{
    cout << random_in_unit_sphere().length_squared();
    return 0;
}
