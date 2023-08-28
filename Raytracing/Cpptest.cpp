#include <iostream>
using namespace std;
class A
{
public:
    void print()
    {
        cout << "(a, b) = (" << a << "," << b << ")" << endl;
    }
    A(){};
    A(int t, int y) : a(t), b(y){};
    int &operator[](size_t x)
    {
        return x == 0 ? a : b;
    }
    const int &operator[](size_t x) const
    {
        return x == 0 ? a : b;
    }

private:
    int a;
    int b;
};
int main()
{
    A a(5, 100);
    int x = a[0];
    x++;
    int &y = a[1];
    y++;
    a.print();

    cout << x << endl;
}