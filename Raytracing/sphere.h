#ifndef SPHERE_H
#define SPHERE_H
#include "hittable.h"
#include "vec3.h"
class sphere : public hittable
{
public:
    point3 center;
    double radius;
    shared_ptr<material> mat_ptr;

public:
    sphere() {}
    sphere(point3 cen, double r, shared_ptr<material> m)
        : center(cen), radius(r), mat_ptr(m) {}
    bool hit(const ray &r,
             double t_min,
             double t_max,
             hit_record &rec) const override;
};
bool sphere::hit(const ray &r,
                 double t_min,
                 double t_max,
                 hit_record &rec) const
{
    vec3 oc = r.origin() - center;
    double a = r.direction().length_squared(); // 大于零
    double half_b = dot(r.direction(), oc);
    double c = oc.length_squared() - radius * radius;
    double delta = half_b * half_b - a * c;
    if (delta < 0)
        return false;
    double sqrtd = sqrt(delta);
    auto root = (-half_b - sqrtd) / a;
    if (root < t_min || t_max < root)
    {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || root > t_max)
        {
            return false;
        }
    }
    // 记录碰撞
    rec.t = root;
    rec.p = r.at(rec.t);
    vec3 outward_normal = (rec.p - center) / radius; // 圆特有的归一化
    rec.set_face_normal(r, outward_normal);          // 设置法向量方向以及判断是否是正面
    rec.mat_ptr = mat_ptr;
    return true;
}
#endif