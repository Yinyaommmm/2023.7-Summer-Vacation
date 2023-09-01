#ifndef MATERIAL_H
#define MATERIAL_H
#include "rtweek.h"

struct hit_record;

class material
{
public:
    // 碰撞已经发生，material计算反射折射，也就是材质对光线的影响。
    // 参数：入射光，碰撞记录(内部的mat需要补充),attenution? , 反射光
    virtual bool scatter(const ray &r_in, const hit_record &rec, color &atenuation, ray &scattered) const = 0;
};

class lambertian : public material
{
public:
    color albedo; // 反射颜色？

public:
    lambertian(const color &a) : albedo(a){};
    virtual bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered) const override
    {
        auto scatter_direction = rec.normal + random_in_unit_sphere();
        if (scatter_direction.near_zero())
        {
            scatter_direction = rec.normal;
        }

        scattered = ray(rec.p, scatter_direction);
        attenuation = albedo;
        return true;
    }
};

class metal : public material
{
public:
    color albedo;
    double fuzz;

public:
    metal(const color &a, double f) : albedo(a), fuzz(f < 1 ? f : 1) {}
    virtual bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered) const override
    {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere());
        attenuation = albedo;
        // 对于反射光在内部的直接吸收掉
        return (dot(scattered.direction(), rec.normal));
    }
};
#endif