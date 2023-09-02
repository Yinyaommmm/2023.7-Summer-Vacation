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
        // 对于模糊反射光在内部的直接吸收掉
        return (dot(scattered.direction(), rec.normal));
    }
};

class dielectric : public material
{
public:
    double ir; // index of refraction;
    dielectric(double index_of_refraction) : ir(index_of_refraction) {}
    virtual bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered) const override
    {
        attenuation = color(1.0, 1.0, 1.0);
        // 默认从空气中射入介质中，如果从背面射出说明是从介质中射入空气中
        double refraction_ratio = rec.front_face ? (1.0 / ir) : ir;

        vec3 unit_direction = unit_vector(r_in.direction());
        double cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
        double sin_theta = sqrt(1 - cos_theta * cos_theta);

        // 考虑全反射发生的情况
        bool cannot_refract = refraction_ratio * sin_theta > 1.0;
        // 考虑反射比例
        double proportion_of_reflect = reflectance(cos_theta, refraction_ratio);
        // 通过proportion_of_reflect与随机值对比完成对部分光线的反射
        vec3 direction = cannot_refract || proportion_of_reflect > random_double()
                             ? reflect(unit_direction, rec.normal)
                             : refract(unit_direction, rec.normal, refraction_ratio);

        scattered = ray(rec.p, direction);
        return true;
    }

private:
    static double reflectance(double cosine, double ref_idx)
    {
        auto r0 = (1 - ref_idx) / (1 + ref_idx);
        r0 *= r0;
        return r0 + (1 - r0) * pow((1 - cosine), 5);
    }
};
#endif