#ifndef CAMERA_H
#define CAMERA_H
#include "rtweek.h"
class camera {
   private:
    point3 origin;
    point3 lower_left_corner;
    // viewport的横纵轴
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    double len_radius;

   public:
    camera(point3 lookfrom,
           point3 lookat,
           vec3 vup,
           double vfov,
           double aspect_ratio,
           double aperture,
           double focus_dist) {
        double theta = degree_to_radians(vfov);
        auto h = tan(theta / 2);
        auto viewport_height = 2.0 * h;
        auto viewport_width = aspect_ratio * viewport_height;

        auto w = unit_vector(lookfrom - lookat);  // +z，注意是向外的
        auto u = unit_vector(cross(vup, w));      // x正半轴
        auto v = unit_vector(cross(w, u));        // y正半轴

        origin = lookfrom;

        horizontal = focus_dist * viewport_width * u;
        vertical = focus_dist * viewport_height * v;

        // 成像平面保持在-z的一个单位的位置
        lower_left_corner =
            origin - horizontal / 2 - vertical / 2 - w * focus_dist;

        len_radius = aperture / 2;
    }
    ray get_ray(double s, double t) const {
        vec3 rd = len_radius * random_in_unit_disk();
        vec3 offset = u * rd.x() + v * rd.y();
        vec3 newStart = origin + offset;
        return ray(newStart, lower_left_corner + s * horizontal + t * vertical -
                                 newStart);
    }
};

#endif