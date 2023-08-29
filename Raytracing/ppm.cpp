#include "color.h"
#include "ray.h"
#include "vec3.h"

#include <iostream>

// 计算是否碰撞球面
double hit_sphere(const point3& center, double radius, const ray& r) {
    vec3 oc = r.origin() - center;
    double a = dot(r.direction(), r.direction());
    double half_b = dot(r.direction(), oc);
    double c = dot(oc, oc) - radius * radius;
    double delta = half_b * half_b - a * c;
    // 返回t的值，-1说明没有交点，但未排除反向
    return delta < 0 ? -1 : (-half_b - sqrt(delta)) / a;
}
// 根据射线高度，线性混合白蓝
color ray_color(const ray& r) {
    static point3 center(0, 0, -1);
    static double radius = 0.5;
    double t = hit_sphere(center, radius, r);
    if (t > 0.0) {
        vec3 N = unit_vector(r.at(t) - center);
        // 将法向量映射到颜色
        return 0.5 * color(N.x() + 1, N.y() + 1, N.z() + 1);
    }

    // 下面在计算背景
    vec3 unit_direction = unit_vector(r.direction());
    t = 0.5 * (unit_direction.y() + 1.0);  // [-1，1] => [0,1];
    return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}

int main() {
    // Image
    const auto aspect_ratio = 16.0 / 9.0;
    const int image_width = 600;
    const int image_height = static_cast<int>(image_width / aspect_ratio);

    // camera
    auto viewport_height = 2.0;
    auto viewport_width = aspect_ratio * viewport_height;
    auto focal_length = 1.0;

    auto origin = point3(0, 0, 0);
    // 视口的两个边界
    auto horizontal = vec3(viewport_width, 0, 0);
    auto vertical = vec3(0, viewport_height, 0);
    // 视口左下角坐标
    auto lower_left_corner =
        origin - horizontal / 2 - vertical / 2 - vec3(0, 0, focal_length);

    // Render
    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";
    // 从左上遍历到右下
    for (int j = image_height - 1; j >= 0; --j) {
        std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
        for (int i = 0; i < image_width; ++i) {
            auto u = double(i) / (image_width - 1);   // 宽度占比
            auto v = double(j) / (image_height - 1);  // 高度占比
            // 两个占比决定在水平和竖直方向位移多少
            ray r(origin,
                  lower_left_corner + u * horizontal + v * vertical - origin);

            color pixel_color = ray_color(r);
            write_color(std::cout, pixel_color);
        }
    }
    std::cerr << "\nDone.\n";
}