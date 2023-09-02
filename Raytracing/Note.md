#### vec3.h

 实现关于数字与向量的+-*/ 重载

 cout 向量

 点乘 叉乘 单位向量 长度

定义了point3和color类型

#### ray.h

定义了射线

#### 设计viewport视窗、视口

通过viewport发出射线，确定交点、物体，计算颜色。

![image-20230828204113181](C:\Users\Dell\AppData\Roaming\Typora\typora-user-images\image-20230828204113181.png)

照相机位于原点，viewport位于距离为1的地方，

铺设背景，实际上是把背景图片放到了viewport之中，但是像素数目保持不变，因此仍然可以做微小的变动

#### 如何visualize法向量

将法向量单位化，(x,y,z)三个值必然都属于[-1,1]，因此可以将其映射到[0,1]

获取法向量，势必要计算交点，因此相比于unit 5中的根据Δ判断有无解，我们需要进一步计算t的值，并且只取小一点的t，毕竟我们只能看见近的点。

【注意，这里保证了球在摄像机前方，这样的话，我们的t才有意义，毕竟在摄像机后方的东西哪怕t有解也是不应该看到的】

#### 对碰撞的抽象

虚基类hittable，子类对应某个具体的物体，例如球。

子类需要复写hit函数，hit函数如果碰撞成功将返回碰撞记录record。

hit函数增加了t_min t_max对t作限制。

#### 判断射线相交的是外面还是里面

由于目前法向量始终是center -> intersects，因此当ray来自外面，两者相向，点积为负；ray来自内部，两者同向，点积为正。

#### 与一组物体碰撞：多态

射线与物体相交的函数

`color ray_color(const ray &r, const hittable &world)`

由于hittable_list继承自hittable，我们可以将其传进去，这会触发多态。因而ray_color内部的hit调用的是hittable_list的hit函数，会返回最近的碰撞记录。

为了让ray_color与具体的碰撞物分开，ray_color的参数时hittable&，而非sphere或者hittable_list.

#### 生成随机数random库

```c++
inline double random_double()
{
    // 均匀分布器
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    // 随机数产生器
    static std::mt19937 generator;
    // distribution重载了()运算符，返回一个double
    return distribution(generator);
}
```



#### 抗锯齿

传统做法：混合像素周围颜色。按理来说是要进行分层的，同层内混色，这里没有做。

此处做法：每个像素加入一定的随机值，重复一定次数取样，随后平均

抗锯齿前：

![image-20230830185736219](C:\Users\卫清渠\AppData\Roaming\Typora\typora-user-images\image-20230830185736219.png)

抗锯齿后：

![image-20230830225747813](C:\Users\卫清渠\AppData\Roaming\Typora\typora-user-images\image-20230830225747813.png)

#### 散射材质Diffuse Materials

问题：散射材质和几何体是否要绑定在一起？这里采用的是分离。

不发光的散射物体的颜色是周围的物体，但也会结合自身颜色进行调制。

反射光在散射材质上会有随机方向，当然光也有可能被吸收，例如黑体。

**反射光随机方向会让物体看起来无金属光泽**

#### 制作随机反射光线

选择以P+N为圆心的单位圆内任意一点S，S-P就是随机的反射方向。

<img src="C:\Users\卫清渠\AppData\Roaming\Typora\typora-user-images\image-20230831000232103.png" alt="image-20230831000232103" style="zoom:50%;" />

方法是rejection，生成随机向量直至长度小于1。

#### 反射光线的颜色

没有反射时，颜色直接是射线触碰到的像素的法线颜色。

现在考虑了反射，颜色是reflector * income ray颜色。

income ray如果来自于天空，那就是天空颜色。如果来自于物体，那又要递归计算.

这样来看color = reflector^n * 天空，或者超过最大递归次数时，直接设为0。

![反射系数0.99](C:\Users\卫清渠\AppData\Roaming\Typora\typora-user-images\image-20230831133800054.png)

#### ★为什么图片这么黑

上面的图递归了20次，下面的图递归50次，也是这么黑。

![image-20230831151150803](C:\Users\卫清渠\AppData\Roaming\Typora\typora-user-images\image-20230831151150803.png)

原因如下：

```c++
// 法向量单位球内一点
point3 target = rec.p + rec.normal + random_in_unit_sphere();
// 0.5残余能量比例 * 反射光颜色
return 0.5 * ray_color(ray(rec.p, target - rec.p), world, depth - 1);


// 反射光的碰撞检测
if (world.hit(r, 0.00, infinity, rec))
```

也就是下面这张图

![image-20230831145455792](C:\Users\卫清渠\AppData\Roaming\Typora\typora-user-images\image-20230831145455792.png)

我们用`ray (rec.p, target - rec.p)`去与世界碰撞，但这条射线由于本身就位于球上，必然与球面有交点(考虑到浮点数的误差 t = 0 oror t = 0.0001 都可能是hit函数的解)，因此我们要小小的修改下hit函数。

```c++
if (world.hit(r, 0.001, infinity, rec)) 
```

让解的下限从0.001开始，防止类似于0.00001这种解所代表的永远在所在球上不断地反射出现。

![image-20230831151500585](C:\Users\卫清渠\AppData\Roaming\Typora\typora-user-images\image-20230831151500585.png)

修改后速度快了很多（原本碰到球就是递归max_depth次），颜色也变得正常了。

#### Gamma修正

r g b值有些太小，可以适当放大，最后输出结果改为sqrt(r,g,b).

![image-20230831151626218](C:\Users\卫清渠\AppData\Roaming\Typora\typora-user-images\image-20230831151626218.png)

#### Lambertian Model

假设入射光在漫反射后向所所有方向辐射亮度相同，反射强度与入射角余弦成正比。

**DiffuseLight = I\ * cosθ**

上面的随机向量其实分布并不是关于角度平均的，是球体内点是平均概率的。

竖直直径的点最多，因此概率最高。越大的Φ角概率越小，越靠近发现概率越高。

将圆内随机点改为，圆上随机一点。

![image-20230831165451986](C:\Users\卫清渠\AppData\Roaming\Typora\typora-user-images\image-20230831165451986.png)

似乎没有什么变化，但是阴影确实更少了。

#### 出射在法线半球区域

![image-20230831191830392](C:\Users\卫清渠\AppData\Roaming\Typora\typora-user-images\image-20230831191830392.png)

#### 材质Material

碰撞归碰撞，光如何反射和折射由材质决定。

材质决定了散射光的形式（漫反射还是镜面反射），同时也决定了物体对各种光的反射率(albedo)

漫反射(lambertian)材质：反射光方向 = noraml +  单位圆

金属材质： 反射光方向 = v - 2 v·n

材质会有一个albedo属性代表对r g b 三种颜色的吸收率。

![image-20230831214844385](C:\Users\卫清渠\AppData\Roaming\Typora\typora-user-images\image-20230831214844385.png)

#### 模糊镜面反射

对于反射光来说，增加一点小小的球状偏移。

![image-20230831215417281](C:\Users\卫清渠\AppData\Roaming\Typora\typora-user-images\image-20230831215417281.png)

值得注意的是，加入偏移后可能反射光方向太朝下，太朝下的就直接吸收掉。（这就是为什么scatter函数会是bool类型，并且在material的scatter函数最后要`return (dot(scattered.direction(), rec.normal));`）

在ray_color函数中，如果发现return false会直接将光线记为黑色（就是对本次光线颜色贡献值为0,如果运气不好可能造成某个不应该是黑色的地方纯黑。)

![image-20230831215351784](C:\Users\卫清渠\AppData\Roaming\Typora\typora-user-images\image-20230831215351784.png)

#### Dielectric 电介质

光射到它们上面可以反射与折射。

Snell's law : n sinθ = n' sinθ'

折射建模： 入射光R、折射率η ，法线n向上；  出射光R‘，折射率η'，法线n'向下。

需要保证R，n，R’，n'是单位向量。

将R’分解为R‘⊥ =  R + cosθ * n, R'∥ = - 根号(1- R'⊥^2) n

cosθ可用 -R·n 来表示。

#### Schlick近似菲涅尔方程

菲涅尔方程描述了光线经过两个介质的界面时，反射率（反射光能量 / 入射光能量）的比值。

![image-20230902000836377](C:\Users\卫清渠\AppData\Roaming\Typora\typora-user-images\image-20230902000836377.png)

它实际上考虑了光p偏振和s偏振两个方向上的反射比。一般的渲染认为光五偏振，也就是p和s偏振量相同，所以反射率R = (Rs + Rp )/2

当入射角θ -> 90°时，反射率->1，公式比较复杂，计算代价高昂。

![image-20230902000958474](C:\Users\卫清渠\AppData\Roaming\Typora\typora-user-images\image-20230902000958474.png)

R(0)表示垂直时，光的反射率，用上面Rs，Rp计算R（0）

![image-20230902001056892](C:\Users\卫清渠\AppData\Roaming\Typora\typora-user-images\image-20230902001056892.png)

全反射+折射

![image-20230902001659656](C:\Users\卫清渠\AppData\Roaming\Typora\typora-user-images\image-20230902001659656.png)

全反射+折射+schlick近似

![image-20230902001846069](C:\Users\卫清渠\AppData\Roaming\Typora\typora-user-images\image-20230902001846069.png)

可以看到最左侧的dielectric顶端（入射角近乎0的地方）变得有些透明（）

#### trick：负半径的玻璃球