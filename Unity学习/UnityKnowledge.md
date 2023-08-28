#### Awake & Start

awake先于start调用，两者在生命周期内都只调用一次.

Awake在游戏对象首次被加载时调用,在游戏对象初始化之前调用.

Start在游戏对象初始化完成后调用.

#### FixedUpdate & Update



#### Activate/deactivate Object

使用gameObject.SetActive()去控制物体是否出现。

重新激活会从上次失活的状态复原。

如果父亲失活，孩子也会消失。

使用activeSelf / activeInHierarchy检查自身是否激活/自身是否在Hierarchy也就是游戏中出现。

父亲失活，孩子即使是activeSelf为True，activeInHierarchy也为False

#### Translate和Rotate

移动与旋转，记得要将设置成speed* Time.deltaTime

如果需要物理效果，建议使用物理的函数而非Translate和Rotate

#### 基准向量

Vector.up/forward 是世界的。

transform.up/forward 是局部的的。

#### 线性插值与平滑过渡

Mathf.Lerp(start, target, weight) 计算线性插值

SmoothDamp 平滑过渡一个float

```c#
void Update()
{
    if ( speed >= 1e-3)
    {
        float x = Mathf.SmoothDamp(transform.position.x, 5, ref speed, 3);
        Vector3 pos = transform.position;
        pos.x = x;
        transform.position = pos;
        Debug.Log(speed);
    }

}
```

Lerp一般比较生硬，建议使用SmoothDamp

#### 销毁

 Destroy(objToDestroy,2); 后面的浮点数用于定时。

也可以用于销毁组件

 `  Destroy(objToDestroy.GetComponent<MeshRenderer>(),2);`

#### GetButton和GetKey

GetButton需要明确按键的”代表名称“，例如”Jump“，Jump键对应具体哪个按键可以进行设置。

GetKey需要明确指定哪个按键。此种方法较为不推荐。

#### GetAxis

GetButton返回Boolean，GetAxis返回Float （-1 ~ 1）

在Input Manager中可以看到三个属性：

Gravity：决定松开按钮后归零的速度

Sensitivity：越大达到1的速度越快

Dead：越大就需要一个更大的值使得GetAxis返回非0值

`Snap：如果同时按下Negative和Positive，将会归0 ?`

Input.GetAxis("Raw")：只有0或1，比较适合方块2d游戏

#### OnMouseDown()

被按下时互动，比如添加一个力

```c#
private void OnMouseDown()
{
   GetComponent<Rigidbody>().AddForce(-transform.forward * 500f);
}
```

#### 获取脚本

使用GetComponent<脚本类名>()，其实这省去了gameObject，也就是自己这个参数，

如果要获取别的对象上的参数，就需要获取该对象实例，anotherObj.GetComponent<>()来获取它身上的脚本。

#### Time.DeltaTime

如果不使用DeltaTime，移动效果放在Update里不好的，因为帧与帧之间的时间不固定，造成移动的不平滑。DeltaTime解决了这个问题，让移动与总时间挂钩。