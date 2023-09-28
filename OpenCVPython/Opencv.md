#### 安装opencv

```
pip install opencv-contrib-python
pip install caer 
```

前者带有社区为opencv贡献的函数

caer用于优化工作流

#### 读入/显示图片

```python
import cv2 as cv
img = cv.imread(path)
cv.show('Window Name', img)
cv.waitKey()   # 等待x毫秒时间, 否则窗口消失
```

cv.waitKey默认参数为0，会表示为无限等待按键时间。

#### 图片宽高RGB

```python
img.shape = (height, width,channel)
```

#### 重塑图片大小

```python
def rescale(frame, scale=0.75):
    height = int(frame.shape[0] * scale) #注意转int
    width = int(frame.shape[1] * scale) 
    return cv.resize(frame, (width, height), interpolation=cv.INTER_LINEAR)
# interpolation 插值方式
```

