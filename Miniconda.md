1. 下载Miniconda

   ```
   https://docs.conda.io/projects/miniconda/en/latest/
   ```

2. 配置清华源
   ```
   1. conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
   2. conda config --set show_channel_urls yes
   ```

3. 创建Env

   ```
   conda create -n ENV_NAME python=3.6
   ```

4. 进入Env

   ```
   conda activate ENV_NAME
   ```

5. 进入Pytorch官网，确定要下载的Pytorch版本。复制其推荐的下载代码
   ```
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   ```

   等待至少10分钟

6. 修改settins.json
   ```json
   # 修改python默认编译器
   "python.defaultInterpreterPath": "C:\\Wqq_Other\\Miniconda\\envs\\PyTorch\\python.exe",
   
   # 在code runner executorMap里修改运行命令
    "python": "C:\\Wqq_Other\\Miniconda\\envs\\PyTorch\\python.exe",
   ```

7. 如果出现里numpy问题

   ```
   UserWarning: mkl-service package failed to import, therefore Intel(R) MKL initialization ensuring its correct out-of-the box operation under condition when Gnu OpenMP had already been loaded by Python process is not assured. Please install mkl-service package
   ```

   参考

   >[UserWarning: mkl-service package failed to import, therefore Intel(R) MKL initialization ensuring it-CSDN博客](https://blog.csdn.net/u010741500/article/details/107141692#:~:text=要解决这个问题，最简单的方法是安装 mkl-service 包。 可以通过pip安装，在Python命令行中执行以下命令： ``` pip install mkl-service,``` 如果您使用的是conda环境，可以使用以下命令： ``` conda install mkl-service ``` 安装完成后，再次运行程序，就不会再出现上述警告信息了。)

