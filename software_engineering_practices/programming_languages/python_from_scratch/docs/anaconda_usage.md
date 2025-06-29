## anaconda的基本使用

```
conda --version
conda info。

```

设置镜像源
```
conda config --add channels http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/

conda config --set show_channel_urls yes
conda config --show channels
conda env list
```

创建jupyter环境(或者直接当前base就可以，这一步可以忽略)：
```
conda create -n pylearn jupyter notebook
```

conda windows下pytorch安装