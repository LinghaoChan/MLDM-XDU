# Machine Learning and Data Mining Tasks (MLDM-XDU)

## 安装依赖

```bash
numpy==1.18.5
matplotlib==3.2.1
xlrd==1.2.0
```

## Task1 [Logistic Regression](Task1-LogisticRegression)

可以使用main.py运行相关程序。

代码使用'LogisticRegression.py'封装成class了，使用方法：

```python
import LogisticRegression as LogisReg

method = "gradientdescent"
optmizer = LogisReg.LogisticRegression(method = method)
beta = optmizer1.fit(X, y)
```

