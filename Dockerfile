# 基于官方 Python 3.11 轻量镜像
FROM python:3.11-slim

# 工作目录
WORKDIR /app

# 先复制依赖并安装（放最前，利用 Docker 缓存：改代码不会重装依赖）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY main.py .
COPY knowledge/ knowledge/
COPY static/ static/

# 暴露端口
EXPOSE 8000

# 启动：host 绑 0.0.0.0 才能被宿主机/外部访问
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
