import subprocess
import os

# 定义环境文件保存路径
requirements_file = 'requirements.txt'

# 使用 pip freeze 获取当前环境中安装的所有库
try:
    with open(requirements_file, 'w') as f:
        subprocess.check_call(['pip', 'freeze'], stdout=f)
    print(f"所有项目库已保存到 {requirements_file}")
except subprocess.CalledProcessError as e:
    print(f"获取项目库时出错: {e}")
