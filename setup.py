from setuptools import setup, find_packages

# 读取requirements.txt文件内容
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='fair-GPD',  # 你的包名，需要在PyPI上唯一
    version='0.0.2',  # 当前包版本
    author='Junximu',  # 你的名字或你的组织/团队的名字
    author_email='mujunxi@126.com',  # 你的电子邮件地址
    description='Graphormer Based Protein Sequence Design Package: GPD',  # 简短描述
    long_description=open('README.md').read(),  # 从README.md读取长描述
    long_description_content_type='text/markdown',  # 长描述内容的类型
    url='https://github.com/decodermu/GPD',  # 项目URL
    packages=find_packages(),  # 自动找到项目中的所有包
    package_data={
        # 如果你的包名是 "my_package"，并且子文件夹中有数据文件
        'GPD': ['parameters/20220607_random_3.pkl'],
    },
    install_requires=requirements,  # 依赖列表
    classifiers=[  # 包的分类索引信息
        'Development Status :: 3 - Alpha',  # 开发的状态，通常是'Alpha', 'Beta'或'Stable'
        'Intended Audience :: Developers',  # 目标用户
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',  # 许可证类型
        'Programming Language :: Python :: 3',  # 编程语言版本
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.8',  # 支持的Python版本
    include_package_data=True,  # 是否包含数据文件
    license='MIT',  # 许可证
    keywords='GPD',  # 包搜索关键词或标签
    # 其他参数...
)

# 注意：你需要替换your_package_name、your_script、Your Name、your.email@example.com等
# 为你项目的实际信息。
