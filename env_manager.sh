#!/bin/bash

# 文件名: env_manager.sh

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 打印带颜色的消息
print_message() {
    echo -e "${2}${1}${NC}"
}

# 检查Python版本
check_python_version() {
    print_message "\nChecking Python version..." "$BLUE"
    python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    print_message "Current Python version: $python_version" "$GREEN"

    read -p "Do you want to use this version? (y/n): " use_current
    if [[ "$use_current" != "y" ]]; then
        print_message "\nAvailable Python versions:" "$BLUE"
        pyenv versions
        read -p "Enter the Python version to use (e.g., 3.8.10): " py_version
        pyenv local $py_version
        print_message "Python version set to $py_version" "$GREEN"
    fi
}

# 创建虚拟环境
create_venv() {
    print_message "\nSetting up virtual environment..." "$BLUE"
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_message "Virtual environment created" "$GREEN"
    else
        print_message "Virtual environment already exists" "$YELLOW"
    fi

    # 激活虚拟环境
    source venv/bin/activate
    print_message "Virtual environment activated" "$GREEN"

    # 更新pip
    pip install --upgrade pip
}

# 检查并安装依赖
check_dependencies() {
    print_message "\nChecking dependencies..." "$BLUE"

    # 检查requirements.txt是否存在
    if [ -f "requirements.txt" ]; then
        print_message "Found requirements.txt" "$GREEN"
        read -p "Do you want to install dependencies from requirements.txt? (y/n): " install_deps
        if [[ "$install_deps" == "y" ]]; then
            pip install -r requirements.txt
        fi
    else
        print_message "No requirements.txt found" "$YELLOW"
        read -p "Do you want to create a new requirements.txt? (y/n): " create_reqs
        if [[ "$create_reqs" == "y" ]]; then
            pip freeze >requirements.txt
            print_message "Created requirements.txt" "$GREEN"
        fi
    fi
}

# 更新依赖
update_dependencies() {
    print_message "\nUpdating dependencies..." "$BLUE"

    # 安装pip-review如果没有
    pip install pip-review

    # 显示可更新的包
    pip-review

    read -p "Do you want to update all packages? (y/n): " update_all
    if [[ "$update_all" == "y" ]]; then
        pip-review --auto
        # 更新requirements.txt
        pip freeze >requirements.txt
        print_message "All packages updated and requirements.txt refreshed" "$GREEN"
    fi
}

# 设置setup.py
setup_package() {
    print_message "\nSetting up package configuration..." "$BLUE"

    if [ ! -f "setup.py" ]; then
        read -p "Package name: " package_name
        read -p "Version (e.g., 0.1.0): " version
        read -p "Author: " author
        read -p "Author email: " author_email
        read -p "Description: " description

        # 创建setup.py
        cat >setup.py <<EOL
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="$package_name",
    version="$version",
    author="$author",
    author_email="$author_email",
    description="$description",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    install_requires=[
$(pip freeze | sed 's/^/        "/g' | sed 's/$/",/g')
    ],
)
EOL
        print_message "Created setup.py" "$GREEN"
    else
        print_message "setup.py already exists" "$YELLOW"
        read -p "Do you want to update it? (y/n): " update_setup
        if [[ "$update_setup" == "y" ]]; then
            ./bump_version.sh
        fi
    fi
}

# 发布到PyPI
publish_package() {
    print_message "\nPublishing package..." "$BLUE"

    # 检查并安装必要的工具
    pip install twine build

    # 清理旧的构建
    rm -rf build/ dist/ *.egg-info/

    # 构建包
    python -m build

    # 检查包
    twine check dist/*

    # 询问是否使用测试PyPI
    read -p "Upload to Test PyPI first? (y/n): " use_test_pypi
    if [[ "$use_test_pypi" == "y" ]]; then
        twine upload --repository testpypi dist/*
        print_message "Uploaded to Test PyPI" "$GREEN"

        read -p "Test successful? Upload to PyPI? (y/n): " upload_to_pypi
        if [[ "$upload_to_pypi" != "y" ]]; then
            print_message "Aborting upload to PyPI" "$RED"
            return
        fi
    fi

    # 上传到PyPI
    twine upload dist/*
    print_message "Package published to PyPI" "$GREEN"
}

# 主菜单
show_menu() {
    print_message "\n=== Environment Manager ===" "$BLUE"
    echo "1) Setup Python environment"
    echo "2) Check/Install dependencies"
    echo "3) Update dependencies"
    echo "4) Setup package configuration"
    echo "5) Publish to PyPI"
    echo "6) Exit"
}

# 主函数
main() {
    while true; do
        show_menu
        read -p "Choose an option (1-6): " choice

        case "$choice" in
        1)
            check_python_version
            create_venv
            ;;
        2)
            check_dependencies
            ;;
        3)
            update_dependencies
            ;;
        4)
            setup_package
            ;;
        5)
            publish_package
            ;;
        6)
            print_message "\nGoodbye!" "$GREEN"
            exit 0
            ;;
        *)
            print_message "Invalid option!" "$RED"
            ;;
        esac
    done
}

# 执行主函数
main
