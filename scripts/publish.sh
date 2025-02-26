#!/bin/bash

# 文件名: publish.sh

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 打印带颜色的消息
print_message() {
    echo -e "${2}${1}${NC}"
}

# 检查并安装必要的工具
check_tools() {
    print_message "Checking required tools..." "$BLUE"

    # 检查并安装 twine
    if ! command -v twine &>/dev/null; then
        print_message "Installing twine..." "$YELLOW"
        pip install twine
    fi

    # 检查并安装 build
    if ! command -v build &>/dev/null; then
        print_message "Installing build..." "$YELLOW"
        pip install build
    fi
}

# 清理旧的构建文件
clean_old_builds() {
    print_message "Cleaning old builds..." "$BLUE"
    rm -rf build/ dist/ *.egg-info/
}

# 构建包
build_package() {
    print_message "Building package..." "$BLUE"
    python -m build
}

# 检查包
check_package() {
    print_message "Checking package..." "$BLUE"
    twine check dist/*
}

# 上传到 PyPI
upload_package() {
    print_message "Uploading to PyPI..." "$BLUE"

    # 询问是否使用测试版 PyPI
    read -p "Upload to Test PyPI first? (y/n): " use_test_pypi

    if [[ "$use_test_pypi" == "y" ]]; then
        print_message "Uploading to Test PyPI..." "$YELLOW"
        twine upload --repository testpypi dist/*

        # 等待确认后再上传到正式版
        read -p "Test successful? Upload to PyPI? (y/n): " upload_to_pypi
        if [[ "$upload_to_pypi" != "y" ]]; then
            print_message "Aborting upload to PyPI" "$RED"
            exit 1
        fi
    fi

    print_message "Uploading to PyPI..." "$GREEN"
    twine upload dist/*
}

# 主函数
main() {
    # 检查工具
    check_tools

    # 清理旧的构建
    clean_old_builds

    # 构建包
    build_package

    # 检查包
    check_package

    # 上传包
    upload_package

    print_message "\n✨ Package published successfully!" "$GREEN"
}

# 执行主函数
main
