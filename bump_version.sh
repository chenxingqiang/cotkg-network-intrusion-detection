#!/bin/bash

# 文件名: bump_version.sh

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

# 检查 setup.py 是否存在
if [ ! -f "setup.py" ]; then
    print_message "Error: setup.py not found!" "$RED"
    exit 1
fi

# 获取当前版本
current_version=$(grep "version=" setup.py | cut -d'"' -f2)
if [ -z "$current_version" ]; then
    print_message "Error: Could not find version in setup.py!" "$RED"
    exit 1
fi

# 询问版本更新类型
print_message "Current version: $current_version" "$BLUE"
echo "Select version bump type:"
echo "1) Major (x.0.0)"
echo "2) Minor (0.x.0)"
echo "3) Patch (0.0.x)"
read -p "Enter choice (1-3): " bump_type

# 更新版本号
IFS='.' read -r major minor patch <<<"$current_version"
case "$bump_type" in
1)
    major=$((major + 1))
    minor=0
    patch=0
    ;;
2)
    minor=$((minor + 1))
    patch=0
    ;;
3)
    patch=$((patch + 1))
    ;;
*)
    print_message "Invalid choice!" "$RED"
    exit 1
    ;;
esac

new_version="$major.$minor.$patch"

# 使用临时文件进行替换
temp_file=$(mktemp)
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    awk -v old_ver="$current_version" -v new_ver="$new_version" '{gsub("version=\""old_ver"\"", "version=\""new_ver"\"")}1' setup.py >"$temp_file" && mv "$temp_file" setup.py
else
    # Linux
    awk -v old_ver="$current_version" -v new_ver="$new_version" '{gsub("version=\""old_ver"\"", "version=\""new_ver"\"")}1' setup.py >"$temp_file" && mv "$temp_file" setup.py
fi

# 验证更新
if grep -q "version=\"${new_version}\"" setup.py; then
    print_message "Version successfully bumped from $current_version to $new_version" "$GREEN"
else
    print_message "Error: Version update failed!" "$RED"
    exit 1
fi

# 询问是否要提交更改
read -p "Do you want to commit this version bump? (y/n): " commit_changes
if [[ "$commit_changes" == "y" ]]; then
    git add setup.py
    git commit -m "build: bump version to ${new_version}"
    print_message "Changes committed successfully!" "$GREEN"
fi

# 询问是否要创建标签
read -p "Do you want to create a git tag for this version? (y/n): " create_tag
if [[ "$create_tag" == "y" ]]; then
    git tag -a "v${new_version}" -m "Version ${new_version}"
    print_message "Tag v${new_version} created successfully!" "$GREEN"

    read -p "Do you want to push the tag to remote? (y/n): " push_tag
    if [[ "$push_tag" == "y" ]]; then
        git push origin "v${new_version}"
        print_message "Tag pushed to remote successfully!" "$GREEN"
    fi
fi
