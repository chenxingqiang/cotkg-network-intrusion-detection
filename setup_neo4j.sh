#!/bin/bash

NEO4J_PASSWORD="neo4jneo4j"

# 检测操作系统类型
detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# macOS 安装配置
setup_macos() {
    echo "Setting up Neo4j on macOS..."

    # 检查 Homebrew
    if ! command -v brew &> /dev/null; then
        echo "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi

    # 检查 Java 17
    if ! command -v java &> /dev/null || [[ $(java -version 2>&1) != *"version \"17"* ]]; then
        echo "Installing Java 17..."
        brew install openjdk@17
        echo 'export JAVA_HOME=$(/usr/libexec/java_home -v 17)' >> ~/.zshrc
        echo 'export PATH="$JAVA_HOME/bin:$PATH"' >> ~/.zshrc
        source ~/.zshrc
    fi

    # 安装/重新安装 Neo4j
    echo "Installing/Reinstalling Neo4j..."
    brew services stop neo4j 2>/dev/null
    brew uninstall neo4j 2>/dev/null
    brew install neo4j

    # 设置权限
    sudo chown -R $(whoami) /opt/homebrew/var/lib/neo4j
    sudo chown -R $(whoami) /opt/homebrew/var/log/neo4j

    # 设置密码
    echo "Setting Neo4j password..."
    /opt/homebrew/bin/neo4j-admin dbms set-initial-password "$NEO4J_PASSWORD"

    # 启动服务
    brew services start neo4j
}

# Linux 安装配置
setup_linux() {
    echo "Setting up Neo4j on Linux..."

    # 添加 Neo4j 仓库
    wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
    echo 'deb https://debian.neo4j.com stable latest' | sudo tee /etc/apt/sources.list.d/neo4j.list

    # 更新包列表并安装 Java 和 Neo4j
    sudo apt-get update
    sudo apt-get install -y openjdk-17-jdk neo4j

    # 设置密码
    sudo neo4j-admin dbms set-initial-password "$NEO4J_PASSWORD"

    # 启动服务
    sudo systemctl enable neo4j
    sudo systemctl start neo4j
}

# Windows 安装配置 (需要管理员权限)
setup_windows() {
    echo "Setting up Neo4j on Windows..."

    # 检查 Chocolatey
    if ! command -v choco &> /dev/null; then
        echo "Please install Chocolatey first. Run this script as administrator."
        echo "Installation instructions: https://chocolatey.org/install"
        exit 1
    }

    # 安装 Java 17 和 Neo4j
    choco install openjdk17 -y
    choco install neo4j-community -y

    # 设置环境变量
    setx JAVA_HOME "C:\Program Files\OpenJDK\jdk-17" /M
    setx PATH "%PATH%;%JAVA_HOME%\bin" /M

    # 设置密码
    "C:\Program Files\Neo4j\bin\neo4j-admin" dbms set-initial-password "$NEO4J_PASSWORD"

    # 启动服务
    net start neo4j
}

# 验证安装
verify_installation() {
    echo "Verifying Neo4j installation..."

    # 等待服务启动
    sleep 10

    # 测试连接
    if curl -s http://localhost:7474 > /dev/null; then
        echo "Neo4j is running successfully!"
        echo "You can access Neo4j Browser at: http://localhost:7474"
        echo "Username: neo4j"
        echo "Password: $NEO4J_PASSWORD"
    else
        echo "Failed to connect to Neo4j. Please check the logs."
        case $(detect_os) in
            "macos")
                echo "Check logs at: /opt/homebrew/var/log/neo4j/"
                ;;
            "linux")
                echo "Check logs at: /var/log/neo4j/"
                ;;
            "windows")
                echo "Check logs at: C:\Program Files\Neo4j\logs\"
                ;;
        esac
    fi
}

# 主函数
main() {
    OS=$(detect_os)
    echo "Detected OS: $OS"

    case $OS in
        "macos")
            setup_macos
            ;;
        "linux")
            setup_linux
            ;;
        "windows")
            setup_windows
            ;;
        *)
            echo "Unsupported operating system"
            exit 1
            ;;
    esac

    verify_installation
}

# 运行主函数
main
