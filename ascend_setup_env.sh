#!/bin/bash

# SSH 安装和配置脚本
# 功能：安装 OpenSSH 服务器，配置 8080 端口，设置免密登录
# 适用系统：Debian/Ubuntu (apt)

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查是否以 root 运行
if [ "$EUID" -ne 0 ]; then 
    print_error "请以 root 用户运行此脚本"
    exit 1
fi

# 1. 安装 OpenSSH 服务器
print_info "安装 OpenSSH 服务器..."
if command -v apt-get &> /dev/null; then
    apt-get update -qq
    apt-get install -y openssh-server openssh-client
elif command -v yum &> /dev/null; then
    yum install -y openssh-server openssh-clients
elif command -v apk &> /dev/null; then
    apk add --no-cache openssh-server openssh-client
else
    print_error "不支持的包管理器"
    exit 1
fi

# 2. 创建必要的目录
print_info "创建 SSH 运行时目录..."
mkdir -p /run/sshd
mkdir -p /var/empty
mkdir -p /var/log
chmod 755 /run/sshd

# 3. 配置 SSH 监听 8080 端口
print_info "配置 SSH 监听 8080 端口..."
SSHD_CONFIG="/etc/ssh/sshd_config"

# 备份原配置
if [ ! -f "${SSHD_CONFIG}.bak" ]; then
    cp $SSHD_CONFIG ${SSHD_CONFIG}.bak
    print_info "已备份原配置到 ${SSHD_CONFIG}.bak"
fi

# 修改端口配置
sed -i 's/^#Port 22/Port 8080/' $SSHD_CONFIG
sed -i 's/^Port 22/Port 8080/' $SSHD_CONFIG
if ! grep -q "^Port 8080" $SSHD_CONFIG; then
    echo "Port 8080" >> $SSHD_CONFIG
fi

# 启用公钥认证
sed -i 's/^#PubkeyAuthentication yes/PubkeyAuthentication yes/' $SSHD_CONFIG
sed -i 's/^#AuthorizedKeysFile/AuthorizedKeysFile/' $SSHD_CONFIG

# 允许 root 登录
sed -i 's/^#PermitRootLogin prohibit-password/PermitRootLogin yes/' $SSHD_CONFIG
sed -i 's/^#PermitRootLogin yes/PermitRootLogin yes/' $SSHD_CONFIG
if ! grep -q "^PermitRootLogin" $SSHD_CONFIG; then
    echo "PermitRootLogin yes" >> $SSHD_CONFIG
fi

print_info "SSH 配置完成"

# 4. 生成主机密钥（如果不存在）


# 5. 设置 root 密码
print_info "设置 root 密码..."
echo -e "${YELLOW}请输入 root 用户的新密码：${NC}"
passwd root

# 6. 配置免密登录
print_info "配置免密登录..."
mkdir -p /root/.ssh
chmod 700 /root/.ssh

touch /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys


# 8. 停止已有的 sshd 进程
print_info "停止已有的 SSH 进程..."
pkill sshd 2>/dev/null || true
sleep 1

# 9. 启动 SSH 服务
print_info "启动 SSH 服务..."
/usr/sbin/sshd -D &

# 等待服务启动
sleep 2

# 10. 验证服务状态
print_info "验证 SSH 服务状态..."
if pgrep -x "sshd" > /dev/null; then
    print_info "SSH 服务已启动"
else
    print_error "SSH 服务启动失败"
    exit 1
fi

# 检查端口监听
if command -v netstat &> /dev/null; then
    if netstat -tlnp | grep -q ":8080.*sshd"; then
        print_info "SSH 正在监听 8080 端口"
    else
        print_warn "未检测到 8080 端口监听"
    fi
elif command -v ss &> /dev/null; then
    if ss -tlnp | grep -q ":8080.*sshd"; then
        print_info "SSH 正在监听 8080 端口"
    else
        print_warn "未检测到 8080 端口监听"
    fi
fi

# 11. 显示信息
echo ""
echo "=========================================="
print_info "SSH 安装和配置完成！"
echo "=========================================="
echo ""
echo "连接信息："
echo "  端口: 8080"
echo "  用户: root"
echo ""
echo "从外部连接："
echo "  ssh -p 8080 root@$(curl -s ifconfig.me 2>/dev/null || echo '宿主机IP')"
echo ""
echo "私钥位置：/root/.ssh/id_rsa"
echo "配置文件：/root/.ssh/config"
echo ""
echo "=========================================="


print_info "脚本执行完毕"
