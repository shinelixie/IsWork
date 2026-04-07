#!/bin/bash

# =================================================================
# 脚本名称: ascend_setup_env.sh
# 适用环境: 华为昇腾 (Ascend) Docker 容器 (已开启 TUN/NET_ADMIN 权限)
# 功能: 自动化配置 ZeroTier 网络与 code-server 网页版 IDE
# =================================================================

echo "[1/5] 更新系统源并安装基础工具..."
apt-get update && apt-get install -y \
    curl \
    wget \
    psmisc \
    iproute2 \
    iputils-ping \
    net-tools

echo "[2/5] 安装 ZeroTier One..."
if ! command -v zerotier-one &> /dev/null; then
    curl -s https://install.zerotier.com | bash
else
    echo "ZeroTier 已安装，跳过..."
fi

echo "[3/5] 启动 ZeroTier 服务 (强制 Root 模式)..."
# 清理旧进程
pkill -9 zerotier-one || true
mkdir -p /var/lib/zerotier-one
# -U 参数防止在容器内降权失败导致僵尸进程
zerotier-one -U -d
sleep 2
echo "ZeroTier 状态: $(zerotier-cli status)"

echo "[4/5] 安装 code-server..."
if ! command -v code-server &> /dev/null; then
    curl -fsSL https://code-server.dev/install.sh | sh
else
    echo "code-server 已安装，跳过..."
fi

echo "[5/5] 启动 code-server (后台运行)..."
pkill -9 node || true
# 配置说明: 监听所有网卡, 端口 8080, 禁用密码以配合私有网使用
nohup code-server --bind-addr 0.0.0.0:8080 --auth none > /root/code-server.log 2>&1 &

echo "-----------------------------------------------------------"
echo "✅ 环境配置完成！"
echo "1. 请执行 'zerotier-cli join [你的NetworkID]' 加入网络"
echo "2. 访问地址: http://10.216.251.164:8080 (请确认你的 ZT IP)"
echo "-----------------------------------------------------------"
