#!/bin/bash

# Configuration
PROJECT_DIR="/Users/tianrunliao/Desktop/pj2 submission"
REPO_URL="git@github.com:tianrunliao/Deep-Learning-Project2.git"
TEMP_DIR="/tmp/pj2_upload_$(date +%s)"

echo "=== 压缩超大文件并上传到GitHub (修复版) ==="
echo "项目目录: $PROJECT_DIR"
echo "临时目录: $TEMP_DIR"
echo "仓库地址: $REPO_URL"
echo

# 检查项目目录是否存在
if [ ! -d "$PROJECT_DIR" ]; then
    echo "错误: 项目目录不存在: $PROJECT_DIR"
    exit 1
fi

# 清理并创建临时目录
echo "创建临时目录..."
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

# 复制项目文件到临时目录（排除.git目录）
echo "复制项目文件到临时目录（排除.git目录）..."
rsync -av --exclude='.git' "$PROJECT_DIR/" "$TEMP_DIR/"

# 进入临时目录
cd "$TEMP_DIR"

# 查找超过100MB的文件（排除.git目录）
echo "正在查找超过100MB的文件..."
LARGE_FILES=$(find . -type f -size +100M ! -path './.git/*')

if [ -z "$LARGE_FILES" ]; then
    echo "没有找到超过100MB的文件，直接上传项目"
else
    echo "找到以下超过100MB的文件:"
    echo "$LARGE_FILES"
    echo
    
    # 创建压缩文件列表
    echo "正在压缩超大文件..."
    echo "# 压缩文件列表" > COMPRESSED_FILES.md
    echo "以下文件已被压缩以符合GitHub文件大小限制:" >> COMPRESSED_FILES.md
    echo >> COMPRESSED_FILES.md
    
    # 压缩每个大文件
    echo "$LARGE_FILES" | while read -r file; do
        if [ -f "$file" ]; then
            # 移除开头的 ./
            clean_file=${file#./}
            echo "压缩文件: $clean_file"
            
            # 获取原始文件大小
            if [ -f "$clean_file" ]; then
                original_size=$(du -h "$clean_file" | cut -f1)
                
                # 压缩文件
                gzip "$clean_file"
                
                # 获取压缩后文件大小
                if [ -f "${clean_file}.gz" ]; then
                    compressed_size=$(du -h "${clean_file}.gz" | cut -f1)
                    echo "- \`$clean_file\` (原始: $original_size, 压缩后: $compressed_size)" >> COMPRESSED_FILES.md
                    echo "  压缩完成: $clean_file -> ${clean_file}.gz"
                else
                    echo "  警告: 压缩失败 $clean_file"
                fi
            else
                echo "  警告: 文件不存在 $clean_file"
            fi
        fi
    done
    
    echo >> COMPRESSED_FILES.md
    echo "**注意**: 这些文件在下载后需要使用 \`gunzip\` 命令解压缩。" >> COMPRESSED_FILES.md
    echo "例如: \`gunzip filename.pt.gz\`" >> COMPRESSED_FILES.md
fi

# 初始化Git仓库
echo "初始化Git仓库..."
git init
git remote add origin "$REPO_URL"

# 创建.gitignore（如果不存在）
if [ ! -f ".gitignore" ]; then
    echo "创建.gitignore文件..."
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.venv/
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Git
.git/
EOF
fi

# 添加所有文件并提交
echo "添加文件到Git..."
git add .
git commit -m "Upload project with compressed large files ($(date))"

# 获取远程分支信息并推送
echo "获取远程分支信息..."
git fetch origin 2>/dev/null || echo "首次推送，跳过fetch"

# 检查远程分支并推送
if git ls-remote --heads origin 2>/dev/null | grep -q "refs/heads/main"; then
    echo "推送到main分支..."
    git push -f origin HEAD:main
elif git ls-remote --heads origin 2>/dev/null | grep -q "refs/heads/master"; then
    echo "推送到master分支..."
    git push -f origin HEAD:master
else
    echo "创建并推送到main分支..."
    git checkout -b main
    git push -u origin main
fi

echo
echo "=== 上传完成 ==="

# 恢复本地文件（解压缩）
if [ ! -z "$LARGE_FILES" ]; then
    echo "恢复本地压缩文件..."
    cd "$PROJECT_DIR"
    
    # 查找所有.gz文件并解压
    find . -name "*.gz" -type f | while read -r gz_file; do
        if [ -f "$gz_file" ]; then
            echo "解压缩: $gz_file"
            gunzip "$gz_file"
        fi
    done
fi

# 清理临时目录
echo "清理临时文件..."
rm -rf "$TEMP_DIR"

echo "所有操作完成！"
echo "- GitHub上的大文件已被压缩"
echo "- 本地文件保持原始状态"
echo "- 压缩文件信息记录在 COMPRESSED_FILES.md 中"
echo "- 仓库地址: https://github.com/tianrunliao/pj2-submission"