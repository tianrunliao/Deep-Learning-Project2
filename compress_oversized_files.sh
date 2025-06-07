#!/bin/bash

# 设置项目路径和GitHub仓库
PROJECT_DIR="/Users/tianrunliao/Desktop/pj2 submission"
REPO_URL="git@github.com:tianrunliao/pj2-submission.git"
TEMP_DIR="/tmp/pj2_submission_compressed"

echo "=== 压缩超大文件并上传到GitHub ==="
echo "项目目录: $PROJECT_DIR"
echo "临时目录: $TEMP_DIR"
echo "仓库地址: $REPO_URL"
echo

# 检查项目目录是否存在
if [ ! -d "$PROJECT_DIR" ]; then
    echo "错误: 项目目录不存在: $PROJECT_DIR"
    exit 1
fi

# 进入项目目录
cd "$PROJECT_DIR"

# 查找超过100MB的文件
echo "正在查找超过100MB的文件..."
LARGE_FILES=$(find . -type f -size +100M)

if [ -z "$LARGE_FILES" ]; then
    echo "没有找到超过100MB的文件，直接上传项目"
else
    echo "找到以下超过100MB的文件:"
    echo "$LARGE_FILES"
    echo
fi

# 创建临时目录
echo "创建临时目录..."
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

# 复制所有文件到临时目录
echo "复制项目文件到临时目录..."
cp -r "$PROJECT_DIR"/* "$TEMP_DIR"/

# 压缩超大文件
if [ ! -z "$LARGE_FILES" ]; then
    echo "正在压缩超大文件..."
    cd "$TEMP_DIR"
    
    # 创建压缩文件列表
    echo "# 压缩文件列表" > COMPRESSED_FILES.md
    echo "以下文件已被压缩以符合GitHub文件大小限制:" >> COMPRESSED_FILES.md
    echo >> COMPRESSED_FILES.md
    
    for file in $LARGE_FILES; do
        # 移除开头的 ./
        clean_file=${file#./}
        echo "压缩文件: $clean_file"
        
        # 压缩文件
        gzip "$clean_file"
        
        # 记录到列表
        original_size=$(du -h "$PROJECT_DIR/$clean_file" | cut -f1)
        compressed_size=$(du -h "${clean_file}.gz" | cut -f1)
        echo "- \`$clean_file\` (原始: $original_size, 压缩后: $compressed_size)" >> COMPRESSED_FILES.md
    done
    
    echo >> COMPRESSED_FILES.md
    echo "**注意**: 这些文件在下载后需要使用 \`gunzip\` 命令解压缩。" >> COMPRESSED_FILES.md
fi

# 进入临时目录并设置Git
cd "$TEMP_DIR"
echo "初始化Git仓库..."
git init
git remote add origin "$REPO_URL"

# 创建.gitignore（如果不存在）
if [ ! -f ".gitignore" ]; then
    echo "创建.gitignore文件..."
    cat > .gitignore << EOF
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
EOF
fi

# 添加所有文件并提交
echo "添加文件到Git..."
git add .
git commit -m "Upload project with compressed large files"

# 获取远程分支信息
echo "获取远程分支信息..."
git fetch origin

# 检查远程分支并推送
if git ls-remote --heads origin | grep -q "refs/heads/main"; then
    echo "推送到main分支..."
    git push -f origin HEAD:main
elif git ls-remote --heads origin | grep -q "refs/heads/master"; then
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
    
    for file in $LARGE_FILES; do
        clean_file=${file#./}
        if [ -f "${clean_file}.gz" ]; then
            echo "解压缩: ${clean_file}.gz"
            gunzip "${clean_file}.gz"
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