#!/bin/bash

# Blog Publish Script
# 用法: ./publish-blog.sh <文章标题> [草稿目录]
#
# 示例:
#   ./publish-blog.sh "我的第一篇博客"
#   ./publish-blog.sh "AI 发展趋势" ~/Blog/drafts

set -e

# 配置
BLOG_DIR="/Users/siyuantao/Github/Blogs/tsyhahaha.github.io"
DRAFT_DIR="${2:-$HOME/Blog/drafts}"
TITLE="$1"
AUTHOR="tsyhahaha"

if [ -z "$TITLE" ]; then
    echo "用法: ./publish-blog.sh <文章标题> [草稿目录]"
    echo "示例: ./publish-blog.sh '我的第一篇博客'"
    exit 1
fi

# 生成 slug（URL 友好的文件名）
SLUG=$(echo "$TITLE" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9\s-]//g' | sed 's/\s+/-/g')

# 生成日期
DATE=$(date +%Y-%m-%d)
FILENAME="${DATE}-${SLUG}.md"
FULL_PATH="${BLOG_DIR}/_posts/${FILENAME}"

echo "📝 正在发布博客: $TITLE"
echo "   文件名: $FILENAME"

# 检查草稿目录
if [ ! -d "$DRAFT_DIR" ]; then
    echo "❌ 草稿目录不存在: $DRAFT_DIR"
    echo "   正在创建..."
    mkdir -p "$DRAFT_DIR"
    echo "   请在 $DRAFT_DIR 目录下创建 $FILENAME 文件"
    exit 1
fi

# 查找草稿文件（支持带日期和不带日期两种格式）
DRAFT_FILE=""
if [ -f "${DRAFT_DIR}/${FILENAME}" ]; then
    DRAFT_FILE="${DRAFT_DIR}/${FILENAME}"
elif [ -f "${DRAFT_DIR}/${SLUG}.md" ]; then
    DRAFT_FILE="${DRAFT_DIR}/${SLUG}.md}"
else
    # 查找最新的 md 文件
    DRAFT_FILE=$(ls -t "$DRAFT_DIR"/*.md 2>/dev/null | head -1)
fi

if [ -z "$DRAFT_FILE" ] || [ ! -f "$DRAFT_FILE" ]; then
    echo "❌ 未找到草稿文件"
    echo "   请在 $DRAFT_DIR 目录下创建博客文件"
    exit 1
fi

echo "   找到草稿: $DRAFT_FILE"

# 读取现有内容（如果有的话，去除已有的 front matter）
CONTENT=$(sed -n '/^---$/,/^---$/d; p' "$DRAFT_FILE")

# 生成新的 front matter
FRONTMATTER=$(cat <<EOF
---
layout: post
title: "$TITLE"
date: $(date +%Y-%m-%d\ %H:%M:%S\ +0800)
author: $AUTHOR
tags: []
categories: []
pin: false
toc: true
mathjax: false
mermaid: false
cover: 
---

EOF
)

# 写入最终文件
echo "$FRONTMATTER$CONTENT" > "$FULL_PATH"

echo "✅ 已生成博客文件: $FULL_PATH"

# 调用 LaTeX 修复脚本
FIX_SCRIPT="/Users/siyuantao/Github/Blogs/scripts/fix_jekyll_latex.py"
if [ -f "$FIX_SCRIPT" ]; then
    echo "🔧 正在自动修复 LaTeX 格式..."
    python3 "$FIX_SCRIPT" "$FULL_PATH"
else
    echo "⚠️ 未找到 LaTeX 修复脚本: $FIX_SCRIPT，跳过修复。"
fi

# Git 操作
if [ "$SKIP_PUSH" != "1" ]; then
    cd "$BLOG_DIR"
    git add "_posts/${FILENAME}"
    git commit -m "feat: 发布博客 - ${TITLE}"
    git push origin master

    echo "🚀 已推送到 GitHub，Netlify 正在部署..."
    echo "   访问 https://tsyhahaha.netlify.app 查看"
else
    echo "⏸️  跳过 Git 推送 (SKIP_PUSH=1)"
fi
