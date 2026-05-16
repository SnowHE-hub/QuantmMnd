#!/bin/bash
# 后台运行：仅索引 reports 目录（无网络依赖）
cd /home/lenovo/projects/quantmind
/home/lenovo/miniforge3/envs/quantmind/bin/python scripts/build_kb.py \
    --source reports \
    --reports-dir reports \
    > /tmp/kb_reports.log 2>&1
echo "DONE: exit=$?" >> /tmp/kb_reports.log
