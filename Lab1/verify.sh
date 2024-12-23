#!/bin/bash

# 设置允许的最大误差
MAX_ERROR=1e-6

# 定义标准文件和实际输出文件路径
EXPECTED_FILE="result/basic_baseline"
ACTUAL_FILE="output"


choice=$1
# read -p "Please select an option(basic, red, mpibasic, mpired)" choice

case $choice in
    "basic")
        echo "Compare with basic."
        EXPECTED_FILE="result/basic"
        ;;
    "red")
        echo "Compare with red."
        EXPECTED_FILE="result/red"
        ;;
    "test")
        echo "Compare with test."
        EXPECTED_FILE="result/test"
        ;;
    *)
        echo "Compare with basic."
        ;;
esac

# 检查两个文件是否存在
if [ ! -f "$EXPECTED_FILE" ]; then
    echo "标准文件 $EXPECTED_FILE 不存在"
    exit 1
fi

if [ ! -f "$ACTUAL_FILE" ]; then
    echo "实际输出文件 $ACTUAL_FILE 不存在"
    exit 1
fi

awk -v max_error="$MAX_ERROR" '
function abs(x) { return x < 0 ? -x : x }
ALLRESULT=0
NR==FNR && NR > 1 {
    expected[NR] = $0
    nlines++
    next
}

FNR > 1 && FNR < nlines {
    split(expected[FNR], expected_values)
    split($0, actual_values)

    for (i = 1; i <= NF; i++) {
        expected_value = expected_values[i]
        actual_value = actual_values[i]

        # 计算绝对误差
        error = abs(expected_value - actual_value)

        # 如果误差超出允许范围，则报告错误
        if (error > max_error) {
            print "行 " FNR-1 ", 列 " i ": 预期值 " expected_value " 实际值 " actual_value " 误差 " error " 超出允许范围 " max_error
            ALLRESULT=1
            exit 1
        }
    }
}
END {
    if (ALLRESULT==0) {
        print "所有数值都在允许误差范围内，验证通过！"
    } else {
        print "存在数值在误差范围外，验证未通过！"
    }
}' "$EXPECTED_FILE" "$ACTUAL_FILE"