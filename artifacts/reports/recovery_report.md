# 截断后验恢复实验

> 原始正式成绩保持不变；本报告仅展示扩展输出预算后的恢复结果。

- 原始结果：495 条
- 截断目标：39 条
- 已完成恢复：39 条
- 第一轮扩展预算：选择题 4096 tokens；指令遵循题 8192 tokens
- 最终重试选择题预算：8192 tokens

## 模型恢复对比

| 模型 | 截断目标 | 已补跑 | 截断恢复率 | 严格端到端正确率 | 恢复后端到端正确率 | 变化 | 格式错误 | 仍截断 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DEEPSEEK | 3 | 3 | 100.00% | 64.24% | 64.24% | +0.00 pp | 1 | 0 |
| GLM | 11 | 11 | 100.00% | 58.18% | 63.64% | +5.45 pp | 0 | 0 |
| KIMI | 25 | 25 | 100.00% | 43.03% | 43.03% | +0.00 pp | 25 | 0 |

## 明细

| 数据段 | 模型 | 题目 | 原状态 | 恢复状态 | 答案 | 正确性 | 预算 |
|---|---|---|---|---|---|---|---:|
| base | DEEPSEEK | mmlu-pro-physics-9160 | truncated | format_error | - | None | 4096 |
| base | GLM | mmlu-pro-math-7901 | truncated | correct | C | True | 4096 |
| base | GLM | ifeval-3718 | truncated | wrong | - | False | 8192 |
| base | KIMI | mmlu-pro-computer-science-10709 | truncated | format_error | - | None | 4096 |
| base | GLM | ifeval-279 | truncated | wrong | - | False | 8192 |
| base | KIMI | mmlu-pro-computer-science-10614 | truncated | format_error | - | None | 4096 |
| base | GLM | ifeval-2275 | truncated | correct | - | True | 8192 |
| base | KIMI | mmlu-pro-physics-9458 | truncated | format_error | - | None | 4096 |
| base | KIMI | mmlu-pro-physics-9978 | truncated | format_error | - | None | 4096 |
| base | KIMI | mmlu-pro-physics-9418 | truncated | format_error | - | None | 4096 |
| base | KIMI | mmlu-pro-physics-9667 | truncated | format_error | - | None | 4096 |
| base | KIMI | mmlu-pro-law-1346 | truncated | format_error | - | None | 4096 |
| base | KIMI | mmlu-pro-law-1569 | truncated | format_error | - | None | 8192 |
| base | KIMI | mmlu-pro-law-1303 | truncated | format_error | - | None | 4096 |
| base | KIMI | mmlu-pro-law-1390 | truncated | format_error | - | None | 4096 |
| stability-1 | GLM | mmlu-pro-physics-9048::run-1 | truncated | correct | J | True | 4096 |
| stability-1 | GLM | mmlu-pro-health-6104::run-1 | truncated | correct | I | True | 4096 |
| stability-1 | GLM | mmlu-pro-law-1231::run-1 | truncated | correct | B | True | 8192 |
| stability-1 | KIMI | mmlu-pro-law-1303::run-1 | truncated | format_error | - | None | 4096 |
| stability-1 | KIMI | mmlu-pro-law-1346::run-1 | truncated | format_error | - | None | 4096 |
| stability-2 | DEEPSEEK | mmlu-pro-physics-9160::run-2 | truncated | wrong | B | False | 4096 |
| stability-2 | GLM | mmlu-pro-computer-science-10403::run-2 | truncated | correct | F | True | 4096 |
| stability-2 | KIMI | mmlu-pro-physics-9160::run-2 | truncated | format_error | - | None | 4096 |
| stability-2 | KIMI | mmlu-pro-health-6104::run-2 | truncated | format_error | - | None | 4096 |
| stability-2 | KIMI | mmlu-pro-law-1303::run-2 | truncated | format_error | - | None | 4096 |
| stability-2 | KIMI | mmlu-pro-law-1346::run-2 | truncated | format_error | - | None | 4096 |
| stability-3 | GLM | mmlu-pro-math-8443::run-3 | truncated | correct | A | True | 4096 |
| stability-3 | DEEPSEEK | mmlu-pro-physics-9160::run-3 | truncated | wrong | A | False | 4096 |
| stability-3 | GLM | mmlu-pro-physics-9048::run-3 | truncated | correct | J | True | 4096 |
| stability-3 | KIMI | mmlu-pro-law-1346::run-3 | truncated | format_error | - | None | 4096 |
| robustness | GLM | robust-mmlu-pro-law-1303-reframe | truncated | correct | A | True | 4096 |
| robustness | KIMI | robust-mmlu-pro-computer-science-10439-shuffle | truncated | format_error | - | None | 4096 |
| robustness | KIMI | robust-mmlu-pro-physics-9160-reframe | truncated | format_error | - | None | 4096 |
| robustness | KIMI | robust-mmlu-pro-physics-9458-distractor | truncated | format_error | - | None | 4096 |
| robustness | KIMI | robust-mmlu-pro-health-6501-reframe | truncated | format_error | - | None | 4096 |
| robustness | KIMI | robust-mmlu-pro-law-1303-reframe | truncated | format_error | - | None | 4096 |
| robustness | KIMI | robust-mmlu-pro-law-1346-distractor | truncated | format_error | - | None | 8192 |
| robustness | KIMI | robust-mmlu-pro-law-1390-reframe | truncated | format_error | - | None | 4096 |
| robustness | KIMI | robust-mmlu-pro-law-1391-distractor | truncated | format_error | - | None | 4096 |

## 解释边界

- 恢复结果是看到截断后进行的后验实验，不替换原始正式成绩。
- 格式错误表示模型完成了输出但未遵守答案格式，不计为内容正确。
- 恢复后端到端正确率仅用于分析扩展预算策略的收益。
