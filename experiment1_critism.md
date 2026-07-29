结论：首轮评估证明了真实数据与 RTX 3060 全流程能够运行，但当前结果只能定义为 `RUN_001_DIAGNOSTIC`，还不能作为“共享标量未被否定”的正式科学证据，更不能据此结束 CUSP 假设。

## 1. 首轮结果概况

正式结果在 [artifacts/cuda](/Users/darkevea/code/place_pulse/artifacts/cuda)，不是根目录下未跟踪的旧 artifacts。

Safety 主分析：

| 模型 | 外层交叉熵 | 相对 M1 |
|---|---:|---:|
| M0 全局频率 | 约 0.9675 | 优于 M1 |
| M1 标量 Davidson | 1.0573 | 基准 |
| M2 连续偏好 | 2.1936 | 恶化 107.5% |
| M3 潜在类别 | 1.5328 | 恶化 45.0% |

来源：[safety_model_comparison.json](/Users/darkevea/code/place_pulse/artifacts/cuda/metrics/safety_model_comparison.json:116)

外层五折方向完全一致，因此不是单个 fold 的偶然波动：

- M2 相对 M1 的 ELPD：−1.136，95% CI [−1.191, −1.095]；
- M3 相对 M1 的 ELPD：−0.476，95% CI [−0.483, −0.468]；
- M3 稳定性 ARI：0.079，远低于预注册阈值 0.70；
- 排序反转比例虽然达到 0.870，但类别本身不稳定，因此不能解释为真实人群分化；
- CUSP 阶段按门控正确跳过。

机器输出为 `SCALAR_NOT_REJECTED`：[verdict.json](/Users/darkevea/code/place_pulse/artifacts/cuda/report/verdict.json:2)。

但更准确的人工审计结论是：

> `BASELINE_CALIBRATION_FAILED — SCIENTIFIC_VERDICT_DEFERRED`

## 2. 为什么当前 verdict 不可靠

### M0 在全部六个维度都优于 M1

这不是 Safety 独有现象：

| 维度 | M0 CE | M1 CE | M1 相对 M0恶化 |
|---|---:|---:|---:|
| Safety | 0.968 | 1.057 | 0.090 |
| Lively | 0.949 | 1.199 | 0.250 |
| Beautiful | 0.962 | 1.268 | 0.306 |
| Wealthy | 0.972 | 1.309 | 0.337 |
| Boring | 0.997 | 1.592 | 0.596 |
| Depressing | 1.001 | 1.445 | 0.444 |

一个拥有图像效用和平局参数的 Davidson 模型系统性输给全局三分类频率，说明 M1 正则化、响应风格建模或优化存在严重失配。只有在 M1 至少胜过 M0 后，M1 对 M2/M3 的比较才具有解释价值。

### M1 没有独立选择正则化

当前代码用 M2 连续模型选择出的 `selected_l2` 直接拟合 M1：

- M2 选择发生在 [pipeline.py](/Users/darkevea/code/place_pulse/src/placepulse_cusp/pipeline.py:419)
- 同一个值随后传给 M1：[pipeline.py](/Users/darkevea/code/place_pulse/src/placepulse_cusp/pipeline.py:434)

M1 因而不是独立调优的“强标量基线”。

更明显的是，六维所有 fold 都选择了候选上限 `0.01`。最优值落在搜索边界通常意味着真正需要的收缩强度位于当前搜索区间之外。

### L2 收缩可能远远不够

Davidson 中图像、评分者左右偏差和平局倾向的惩罚都使用参数平方的均值：[davidson.py](/Users/darkevea/code/place_pulse/src/placepulse_cusp/models/davidson.py:56)。

在约 111,000 张图像和近 49,000 名 Safety 评分者的规模下，`mean()` 会让单个参数的正则梯度随参数数量缩小。最终效用范围达到约 −4.92 至 4.85；M3 各类别效用甚至达到约 −8.82 至 9.11，符合弱收缩下稀疏项目过拟合的表现。

### 已知评分者偏好明显过拟合

边留出中，M2/M3 使用同一评分者训练历史推断其偏好或类别，预测极差；但在新评分者留出中，对评分者偏好积分后反而有所改善：

- Safety M3 新评分者 ELPD：+0.058；
- Safety M3 时间留出 ELPD：+0.078；
- 同一个模型在边留出 ELPD：−0.476。

最合理的解释不是“异质性只对新评分者有效”，而是：

> 评分者特定因子和类别后验在稀疏个人历史上严重过拟合；人口平均预测反而规避了这种过拟合。

最终 M3 中约 51.9% 的评分者最大类别概率超过 0.99，但跨初始化 ARI 只有 0.079。这种“后验非常确定、类别极不稳定”的组合是伪分群的典型警报。

## 3. 与原计划存在的实现偏差

这些问题不会改变本轮阴性方向，但阻止它成为确认性结果：

1. `inner_folds: 3` 没有真正使用。M2 使用单次随机 80/20 事件切分，M3 使用单次评分者切分：[pipeline.py](/Users/darkevea/code/place_pulse/src/placepulse_cusp/pipeline.py:294)。

2. `random_starts: 5`、`lbfgs_steps: 40` 和 `batch_size` 都存在于配置，但代码中没有被模型训练使用：[confirmatory.yaml](/Users/darkevea/code/place_pulse/configs/confirmatory.yaml:35)。

3. M3 的“50 次 bootstrap 稳定性”并没有 bootstrap 评分者；它是在同一完整数据上重复初始化：[pipeline.py](/Users/darkevea/code/place_pulse/src/placepulse_cusp/pipeline.py:627)。

4. 排序反转没有限制为高信息图像对，也没有计算预注册的反转后验概率 ≥0.90。

5. 模拟校准只测试 CUSP 与普通混合密度，没有运行：

   - 纯标量数据的错误否定率；
   - 连续偏好恢复；
   - 潜在类别数恢复；
   - 类别 ARI 恢复。

   当前 `cusp_recovery_rate=1.0` 只能证明密度代码能区分专门生成的 CUSP 与简单混合，不能验证 M1–M3 选择程序。[recovery.py](/Users/darkevea/code/place_pulse/src/placepulse_cusp/simulation/recovery.py:28)

6. `simulation_ok` 在异质性门控中没有接收 M1–M3 的实际校准结果。

7. CUSP 分支若未来被触发，目前会重新在完整数据上拟合六维 mixture，而不是使用计划规定的外层交叉拟合效用：[pipeline.py](/Users/darkevea/code/place_pulse/src/placepulse_cusp/pipeline.py:701)。

## 4. 代码库变化审计

从先前版本到首轮运行主要有两类变化。

提交 `f23a202`：

- 从 uv/`uv.lock` 迁移到 Conda/Pip 和宽版本范围的 `requirements.txt`；
- 分析模型代码没有改变；
- 但删除锁文件降低了环境复现精度；
- 实际运行使用 PyTorch `2.7.1+cu118`，而原锁定环境是另一版本。

提交 `05efcd2`：

- 提交了 CUDA artifacts；
- 修复 KaggleHub 返回“以 CSV 命名的 ZIP”时的解压处理；
- 没有修改 M0–M3 分析逻辑；
- 结果记录的运行代码提交正确指向 `f23a202`，因为 `05efcd2` 是结果生成后的归档提交。

因此，这轮异常来自原有模型/评估实现，不是最近的 Kaggle 下载修复造成的。

## 5. 可复现性状态

做得好的部分：

- 五个外层 fold 全部完成；
- 测试图像覆盖率全部为 100%；
- 六维结果均生成；
- CUDA 设备记录完整：RTX 3060、12 GB、CUDA 11.8；
- 没有 NaN 或中途失败；
- 配置哈希和运行提交哈希一致。

仍需修正：

- 结果的 `provenance.inputs` 为空：[safety_model_comparison.json](/Users/darkevea/code/place_pulse/artifacts/cuda/metrics/safety_model_comparison.json:185)；
- GPU 端的 vote、split 与数据验证哈希没有随 artifacts 归档；
- 没有归档 `pip freeze`；
- 日志报告 CuBLAS 操作非确定性，因为运行前没有设置 `CUBLAS_WORKSPACE_CONFIG`；
- 根目录存在四组未跟踪旧结果，容易与 CUDA 结果混淆；
-自动报告仍只有一行 Results、Methods 和 Limitations，尚不具备论文报告质量。

## 6. 下一轮应如何进行

不要立刻重跑六个维度。优先顺序应是：

1. 将当前结果永久标记为 `RUN_001_DIAGNOSTIC`，保留但不覆盖。

2. 增加硬门控：

   > 如果 M1 没有在留出数据上优于 M0，则输出 `MODEL_CALIBRATION_FAILED`，不得解释标量充分性。

3. 将 M1 拆成两个基线：

   - M1a：纯图像标量 + 总体 tie；
   - M1b：图像标量 + 分层收缩的评分者左右偏差/tie 风格。

4. 为 M1、M2、M3 分别选择正则，不共享 M2 的 L2。

5. 扩大收缩搜索区间。当前结果表明至少应继续搜索 `0.1、1、10` 等数量级，但应先通过合成数据确定参数化与惩罚尺度，而不是直接根据真实结果选择最终阈值。

6. 真正执行三折内层验证和五个随机初始化。

7. 先完成三类模型恢复模拟，再碰真实数据：

   - 纯标量生成；
   - 连续异质性生成；
   - 稳定离散类别生成。

8. 只重跑 Safety。验收条件首先是：

   - M1 > M0；
   - 标量生成模拟错误否定率 ≤5%；
   - 训练/验证损失和效用尺度合理；
   - 正则最优值不再位于搜索边界。

9. CUDA 命令前设置：

```powershell
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
```

10. Safety 经审计通过后，再决定是否运行五维复制和 CUSP。


如果仍希望保留原研究方向，最有价值的下一步不是追求 CUSP 阳性，而是先把 M0–M1 基线异常修正。只有这一关通过，后续任何双峰或 CUSP 结果才有科学意义。