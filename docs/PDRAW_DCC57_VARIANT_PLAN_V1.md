# PDraw DCC5/DCC7 variant CPU 规划 v1

## 正式执行结果

2026-08-12 已使用 `configs/pdraw_dcc57_variant_execute_2a_v1.yaml` 完成冻结 formal
CPU 流程。四个候选的 K64/K128 均为 `45/45 PASS`，且四个 bank 均通过 parent
loader/renderer 的 `5/5` 光圈 smoke；没有失败项，也没有放宽任何门禁。

| atom/variant | K64 slope / R² min | K128 slope / R² min | `|Δslope|` mean/P95 | `|ΔR²|` mean/P95 |
|---|---|---|---|---|
| MP00 JP2018 / DCC5 | `0.993720–1.006121` / `0.999899015` | `0.993967–1.006077` / `0.999929065` | `0.000529924/0.000732917` | `5.319e-6/2.050e-5` |
| MP00 JP2018 / DCC7 | `0.992322–1.008635` / `0.999794956` | `0.992678–1.008493` / `0.999844740` | `0.000616618/0.000800356` | `1.004e-5/3.802e-5` |
| MP01 JP2013 / DCC5 | `0.984069–1.009187` / `0.999914513` | `0.990784–1.009074` / `0.999918039` | `0.000870280/0.003230599` | `4.822e-6/2.855e-5` |
| MP01 JP2013 / DCC7 | `0.977788–1.011758` / `0.999855570` | `0.987149–1.011618` / `0.999860153` | `0.001110600/0.004430539` | `8.764e-6/5.761e-5` |

正式总账：

`/mnt/data/lsj/artifacts/pdraw-stereo-v2/render/cldefocus/cldefocus_dcc57_variants_2a_v1`

- aggregate manifest SHA256：`a1feb81d1a0df8775a23d7820426b0e134671d137595cfed487b6c6e57fce9ec`；
- aggregate summary SHA256：`93a17eca2e3dd2e7b406e9132aa941f8e34b7be58fec75b465d51e4958e0af8e`；
- `candidate_pass_count=4/4`，`training_admitted=true`；该准入只针对本节四个
  冻结 variant bank，不改变其他样本集或实验的 NO-GO 结论。

正式 bank 与 manifest SHA256：

- MP00/DCC5：`MP00_JP2018-205527_Example01P_DCC6/dcc5/paircommon`，
  `d45affbf4182e4b9fc35ec22cea877d8e627a182675e3592df983bf5c14c9cc9`；
- MP00/DCC7：`MP00_JP2018-205527_Example01P_DCC6/dcc7/paircommon`，
  `7bd7c5d15b12a848bed723539c56530137383f6bd7e7446b4287c4cc8faf79ed`；
- MP01/DCC5：`MP01_JP2013-003324_Example02P_DCC6/dcc5/paircommon`，
  `b5c89b878f89af36f79ee2a637181a89533197acfaf87402a03f11019a543169`；
- MP01/DCC7：`MP01_JP2013-003324_Example02P_DCC6/dcc7/paircommon`，
  `9b3b70091e8085e522e2cd22394cf4cd10b00a1fb1beefd030cbcc456646fade`。

执行器强制要求 `CUDA_VISIBLE_DEVICES=''`。本次总账记录
`gpu_used=false`、`new_optical_propagation_run=false`、`pdoffset_embedded=false`；
未访问真实 PDraw、Google dev/holdout 或 DP5K，未启动训练。

## 结论

DCC5 和 DCC7 可以低成本从两个已冻结的 `raw-optical` atom 独立分叉，无需重新运行
CLDefocus 光学传播，也不应从 DCC6 成品做二次 retarget。独立从 raw 分叉可以保留每个
aperture×field 的原生 `CoC=0` optical bias，避免把 DCC6 的插值与 PairCommon 结果再次
卷入 DCC5/DCC7。

冻结合同保持不变：

- DCC5 slope 为 `1/5 px/CoC`，DCC7 slope 为 `1/7 px/CoC`；
- `anchor_mode=preserve_native_coc0`；
- PairCommon 为 direct mean、bilinear center、Fourier nonnegative place、共同
  `sigma=1 px` MTF、`tc16`；
- 解析标签始终从最终 kernel 的 `mu_left_x-mu_right_x` 重算；
- NCC 只用于准入，绝不修改 kernel、centroid 或 label；
- PDOFFSET 不嵌入 bank，仍使用
  `disp_gt=analytic_centroid_disp+independent_pd_offset_px`。

## CPU 静态复核

输入是已通过正式 DCC6 K64/K128 与 parent smoke 的两个 active atom：

- `MP00_JP2018-205527_Example01P_DCC6`；
- `MP01_JP2013-003324_Example02P_DCC6`。

规划器先从 raw-optical 重放 DCC6。两个 atom 的重放 kernel 与解析 label 均和现有正式
DCC6 bank **bitwise 一致**，最大误差均为 `0.0`。这验证了 DCC5/DCC7 计划使用的变换
顺序、数值实现和来源 lineage 与已准入 DCC6 相同。

四个新候选的静态结果如下：

| atom | variant | centroid 合同最大误差 | retained mass min | Fourier negative fraction max | energy 最大误差 |
|---|---|---:|---:|---:|---:|
| MP00 JP2018 | DCC5 | `9.991e-8 px` | `0.9999620` | `0.0038603` | `6.999e-8` |
| MP00 JP2018 | DCC7 | `9.752e-8 px` | `0.9999667` | `0.0038761` | `6.346e-8` |
| MP01 JP2013 | DCC5 | `9.414e-8 px` | `0.9999858` | `0.0034398` | `6.779e-8` |
| MP01 JP2013 | DCC7 | `9.932e-8 px` | `0.9999822` | `0.0031767` | `6.786e-8` |

这些静态结果本身只证明确定性变换的数值可行性；正式准入由上节记录的 K64/K128
审计与 parent smoke 决定。CPU plan 内保留的
`pending_k64_k128_not_admitted` 是执行前快照，不应当作正式输出的当前状态。

## 版本化入口与产物

版本化入口：

- `configs/pdraw_dcc57_variant_prepare_2a_v1.yaml`；
- `pdraw_adapter/dcc_variant_plan.py`。

CPU plan：

`/mnt/data/lsj/artifacts/pdraw-stereo-v2/render/cldefocus/cldefocus_dcc57_variant_prepare_2a_v1_cpu_plan`

manifest SHA256：

`41978eaabe29f65076b18aacac1e3cfe4b5b22740f8d8025f4ba6c9a65be9ece`

该 plan 已为 `2 atom × 2 DCC × K64/K128` 生成八份可被现有
`raw_family_continuous_audit` 严格加载的 YAML，并为四个候选生成
`candidate_export_recipe.json`。manifest 明确记录：

- `new_optical_propagation_required=false`；
- `formal_k64_k128_run=false`；
- `variant_assets_exported=false`；
- `training_admitted=false`；
- `pdoffset_embedded=false`。

## 后续成本与最小执行改动

已有 DCC6 实测中，两个 atom 的 K64 分别耗时 `20.14/20.10 s`，K128 分别耗时
`57.69/58.43 s`。据此估算四个 DCC5/DCC7 候选的八次 formal audit 合计约
`313 s CPU`；加四次候选变换、SHA 校验、parent 5-aperture loader/render smoke 和落盘，
空闲 CPU 下约 `6–7 min`，与训练/DataLoader 争用时按 `8–12 min` 预算。GPU 时间为
`0`，新 optical propagation 时间为 `0`。正式输出新增约四个 19 MB 级 PSF bank，外加
审计 JSON/CSV。

现有 `raw_family_continuous_audit.py` 和 `raw_family_candidate_export.py` 已支持所需操作，
不需要修改 renderer 或变换算法。最小正式执行流程是：

1. 逐一运行 CPU plan 中的八份 audit YAML；
2. 读取每份 `summary/provenance/screens` 的 SHA256；
3. 把 SHA 写入对应 export recipe 生成四份 candidate-export YAML；
4. 调用现有 candidate exporter 完成 parent smoke；
5. 只有单个候选 K64/K128 45/45 field 全过时，才允许其独立 manifest 置为 PASS。

若希望全自动执行，只需新增一个小型 orchestration helper 来完成第 2–4 步的 SHA 解析与
断点续跑；无需改动 PSF 算法、parent loader 或 renderer。

本节描述的是执行前成本估算；实际 formal CPU 执行耗时 `326.329 s`，估算有效。
