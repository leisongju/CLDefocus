# PDraw pair-common 固定 DCC 全场审计

## 结论

在冻结 raw-optical family16 上，`DCC=1/6` 与 `DCC=1/5 px/CoC` 均通过 K64/K128
3×3 全场 continuous NCC 门禁，并已导出 parent 可直接加载的外部资产。该变换只调整最终
L/R PSF 的相对 centroid slope，保留各 profile×aperture×field 的原生 `CoC=0` optical
bias；独立 PDOFFSET 不写入资产。

NCC 的角色仅是检查生成图的 apparent shift 是否跟最终 kernel 解析 centroid 一致。它不
回写 kernel、不生成训练 label，也不进入训练网络或手机部署。

## 三个 DCC 候选

| 候选 | DCC (px/CoC) | K64 profile 全过 | K64 slope / R² min | K128 profile 全过 | K128 slope / R² min |
|---|---:|---:|---:|---:|---:|
| DCC7 | 1/7 | 14/16 | 0.990480–1.021343 / 0.999658 | 16/16 | 0.990526–1.016768 / 0.999802 |
| DCC6 | 1/6 | 16/16 | 0.991396–1.018436 / 0.999753 | 16/16 | 0.991481–1.014515 / 0.999858 |
| DCC5 | 1/5 | 16/16 | 0.992449–1.011434 / 0.999873 | 16/16 | 0.992612–1.012231 / 0.999899 |

DCC7 的 K64 两个 profile 只在最严格的符号/阈值门禁上失败，因此不导出；DCC6 与 DCC5
各自覆盖 `16×5×3×3=720` 个 field cell 并全部通过。

## 正式外部资产

### DCC6

- 根目录：
  `/mnt/data/lsj/artifacts/pdraw-stereo-v2/render/cldefocus/cldefocus_pdraw_family_16p5a_11c_3x3_k49_paircommon_s1_tc16_dcc6_contncc_v1`
- manifest/summary/REPORT SHA256：
  `25155c61…99bc` / `9d0061d9…154e` / `890772e5…9a3b`。
- K64→K128 slope 绝对差均值/P95：`0.000618/0.001452`。
- Parent profile×aperture：80/80；kernel/analytic label 最大差：`3.576e-7 px`。
- export-time DCC target 最大误差：`9.995e-8 px`。

### DCC5

- 根目录：
  `/mnt/data/lsj/artifacts/pdraw-stereo-v2/render/cldefocus/cldefocus_pdraw_family_16p5a_11c_3x3_k49_paircommon_s1_tc16_dcc5_contncc_v1`
- manifest/summary/REPORT SHA256：
  `333103cc…8b0f` / `d136aa05…21d` / `a2771ebd…985c`。
- K64→K128 slope 绝对差均值/P95：`0.000558/0.001275`。
- Parent profile×aperture：80/80；kernel/analytic label 最大差：`3.576e-7 px`。

大 PSF 数组不纳入 Git；本仓库只保存可复现配置、代码、路径、资产名和哈希。

## 训练判定

DCC6 已在 parent V36 与 native-DCC 的 V34 做 matched 125-step 消融：DP5K test EPE
`0.256443` 对 `0.256367`，差异只有 `+0.000076 px`。因此固定 DCC6 不晋级，DCC5 暂不因
renderer 门禁通过而自动启动训练。结论只说明一个全局 DCC 常数不是当前主要 sim2real
瓶颈，不否定未来按设备/光圈分布采样 DCC。

## 审计身份

- K64 summary/provenance/screens：
  `5c57fdc1…f7d` / `23eb499e…8d0` / `424acdbe…703`。
- K128 summary/provenance/screens：
  `d5a1062e…c48` / `783e47d6…7af` / `ad7783ec…ed7`。
- 真实 PDraw、Google dev/holdout、DP5K 访问均为 0；审计没有启动 stereo training。
