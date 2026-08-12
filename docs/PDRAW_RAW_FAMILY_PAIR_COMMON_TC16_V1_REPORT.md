# PDraw raw-family pair-common tc16 continuous NCC 审计报告

## 结论

状态：**PASS**。

冻结 raw-optical family 的原始左右 PSF 在 small `|CoC|<=1 px` 区域存在稳定的
NCC apparent-disparity 压缩，中心视场 16 profiles 的 slope 约为
`0.255–0.666`。简单 shifted-impulse 或 bilinear pair-common 虽保持解析 centroid，
仍不能通过 `slope in [0.95,1.05]` 门禁。

最终准入候选采用：

- L/R 各自按原 centroid 去中心；
- zero-centered L/R shape 直接求共同形态，不做镜像；
- 对共同形态施加左右相同、零 centroid 的 `sigma=1.0 px` Gaussian MTF；
- 用 Fourier phase shift 把共同形态放回原生 `mu_L/mu_R`，并审计非负投影质量；
- 按 `smoothstep(|CoC|/16)` 混回 raw PSF；
- 每侧能量、原生 optical centroid、profile/aperture/field variation 均保留；
- PDOFFSET 始终在资产外由 parent 独立加入；
- analytic label 只从最终 kernel centroid 重算，NCC 只作诊断准入。

正式资产：

`/mnt/data/lsj/artifacts/pdraw-stereo-v2/render/cldefocus/cldefocus_pdraw_family_16p5a_11c_3x3_k49_paircommon_s1_tc16_contncc_v1`

## 分级诊断结果

### Center K64

- profile：16/16 五档光圈全过；
- profile×aperture：80/80；
- slope：`0.997684909–1.004037911`，均值 `0.998801274`；
- R² 最低：`0.999983149`；
- 用时：`33.653 s`。

### 3×3 full-field K64

- profile×aperture×field：720/720；
- slope：`0.978144853–1.019841791`，均值 `1.001795950`；
- R² 最低：`0.999705515`；
- 最差 slope cell：`F009 / aperture_index=3(f/4) / field=(0,0)`，
  slope `0.978144853`、R² `0.999899331`；
- 用时：`314.811 s`。

### 3×3 full-field K128

- profile×aperture×field：720/720；
- slope：`0.989063868–1.019722544`，均值 `1.002514255`；
- R² 最低：`0.999840399`；
- 用时：`927.879 s`。

K128−K64 收敛统计：

- slope 差：`[-0.002740098,+0.010919015]`；
- slope 绝对差均值/P95：`0.000900773/0.002651295`；
- R² 绝对差均值/P95：`0.000004007/0.000013058`；
- 最大 slope 差仍位于 `F009/f4/field(0,0)`，从 `0.978144853`
  收敛至 `0.989063868`。

逐 cell 的 K64/K128 对齐结果保存在正式资产的
`k64_k128_field_convergence.csv`。

## 资产合同

- 形状：每 profile `[5,11,3,3,2,49,49]`；
- profile 数：16；光圈数：5；
- `centroid_policy=native`；
- `centroid_variant=paircommon_direct_s1_tc16_contncc_v1`；
- `analytic_label_source=final_kernel_centroid_mu_left_x_minus_mu_right_x`；
- `pdoffset_embedded=false`；
- `ncc_role=diagnostic_admission_only_never_label_writeback`；
- 最终/源 optical centroid 最大误差：`1.8471e-7 px`；
- Parent loader 后 kernel/label 最大误差：`3.5763e-7 px`；
- Fourier 非负投影质量最大值：`0.0046211`；
- Parent profile×aperture loader：80/80 PASS；
- Parent 首 profile 五档 `physical_psf_fast` renderer smoke：5/5 PASS。

Parent 应直接按光圈加载 `profiles/Fxxx/psf_bank.pt`，使用 bank centroid，并保持
`focus_zero_calibrate=false`。不得把 source raw profile 中仅供 export-retarget 的
`centroid_slope_px_per_coc` 解释为 raw-active optical randomization axis。

## 测试

统一使用 Genfocus：

```bash
PYTHONPATH=/mnt/data/lsj/model/pytorchlightning:/mnt/data/lsj/model/pytorchlightning/render/CLDefocus \
  /home/wang/anaconda3/envs/Genfocus/bin/python -m pytest -q tests
```

结果：`14 passed`。

Parent 相关回归：

```bash
PYTHONPATH=/mnt/data/lsj/model/pytorchlightning:/mnt/data/lsj/model/pytorchlightning/render/CLDefocus \
  /home/wang/anaconda3/envs/Genfocus/bin/python -m pytest -q \
  tests/test_cldefocus_continuous_ncc_v2.py \
  tests/test_cldefocus_pdraw.py \
  tests/test_dp_renderer.py
```

结果：`68 passed`。

子模块根目录裸 `pytest -q` 会在收集四个上游 `deblurring/scripts/test_*.py` 时，
因可选的 `deblurring.models` 包未安装而失败；该错误发生在本次测试之外的上游
deblurring 示例脚本收集阶段。本次相关测试均已显式通过。

## SHA256

- 正式资产 `artifact_manifest.json`：
  `e1836368418a2c802b085e7d2a1b1bbe1d83331a58fa545082dfff7fd66326a7`
- 正式资产 `summary.json`：
  `780dfee211b5eb0c220acda4fb6d2ae6e21879d74840f7e84498a5ad87890744`
- 正式资产 `REPORT.md`：
  `66042539b6055381440eb4b239dbf679f8f525a103bd66f027195ab700ee76da`
- `k64_k128_field_convergence.csv`：
  `92468db58c5a2c9cdb8f6b7e817d75c706349d9594fde6d7fd3b6ebbe5ccc381`
- K64 `screens.json`：
  `91e1a685348c9abf1342c589da9a48afeb65f5d0fe39aab18e4ef40e27ea778d`
- K128 `screens.json`：
  `939a2f02d3884ab4d565788cf433a81f97ffddac517632131d23d53abb1d6e30`

本次未访问真实 PDraw、Google dev/holdout 或 DP5K，未训练模型，也未修改 parent
仓库的数据集或训练配置。
