# CLDefocus PDraw 同族多 Profile Bank

## 目标

`pdraw_adapter/family_bank.py` 从一份冻结的复合镜头处方离线生成同一光学
family 内的多 profile、多光圈 PDraw PSF bank。它不读取真实 PDraw、Google 或
DP5K，不启动训练，也不使用 NCC 改写标签。

本实现把随机因素拆成四组互相独立的轴：

1. pupil/sensor response：PD 分界、transition、cross-talk 和 throughput；
2. 低阶像差：astigmatism、coma、spherical 及缓慢 field variation；
3. PD 分离强度：`centroid_slope_px_per_coc`；
4. aperture：同一镜头 stop 的五档真实 f-number，并允许 slope 随光圈缩放。

所有随机轴使用独立的命名 RNG 和一维 LHS。只修改一个轴的范围，不会改变其他轴
已经采样的数值。每个 profile 保存完整参数和 `parameter_sha256`，整个 family 另有
`family_sha256`。

## 标签约定

统一采用：

```text
d = x_L - x_R
disp_gt = analytic_centroid_disparity_px + independent_pd_offset_px
```

`analytic_centroid_disparity_px` 始终从最终导出 PSF 的左右解析质心重新计算。低阶
相位屏不包含 defocus；CoC=0 时 morphology bank 的解析左右质心差被校准为 0。

PDOFFSET 是独立的相机/标签项，绝不烘进 zero-centroid morphology，也不会被 NCC
估计覆盖。因此 CoC=0 可以通过非零 PDOFFSET 得到非零最终标签，同时 bank 本身保持
不变。这正是部署时应由显式标定量控制的行为。

## 传播复用

每个 `aperture × field × CoC` cell 只构造一次波前、sensor sample 和
Rayleigh–Sommerfeld Green 几何；同一 cell 内全部 profile 的左右复振幅在一个 batch
中传播。不同 profile 使用同一镜头处方，仅叠加各自的 pupil response 与不含 defocus
的低阶相位屏，不混用不相关镜头资产。

## 冒烟运行

必须使用 Genfocus：

```bash
cd /mnt/data/lsj/model/pytorchlightning/render/CLDefocus
/home/wang/anaconda3/envs/Genfocus/bin/python -m pdraw_adapter.family_bank \
  --config configs/pdraw_family_smoke_v1.yaml
```

冻结冒烟配置生成 `2 profile × 5 aperture × 3 CoC × 2 field`。输出位于仓库外部，
包含：

- `psf_bank.pt`：8 维 bank、解析标签、throughput 和 profile metadata；
- `profile_manifest.json`：逐 profile 参数与 SHA256；
- `asset_recipe.yaml`：下游引用与 PDOFFSET 合成契约；
- `summary.json`：速度、support、energy、centroid、CoC 连续性门禁；
- `resolved_config.yaml`、`provenance.json`、`artifact_manifest.json`。

冒烟只验证资产构造契约，不代表真实域收益；通过后仍需由上层固定训练协议单独做
ablation。
