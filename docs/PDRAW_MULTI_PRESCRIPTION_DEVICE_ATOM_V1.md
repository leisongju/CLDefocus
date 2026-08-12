# PDraw multi-prescription device-atom v1 CPU 规划

## 目标与边界

本阶段把 PSF 多样性从“同一镜头 family 的 morphology scale”改为四个不同复合镜头
处方。当前只完成 CPU 处方筛选、近轴/JAX 等价 smoke、五光圈 lineage 合同和后续产物
manifest；没有启动 GPU raytrace、PSF 导出、continuous NCC 或 stereo 训练。

正式 device atom 固定以下合同：

- 五档光圈为 `f/1.8、2.0、2.8、4.0、5.6`，必须由同一处方的原生 stop 仅收缩得到；
- 每个镜头首期只使用同一个中性 sensor head，固定 `DCC=1/6 px/CoC`；
- PairCommon 固定为 direct mean、共同 `sigma=1 px` MTF、`tc16`，并保留原生
  `CoC=0` optical bias；
- label 只从最终左右 kernel centroid 解析；NCC 只作准入诊断，不回写 label；
- `PDOFFSET` 永远位于 bank 外，标签合同为
  `disp_gt=analytic_centroid_disp+independent_pd_offset_px`；
- 当前状态一律为 `not_admitted`，完成 K64/K128 full-field continuous NCC 前不得训练。

## CPU 选镜结果

CPU selector 扫描 `prime/*.mytable` 共 946 个处方。159 个处方可在不放大原生 stop 的
前提下覆盖五档目标光圈；再按参考处方的 EFL/image-radius envelope 筛得 9 个候选，使用
冻结的 reference-first compatible-maximin 规则选择以下四个：

| 顺序 | lens ID | 原生 f/# | EFL (mm) | image radius (mm) | surface/asphere | stop index |
|---:|---|---:|---:|---:|---:|---:|
| 0 | `JP2018-205527_Example01P` | 1.325346 | 16.446231 | 14.200000 | 33/2 | 18 |
| 1 | `US08503096-1` | 1.546101 | 25.224884 | 10.969512 | 15/3 | 5 |
| 2 | `CN107272157_Example01P` | 1.002319 | 25.939240 | 10.840000 | 24/0 | 7 |
| 3 | `US20200371325_Example02P` | 1.680543 | 10.298606 | 10.815000 | 31/6 | 15 |

四个 lens 文件 SHA256 已写死在
`configs/pdraw_multi_prescription_device_atom_v1.yaml`。规划器先用纯 NumPy 复现
CLDefocus 的近轴矩阵，再只对四个选中处方用 JAX CPU 复核 EFL、f-number 和
`build_aperture_plan`；相对误差门为 `1e-9`。

## 规划产物

版本化入口：

```bash
PYTHONPATH=/mnt/data/lsj/model/pytorchlightning/render/CLDefocus \
  /home/wang/anaconda3/envs/Genfocus/bin/python \
  -m pdraw_adapter.multi_prescription_plan \
  --config configs/pdraw_multi_prescription_device_atom_v1.yaml
```

外部 CPU plan 根目录为：

`/mnt/data/lsj/artifacts/pdraw-stereo-v2/render/cldefocus/cldefocus_multi_prescription_device_atom_4l_dcc6_v1_cpu_plan`

其中：

- `lens_catalog.jsonl`：946 个输入处方的通过/拒绝账本；
- `selected_lenses.json`：四个选中处方、近轴参数和 JAX CPU 等价结果；
- `device_atom_plan.json`：四个共享五光圈 lineage 的 staged GPU 输出计划；
- `artifact_manifest.json`：所有小型 CPU 规划产物的 SHA256，并显式声明
  `gpu_propagation_started=false`、`training_admitted=false`、`pdoffset_embedded=false`。

冻结 SHA256：

- `artifact_manifest.json`：
  `3adff4d37e0960556163dda07d227024fd22dd3d42ca8e6072fdd43d96795b3f`；
- `summary.json`：
  `7ba58d27f93b60edbe013439b33c92f4ef6ac76f2d219edaa07c43b6ca39c119`；
- `selected_lenses.json`：
  `5ef2136bc78391256d72c5833b184c10483fbbf12fbba87ef5ded6f03d0df807`。

## 后续 GPU 阶段

正式执行使用版本化入口
`configs/pdraw_multi_prescription_execute_4l_dcc6_v1.yaml`。执行器先复核 CPU plan、
CLDefocus 六个传播/变换实现和 parent loader/renderer 的 SHA256，再为每个 lens 依次执行：
raw family propagation、raw-optical export、DCC6 retarget、PairCommon、K64 full-field、
K128 full-field 和 parent loader/render smoke。每个 atom 必须覆盖
`5 aperture × 11 CoC × 3×3 field × 2 side`，且五光圈必须共享同一个
`shared_lineage_id`。中断后可以从已经通过 manifest SHA256 校验的 stage 继续。

只做 CPU/静态准备、绝不触发 propagation：

```bash
PYTHONPATH=/mnt/data/lsj/model/pytorchlightning/render/CLDefocus \
  /home/wang/anaconda3/envs/Genfocus/bin/python \
  -m pdraw_adapter.multi_prescription_execute \
  --config configs/pdraw_multi_prescription_execute_4l_dcc6_v1.yaml \
  --prepare-only
```

正式执行单个 atom 时增加 `--atom-id MPxx_...`；不指定 `--atom-id` 时按 CPU plan
冻结顺序执行全部四个。执行器会在外部 atom 根目录的 `generated_configs/` 中保留每个
解析后子配置。只有四个 atom 的 raw、K64、K128、PairCommon 及 parent smoke 全部通过，
聚合 `artifact_manifest.json` 才允许写出 `training_admitted=true`；任何部分完成、门禁失败
或中断状态都保持 `false`。

现有 JP2018 family16 的两个 8-profile raw chunk 各约 `927 s`，折算单 profile 约
`116 s`，但四个新处方会分别触发 JAX 编译。预计四镜头 GPU propagation 合计约
`12–25 min`；加 PairCommon 导出、K64/K128 continuous NCC 与 parent smoke，完整资产
准入约 `20–40 min`。该估计会在第一个 lens 完成后按实测更新。

本阶段没有访问真实 PDraw、Google dev/holdout 或 DP5K，也没有启动 stereo 训练。

## 正式执行进度

`MP00_JP2018-205527_Example01P_DCC6` 已完成整条链路并通过：

- raw propagation：`495` 个 reference cells，`916.145 s`；原子 chunk SHA256 为
  `de1c47f86f5ac381f3a7a3c820be2d9c86cabce138e352adfdecaee99c7c3267`；
- K64：`45/45` aperture-field 全过，slope 为 `0.993016–1.007370`，最低 R²
  `0.999851`；
- K128：`45/45` aperture-field 全过，slope 为 `0.993315–1.007278`，最低 R²
  `0.999891`；
- K64→K128 slope 差绝对值 P95 为 `0.0007684`；
- 最终 PairCommon+DCC6 的 parent loader/render 为 `5/5` 组合全过，kernel/label
  centroid 最大误差 `2.384e-7 px`，没有执行 dense centroid retarget；
- 最终 asset manifest SHA256 为
  `96571e56972c45b4fff7663cc124cee746053cf9ccdbd17e194d9abff9777879`。

当前按 GPU 调度要求暂停在 `1/4` atom；聚合状态仍是
`in_progress_not_admitted`，`training_admitted=false`。未启动 MP01，也没有访问任何真实
数据。候选导出器原先有一个只影响报告的旧硬编码 `80/80`，真正 parent gate/manifest
从始至终均为 `5/5`；该字段已改为按实际组合数生成，旧包保留在带
`superseded_hardcoded_80of80` 后缀的外部目录以便审计。

MP01 首次 setup 暴露出处方记录的 image plane 离实际零 CoC 约 `2.58 mm`，统一
`±1.5 mm` sensor-offset 求根 bracket 无法包围目标。该失败发生在 raytrace 前，没有生成
raw chunk。随后用 JAX CPU 对 MP01 的 `5 aperture × 3×3 field` 检查 `±4 mm` 端点：
45/45 cell 均完整包围 `[-8,+8] CoC`，最窄一端仍有 `+17.757/−125.655 px`。
因此只为 MP01 冻结 `4 mm` 求根 envelope；它只扩大数值求根 bracket，不改变最终 CoC
bins、sensor head、DCC、PairCommon 或标签定义。

MP01 随后完成 495-cell raw propagation（`912.657 s`，chunk SHA256
`1a0530b0d10fcb53857d0deeb44440a047405f126b24e74c5e1ea3d69eac6ffc`），raw-family、
raw-optical 和 parent smoke 通过；但 K64 的四个角场存在 `0.913–3.184 px` 左右核垂直
centroid 差，超过冻结 `0.75 px` vertical gate。20/45 aperture-field 的五个 small-CoC
记录全部无效，aggregate valid fraction 只有 `0.555556`。因此 US08503096-1 已记为正式
负向镜头，禁止放宽 gate、禁止训练。

从原 compatible pool 排除该失败 lens 后，使用相同 reference-first deterministic
compatible-maximin 规则得到首个替补 `prime/JP2013-003324_Example02P.mytable`，SHA256
为 `49130b317e3b719bd78e566ca1b1ee53bb3d51995a08ab7f5c65738acb7bbcc1`。替补先执行
`1 aperture × 5 small-CoC × 3×3 field = 45 cells` 的冻结 K64 screen，通过后才允许
运行完整 495-cell 链路。

JP2013 替补的 45-cell screen 已通过：9/9 field PASS，slope
`0.980945–1.003006`，最低 R² `0.999887`。随后完成正式 495-cell 链路：

- raw propagation：`911.241 s`，chunk SHA256
  `9fd9084af4a26d573b41d03c179c46ea717b3ce31620648632782c6c77fa6b92`；
- K64：45/45 field PASS，slope `0.980945–1.010519`，最低 R² `0.999886893`；
- K128：45/45 field PASS，slope `0.988981–1.010393`，最低 R² `0.999890896`；
- K64→K128 slope 差绝对值 P95：`0.0038312`；
- parent：5/5 loader + 5/5 render smoke PASS，centroid 最大误差 `2.384e-7 px`，
  没有 dense retarget；
- candidate bank SHA256：
  `db789074b1d2f4da23f93f19d2f72924424a8fd5843f0f280a0134515adfcffc`；
- final asset manifest SHA256：
  `16c5ceccb52b94713551d92913dc6ddc4be479a4f7f351f2e73910cc1c413e1b`。

当前正式进度为 2/4 active atoms 通过，聚合仍严格保持
`training_admitted=false`；MP02 尚未启动。
