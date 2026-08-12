# CLDefocus PDraw 16 Profile 正式 Bank 报告

## 结论

`cldefocus_pdraw_family_16p5a_11c_3x3_k49_v1` 已完成并通过全部光学、标签、
支持域、连续性和跨光圈同谱系门禁。该资产可以进入下一步严格 matched A/B：

- A：只使用 medoid profile `F005`；
- B：使用全部 16 个 profile；
- 两组保持模型、样本、seed、步数、PD split/PDOFFSET 分布和训练合同一致。

这里的“可以训练”只表示资产合同合格，不表示它已优于已有数据或可直接晋升冠军；
最终结论仍必须来自冻结的真实 PDraw/DP5K 评估协议。

## 资产范围

- 冻结镜头：`prime/JP2018-205527_Example01P.mytable`；
- 镜头 SHA256：`44ecff38064ae75429406eb466343ac2eeeec61a078bbde5dd67ad691dc7303a`；
- profile：16 个独立低阶像差、pupil response、throughput 与 centroid-slope 参数；
- aperture：`f/1.8, f/2.0, f/2.8, f/4.0, f/5.6`；
- signed CoC：`[-8,-4,-2,-1,-0.5,0,0.5,1,2,4,8] px`；
- field：`3×3`，x/y 均为 `[-0.65,0,0.65]`；
- support：57×57 reference propagation，门禁后中心裁为 49×49；
- combined shape：`[16,5,11,3,3,2,49,49]`。

所有 profile 固定同一镜头和同一组 ray geometry。低阶相位屏不含 defocus；
CoC 仍由 sensor propagation 定义。PDOFFSET 不进入 kernel：

```text
disp_gt = analytic_centroid_disparity_px + independent_pd_offset_px
```

## 门禁结果

16/16 profile 全部通过：

| 指标 | 全 profile 范围 |
|---|---:|
| 57→49 reference retained mass min | 0.997063 ～ 0.998000 |
| combined support retained mass min | 0.997013 ～ 0.997952 |
| final energy | 0.99999970 ～ 1.00000024 |
| CoC=0 centroid disparity abs max | 3.44e-8 ～ 9.97e-8 px |
| centroid target abs max error | 9.69e-8 ～ 1.00e-7 px |
| adjacent-CoC PSF L1 mean | 0.8687 ～ 0.8874 |
| adjacent-CoC PSF L1 max | 1.9464 ～ 1.9557 |
| raw physical centroid/CoC slope | 0.1114 ～ 0.2746 px/CoC |

固定 49×49 support 下，一次 centroid translation 会因极少量边缘质量损失留下约
0.001～0.0025 px 残差。因此最终实现从导出 kernel 的解析质心闭环校正，最多使用
2 次 refinement；没有扩大 kernel，也没有用 NCC 或真实数据修正标签。

## Matched single-profile medoid

medoid 为 `F005`。选择方法是把 20 个独立随机轴分别按配置范围归一化到 `[0,1]`，
计算到 family 中心 `0.5` 的欧氏距离，并只在正式门禁通过且 support/centroid 距硬
门限至少保留 5% 归一化余量的 profile 中选择。

- 到 family 中心的归一化距离：`0.992632`；
- 最小 support/centroid 归一化余量：`0.591648`；
- 16 个 profile 全部满足非门禁边缘要求；
- `F005` 单 profile 7D SHA256：
  `28d83f0d0bfed229a8371cd5bd9283b15943beebd9658459cf466155b970a3c1`。

完整轴坐标、候选排序和逐门禁余量位于 `profile_manifest.json` 与 `summary.json`。

## 性能与恢复

正式传播分为两个 8-profile batch：

- `F000–F007`：926.55 s；
- `F008–F015`：928.30 s；
- source propagation 总计：1854.85 s（30.91 分钟）；
- 每批 495 个共享 reference cell，约 1.87 s/cell。

每批完成后原子写入 `raw_chunks/chunk_*.pt`，再生成 SHA sidecar。恢复时严格核对
config、family、lens、profile IDs、profile 参数 SHA、kernel 和 tensor shape。最终从
cache 恢复并完成门禁与全部格式导出耗时约 43.75 s。

## 输出与兼容性

外部根目录：

```text
/mnt/data/lsj/artifacts/pdraw-stereo-v2/render/cldefocus/
cldefocus_pdraw_family_16p5a_11c_3x3_k49_v1
```

主要文件：

- `psf_bank.pt`：combined 8D bank；
- `combined_psf_bank.npy`：combined NumPy archive；
- `profiles/Fxxx/psf_bank.pt`：parent loader 可直接消费的单 profile 7D bank；
- `profiles/Fxxx/psf_bank.npy`：对应 7D NumPy array；
- `profiles/Fxxx/pdraw_mono_v1/f*/`：逐 aperture 的标准 shape-only bank；
- `profile_manifest.json`：profile 参数、SHA、medoid 和逐 profile 路径；
- `artifact_manifest.json`：全部 204 个版本化资产文件的 SHA；
- `summary.json`、`preexport_diagnostics.json`、`provenance.json`。

验证结果：

- artifact manifest：204/204 SHA 匹配；
- 单 profile 7D parent-loader：80/80 aperture 组合通过；
- pdraw_mono_v1 parent-loader：80/80 通过；
- combined bank profile 选择：通过；
- 子模块测试：7/7；
- parent CLDefocus + pdraw_mono 回归：34/34。

关键 SHA256：

- combined `psf_bank.pt`：`611c220750e78d5abdb4b518784a1724773e3fefc353bf2f2574f913367af493`；
- combined `.npy`：`c514aff601104740bbd2a3ef99555d310fe864566ceef2d931ca4039038caa03`；
- `profile_manifest.json`：`c32b60dd3760bb10889d6a44e19bfb22bd69d1fd0f024af0d6202f7ae97879b0`；
- `artifact_manifest.json`：`d00cf0c92550990abde3fc56c183421245bfc4270a345f82d36a8e97c9c64a52`；
- `summary.json`：`dfc78f2c5d39a21db30f89516a8b498e3f380fac8b63ca015eb88de5f9a907c8`；
- production config：`00e4bbaf8dd40d7be0ccf66a58558e0b0a103df098fc87e9fbcb7c8628fa50c1`。

数据隔离：未读取 Google dev/holdout、DP5K 或真实 PDraw，未启动 stereo training。
