# 解析 parity fixture（LI → Lumice）

[English](analytic-parity-fixtures.md)

Lumice Integral（LI）按固定 rev 导出一个 JSON fixture 目录，Lumice 把它拷进自己的仓库，在 CI 里对
`liblumice_analytic` 模块 A v0 回放（Lumice `doc/analytic-api.md` §4：`EvaluatePath`、seed 搜索、
`TraceFiber[Batch]`，只返回点列）。方向是单向的（owner 2026-09-28 裁定）：LI 是参考，Lumice 跟随。
本页的写法要让 Lumice 侧不读 LI 源码就能写出读取器。fixture 背后的规范语义是
`docs/phase1-math-contract.md` §9（continuation）与 §9.5（discovery），约定以 `docs/conventions.md` 为准。

## 1. 生成

```bash
uv run python scripts/export_analytic_parity.py --output-dir artifacts/analytic-parity --verify
```

- 一条命令写出矩阵（§6）的全部 fixture 和一个 `manifest.json`，M2 Max 上全矩阵约 16 s。
- **确定性**：同一 rev 重跑逐字节相同。LI 的 `tests/test_parity_export.py` 在两个独立解释器里各导出一次，
  逐文件比较。fixture 不记录时间、主机和路径。
- `--verify` 读回每个 fixture，用当前 checkout 重算，按 fixture 自带容差（§5）比较，任一失败即非零退出。
  `--cells 3-5__random ...` 只导出子集。
- 产物在 `artifacts/` 下，LI 不纳入版本控制。钉住的那一份在 Lumice 仓库里，每个 fixture 自己记录来源的
  LI rev。
- 代码：`lumice_integral.parity_export`（格式、选点、校验器）与 `scripts/export_analytic_parity.py`
  （矩阵定义，瘦 CLI）。

## 2. 文件与公共字段

一个格（cell）是矩阵的一行，即一条路径加一个点类别。它的文件名为
`<path>__<category>__<kind>[__<label>].json`：路径用连字符（`3-5-6-7`），类别为 `random` / `critical` /
`near_boundary`，种类为 `evaluate_path` / `trace_fiber` / `seed_search`。`manifest.json` 列出每个格的
`files`、带原因的 `skipped`、`rationale` 以及 §6 的 `selection` 记录。

JSON 为 UTF-8，键排序。浮点写成能往返回同一个 float64 的最短形式，任何符合规范的 JSON 读取器
（`strtod`）解析后都得到导出时的原始位，`-0.0` 保留，不含 NaN 和无穷。每个 fixture 都有下列字段：

| 字段 | 含义 |
|---|---|
| `format` | `"lumice-integral/analytic-parity"` |
| `schema_version` | `1`。波次 2（诊断与权重）只新增可选字段，不改名也不删已有字段；破坏性修改才升版本。 |
| `fixture_kind` | `evaluate_path`、`trace_fiber` 或 `seed_search` |
| `symmetry_semantics` | 恒为 `"none"`：输入是一个具体面序列，不涉及任何对称约化（Lumice `doc/analytic-api.md` §3.3 规则 2；`docs/conventions.md` #21）。 |
| `provenance.li_rev` | 导出时 LI 的完整 commit SHA |
| `provenance.li_tracked_tree_clean` | 导出时受控文件与该 commit 不一致则为 `false`，这样的 fixture 不应拷入 Lumice |
| `provenance.conventions_sha256` | LI `docs/conventions.md` 的 SHA-256，变化提示约定可能已经变动 |
| `cell` | `name`、`path`、`category`、`rationale`、`selection`，`evaluate_path` 另有 `pose_label` |
| `input` | 调用参数（见下） |
| `expected` | 参考输出（见下） |
| `tolerance` | 每个被比较的量一项：`{"value": 数值, "basis": 依据}` |

### 2.1 公共输入字段

| 字段 | 含义 |
|---|---|
| `crystal` | 与 `LUMICE_ANALYTIC_Crystal` 逐字段对应：`kind`（`"prism"` / `"pyramid"`）、`height`（柱高，或锥晶的棱柱段 `prism_h`，以参考外接圆直径为单位）、`face_distance[6]`、`upper_h`、`lower_h`、`upper_wedge_deg`、`lower_wedge_deg`；该种类不用的字段为 `0.0`。语义见 Lumice `doc/configuration.md` §prism / §pyramid。LI 用 `HexPrism.from_lumice(height, face_distance)` / `Pyramid.from_lumice(height, upper_h, lower_h, face_distance=..., upper_wedge_deg=..., lower_wedge_deg=...)` 构造，六边形边长 `a = 1`；下面所有输出都与尺度无关。 |
| `faces` | Lumice 面编号的具体面序列（入射面、各内反射面、出射面） |
| `refractive_index` | 本次调用的冰折射率 |
| `incident_direction` | 世界系太阳光传播方向，太阳 → 晶体（`s = -ŝ`） |
| `pose` / `seed_pose` | 九个数，行主序 3×3 旋转，体 → 世界（`v_W = R v_B`） |
| `target_direction` | 世界系出射光传播方向，晶体 → 观察者（`d`） |
| `continuation` | LI 的 `ContinuationOptions`（契约 §9.2，§10.1 的参考默认值），全字段。Lumice v0 的选项块字段更少，后端用自己的对应项并写明映射。 |

## 3. 三类 fixture

### 3.1 `evaluate_path`

输入：`crystal`、`faces`、`refractive_index`、`incident_direction`、`pose`。

| `expected` 字段 | 含义 |
|---|---|
| `valid` | 该 pose 下路径可实现：入射、出射的 Snell 判别式与各入射余弦（入射面、每个从内部到达的内反射面、出射面）都为正（契约 §5.2）。内反射的 TIR 判别式不参与判定（conventions #18）。 |
| `outgoing_direction` | 世界系出射传播方向（仅 `valid` 时） |
| `segment_directions` | `(face_count + 1) × 3`，体坐标传播方向 `Rᵀ v`：入射光、每个面之后的光线、出射光（仅 `valid` 时） |
| `interface_transmittances` | `face_count` 个值：入射、出射面的非偏振（s/p 平均）透射率 `T`，每个内反射面的反射率 `R`，TIR 时为 `1`（仅 `valid` 时） |
| `fresnel_transmission` | `T_entry · Π R_k · T_exit`，非 `valid` 时为 `0.0` |
| `diagnostics` | 不参与比较：LI 算出的全部 margin（`validity_margins`）、最小 validity margin 及其名字，非法 pose 另附失败信息 |

非法 pose 只比较 `valid` 与 `fresnel_transmission`。

### 3.2 `trace_fiber`

输入：公共字段加 `target_direction`、`seed_pose`、`continuation`。种子在 fiber 上：`target_direction`
由种子 pose 算出，残差是舍入量级。

`expected.traces` 含一条或两条 trace。第一条用 `initial_tangent_sign = +1` 追踪；它不闭合时，从同一种子
反号再追一条（契约 §9.5.5）。每条 trace 有 `initial_tangent_sign`、`status`、`reason`、`poses` `(N, 9)`、
`crystal_frame_sun_directions` `u = Rᵀ(−s)` `(N, 3)`、`arclength_increments` `(N − 1)`、`residual_norms`
`(N)`、`tangents` `(N, 3)`（契约 §9.3）与 `arclength`。一条 trace 可以只有一个 pose（没有被接受的步）。

fixture 的**曲线**：第一条闭合时就是它；否则是第二条 trace 种子之后的 pose 倒序，接上第一条 trace 的全部
pose，得到一条过种子的开折线（契约 §9.5.5，`resample.stitch_open_arc`）。种子切向的符号只在同一个 LAPACK
构建内确定，跨构建不保证（契约 §10.1），所以后端的 `+1` 可能是 LI 的 `−1`，§4 的比较方法因此与方向无关。

### 3.3 `seed_search`

输入：公共字段加 `target_direction`、`extra_seeds`（v0 为空）、`band_half_width_deg`、`cluster_radius_rad`、
`distance_threshold`、`continuation`，以及 `sample`：

| `sample` 字段 | 含义 |
|---|---|
| `sampler`、`n` | LI 生成样本的方式：契约 §9.5.2 的对径 Fibonacci 格点，`n` 个点，保留 `w = A T > 0` 的事件 |
| `band_u`、`band_phi`、`band_deviation` | band 本身，按 pool 顺序（`D_i` 递增）：`u_i`（体坐标太阳方向）、`phi_i`（体坐标出射传播方向）、`D_i`（弧度） |

后端在这条 band 上重放契约 §9.5.3–§9.5.5：先构造候选 pose `R_i = W F_iᵀ`（§9.5.3，取 `ŝ = −incident_direction`、
`d = target_direction`），再聚类、校正、闸门、去重、追踪、分类。采样器本身另按 §9.5.2 的公式比较，不经过这些
fixture（§9.5.8）。

| `expected` 字段 | 含义 |
|---|---|
| `completeness` | `complete` / `unknown`（程序性信号，§9.5.6） |
| `pool_count`、`extra_seed_count`、`raw_cluster_count`、`admissible_count` | 漏斗计数 |
| `events` | 六个计数器，全部出现 |
| `components` | 按追踪顺序：`kind`（`closed` / `arc`）、`seed`（9）、`status`、`reason`、`start_reason`（闭合时为 `null`）、`arclength`、`curve_poses` `(N, 9)`（闭合 trace 或缝合后的弧）；弧另有 `seed_index` |
| `incomplete` | `cause`、`seed`、`status`、`reason` |

## 4. 比较方法

- **向量与标量**：逐分量绝对差的最大值不超过容差。
- **旋转距离**：`angle(Aᵀ B) = atan2(|vee(skew)|, (tr(Aᵀ B) − 1)/2)`，其中 `skew = (M − Mᵀ)/2`，`M = Aᵀ B`。
- **曲线距离**（两条 pose 折线 `P`、`Q`，各自为开或闭）：沿测地弦（`P_i exp(t log(P_iᵀ P_{i+1}))`）把每条
  折线加密到间距不超过 `1e-3` rad，闭合曲线另加最后一个 pose 回到第一个的弦；然后取对称 Hausdorff 距离，即两个
  方向上「一条曲线的 pose 到另一条加密曲线的最大距离」中较大的那个。加密去掉了弦采样带来的那部分距离，同一条
  曲线的两种采样之间最多差半个间距。参考实现：`parity_export.curve_distance`。
- **`trace_fiber`**：trace 条数相等；`(status, reason)` 对作为多重集相等；曲线距离不超过 `curve_distance_rad`；
  `arclength` 之和相对差不超过 `arclength_relative`；每个 `residual_norms` 不超过 `residual_norm_bound`。
  不比较步数，也不比较单个 pose。
- **`seed_search`**：`completeness`、四个计数、六个计数器相等。分量按顺序比较：`kind`、`status` 相等，非空的
  两端 `{reason, start_reason}` 作为无序集合相等；后端的种子到 fixture 曲线的距离不超过 `seed_to_curve_rad`
  （契约 §9.5.4 第 5 步的 pose 到采样点距离）；曲线距离不超过 `curve_distance_rad`；闭合分量的 `arclength`
  相对差不超过 `closed_arclength_relative`。`incomplete` 的 cause 按顺序相等。

## 5. 容差及依据

| 量 | 容差 | 依据 |
|---|---|---|
| `valid` | 精确 | 每个 fixture pose 在 `diagnostics` 里记录了最近那道闸门的 margin；当前矩阵上它的绝对值至少 `1e-8`，远大于 margin 本身的 float64 舍入（`~1e-15`）。 |
| 方向、透射率、Fresnel 因子 | `1e-12 · max(1, 1/(2√d))`，`d` 为该 pose 最小的 Snell 判别式 | 同一条闭式链、逐位相同的输入，每个界面只贡献几个 ulp，`1e-12` 约留 `1e3` ulp 余量；折射余弦为 `√d`，其导数 `1/(2√d)` 在 Snell 边界附近放大舍入。矩阵里弧端 pose 的 `d ≈ 1.2e-8`（放大约 4600 倍）。 |
| 曲线距离 | `0.012` rad | 契约 §10.1：3-5 环在两组步长控制参数之间的双向采样 pose 集合距离低于 `0.012` rad（实测 `0.0093`）。 |
| 弧长（一条 trace；闭合分量） | 相对 `2e-3` | 契约 §10.1：同一 3-5 环在初始步长 `0.03`–`0.08` 下的长度为 `0.96432`–`0.96472`，相对跨度 `1.8e-3`。弧长只在同一种子下比较（§9.5.6），`trace_fiber` 满足这一条，`seed_search` 不满足。 |
| 残差范数 | `≤ residual_tolerance + relative_residual_tolerance`（`1e-11`） | 契约 §9.5.4 第 3 步与 §10.1；这是对后端的界，不要求与 LI 相等。 |
| seed 搜索的计数、计数器、kind、reason | 精确 | 契约 §9.5.8：pool 就是导出的 band，步骤顺序与闸门是规范性的。 |
| 被发现分量的种子 | 到 fixture 曲线的距离不超过 `distance_threshold`（`0.08` rad） | 契约自己判定「这个 pose 属于那个分量」的准则（§9.5.4 第 5 步）。 |

LI 自己的读回（`--verify`）在这些容差下全部通过；第一次跨后端运行才是这些容差在 Lumice 侧的第一份证据。
若某个容差被证明过紧或过松，改动发生在 LI（新证据记入本页），然后重新导出；不在 Lumice 读取器里悄悄放宽。

## 6. 矩阵

路径拓扑 × 点类别（Lumice `doc/raypath-analysis.md` §5.1.6 第 3 点）。太阳高度 15°、方位 0（ch06 场景），`n = 1.31`。

| 路径 | 晶体 | 拓扑 |
|---|---|---|
| `3-5` | 规范柱晶：prism，`height = 1`，正六棱柱 | 无内反射（22° 晕） |
| `3-5-6-7` | 同上 | 两次内反射（部分或全反射，按 Fresnel `R` 加权），弧止于出射 TIR 或路径不可行 |
| `13-15-26-28` | 非对称锥晶：`prism_h = 0.5`、`upper_h = 0.25`、`lower_h = 0.6`，楔角取 Miller `(1,0,1)` / `(2,0,3)`（`miller_wedge_deg`），`face_distance = (1, 1.1, 0.9, 1, 1.2, 0.95)`（LI `tests/test_optics_crystal_native.py::ASYMMETRIC_PYRAMID`） | 锥晶族外的锥面路径 |

选点在 `u`（体坐标太阳方向）的球面上进行（`parity_export.choose_point`）。由 `u` 得到的 pose 把 `u` 转到 `ŝ`、
出射方向落在 `s` 上方的竖直半平面内（`s2_store.event_rotations`，方位参考 `+z`）；target 就是该 pose 的出射方向，
因此每个种子都在自己的 fiber 上。

- **random（随机）**：`numpy.random.default_rng(20260928)` 正态抽样中第一个通过 discovery 闸门的方向（路径有效、
  entry measure 为正）。
- **critical（临界点）**：`D_P` 的第一个内部极值 `D*`（`DPField.interior_critical_points`）。点取在它的亮侧
  `D* ± 0.5°`，从极值点沿测地线走过去再二分，fiber 因此是绕一个 Jacobian 退化点的短环。
- **near_boundary（近边界）**：从随机点沿固定测地线走到 `U_P` 边缘，再二分到最小 validity margin 为 `1e-3`
  的 pose；它关于边缘的镜像作为非法的 `evaluate_path` 导出（`__outside`）。

每个格导出 `evaluate_path__point`（及 `__outside`）、从该 pose 出发的 `trace_fiber`、
`evaluate_path__curve_min_margin`（追踪曲线上 validity margin 最小的 pose，通常是事件容差内的弧端点），以及该
target 上的 `seed_search`（`N = 1e5`、band `0.2°`、聚类半径 `0.3` rad、`distance_threshold = 0.08`）。

跳过项（记入 `manifest.json`）：

- `3-5-6-7__critical`：在该晶体上 `3-5-6-7` 的 `D_P` 没有内部极值，`interior_critical_points` 为空，它的临界值
  （`DPField.critical_values`：50.06°、141.84°、163.47°）全是边界临界值。
- 三个 `13-15-26-28` 格的 `seed_search`：参考 discovery 在 discovery 之前就拒绝锥晶，因为它的样本 store 不支持锥晶
  （契约 §9.5.10）。这些格仍有 `evaluate_path` 与 `trace_fiber`。

首次导出（2026-09-28）共 32 个 fixture：

| 格 | `trace_fiber` | `seed_search` |
|---|---|---|
| `3-5__random`（`D = 39.32°`） | 闭合，5.670 | 1 个闭合分量；pool 172，13 簇，折叠 12 |
| `3-5__critical`（`D = 22.34°`，极小 21.84°） | 闭合，1.160 | 1 个闭合分量；pool 579 |
| `3-5__near_boundary`（`D = 43.23°`，入射余弦 `1e-3`） | 弧 path_infeasible / path_infeasible，0.067 + 0.916 | 4 条弧，其中一条单 pose 弧（两端 TIR，§9.5.6） |
| `3-5-6-7__random`（`D = 143.87°`） | 弧 TIR / TIR，0.922 + 1.545 | 1 条弧 |
| `3-5-6-7__near_boundary`（`D = 140.72°`，内反射入射余弦 `1e-3`） | 一侧在种子处即停（path_infeasible），另一侧 1.189 后止于 TIR | 2 条弧 |
| `13-15-26-28__random`（`D = 142.54°`） | 闭合，2.735 | 跳过 |
| `13-15-26-28__critical`（`D = 148.74°`，极大 149.24°） | 闭合，0.808 | 跳过 |
| `13-15-26-28__near_boundary`（`D = 87.33°`） | 一侧在种子处即停，另一侧 0.428 后止于 TIR | 跳过 |

## 7. 更新流程

1. LI 改行为（求解器、约定、默认值）并提交。
2. LI 在新 rev 上用 `--verify` 重新导出，必须全部通过，且在干净工作树上导出（`li_tracked_tree_clean = true`）。
3. Lumice 整目录拷入自己的仓库，替换上一版。
4. C++ 行为有差异的地方，Lumice 的 parity CI 变红。
5. Lumice 改 C++ 直到变绿，不改 fixture；对容差或语义有异议则回到第 1 步。

格式变更（字段、文件名、比较方法）必须在同一个 LI commit 里同时改本页与 `parity_export.SCHEMA_VERSION`。

## 8. 不在这些 fixture 里的内容

- 诊断与权重（`jacobian_diagnostics`、`terminal_payload`、`weight_observables` 等），属于波次 2
  （`explore-fiber-diagnostics-contract`），届时以可选字段加入。
- 采样器本身、`check_band_coverage`、加密行为（§9.5.7）。
- 任何对称约化：`symmetry_semantics` 处处为 `none`。
