# 解析 parity fixture（LI → Lumice）

[English](analytic-parity-fixtures.md)

Lumice Integral（LI）按固定 rev 导出一个 JSON fixture 目录，Lumice 把它拷进自己的仓库，在 CI 里对
`liblumice_analytic` 模块 A v0 回放（Lumice `doc/analytic-api.md` §4：`EvaluatePath`、seed 搜索、
`TraceFiber[Batch]`，只返回点列）；自波次 2 起也对模块 B（单光路带求和，
[band-sum-contract.md](band-sum-contract.md)）回放。方向是单向的（owner 2026-09-28 裁定）：LI 是参考，Lumice 跟随。
本页的写法要让 Lumice 侧不读 LI 源码就能写出读取器。fixture 背后的规范语义是
`docs/phase1-math-contract.md` §9（continuation）、§9.5（discovery）与 `docs/band-sum-contract.md`（带求和），约定以
`docs/conventions.md` 为准。

## 1. 生成

```bash
uv run python scripts/export_analytic_parity.py --output-dir artifacts/analytic-parity --verify
```

- 一条命令写出矩阵（§6）、边缘情形（§6.1）与带求和格（§6.2）的全部 fixture 和一个 `manifest.json`，共 89 个。
  M2 Max 在负载 20–30 下（2026-09-29）导出 50 s、读回 153 s，其中七个带求和 fixture 读回 38 s（统计性秩 0 格要重放
  `4e6` 个姿态的 Haar 流）；此前的 82 个在空闲机器上连读回约 1 min，波次 2 之前只有矩阵，约 16 s。
- **确定性**：同一 rev 重跑逐字节相同。LI 的 `tests/test_parity_export.py` 在两个独立解释器里各导出一次，
  逐文件比较。fixture 不记录时间、主机和路径。
- `--verify` 读回每个 fixture，用当前 checkout 重算，按 fixture 自带容差（§5）比较，任一失败即非零退出。
  `--cells 3-5__random 3-5__limits 3-5__band_sum_plate ...` 只导出子集（矩阵格、边缘格与带求和格都按名字）。
- 产物在 `artifacts/` 下，LI 不纳入版本控制。钉住的那一份在 Lumice 仓库里，每个 fixture 自己记录来源的
  LI rev。
- 代码：`lumice_integral.parity_export`（格式、选点、校验器）与 `scripts/export_analytic_parity.py`
  （矩阵定义，瘦 CLI）。

## 2. 文件与公共字段

一个格（cell）是矩阵的一行，即一条路径加一个点类别。它的文件名为
`<path>__<category>__<kind>[__<label>].json`：路径用连字符（`3-5-6-7`），类别为 `random` / `critical` /
`near_boundary`，种类为 `evaluate_path` / `trace_fiber` / `seed_search`。边缘格（§6.1）改名为
`<path>__<label>`，文件为 `<path>__<label>__<kind>[__<suffix>].json`。`manifest.json` 在 `cells` 下列出矩阵格，
在 `edge_cells` 下列出边缘格，每个都有 `files`、带原因的 `skipped`、`rationale` 以及 §6 的 `selection` 记录；
边缘格另有 `label`、`point`（种子怎么选）与 `serves`（它认证的契约 §11 条目，见
`docs/phase1-math-contract.md` §11.1）。只导出矩阵时没有 `edge_cells` 键。带求和格（§6.2）只有一个文件
`<path>__band_sum_<label>.json`，列在 `band_sum_cells` 下，带 `label`、`rank`、`pose_density` 族名、`projection` 种类与
`rationale`；不导出带求和格时没有 `band_sum_cells` 键，§6 与 §6.1 的文件和 manifest 字节不变。读取器必须忽略自己
不认识的 manifest 顶层键：后续波次会在新键下增加 fixture 种类。

JSON 为 UTF-8，键排序。浮点写成能往返回同一个 float64 的最短形式，任何符合规范的 JSON 读取器
（`strtod`）解析后都得到导出时的原始位，`-0.0` 保留，不含 NaN 和无穷。每个 fixture 都有下列字段：

| 字段 | 含义 |
|---|---|
| `format` | `"lumice-integral/analytic-parity"` |
| `schema_version` | `1`。波次 2 只新增了可选字段（§3.1、§3.2），没有改名或删除，v0 的每个键字节不变（对照 2026-09-29 `5ea2bde` 的导出核过）。没有波次 2 字段的 fixture 来自旧导出，按 v0 的比较方法。破坏性修改才升版本。 |
| `fixture_kind` | `evaluate_path`、`trace_fiber`、`seed_search` 或 `band_sum` |
| `symmetry_semantics` | 恒为 `"none"`：输入是一个具体面序列，不涉及任何对称约化（Lumice `doc/analytic-api.md` §3.3 规则 2；`docs/conventions.md` #21）。 |
| `provenance.li_rev` | 导出时 LI 的完整 commit SHA |
| `provenance.li_tracked_tree_clean` | 导出时受控文件与该 commit 不一致则为 `false`，这样的 fixture 不应拷入 Lumice |
| `provenance.conventions_sha256` | LI `docs/conventions.md` 的 SHA-256，变化提示约定可能已经变动 |
| `cell` | `name`、`path`、`category`、`rationale`、`selection`，`evaluate_path` 另有 `pose_label`；`band_sum` 为 `name`、`path`、`label`、`rationale`、`rank` |
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

## 3. 四类 fixture

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
| `branch_margins` | 波次 2。`valid` 时按名字给出有符号的 validity 余量：`entry_incidence_cosine`、`entry_snell_discriminant`、每个内反射 `k = 1, 2, ...` 的 `internal_<k>_incidence_cosine`、`exit_incidence_cosine`、`exit_snell_discriminant`（契约 §9.3 的 `branch_diagnostics`）。前者是光线与面法向的夹角余弦，后者是 Snell 判别式 `1 − n_rel²(1 − cos²)`，无量纲，在光滑分支上为正。非 `valid` 时为 `null`。 |
| `failed_gate` | 波次 2。非 `valid` 时为 `{name, value}`：按上面的顺序第一个余量不为正的闸门（`value ≤ 0`）。`valid` 时为 `null`。 |
| `jacobian_available` | 波次 2。恰在 `valid` 时为 `true`：法向 Jacobian 只在光滑分支上有定义。 |
| `normal_jacobian` | 波次 2。该 pose 下 `2 × 3` 残差 Jacobian `A` 的 `J_perp = σ₁ σ₂`，目标取该 pose 自己的出射方向（契约 §5.4）。`A` 对出射方向在右平凡化旋转 `R exp([δ]×)` 下求导（δ 以弧度计，§5.1 度量），再投影到该方向切平面的一组正交基上。无量纲（每弧度姿态对应的方向弧度），与基的选取无关，只依赖 pose 和路径。不可用时为 `null`。它可以极小（`D_P` 极值点上约 `4e-17`），也可以在 Snell 边界附近很大（`d ≈ 2e-8` 处约 `2600`）。 |
| `singular_values` | 波次 2。同一个 `A` 的 `[σ₁, σ₂]`，`σ₁ ≥ σ₂ ≥ 0`。不可用时为 `null`。 |

非法 pose 只比较 `valid`、`fresnel_transmission`、`failed_gate` 与可用性（`jacobian_available = false`，余量与
Jacobian 为 `null`）。

### 3.2 `trace_fiber`

输入：公共字段加 `target_direction`、`seed_pose`、`continuation`。种子在 fiber 上：`target_direction`
由种子 pose 算出，残差是舍入量级。

`expected.traces` 含一条或两条 trace。第一条用 `initial_tangent_sign = +1` 追踪；它不闭合时，从同一种子
反号再追一条（契约 §9.5.5）。每条 trace 有 `initial_tangent_sign`、`status`、`reason`、`poses` `(N, 9)`、
`crystal_frame_sun_directions` `u = Rᵀ(−s)` `(N, 3)`、`arclength_increments` `(N − 1)`、`residual_norms`
`(N)`、`tangents` `(N, 3)`（契约 §9.3）与 `arclength`。一条 trace 可以只有一个 pose（没有被接受的步），
种子本身被拒时（`rank_loss`、`chart_boundary`，见 §6.1 的边缘格）可以一个也没有。

波次 2 加入与 `poses` 对齐的逐点数组（契约 §9.3 的 `jacobian_diagnostics` 与 `branch_diagnostics`）：
`branch_margin_names`（§3.1 `branch_margins` 的 `k` 个名字，按该顺序）、`branch_margins` `(N, k)`、
`jacobian_available` `(N)`、`normal_jacobian` `(N)` 与 `singular_values` `(N, 2)`，含义同 §3.1 该字段在对应
pose 上的值。

trace 以 `budget_exhausted` 结束的 fixture 另带 `expected.reference_curve`（`poses` `(M, 9)` 与 `closed`：同一
种子在默认选项下追出的曲线，定义见下）以及一个 `budget_extent` 容差。

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

### 3.4 `band_sum`

模块 B（`docs/band-sum-contract.md`）：一条具体光路、一种姿态密度、一张以方向给出的像素表。输入：`crystal`、
`faces`、`refractive_index`、`incident_direction`（§2.1），以及

| `input` 字段 | 含义 |
|---|---|
| `pose_density` | `family`（`random` / `column` / `plate` / `parry` / `lowitz`）及其解析后的参数（度，契约 §2.2）。`normalization_informative` 是 LI 的 `I`（与 `Q`），不比较。 |
| `sample` | `sampler` 与 `n`：`n` 点的对径 Fibonacci 格点（契约 §3）。 |
| `pixels` | `labels` `(P, 2)`、`centre` `(P, 3)`、按环绕顺序的 `corners` `(P, 4, 3)`、`solid_angle` `(P)`：出射传播方向，天空点是其负向（契约 §2.3）。`projection` 记录 LI 如何展开（Lumice 线性镜头的 `render` 块或单圆盘 Lambert 视图），仅供参考。 |
| `events` | 层 1（契约 §7.1）。秩 2：落在任一非奇异像素带内的保留事件的 `u`、`phi`、`deviation`、`w`，按 LI 的顺序（`D` 递增）。随机密度下的秩 0：全部保留事件的 `w`。统计性秩 0 格没有此字段。 |

| `expected` 字段 | 含义 |
|---|---|
| `rank` | `2`（带求和）或 `0`（点质量，契约 §5） |
| `pixels` | 按表序：`label`、`status`、`value`；秩 2 另有 `delta`、`delta_lo`、`delta_hi`、`K`、`K_rho_pos`、`K_eff`（`singular` 时为 null）、`total`、`square`（仅供参考）与 `allowance`（下行） |
| `pixels[].allowance` | 秩 2 非奇异像素：`K_rho_pos_subnormal`（`c_i` 在 LI 中为次正规数的带内事件数），层 2 的放宽量 `K_layer2`、`K_rho_pos_layer2`、`value_layer2`、`K_eff_layer2`，以及它们的来源 `candidates`（`band_end`、`gate`、`gate_without_finite_D`；契约 §7.2） |
| `point_mass` | 秩 0：`m`、`method`（`lattice_mean`，或带 `error_estimate`、`sample_count`、`rng_seed` 的 `haar_stream`）；格点均值格另记 LI 的 `haar_check_informative` |

## 4. 比较方法

- **向量与标量**：逐分量绝对差的最大值不超过容差。
- **旋转距离**：`angle(Aᵀ B) = atan2(|vee(skew)|, (tr(Aᵀ B) − 1)/2)`，其中 `skew = (M − Mᵀ)/2`，`M = Aᵀ B`。
- **曲线距离**（两条 pose 折线 `P`、`Q`，各自为开或闭）：沿测地弦（`P_i exp(t log(P_iᵀ P_{i+1}))`）把每条
  折线加密到间距不超过 `1e-3` rad，闭合曲线另加最后一个 pose 回到第一个的弦；然后取对称 Hausdorff 距离，即两个
  方向上「一条曲线的 pose 到另一条加密曲线的最大距离」中较大的那个。加密去掉了弦采样带来的那部分距离，同一条
  曲线的两种采样之间最多差半个间距。参考实现：`parity_export.curve_distance`。
- **`trace_fiber`**：trace 条数相等；`(status, reason)` 对作为多重集相等；曲线距离不超过 `curve_distance_rad`；
  `arclength` 之和相对差不超过 `arclength_relative`；每个 `residual_norms` 不超过 `residual_norm_bound`。
  不比较步数，也不比较单个 pose。两边都没有被接受的 pose 时距离为 0；一边有一边没有时距离为无穷。
- **`evaluate_path` 的波次 2 字段**：`jacobian_available` 相等，`branch_margins` 的名字集合相等；余量按
  `branch_margins` 容差（绝对）比较，`normal_jacobian` 与每个奇异值按各自容差相对 `max(1, |value|)` 比较。
  非法 pose 的 `failed_gate.name` 相等，其值按余量比较。
- **`trace_fiber` 的逐点数组**（`pointwise_consistency`、`accepted_pose_regularity`）：两个后端步进不同，
  所以逐点数组从不按样本和 LI 比。后端在自己的每个 pose `i` 上，`normal_jacobian[i]`、`singular_values[i]`、
  `branch_margins[i]` 必须等于它自己的 `EvaluatePath` 在 `poses[i]` 处的返回值（容差为 §3.1 在该 pose 上的
  容差）；`EvaluatePath` 本身在固定 pose 上对 LI 认证，两者合起来认证逐点数组，包括它与 `poses` 的对齐。每个被接受
  的 pose 还必须正则：`jacobian_available`、每个余量 `> 0`、`normal_jacobian > 0`。
- **`trace_fiber` 的预算**（有 `budget_extent`）：`(status, reason)` 多重集与残差界同上，但不比曲线与长度是否
  相等（两个控制器在同一预算内到不了同一处），改为：`step_budget` 的 trace 恰有 `maximum_accepted_steps + 1`
  个 pose（种子加每个被接受的步一个）；`arclength_budget` 的 trace 满足 `maximum_arclength − maximum_advance <
  arclength ≤ maximum_arclength`（下一条边会越过预算）；`evaluation_budget` 的 trace 至少一个 pose（一个评估
  单位算什么属于后端内部，契约 §9.2）；每条 trace 的每个 pose 到加密后的 `reference_curve` 的距离不超过
  `curve_distance_rad`。后端的步数记法若不同，自己写明映射，按区间比较步数预算；fixture 不改。
- **选项变体**（边缘格的 `trace_fiber__<variant>`）：`continuation` 与默认不同的普通 `trace_fiber` fixture。
  对 `perturbation` 变体，导出器检查 LI 追出的曲线按上面的方法仍与默认 trace 一致，不一致就拒绝写出；后端每个
  变体都通过，就说明它在每组设置下都复现同一个分量（契约 C06）。
- **`seed_search`**：`completeness`、四个计数、六个计数器相等。分量按顺序比较：`kind`、`status` 相等，非空的
  两端 `{reason, start_reason}` 作为无序集合相等；后端的种子到 fixture 曲线的距离不超过 `seed_to_curve_rad`
  （契约 §9.5.4 第 5 步的 pose 到采样点距离）；曲线距离不超过 `curve_distance_rad`；闭合分量的 `arclength`
  相对差不超过 `closed_arclength_relative`。`incomplete` 的 cause 按顺序相等。
- **`band_sum`**，分两层（契约 §7）。层 1 在 `input.events` 上跑估计器；层 2 重新生成样本、跑完整调用。两层都要求
  每个像素的 `status` 相等（奇异像素没有值）。秩 2 格的 `ok` 像素：`K` 层 1 相等、层 2 差不超过
  `allowance.K_layer2`；`K_rho_pos` 差不超过 `allowance.K_rho_pos_subnormal`（层 1）或 `allowance.K_rho_pos_layer2`
  （层 2）；`value` 与 `K_eff` 差不超过 `value_relative` × |期望值|，层 2 另加 `allowance.value_layer2` /
  `allowance.K_eff_layer2`。秩 0 格按 `point_mass_relative` 比较 `m` 与各像素值；`haar_stream` 格的该容差是 LI 的
  五个标准误（相对），自带估计误差 `σ` 的后端放宽为 `5 √(σ_LI² + σ²)`。LI 的 `--verify` 用自己的两种求和形式
  （生产的 scatter 与逐像素 gather）跑层 1，核对 fixture 的事件就是它重新生成的样本里的事件，再跑层 2。

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
| 分支余量、`failed_gate` 的值 | `1e-12 · max(1, 1/(2√d))`，与方向相同 | 余量是同一条闭式链上的余弦与判别式。 |
| `normal_jacobian`、奇异值 | 相对 `max(1, |value|)`：`1e-12 · max(1, 1/(4d))` | `J_perp` 比方向高一阶导数，方向容差中的 Snell 放大 `1/(2√d)` 以平方进入。实测：LI 的 AD 值与独立中心差分（`h = 1e-6`）相对差 `6e-9`；远离 Snell 边界时 `1e-14` rad 的姿态扰动使其相对变化至多 `2.5e-12`（`d ~ 1e-8` 处 `4e-8`，该处容差为 `2.5e-5`）。 |
| trace 的逐点数组 | 上两行，在每个 pose 上取值 | 比较对象是后端自己的 `EvaluatePath`（§4）。 |
| 被接受 pose 的正则性、可用性、闸门名 | 精确 | 契约 §6.1 与 §5.4：被接受的 pose 在光滑分支上且正则。 |
| 预算范围 | 计数与界精确（§4） | 契约 §9.4：`budget_exhausted` 保留部分几何；计数规则写在契约 §11.1。 |
| 带求和：状态、层 1 的 `K` | 精确 | 像素含 `s` 或 `−s` 时为奇异；带内判定是在 fixture 自己的 `D` 上取 `delta_lo ≤ D < delta_hi`（契约 §4.1、§4.2）。 |
| 带求和：值、`K_eff`（层 1） | 相对 `1e-10` | 在逐位相同的事件上求同一个和：只差求和顺序与 `arccos`、`atan2`、`exp` 的舍入。LI 的 scatter 与 gather 相差 `1e-12`（`K_eff` 接近 1 时 `3e-12`）；只依据规格写的参考实现（`tests/test_band_sum_spec_reference.py`，`I` 用 scipy 自适应积分、`Q` 用闭式）与每个导出值相差不超过 `8e-14`。 |
| 带求和：`K_rho_pos`（层 1） | 逐像素 `allowance.K_rho_pos_subnormal` | 契约 §4.5 要求渐进下溢；只有 `c_i` 在 LI 中为次正规数的带内事件会被冲零（flush-to-zero）的后端数错（Parry 格单像素最多 12 个）。 |
| 带求和：层 2 | 逐像素 `allowance.*_layer2` | 契约 §7.2：距带端不超过 `edge_epsilon_rad = 1e-9`、距有效性闸门不超过 `gate_epsilon = 1e-9`、或 `0 < w ≤ weight_epsilon = 1e-9` 的格点数，以及它们对各量的最大影响。导出的各格上这些计数全为 0，所以层 2 与层 1 一样紧。 |
| 秩 0 点质量 | 相对 `1e-10`（格点均值）；相对 `5 σ_LI`（Haar 流） | 契约 §5：格点均值是确定性的；LI 的 Haar 流其他后端无法复现。 |

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

锥晶三格自任务 `s2-store-pyramid-seeds`（2026-09-29）起带 `seed_search`（此前参考 store 拒绝锥晶）。
`13-15-26-28__near_boundary` 的 target 落在 store 亮区之外（`D = 87.33°`；保留事件覆盖 121.51°–149.23°）：它的点
贴着 `U_P` 的入射门，而这道门只看面法向，此处有限晶体的 entry measure 已经为零（`corridor_empty`），所以 band 为空，
期望结果是 complete、无分量。它作为负例保留：seed 搜索不得在这里凭空造出分量。

样本 store 用 `PyramidShape` 描述锥晶（`s2_store.crystal_description`：`prism_h`、`upper_h`、`lower_h`、
`upper_c_over_a`、`lower_c_over_a`、`face_distance`、`a`），而不是上面 `crystal` 的楔角：`c_over_a` 是
`Pyramid.from_lumice` 把 Miller 指数或楔角换算成的量，可以比特级重建晶体。两者描述同一个晶体
（`c_over_a = (√3/2) cot(wedge)`）；store 的形式是 cache key，不是交换格式。

首次导出（2026-09-28）共 32 个 fixture；2026-09-29 起共 35 个：

| 格 | `trace_fiber` | `seed_search` |
|---|---|---|
| `3-5__random`（`D = 39.32°`） | 闭合，5.670 | 1 个闭合分量；pool 172，13 簇，折叠 12 |
| `3-5__critical`（`D = 22.34°`，极小 21.84°） | 闭合，1.160 | 1 个闭合分量；pool 579 |
| `3-5__near_boundary`（`D = 43.23°`，入射余弦 `1e-3`） | 弧 path_infeasible / path_infeasible，0.067 + 0.916 | 4 条弧，其中一条单 pose 弧（两端 TIR，§9.5.6） |
| `3-5-6-7__random`（`D = 143.87°`） | 弧 TIR / TIR，0.922 + 1.545 | 1 条弧 |
| `3-5-6-7__near_boundary`（`D = 140.72°`，内反射入射余弦 `1e-3`） | 一侧在种子处即停（path_infeasible），另一侧 1.189 后止于 TIR | 2 条弧 |
| `13-15-26-28__random`（`D = 142.54°`） | 闭合，2.735 | 1 个闭合分量；pool 29，5 簇，折叠 4 |
| `13-15-26-28__critical`（`D = 148.74°`，极大 149.24°） | 闭合，0.808 | 1 个闭合分量；pool 71，2 簇，折叠 1 |
| `13-15-26-28__near_boundary`（`D = 87.33°`） | 一侧在种子处即停，另一侧 0.428 后止于 TIR | 空 band：pool 0，无分量（亮区之外，见上） |

### 6.1 边缘格（波次 2）

矩阵采的是一般点。边缘格接在矩阵之后，是契约 §11 的不变量真正要被判定的情形：极短环、贴边界走的环、TIR 截断
的弧、秩亏、预算、锥晶。每个格服务的 §11 条目写在 manifest 条目里，逐条说明见契约 §11.1。下表数字取自本 rev
的导出。

种子来源 `point` 有五种：`critical_offset`（`D_P` 内部极值点亮侧给定偏移处，同矩阵的 `critical` 类别）；
`extremum`（极值点本身）；`random`（矩阵的随机抽样）；`antipodal_target`（同一 pose 但目标取 `−d`）；
`target`（给定目标，种子是在 `N = 1e5` 样本上做种子搜索返回的各分量，每个一份
`trace_fiber__component_<k>`）。像素目标来自 `camera.linear_pixel_outgoing_direction` 配
`canonical_scene.CANONICAL_RENDER`；偏向角目标在过 `s` 的竖直平面内、
`s` 上方。未另行说明时，一个格导出 `trace_fiber`（默认选项）、`evaluate_path__curve_min_margin` 与
`__curve_min_jacobian`（trace 上余量最小、`J_perp` 最小的 pose），非 target 类的点另有 `evaluate_path__point`。

| 格 | 服务 | 为什么选它 | 实测（LI，本 rev） |
|---|---|---|---|
| `3-5__short_loop` | C05 C06 C14 | 3-5 最小偏向角上方 0.01°：环长约四个初始步长，按步长的闭合判据必须在第一圈闭合。环长短于闭合最小弧长 `2 × initial_step` 时按设计会走两圈（契约 §6.4：0.001° 处的环用 `initial_step 0.01` 量得 0.0523，默认 0.04 量得 0.1046）。`J_perp` 在这里最小。变体 `initial_step_0.03`、`initial_step_0.08` 与 `controller_thresholds`（`minimum_step 2e-5`、`maximum_step 0.10`、`shrink 0.4`、`growth 1.15`、`maximum_retries 10`）。 | 闭合，0.1652（55 个 pose；三个变体下 0.1653 / 0.1652 / 0.1653）。`J_perp` 最小 0.00405。 |
| `3-5__strip_short_loop_r100_c126` | C06 C15 | ch06 像素 (100, 126)：短于 π 的环，已退役的绝对闭合闸会走两圈。 | 闭合，1.6452；种子搜索 pool 553，3 簇，1 个闭合分量。 |
| `3-5__caustic_loop_r49_c0` | C06 C15 | ch06 像素 (49, 0)：焦散边缘 0.19 的环，它的多余种子曾经止于虚假事件。 | 闭合，0.1898；pool 320，1 簇。 |
| `3-5__boundary_hugging_r700_c150`、`__r780_c150` | C06 C08 C16 | 贴着出射 TIR 边界走的环（最小余量 0.0142 / 0.0062），在按速率的事件减速（契约 §6.3）之前会耗尽步数预算。 | 闭合，5.4085 / 5.6359；pool 185 / 175，13 / 15 簇，其余候选全部折叠，`complete`。 |
| `1-3__two_arcs_60deg` | C06 C08 C17 C18 | 路径 `1-3` 在 δ = 60°：两个不同分量，各是一条弧，一端被出射 TIR 截断，另一端是入射光离开面 1。分量 0 上有 `initial_step_0.03` 与 `initial_step_0.08` 两个变体。 | `path_infeasible` 0.1397 + `tir_boundary` 0.4749，以及 `tir_boundary` 0.3820 + `path_infeasible` 0.2325；种子搜索 2 条弧（0.6146、0.6145），折叠 1；TIR 端（`d = 2.4e-8`）`J_perp` 2581。 |
| `3-5__rank_loss_extremum` | C07 | 种子取 `D_P` 的内部极小点（21.84°）本身，fiber 在这里退化成一个点。 | 两个方向都是 `rank_loss`，没有 pose；种子处 `σ₂ = 1.1e-16`，`J_perp = 4.2e-17`。 |
| `3-5__limits` | C09 C11 | 矩阵的 3-5 随机种子，在让 trace 提前结束的选项下：变体 `step_budget`（`maximum_accepted_steps 5`）、`arclength_budget`（`maximum_arclength 0.3`）、`evaluation_budget`（`maximum_evaluations 15`）、`corrector_failure`（`maximum_advance 0.01`、`maximum_retries 0`）、`step_underflow`（`initial_step 0.04`、`minimum_step 0.03`、`maximum_step 0.04`、`maximum_advance 0.01`）。它的默认 `trace_fiber` 就是各预算变体的 `reference_curve`。 | 每个方向：6 个 pose（0.2000）；8 个 pose（0.2800）；3 个 pose（0.0800）；1 个 pose `corrector_failure`；1 个 pose `step_underflow`。 |
| `3-5__antipodal_target` | C04 | 3-5 随机 pose，目标取它自己出射方向的对径 `−d`：投影残差的代数零点。 | 两个方向都是 `chart_boundary`，没有 pose。 |
| `13-24-26__boundary_arc_90deg` | C18 C19 | 锥晶路径 `13-24-26`，晶体是 LI `tests/test_discovery.py` 的参考锥晶（`prism_h = upper_h = lower_h = 0.5`，两个楔角都是 `90° − pyramid_face_angle()`，正规）。90° 处是一条两端被路径域截断的弧，路径没有内部临界点。 | `path_infeasible` 1.4403 + `path_infeasible` 0.6801；种子搜索 3 簇得 1 条弧（2.1205）。 |

未覆盖：确定性的 `linear_solve_failure`（契约 §11 C09，未决）与 C19 的 `D3h` 棱柱。锥晶上的短环是矩阵的
`13-15-26-28__critical`（0.808）。

### 6.2 带求和格（模块 B）

太阳与矩阵相同（高度 15°、方位 0），`n = 1.31`。像素表来自三种视图：Lumice 线性镜头（41 × 41，视场 60°，仰角
15°，太阳在像素 (20, 20)，每像素约 1.5°）、以太阳为中心的 Lambert 视图（65 × 65，视场半径 30°，每像素约
0.93°，太阳在 (32, 32) 中心）与以反日点为中心的 Lambert 视图（65 × 65，60°，约 1.8°）；秩 0 格用以太阳为中心的
小 Lambert 视图（9 × 9，5°，太阳在 (4, 4)）。各格的像素从对应视图的扫描中挑出，以表内 `(row, column)` 或
`(y, x)` 标记。

| 格 | 路径、晶体、`N` | 密度、视图 | 覆盖 | 观测（LI，本 rev） |
|---|---|---|---|---|
| `3-5__band_sum_random` | 3-5，规范柱，`2e4` | random，线性 | 无内反射 | 太阳像素奇异；两个空带（晕内）；亮像素 `K` 112–384，`K_eff/K` 0.88–1.00 |
| `3-5__band_sum_plate` | 同上 | plate 1°，Lambert（太阳） | 窄天顶族 | 单条光路的幻日（只在一侧），值 108、`K_eff` 21.5；尾部到 `9e-88`；镜像一侧 `K = 258`、`K_rho_pos = 0`；次正规放宽最多 6 |
| `3-5__band_sum_parry` | 同上 | Parry 1° / 1°，线性 | 滚转锁定族（读 `e1`、`e2`、`e3`） | 上 Parry 弧 0.017–1.6，`K_eff` 2.4–4.7；尾部到 `8e-238`，`K_eff = 0`（平方下溢）；次正规放宽最多 12 |
| `3-5-6-7__band_sum_random` | 3-5-6-7，规范柱，`5e4` | random，Lambert（反日点） | 两次内反射（Fresnel `R`） | 反日点像素奇异（`δ = π`）；亮像素 `K` 12–103；最大偏折角之外的暗像素 |
| `13-15-26-28__band_sum_random` | 矩阵的非对称锥晶，`5e4` | random，Lambert（反日点） | 锥面 | 亮环（`D` 121–149°）`K` 44–107；两侧皆暗 |
| `3-6__band_sum_rank0` | 3-6，规范柱，`2e4` | random，小 Lambert（太阳） | 秩 0 点质量，确定性 | `m = 0.118166`（格点均值；LI 的 Haar 核对 `0.11845 ± 0.00030`），只在像素 (4, 4) |
| `3-6__band_sum_rank0_plate` | 同上 | plate 1° | 秩 0 点质量，统计性 | `m = 0.157 ± 0.011`（Haar 流，`4e6` 个姿态） |

Lambert 格的 LI 侧与线性格走同样两种求和形式：相机只经像素的方向进入 LI 的估计器（契约 §10）。

## 7. 更新流程

1. LI 改行为（求解器、约定、默认值）并提交。
2. LI 在新 rev 上用 `--verify` 重新导出，必须全部通过，且在干净工作树上导出（`li_tracked_tree_clean = true`）。
3. Lumice 整目录拷入自己的仓库，替换上一版。
4. C++ 行为有差异的地方，Lumice 的 parity CI 变红。
5. Lumice 改 C++ 直到变绿，不改 fixture；对容差或语义有异议则回到第 1 步。

格式变更（字段、文件名、比较方法）必须在同一个 LI commit 里同时改本页与 `parity_export.SCHEMA_VERSION`。

## 8. 不在这些 fixture 里的内容

- 控制器内部记录（`step_diagnostics`、`closure_diagnostics`、`terminal_payload`）与权重
  （`weight_observables`）：按 2026-09-29 的作者裁定，后端只凭输出认证，权重由 LI 在返回的 pose 上自算。契约
  §11.1 对 §11 每一条写明输出认证了什么、哪些留给后端自己的测试。诊断里 fixture 只带 `J_perp`、奇异值与分支余量
  （§3.1、§3.2）。
- 采样器本身、`check_band_coverage`、加密行为（§9.5.7）。带求和后端的采样器只经层 2 的结果间接受检（§4）。
- 任何对称约化：`symmetry_semantics` 处处为 `none`；一个类（L2 行）由调用方对成员求和（契约 §1）。
- 多波长带求和、发散光，以及非随机密度下确定性的秩 0 点质量（契约 §1、§5）。
