# S6 junction：结构交的机会式枚举

[English](junction.md) · [Phase II](phase2_zh.md#31b-s6-junction) · [Parity 合同](analytic-parity-fixtures_zh.md#65-s6-junction波次-3-第二批)

## 对象与边界

固定晶体、literal 面序列 P、折射率 n，在晶体系 S² 上求已有结构分层的交，并裁在 U_P 闭包内。这不是亮度/可见性标签，不负责发现一般临界流形，也不是 SO(3) 目标纤维交；后者属于独立的 C16 `family_target`。

`lumice_integral.junction.enumerate_junctions` 消费：

- **kind-1**：`DPField.interior_critical_points`、已知 slab 轴点和解析折痕圈 `axis·u=0`。孤立点直接做成员检验，不把 `|grad D|²=0` 冒充正则曲线方程。
- **kind-2**：边界 pieces 及 coincident validity gates，保留原 margin 与 piece 身份。即使行走器剔除了代数同义门，验收仍检查全部原始 validity gates。
- **kind-3**：每个内反射的 TIR onset，保留 step、margin、arc、同义身份及 upstream failed seeds。它是 Fresnel 权重折线，**不是**新增有效域门。
- **精确族像**：复用 [ExactFamily](family-restricted.md) 的四族支撑。与一维像通常得到点或重合；与二维像通常仍是一段曲线，不能把其采样点冒充有限 junction 集。同 u 的全部真实姿态原像必须保留。

顶层恒为 `exhaustive=false`、`coverage="opportunistic_existing_strata_only"`。`omitted` 明列走廊 A=0、未表示临界流形、上游未发现分量。单个闭环或 `KinkCurve.complete` 都不升级这个声明。共享 `family_restricted.circle_intersections/margin_plane/unit_sun` 正式公开为单权威原语，不维护第二份圆交求解器。几何存在性不新增 A>0 条件；有限晶体权重仍属另一维度。

## 输入与身份

入口 `enumerate_junctions(field, pairs=None, family=None, sun=None, ...)` 接收已有 `DPField`。`pairs` 选择两个 source ID；省略时枚举已有结构对及所请求的族交。交换/重复选择只物化一次；未知 ID 是调用错误，不是正常空集。

ID 为 `kind_2:<原margin>`、`kind_3:<原margin>`、`kind_1:interior:<index>`、`kind_1:axis:<index>`、`kind_1:crease`；族为 `family:<name>`。临界点 index 仅在本波长标识 upstream 对象，不能据此跨 λ 配对。source descriptor 保留 `kind/dimension/margin/origin/equivalent_to/failed_seeds/model`。点记录另列匹配的 upstream origins；同位置的不同物理事件不能因位置去重而失去身份。

`sun` 是**指向太阳**的有限单位向量，R 从晶体系到世界系，`Ru=sun`。族参数顺序、sheet、周期接缝与端点完全沿用 `ExactFamily`，不另写逆映射。角度/场值单位为弧度，margin 与单位向量分量无量纲。

| 搜索参数 | 缺省 | 含义 |
|---|---:|---|
| `budget` | 64 | 每 source 对最多尝试的局部候选数 |
| `nearby` | 0.02 | 候选近邻弦距，不是根接受阈值 |
| `max_iterations` | 30 | 每候选 Newton 次数 |
| `max_step` | 0.001 | 切平面大步拒绝上限（一阶弧度） |
| `family_samples` | 32 | 既有族求根/裁剪采样 |
| `family_budget` | 4096 | 既有族数值求根预算 |

这些预算不包含上游结构构建成本；其输入由 `DPField.lattice_n` 单独记录。零预算用于受控失败，不能解释为成功空交。

## 点的接受证据

近邻、变号、近零局部极小、端点只提供候选；polyline 相近不是交点证书。数值搜索无候选返回 `not_found`，不宣称全局不存在。

低层 `spherical_newton.refine_pair` 是与 `Walker.refine_corner` 共享的**唯一**机械 Newton 实现。旧适配保持原回调顺序、浮点运算、30次上限、0.001大步拒绝、1e-14 determinant 闸、1e-16更新后停止和严格最佳残差点，失败后仍返回旧点。S6 则要求成功的局部终止（残差1e-11）后另验：

1. 有限单位 u，`|u·u-1|<=1e-12`。
2. **原始约束**而非仅等价平面残差均<=1e-11；全部原始 validity gates 均>=-1e-13（旧 walker 的闭包阈值）。因此 kink 也必须在正确反射入射支。
3. 指定 source arc 的局部成员及种子位移：0.012弧度以内（arc 采样成员用相应弦距邻域）。它略大于两个 upstream 0.25°步长，仅约束局部性，不充当根精度尺子。
4. 数值横交的原函数切空间 Jacobian 有限，最小/最大奇异值比>1e-7。即使用等价平面细化，输出秩证据仍来自原 margin Jacobian。

非有限、奇异、耗尽预算、大步及迭代失败不得因 best-so-far 点看似合理而成功。小 determinant 不证明相切；切触须由既有解析圆交或族 jet+双侧核验承担，未解决接触保持 unresolved。解析正判别式双根不按近邻并根；数值重复种子只在接受后按1e-10角距去重，近退秩仍拒答。

D_P 值沿用 walker 的闭包求值：exit-Snell 舍入负判别式用既有丢根极限，不 epsilon 内推、不伪造奇异梯度。未提供数值正则 Jacobian 时明确 `differential.available=false`：解析圆证据用 `analytic_evidence_no_differential`，点分层成员检验用 `upstream_stratum_membership_no_differential`。这表示未计算数值微分，不声称几何非光滑或导数为零。

## 连续对象与结果状态

连续重合须有结构/代数身份：同义 margin、同一解析圆平面、或展开入射法向平方相同。例如 `3-5-6-7` internal-1 TIR 与 exit-Snell 判别式符号相反、**零集相同**，输出经过全部门校验的连续 upstream runs，不把多个落点当多个 junction。裁剪仍有上游采样覆盖限定。跨周期采样接缝的 accepted run 合并，`closed=true` 区分完整闭环与裁剪开弧。幸存单样本不能证明连续弧或切触，须以 `dropped_short_run` 和位置给出诊断，不静默丢弃。`3-5-6-7-3` 的 internal-1/internal-3 重合也保留两个反射事件身份。

每对给 `sources/status/coverage/records/curves/diagnostics`；数值对另有 candidate/attempted/converged/rejected 计数。点含 u、来源/原始残差、最小门、D_P、交型、证据、微分可用性；连续对象含 arc/sheet、有序样本、维数及证据。

| status | 语义 |
|---|---|
| `computed` | 局部接受的孤立点，不代表穷尽 |
| `continuous` | 一维对象，不是有限点列表 |
| `disjoint` | 解析圆模型在球面无根，coverage 限定模型范围 |
| `not_found` | 在既有选定结构/搜索中未发现可接受交 |
| `unresolved` | 候选失败或预算耗尽，未接受点 |
| `partial` | 已接受对象，同时存在显式未解/上游失败 |
| `unsupported` | 精确族或临界分层没有可信适配 |
| `unsupported_continuous` | 已知临界圆重合，但没有支持的裁剪适配 |

后两者不是 computed empty。仅定义好的 `RestrictedEscape` 映射为具名诊断；其它异常继续使导出失败，不用宽泛 catch 吞编程错误。部分成功不得吞上游失败或未尝试候选。解析模型的 `disjoint` 证明独立于 upstream arc 的采样成败：失败诊断继续保留，但不把已证无交降级为 `unresolved`。点成员检验不执行 Newton，因此即使命中，其 candidate/attempted/converged 计数仍为零。

## 实测覆盖与交付

M1 有界搜索取11条既有路径×完整 n(400/550/700)，共33例，lattice20000；每对至多8个独立 scipy roots、每根100函数调用、总预算900秒。约21秒即找到所需存在性 witness，无需第二遍搜索；不据此判断其它类型/路径无交。

| fixture label | 路径 | 核验 |
|---|---|---|
| `mc_s6_corner` | 3-5 | 旧 gate×gate 角点 |
| `mc_s6_gate_kink` | 3-1-6 | entry gate×基面 TIR onset |
| `mc_s6_kink_pair` | 4-8-1-7-5 | 不同反射1×2的内域正则交 |
| `mc_s6_continuous` | 3-5-6-7 | exit-Snell×internal-1 连续重合 |
| `mc_s6_plate_disjoint` | 3-1-6 | 族圆与kink圆解析相离 |
| `mc_s6_column` | 3-1-6 | 二维族像交仍为曲线，两个姿态sheet |
| `mc_s6_iteration_failure` | 4-8-1-7-5 | 零迭代受控失败，不伪造根 |

n(550) 下，独立三约束求根的反射1×2位置约为 `[-0.2824757039268523,0.44883472231481775,-0.8477941193094051]`，原始残差约2.1e-16、最小门约0.02699、原函数切空间最小奇异值约0.2766。fixture 不钉奇异集 A/T 正性。

加法 `junction` kind 保持 schema v1 与全部旧配方。7个文件各含400/550/700nm独立完整指数段。输入、可用性、比较与第二链 witness 规范见 [parity 合同](analytic-parity-fixtures_zh.md#65-s6-junction波次-3-第二批)。不能只靠自重算：还须独立核原始光学 margin、全部门、球面与真实 SO(3) 族原像。
