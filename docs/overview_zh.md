# Lumice Integral 总览

English version: [overview.md](overview.md)。两份内容同步维护；若有出入，以英文版为准。

Lumice Integral 用确定性的数值积分计算冰晕的辐射亮度。它不是通用的求积库，也不是又一个 Monte Carlo 光线追踪器。它要解决的核心问题是：

> 给定入射光方向、一个出射的像方向和一条光路，找出实现这一方向映射的**全部**晶体姿态，并在这个姿态集合上积分物理贡献。

项目起点是重建作者当年为写作系列《现代冰晕研究漫谈》第 6 章做的直接积分原型。原型的求解器源码已经遗失，只留下渲染数据和诊断图。现在它也为该系列后续章节（halo map、Jacobian、焦散、光路类、姿态族）提供数值工具。

## 1. 渲染冰晕的两种方式

两种方式计算的是同一个量：一群随机取向的晶体向每个天空方向送出的光，

$$
I(\mathbf d) = \sum_P \int_{\mathrm{SO}(3)} \rho(R)\,A_P(R)\,T_P(R)\;
\delta_{\mathrm{Dirac}}\big(\mathbf d,\ F_P(R)\big)\,d\mu_{\mathrm{Haar}}(R),
$$

对光路 $P$ 求和。$R$ 是晶体姿态，$\rho$ 是姿态密度，$A_P$ 是入射测度（晶体截面中有多少能喂进这条光路），$T_P$ 是光学透过率，$F_P$ 是从姿态到出射方向的 halo map。

**Monte Carlo 正向渲染**（[Lumice](https://github.com/LoveDaisy/ice_halo_sim)）：采样姿态和光线，让每条光线穿过晶体，在出射处累加。δ 函数靠像素分箱来处理。它通用（任意晶体、任意光路、多次散射、任意光源），也容易让人信任；但有噪声：精度只随光线数的平方根增长，暗的或窄的特征需要非常多的光线。它也很少告诉你冰晕*为什么*长这个样子。

**直接积分**（本项目）：固定出射方向，反解姿态。对一条光路，δ 函数把 $\mathrm{SO}(3)$ 切成姿态集合 $F_P^{-1}(\mathbf d)$，一般是若干条闭曲线；像素值是沿这些曲线、以法向 Jacobian 的倒数为权的线积分（coarea 公式）。它是确定性的：误差是数值误差并被报告出来，而不是采样噪声。它还把结构暴露出来：纤维、Jacobian、fold 焦散、每条光路和每个姿态族各自的作用。代价是求解器必须找到每个纤维的每个连通分量，而且每条光路都要显式处理。

两者是同一物理的两份独立实现，所以它们一致本身就是证据：Lumice 是本项目在原语层与 LI 仍在研究的模块上的外部验证 oracle；成熟的算法模块之后可以改经 Lumice 的共享库消费（§5.1、§5.3）。

## 2. Phase I 与 Phase II

**Phase I**（[phase1_zh.md](phase1_zh.md)，2026-09-23 收口）沿用原型：对每个像素，在 $\mathrm{SO}(3)$ 上用带自动微分的预测-校正 continuation 追踪姿态纤维，用一张预计算的姿态表发现全部分量，再沿纤维对各具名因子积分。它复现了第 6 章的切弧条带（251 × 801），与 Lumice 在形状上一致到 Monte Carlo 噪声底，绝对尺度一致到 `0.997-0.999`。它是逐点参考。

**Phase II**（[phase2_zh.md](phase2_zh.md)）利用了一个 Phase I 看不见的事实：所有几何与光学权重只经由 $\mathbf u = R^{-1}\hat{\mathbf s}$（光源在晶体系中的方向）依赖于姿态。积分因此搬到球面 $S^2$ 上，像素值变成一个标量场——偏折角 $D_P(\mathbf u)$——的水平集上的线积分。它有两种求积：

- **带求和**（2026-09-23 起投入生产）：对每种晶体、光路和波长，在 $S^2$ 上预计算一次 $N$ 个事件（与光源无关），再对落在每个像素偏折角带内的事件求和。canonical 条带在 Mac 4 个 worker 上 `31 s`（2026-09-24 的 scatter 形态；此前 `169 s`），Phase I 需要 30 个 worker 跑 `34.7 min`；
- **等值线法**（里程碑 M2，2026-09-25 完成）：追踪水平集并沿其积分；配合 $D_P$ 的临界点，它能证明所有分量都已找到，这是 Phase I 做不到的。

同一个积分的三种求积（Phase I 纤维、带求和、等值线），失败模式互不共享，再加上 Lumice 的 Monte Carlo，构成本项目的验证网。

## 3. 规划

里程碑及其对应的写作章节（当前状态与近期队列见 [roadmap.md](roadmap.md) §0，决策见 roadmap §9）：

| 里程碑 | 内容 | 章节 | 状态 |
|---|---|---|---|
| Phase I | $\mathrm{SO}(3)$ continuation 渲染器、第 6 章条带、光路类、五个姿态族、绝对尺度 | 6、7-9、11 | 2026-09-23 收口 |
| M1 | $S^2$ 事件仓库与带求和渲染器投入生产；$D_{6h}$ 搬运；约定与对称性权威 | 6、11 | 2026-09-24 完成 |
| M2 | 等值线求积、临界点与完整性证书、交叉验证、第 10 章数值裁定 | 10 | 2026-09-25 完成 |
| — | 内部反射按 Fresnel 分裂（原 TIR-only 门丢掉了全部部分反射支），仓库 schema 4 | 8、10、11 | 2026-09-25 完成 |
| 下一步 | 第 11 章全表（光路类 × 姿态族），配合按偏折角组织的带求和 | 11 | 排队 |
| 以后 | 发散光（街灯）、有限日盘、多波长 | — | backlog |

明确的非目标：

- 通用的数值积分或微分几何库；
- 取代 Lumice 的 Monte Carlo 渲染器；
- 多次散射场景；
- 精确的低维取向测度；
- 生产级 GUI；
- 在 CPU 参考结果可信之前做 GPU 优化；
- 原语层或 LI 仍在研究的模块对 Lumice 的生产依赖，或在 §5.1 / §5.3 界定的有界共享库消费之外抽取 Lumice 引擎 API。

## 4. 架构分层

项目大概需要以下几个概念层，但这不意味着每层都要立刻成为一个包：

1. **独立的可微光学求值器**：项目自有的几何与光学，把姿态、光路、波长映射到出射方向、有效性、具名物理权重，以及 AD 需要的光滑计算图。有限晶体几何（凸多面体、光路展开、走廊投影求交、光路枚举、单姿态有效入射截面 `entry_measure`）归 `lumice_integral.geometry` 所有。该子包是纯 numpy，位于可微图之外；它提供具名物理权重所乘的几何因子 $A_P(R)$，是指定的权威实现。写作项目在其任务 `geometry-depend-on-lumice-integral`（W1，roadmap §3.6）切换之前仍自带一份副本。
2. **对称性与组合分类**：晶体点群及其分类的对象归 `lumice_integral.symmetry` 所有（任务 `symmetry-authority`，2026-09-24）：作为面置换的 `D6`、带公开编号 #1–#12 的折叠群 `G`、共轭类与特征值类、`D6h` 元素表 `signature.D6H`（仓库内唯一一份；`path_class.hexprism_symmetry_matrices` 返回它）、canonical signature 与 `Phi` 类（棱柱上 34 类）、第 3 章 ground-truth 列表和柱晶姿态。纯 numpy，依赖 `geometry` 而不反向依赖（有静态测试）。有一条跨子包的边是有意保留的：`geometry.unfold._wedge_angle_deg_from_normals` 的存在是为了让 `geometry.wedge_angle_deg` 与 `symmetry.signature.wedge_angle` 共用同一个楔角内核；它不是 geometry 专用的辅助函数，不能因为 geometry 内部没用到就删掉。写作项目在其任务 33（W3，roadmap §3.6）中切换过来。
3. **微分求值器**：对局部 SO(3) 坐标的导数，最好与值求值器出自同一组方程。
4. **纤维求解器**：seed 搜索、预测-校正 continuation、分量发现、闭合与诊断。
5. **积分器**：感知参数化的线求积与误差估计。
6. **图像驱动**：光源、光谱、姿态分布、像素模型、缓存与输出组装。

边界应当从第一个能跑通的切片中长出来，而不是要求在追踪第一个环之前先搭好六个框架。

### 4.1 晶体的闭式构造与晶体自身对称群

两个 phase 都从 `lumice_integral.geometry` 取晶体，所以它的构造写在这里而不是某个 phase 文档里（任务 `crystal-closed-form`，scrum `crystal-native-geometry`，2026-09-27；conventions #19）。

- **语义跟 Lumice**（`doc/configuration.md`）：棱柱有高度与 6 个 `face_distance`，后者以正六边形边心距 `a·√3/2` 为单位、允许负值；`HexPrism(a, h, face_distance)`，Lumice 的 `height = h / (2a)` 用 `HexPrism.from_lumice(height, face_distance, a)`。锥晶按 Lumice `pyramid` 的全部语义由 `Pyramid.from_lumice(prism_h, upper_h, lower_h, upper_indices, lower_indices, face_distance, a)` 构造：上下高度独立（0 = 无锥、以底面封口，`(0, 1)` = 截顶，`>= 1` = 锥到顶点，负值折叠），上下 Miller 指数或楔角独立，六个 `face_distance` 作用于棱柱段与两个锥台的每一层截面（task `pyramid-lumice-semantics`）。`Pyramid(a, h, c_over_a, tip_ratio)` 仍是对称、正六边形截面的子集，`from_lumice` 在该子集上直接委托给它。
- **闭式，不从数值上发现拓扑**（Lumice 在 PR #214 弃用的路线，`doc/crystal-geometry-representation.md` §1、§4）：6 个侧面是固定的方向星，每个侧面的直线被其余 5 个半平面截成一个解析区间，区间长度为正（尺度相对 `1e-9`）即该面*存在*。角点是相邻存在面的交点，写成正六边形角点加精确的线性修正，所以正六棱柱与历史实现逐位相同。面号是常量表 1/2、3–8（锥晶另有 13–18、23–28）中存在的那部分；查不存在的面号是 `KeyError`，不会静默返回错误法向。
- **锥台是同一个方向星的侵蚀**（Lumice 的模型，`src/core/geo3d_closedform.hpp`）：肩部之上侵蚀量 `m` 处的截面是比例为 `face_distance − m` 的方向星，高度 `h/2 + a·c_over_a·m`。每条边以闭式速率缩短，组合结构只在某条边缩到零长（角点死亡事件，三条侧线共点）处改变，直到自然锥顶——一点或一条棱脊（`closed_form.cone_sweep`，与 LP 对照）。截顶 `upper_h` 位于 `m = upper_h·m_apex`；顶点是肩部环、每个事件一点、截顶环或锥顶。
- **拒绝即异常**：存在的侧面少于 3 个意味着横截面没有面积，构造函数抛 `ValueError`（Lumice 是丢弃晶体）。只看每对对置面宽度为正是不够的：`[1, 1, -0.5, -0.9, -0.9, 1]` 三条板带宽度都为正，却没有公共部分。
- **`G_true`**（`symmetry.crystal_group.true_symmetry_group`）：所有法向都在六方向星、`±c` 或方向星上方的锥面上，只被 `D6h` 置换，所以晶体自身对称群就是 `signature.D6H` 中把存在面的（法向，相对顶点质心的偏移）集合映到自身的那些元素，并核验为群。正六棱柱与对称双锥 24，`[1, d, 1, d, 1, d]` 12，`[1.9, 1, 1, 1.9, 1, 1]` 与 `[2, 1, 1, 2, 1, 1]` 8，一般棱柱 2（`{E, σh}`）；上下锥不同（高度或指数）或单侧锥的锥晶 12（`C6v`），再配上述截面分别为 6、4、1。
- **法向从晶体读**（任务 `optics-reads-crystal`，2026-09-27）：`optics` 的全部单光路函数（`path_direction`、`path_domain[_batch]`、`fresnel_transmission_path[_batch]`、`path_problem` 及 `path_3_5*` 薄包装）接受 `crystal`（默认正六棱柱 `HexPrism()`），面 `f` 的法向经唯一查询 `optics.face_normals` 取 `crystal.normal(crystal.face(f))`；面 1–8 的法向若与其闭式星方向（`±c`、方位 `i·60°`）相差不超过 `1e-12` 则取精确星方向（Newell 公式与之差约 1e-16），所以正六棱柱与 ch06 fixture、strip 管线、$S^2$ store 所钉的历史常量表逐位一致，偏离星方向的面保留自身法向；`HEXPRISM_BODY_NORMALS` 是这一查询在正六棱柱上的结果，留给按面号查表的调用方。`D_P` 核函数（`dp_field`、`contour`、`contour_quadrature`、`ch10_verdicts`）把法向当作被追踪的数组参数（`DPField.normals`），同一面序列只编译一次、所有晶体复用。凡已持有晶体的消费者都把它传到底（`DPField`、`focusing.classify`、`weights`、`s2_store.evaluate_fields`、discovery、strip 场景）；引用晶体没有的面是 `ValueError`，锥晶面 13–28 能跑通 `focusing.classify`。边界行走的 margin 恒等（`identical_margins`）按晶体法向核实，不再只凭面号假定。
- **约化簇改用晶体自身的 `G_true`**（任务 `reduction-cluster-g-true`，scrum `crystal-reduction-generalize`，2026-09-27；是上面 fail-fast 裁定的阶段 2）：`path_class` 的 `pbd_orbit_hexprism`、`phi_key`、`path_class_symmetry`（`s2_store`、`strip_pixel`、`band_sum` 只经由它们接触晶体对称性）不再要求 `|G_true| = 24`，改用 `true_symmetry_group` 作候选群；显式传入的 `symmetry_elements` 仍须属于 `G_true`。`D6h` 折叠表与六方向星（`phi_key` 的方向编号）保留为与晶体无关的查找表——光路引用晶体不存在的面是 `ValueError`（`_require_faces_of`），不是静默合并。独立 oracle（某成员自建 store 对照另一成员的对称搬运）在 `D3h`（阶 12）、`D2h`（阶 8）与一般阶 2 棱柱上一致到 `1e-12`（`3-5` 反例：12 条 `D6h` 像收敛为晶体真实的 1 成员轨道）；正六棱柱渲染逐位不变，只有 store 缓存 key 变化（schema 4→5，新增 `face_distance`）。对两种低对称棱柱做了对 Lumice Monte Carlo 的绝对尺度独立验证（任务 `low-symmetry-lumice-validation`）：6 组"晶体 × 姿态族 × 光路类"组合（random/Parry 姿态，类 `[3,5]`，`D3h` 棱柱另加 `[3,5,6,7]`）总通量比一致到 `4e-4`，均在两 seed 噪声底内，未拟合任何常数；该验证同时发现 Lumice 自身的 `P`/`B`/`D` 约化在 `D6h` 以下会把不等价光路错误合并（`D3h` 上 `[3,5]` 实测 `1.41` 倍），是 Lumice 侧缺陷，已记入其自身 backlog，不是 LI 的缺陷。锥晶的 store 路径仍保持 fail-fast（任务 `pyramid-lumice-semantics`）。
- **在验证边界上对照 Lumice**：`scripts/verify_crystal_closed_form.py` 经 `ctypes` 调 Lumice 的 `LUMICE_GetCrystalMesh`，在临界形状（远面恰擦角点）和 3000 组随机 `face_distance` 上对照存在面、法向与角点（含拒绝判定，全部一致），并行跑独立几何校验（半空间包含、`V − E + F = 2`）。

## 5. 与其他项目的关系

### 5.1 Lumice

Lumice 是正向 Monte Carlo 模拟器，也是首要的独立验证 oracle。Lumice Integral 是采用不同数值方法的兄弟产品线。

Lumice Integral 对 Lumice 的依赖按角色与模块成熟度分层（仓库定位裁定，§5.3）。**原语与约定层**——晶体几何、面编号、对称约化与 `G_true`、Snell / Fresnel 光学——保持永久独立：LI 在这一层**不**使用 Lumice 的源码、C API、库或可执行文件，求解器自己拥有从姿态经几何、光学到方向、物理权重和导数的完整可微路径。在这一层保留两份独立实现是有意为之，不是等待未来共享引擎的临时重复：2026-09-27，Lumice 把 `fn_period_` 硬编码为 6（不读 `face_distance`）的缺陷，正是因为与 LI 的独立实现出现分歧才被发现（Lumice PR #429）；若两边共用一份原语实现，这类缺陷会让两边一起错而不可见。这条实证具体针对的是光路分析面板的**物理**分组（L2，`docs/conventions.md` #21）：周期 6 正是 Lumice `symmetry: "PBD"` 标号 filter（L1）的本意——不受约束、与晶体形状无关；PR #429/#430 对 filter 本身的改动已被 Lumice PR #436 恢复，该 PR 同时点名了两种含义（`doc/raypath-symmetry.md` §1.1），把 filter 恢复为原本无条件的周期 6。实证本身仍然成立，因为当时对照的正是面板行（L2）口径。对称约化不在两仓共享边界之内：Lumice 即将发布的 `liblumice_analytic` 只接受具体面序列，不提供任何对称约化，两仓各自按自身消费者的需要（L1 或 L2）保留一份对称约化层。

**算法层**——单光路反解与 fiber 行走、$S^2$ 事件仓库、带求和、`dp_field` / `focusing` 临界点分类——遵循不同的规则：模块还年轻、仍在研究驱动下变化时，两边各自实现（JAX 权威、C++ 派生，由 parity fixture 锁定；§5.3「计算落点」裁定）。模块成熟、且 LI 不再研究它之后，收敛为 Lumice 正式发布的一份共享库背后的单一 C++ 实现；LI 经 Python binding 消费它，自己的 JAX 版随之退役。LI 仍在研究的模块，不论 Lumice 一侧状态如何，继续按 JAX 权威维护。

下面这段原因只适用于原语层与 LI 仍在研究的模块。Lumice 为正向随机采样和图像累加而优化；LI 这部分需要一个固定光路、分片光滑、适合自动微分和 continuation 的计算图。换面、遮挡、折射域界限和入射/出射面 TIR 边界必须表示为光滑分支周围的显式事件；一个不透明的 Lumice 调用会切断这张图，而穿过它做有限差分，在这些边界和 halo map 奇点附近提供不了可信的基础。

Lumice 只能跨越显式的验证边界使用：

- 独立运行，产出收敛的 Monte Carlo 图像或剖面；
- 导出光路分析数据用于比较；
- 核对共享的物理与坐标约定，如面编号、方向符号、折射率、Fresnel 因子和姿态分布。

验证工具在明确要求时可以调用 Lumice 并读取其文件，但构建或运行 LI 的原语层、或 LI 仍在研究的模块，绝不能以 Lumice 为前提。这一层的两份独立实现加强了交叉验证：当一致性不可能来自共享的几何或光学 bug 时，它才更有意义。

### 5.2 《现代冰晕研究漫谈》

写作项目为求解器提供动机并消费其结果，但不拥有其实现。Lumice Integral 最终应重新生成第 6 章的直接积分条带和状态空间诊断图，并为后续章节中关于 halo map、Jacobian、纤维和焦散的讨论提供数值证据。

写作项目有两层代码现在住在这里并以此为权威：晶体几何（`lumice_integral.geometry`，2026-09-16）与对称性和分类层（`lumice_integral.symmetry`，2026-09-24，从 `halo_notes.math` 迁来，公开名称 1:1 保留）。用本包重算的第 8、9 章 signature 表与已发表的 CSV 逐字节一致（`tests/test_symmetry_signature_table_regression.py`，只读写作仓的数据文件）。

### 5.3 Ice Halo 光路分析面板

Owner 需求（权威记录在 Ice Halo 仓
[`doc/raypath-analysis.md`](https://github.com/LoveDaisy/ice_halo_sim) §5.1，
由该仓 chore `raypath-analysis-lumice-integral-plan` 同日（2026-09-27）改写）：
用户在面板中选定一条光路之后，跟进三个功能：

1. **光路详情**：一维姿态族（fiber）上的晶体姿态、晶体内光路轨迹（可视化 / 动画）、
   逐段衰减的能量分配（beam tube 是它的一种可视化形式）；
2. **单光路全天亮度图**：单独这条光路在全天的亮度分布，并可在图上重新选点回到功能 1
   （功能 2 的产品形态 owner 暂 hold）；
3. **预设点**：最亮点、临界点 / 线（$D_P$ 极值 / 鞍点、$U_P$ 边界，如切弧边缘）、
   波长相关临界点（例如蓝光已过 TIR 而红光仍透过的位置）。

多次散射链需要支持。

三个功能是同一个 $S^2$ 对象的三种读法（这是应当写进 LI 自己文档的部分，不是复述面板的
产品设计）：功能 2 = 带求和；功能 1 = 该像素对应的水平集
$\{D_P(\mathbf u) = \delta\}$（带求和里落进该像素偏折角带内的事件即这条 fiber 的离散化，
`band_sum.band_poses` 已能取出这些姿态）；功能 3 = $D_P$ 的临界结构加 $U_P$ 边界
（`dp_field`、`focusing.classify`），波长临界即 $U_P$ 边界随 $n(\lambda)$ 移动。三者
共享同一张按「晶体 × 光路 × 波长」预计算、与光源无关的 $S^2$ 场。「为什么冰晕长在这里」
这一产品形态 = 功能 3 的预设点加机制标签——这是 Lumice 的 Monte Carlo 形态给不出的、
LI 独有的价值。

多次散射**场景渲染**仍在 LI 范围之外（§3 的非目标条目不改）；但两层散射链可以在消费层
经中间方向 $\mathbf m$ 把单层结果组合起来：$S^2$ 仓库与光源无关，所以第二层可以把
$\mathbf m$ 当作光源复用同一个仓库。这一组合层是否由 LI 自己承担尚待定；本段只记录
需求来源，不构成范围变更。

**计算落点裁定（2026-09-27 owner 裁定，收敛 explore `panel-inverse-probe` 与
`ad-port-probe`；本地记录
`scratchpad/explore-panel-inverse-probe/SUMMARY.md`、
`scratchpad/explore-ad-port-probe/SUMMARY.md`）：** 这不是一次性的「单实现 vs
双实现」架构选择，而是按模块成熟度决定何时移植。

实测：面板精度下算力充裕（$161\times81$ 窗口、$N=10^6$、全链路 $3.4\,\mathrm{s}$；
单像素取 fiber $1$–$2\,\mathrm{ms}$；对比：Python 渲染约
$74\,\mu\mathrm{s}/\mathrm{px}$，$512^2$ 全天图约 $20\,\mathrm{s}$）。功能 1/2
核心（`band_sum`、`s2_store`、`weights`、`geometry`）零 JAX 依赖。功能 3 的 AD
可精确规约为 C++ 前向 hyper-dual 模板（`Jet2<3>`，约 $150$ 行零依赖）：3 条路径
（含一条内反射、一条锥晶族 `13-15-26-28`）与 JAX 的值 / 梯度 / Hessian 相对误差
$\le 10^{-11}$；单点 C++ $193\,\mathrm{ns}$ vs JAX $104\,\mu\mathrm{s}$。功能 3
的 C++ 估 $2800$–$3600$ 行，大头是边界 / 水平集 walk 状态机而非 AD 本身。

裁定：

1. **现在不移植。** 全部留在 LI 的 JAX 实现里，研究与写作自由改动。
2. **移植触发** = 两个条件同时满足：面板功能真正排期；该模块一段时间无语义改动
   （参考：阶段 2、非对称锥、多波长、多次散射组合都已落地之后）。
3. **移植后：JAX 是权威、C++ 是派生实现**，由 parity fixture 锁定（路径拓扑：
   无内反射 / 含内反射 / 锥晶族外，交叉点类别：随机 / 临界点 / 近边界；模板 =
   explore `ad-port-probe` 的 `compare_*.py`），CI 运行；改动单向流动——先改
   JAX → parity 红 → 再改 C++。这不是两份对等实现静默分叉：分叉被自动检出，
   不存在「哪边是对的」之问。
4. 功能 1/2 与功能 3 同样处理（`band_sum` / `s2_store` 一样年轻）；C++ 侧同样
   采用前向 hyper-dual 模板做标量求导，不引入 AD 框架。
5. 某模块长期稳定后，是否让 LI 也改调 C++ 并退役 JAX 版：由下方的「仓库定位」裁定回答——算法层是，原语层否。
6. 取代此前「主 session 倾向 (c)、抽可移植 C++ 核心」的表述：(c) 的形态保留为
   移植后的终态候选，但时机由第 2 点的触发条件决定，而非预先定死。

**对 LI 现阶段的约束：** 面板相关模块（`optics`、`weights`、`geometry`、
`s2_store`、`band_sum`、`dp_field`、`contour`、`focusing`）现阶段照常演进，
不因面板而冻结；某模块移植后，其改动须先落在 JAX 侧并经 parity fixture 同步
校验，再触碰 C++。

已于 2026-09-28 裁定（`doc/raypath-symmetry.md` §1.1，chore
`symmetry-two-meanings-docs-and-comments`）：默认**不是**同一口径，不可混用。
面板的对称约化行是**物理**含义（L2）；Lumice `PBD` 标号 filter 是另一种、与形状无关的
标号重写（L1），只在晶体 `G_true` 为整个 `D6h` 时与 L2 重合。LI 的 `G_true` 轨道
（`symmetry.crystal_group.true_symmetry_group`）是 L2 的形状半边；LI 自己的 L1
（`symmetry.reflection_group.pbd_orbit`）是另一份独立实现。完整对照表见
`docs/conventions.md` #21。

**仓库定位（owner 裁定 2026-09-28）：** 该裁定在 Lumice 一侧的对应改动记在该仓
[`doc/raypath-analysis.md`](https://github.com/LoveDaisy/ice_halo_sim) §5.1.6 与
[`doc/api-layering-and-product-lines.md`](https://github.com/LoveDaisy/ice_halo_sim)，
由该仓自己的 chore 同日改写；引用章节号即可，不要断言其内容已合入。

1. **按角色划分，替代按「正向 / 逆向」划分。** 原先的二元划分其实是三个维度碰巧
   重合：数值方法（正向 MC / 逆向确定性积分）、角色（产品 / 研究）、运行时
   （C++ / Python-JAX）。光路分析升级为 Lumice 的第二产品核心，引入「逆向方法
   + 产品角色」这个新组合，按方法划分失效，按角色划分仍成立。**Lumice = 产品**：
   终端用户能用到的全部计算，C++，每个语义一份产品实现——包括光路分析运行时要
   用的逆向能力（单光路反解与 fiber、$S^2$ 场、带求和、临界点分类）。
   **LI = 研究与参照**：新方法诞生之处、数学规格、写作系列支撑，同时是 Lumice
   的独立校验对象。LI 的其余非目标不变（不做产品 GUI、不做多次散射场景渲染、
   不替代 Lumice 的 MC 渲染器）。
2. **共享判据：「稳定，且不是两边互相校验的对象」。** 原语与约定层两边刻意各留
   一份，作为互相校验的对象（§5.1 的 `fn_period_` 实证）；算法层在模块成熟、
   LI 不再研究它之后，收敛为 Lumice 一份 C++ 实现背后的共享库，详见上文 §5.1。
3. **共享库与时序。** Lumice 发布的是一个新的窄接口，只含稳定一侧（几何 / 光路 /
   逆向求解相关），不是 Lumice 现有面向 GUI 的 C API；LI 是它的第一个外部消费者。
   Lumice 先做发布基础设施；共享库的实际内容随第一个成熟算法模块一起进去。
   **第一个候选：单光路反解 + fiber 行走**（LI Phase I 已于 2026-09-23 收官，
   语义稳定；Lumice 的 Analyze 工作区一期本来就要在 C++ 里实现它，对照 LI 做
   parity）。LI 对 Lumice 的依赖因此是有界的：只针对已退役 JAX 版的成熟算法
   模块，经 binding；原语层与仍在研究中的模块，LI 的构建与运行不依赖 Lumice。
   验证工具调用 Lumice 做对照的既有规则不变。

**波次推进（owner 裁定 2026-09-28，作者与 owner 跨仓讨论定下；权威记录本仓
`scratchpad/scrum-analytic-lib-wave1-spec/scrum.md` §1）：** 上述共享库时序
具体化为三个波次，每波 Lumice 先落地一个模块，LI 晚一波切换依赖：

| 波次 | Lumice 共享库 | 服务的 Analyze 功能 | LI 侧 |
|---|---|---|---|
| 1 | 模块 A v0：`EvaluatePath` + **seed 搜索** + `TraceFiber[Batch]`，只返回点列 | 功能 1 光路详情 | 写 seed 搜索（discovery）契约；导出 parity fixture；研究并稳定诊断/权重契约。**不切换** |
| 2 | 模块 A v1：加诊断 + 权重（`struct_size` 兼容扩展）；模块 B：单光路 S² 仓库 + 带求和 | 功能 2 单光路全天图 | 按 `docs/phase1-math-contract.md` §11 conformance 认证 A v1 → 切换 fiber 求解、退役 JAX continuation；写作仓传递依赖按 `.lumice` release 拉取模式接入；B 只做 parity 不切换 |
| 3 | 模块 C：`dp_field`/`contour`/`focusing`（C++ 用 `Jet2` 前向 hyper-dual） | 功能 3 预设点与机制标签 | ch12/12.1 用完、不再研究后，先切 B 再切 C |

1. **v0 含 seed 搜索**（作者判断：合理）；不需要 Lumice MC 记录光线姿态——Analyze
   全天图低分辨率，seed 密度可先低后渐进加密，交互上不构成 blocker。
2. **诊断字段两步走**：v0 只返回点列；LI 同时把 `docs/phase1-math-contract.md`
   §9.3 的诊断/权重契约研究稳定，波次 2 再进库。
3. **parity fixture 改动方向 LI → Lumice**：LI 按固定 rev 导出，Lumice 拷入并在
   CI 跑。
4. **写作仓对 C++ 库的传递依赖**沿用 `halo_notes.sim` 的 `.lumice` 按版本拉
   release 模式（波次 2 落地）。
5. **LI 切换某模块的判据**（三条同时满足）：conformance 认证通过；LI tasks 中
   无进行中的针对该模块的研究；无计划中的需求要对它做 AD（例如 ch14 可微渲染
   若要对晶体形状参数求梯度，相关模块不得退役 JAX 版；只对姿态密度 ρ 求导不受
   影响——前向模型对 ρ 线性）。

波次 1 的落地任务见 scrum `analytic-lib-wave1-spec`（本仓 `scratchpad/`）。

待后续核实（本 chore 不做，只标记）：Lumice 的 Analyze 工作区设计里「太阳方向球
上的水平集 = fiber」的一一对应，在锥晶与含内反射光路上是否成立，由 LI 核对
（来源：Lumice `doc/raypath-analysis.md` §5.1.8，已标为 assistant 推断）。

## 6. 验证策略

验证必须组合几类相互独立的证据：

- 可解析处理的映射与对称情形；
- 沿每条已追踪纤维的局部残差与切向检查；
- 在条件良好的姿态上，AD 导数对有限差分；
- 求积加密与重参数化不变性；
- 遗留的历史直接积分输出，只作形态与来源记录（作者 2026-09-20 裁定：历史 raw 不再作为正确性参考，以与 Lumice 一致为准）；
- 在对齐 Monte Carlo 不确定度、光源模型、光谱、姿态密度、投影和辐射度归一化之后，与 Lumice 一致；
- 在验证边界上显式的约定 fixture，不共享几何或光学实现；
- Phase II 交叉验证：带求和对 Phase I（已完成，M1），等值线法对前两者（M2）。

一张看起来光滑的图不是充分证据：漏掉的纤维分量会产出看似合理、实则系统性错误的亮度。

## 7. 文档

| 文档 | 作用 |
|---|---|
| `overview_zh.md`（本文） | 项目是什么、与 Monte Carlo 的区别、各阶段、规划 |
| [phase1_zh.md](phase1_zh.md) | Phase I 设计、流水线、关键转折及其理由（实测记录见英文版附录） |
| [phase2_zh.md](phase2_zh.md) | Phase II 设计：$S^2$ 积分、两种求积、事件仓库、成本、按偏折角组织的带求和、发散光（实测记录见英文版附录） |
| [roadmap.md](roadmap.md) | 状态、近期队列、与写作项目的耦合、决策日志（英文） |
| [phase1-math-contract.md](phase1-math-contract.md) | Phase I 规范性契约（坐标、测度、事件、接口、conformance；英文） |
| [conventions.md](conventions.md) | 每条坐标、符号与记号约定及其权威与检查（英文） |
| [ch06-reference-fixture.md](ch06-reference-fixture.md) | 第 6 章 fixture、来源分级、验收阶段、与 Lumice 的对照（英文） |
| [ch11-pose-density-families.md](ch11-pose-density-families.md) | 五个姿态密度族（英文） |
| [decisions/](decisions/) | 架构决策记录（英文） |
