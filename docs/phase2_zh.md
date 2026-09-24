# Phase II：$S^2$ 上的积分

English version: [phase2.md](phase2.md)。两份正文同步维护；若有出入，以英文版为准。实测记录只在英文版附录中保留，不翻译，以免两份数字各自改动后对不上。

Phase I 在 $\mathrm{SO}(3)$ 上为每个像素求解姿态纤维（[phase1_zh.md](phase1_zh.md)）。Phase II 利用问题的结构，把积分搬到「晶体系中光源方向」构成的球面上；在那里，它变成一个标量场的水平集积分。这个积分有两种求积：追踪水平集（等值线法），以及按偏折角带对预计算事件求和（带求和）。带求和已投入生产（里程碑 M1，2026-09-23）；等值线法是里程碑 M2（scrum `phase2-contour-quadrature`）。本文讲设计；决策按日期记在 [roadmap.md](roadmap.md) §9，实测记录见英文版附录。

记号遵循 [conventions.md](conventions.md)：$\hat{\mathbf s}$ 指向光源，入射传播方向是 $\mathbf s = -\hat{\mathbf s}$，$\Phi_P$ 在晶体系中把入射传播方向映到出射传播方向，$\psi$ 是绕 $\hat{\mathbf s}$ 的扭转，$\rho$ 是相对 Haar 的姿态密度。必要处标注每条陈述的性质：**实测**（附证据）、**推导**（由模型推出，尚未数值核对）、**设计**（拟议的组织方式，成本未测）。

## 1. 从 $\mathrm{SO}(3)$ 到 $S^2$

令

$$
\mathbf u=R^{-1}\hat{\mathbf s},
\qquad
D_P(\mathbf u)=\angle\big(-\Phi_P(-\mathbf u),\mathbf u\big)
$$

（写作系列 framework 定理 8）：$\mathbf u$ 是晶体系中的光源方向，$-R\,\Phi_P(-\mathbf u)$ 是天空上的亮点，$D_P$ 是它到光源的角距离。$\mathrm{SO}(3)$ 经 $R \mapsto \mathbf u$ 纤维化到 $S^2$ 上；剩下的坐标是绕 $\hat{\mathbf s}$ 的扭转 $\psi$，Haar 概率测度分解为 $d\mu_{\mathrm{Haar}} = dA(\mathbf u)/4\pi \cdot d\psi/2\pi$。

**所有几何与光学权重都是 $S^2$ 上的函数。** `entry_measure` 在第一行就把姿态约化为 `s_body = R.T @ incident` $= -\mathbf u$，此后别的什么都不用；TIR 门和 Fresnel 因子依赖入射角，也就只依赖 $\mathbf u$。让晶体绕光源方向转动，这些量一个都不变：扭转只是让出射方向绕 $\hat{\mathbf s}$ 刚性转动。所以入射测度 $A_P$、透过率 $T_P$、窗口 $w_P = A_P T_P$、有效性门、可行域 $V_P$ 和偏折角 $D_P$ 都是底空间 $S^2$ 上的场；只有 $\rho$ 看得见完整姿态。Phase I 沿一条 $\mathrm{SO}(3)$ 曲线逐点求这些量，是因为它看不见这个结构。

**实测**：在绕 $\hat{\mathbf s}$ 的三次扭转下，有效性、$A_P$、$T_P$、$\Phi_P$、$D_P$ 不变，差 `1.5e-14`（任务 `band-sum-quadrature-probe`）。

### 1.1 预计算依赖什么

因此一个事件 $(\mathbf u_i, \Phi_i, D_i, w_i)$ 是晶体系中一条纯粹的「入射方向 → 出射方向」记录。这类事件表依赖**晶体形状、光路（其 $\Phi$ 组，§3.2）、折射率（即波长）和采样（$N$、格点或采样器、偏折角窗口、dtype），与光源的任何性质都无关**。

- **实测**（owner 探针，2026-09-24，`scratchpad/task-s2-store-schema-3/owner_probe_sun_indep.py`）：$h/a = 2$、$n = 1.31$、$N = 2\times10^5$，太阳取 (15°, 0°)、(60°, 37°)、(−30°, 200°)；光路 `[3,5]` 与 `[1,3,2]`：事件数相同，$\mathbf u$ 逐位相等，$\Phi$、$D$、$w$ 差在 `1.6e-11` 以内。
- 推论：换太阳高度或方位、换姿态密度、换相机，都复用仓库、只需重新渲染；换波长要一份新仓库；换晶体形状或光路也要新仓库。
- 已投产（schema 3，任务 `s2-store-schema-3`，2026-09-24）：`S2StoreSpec` 与缓存 key 不再含太阳方向；构建时把 $R\mathbf u = \hat{\mathbf s}_0$ 对齐到一个固定参考方向，其数值取 canonical 太阳，以便逐位复现 schema 2 的 canonical 构建；一份仓库服务所有太阳高度。`tests/test_s2_store.py::test_events_are_independent_of_the_reference_direction` 把探针转成回归（三个方向 × `[3,5]`、`[1,3,2]`、`[1,3,5,2]`，保留的点相同，$\Phi$、$D$、$w$ 差在 `1e-10` 以内），`tests/test_band_sum.py::test_one_store_serves_every_sun_altitude` 钉住换一个太阳高度不重建。schema 2（key 含太阳方向、单个 `events.npz`）已逐位复现，加载时拒绝。

光源进入渲染只有两处：把事件放到某个像素上的那个姿态，以及该姿态下的 $\rho$。

## 2. 一个积分，两种求积

对偏折角 $\delta$、绕光源方位角 $\alpha$ 处的像素，把 $\rho A_P T_P\,d\mu_{\mathrm{Haar}}$ 前推到天空上。固定 $\mathbf u$ 时出射方位角是 $\alpha = \alpha_0(\mathbf u) + \psi$，所以 $d\psi = d\alpha$，而偏折角保持为 $D_P(\mathbf u)$。天空上 $dA(\mathbf d) = \sin\delta\,d\delta\,d\alpha$，由此得到 Phase I 的像素值（与契约 §7 同一个 $1/(8\pi^2)$）：

$$
I(\delta,\alpha)\,\sin\delta
= \frac{1}{8\pi^2}\int_{S^2}\delta_{\mathrm{Dirac}}\big(D_P(\mathbf u)-\delta\big)\,
  \rho\,A_P T_P\,dA(\mathbf u)
= \frac{1}{8\pi^2}\int_{D_P=\delta}
  \frac{\rho\,A_P T_P}{|\nabla_{S^2}D_P|}\,d\ell .
$$

固定像素时没有单独的 $d\psi/2\pi$：$\psi$ 被 $d\psi = d\alpha$ 用掉了，$\rho$ 在单值的姿态 $R(\mathbf u, \psi(\mathbf u,\alpha))$ 处求值。

右端有两种离散方式，它们是同一个积分的两种求积，不是互相替代的方案：

- **等值线法**（§4）：追踪水平集 $\{D_P = \delta\}$ 并沿其积分。逐点、确定性、高阶；配合 $D_P$ 的临界点，能证明所有分量都已找到。
- **带求和**（§5）：改为在一个带 $[\delta_{\mathrm{lo}}, \delta_{\mathrm{hi}}]$ 上积分，线积分就变成 $S^2$ 上的面积分，再用 $N$ 个预计算的等面积点求值。每个像素一次排序查找、一次区间查询、一次批量 $\rho$ 求值。

**边界不需要事件处理。** 被积函数在每条边界上都连续地归零：走廊边界（两个多边形分离，$A_P$ 连续趋于 0）；公式域 $U_P$ 的出射面 TIR 边界（Fresnel 透过率在临界角处趋于 0）；入射没有临界角；内部反射的部分反射是连续权重，不是边界。被 $\partial V_P$ 截断的等值线在 $U_P$ 上照常追踪，由 $A_P T_P$ 去掉不可行的部分；Phase I 的规则「继续追踪、权重归零」（契约 §6.3）就是把同一个事实放进了追踪器里。剩下的唯一非光滑性是 $A_P$ 的折点（某个顶点越过某条棱），它和 Phase I 一样会让局部求积降阶。

## 3. 球面揭示的结构

### 3.1 拓扑与完整性

标量场的水平集由其临界点支配：$\nabla D_P = 0$ 的点（有限个，由格点 seed 出发做 AD Newton 求得），加上 $D_P|_{\partial U_P}$ 的临界点，把 $\delta$ 轴切成若干区间，每个区间内水平集的拓扑不变。每个区间 marching 一次再 Newton 细化，就能得到*全部*分量；于是 Phase I 给不出的完整性证书（契约 C11：`completeness` 只是过程性的）变成了一个可检验的陈述。这是 Phase II 更大的收益，速度是较小的那一个。（**设计**；M2 子任务 `dp-field-topology`、`dp-field-layer`、`s2-contour-extraction`。）

**已实现：场层**（`lumice_integral.dp_field`，任务 `dp-field-layer`；对外只有 `DPField` 与 `TopologyEscape`，秩 0 路径在 `DPField.build` 这一处被拒绝，无法绕过）。针对一条面序列，与太阳方向无关：

- *求值。* 单位姿态下 $D_P(\mathbf u) = \angle(\Phi_P(-\mathbf u), -\mathbf u)$，批量（`jax.vmap`）给出值、切向梯度与 Riemannian Hessian $P(H - (\mathbf u\cdot\mathbf g)I)P$——曲率项使它与 $D_P$ 在球外如何延拓无关（缺了它，3-5 极小点会随延拓方式被读成极大或鞍点）。角度取 `atan2` 形式，与事件仓库的 `evaluate_fields` 一致到 `1e-12`。
- *折叠前置分流。* $|\mathbf n_a\cdot M^{\mathsf T}\mathbf n_b| = 1$ 为平板（slab）：$D_P(\mathbf u) = \angle(M\mathbf u, \mathbf u)$，按此闭式求值，其临界集为 $\pm\mathbf n_M$ 与折痕 $\mathbf u\cdot\mathbf n_M = 0$；否则内部临界点由阻尼切空间 Newton 给出（对 Fibonacci 格点在 $U_P$ 内的点批量迭代），按 Hessian 分类。
- *边界。* 沿 $\partial U_P$ 行走：沿当前 active margin 的零集前进、$U_P$ 在左侧，另一个 margin 变负处即角点（二分后做双 margin Newton，残差 `<= 2.3e-16`），再沿过该角点、能延续边界的唯一 margin 继续，直到回到第一个角点——行走闭合即该边界环完整的陈述。入射余弦 margin 的法向 $\mathbf m = R_{k-1}^{\mathsf T}\mathbf n_k$ 满足 $\mathbf m\cdot\mathbf n_a = 0$ 时它对 $\mathbf u$ 线性，按大圆闭式行走（每条路径运行时核验），否则与各判别式一样 marching。与前面某个 margin 恒等的（三次侧面反射、方位角步长相同 ±60°）先删去；沿整段为零的 margin（平方型，如 `3-5-6-7-3` 上 `exit_snell_discriminant` $=$ `entry_incidence_cosine`$^2$）记为重合。角点记录在该处为零的全部 margin、围出它的两条、其余与之相切或横截。
- *区间划分。* 临界值取内部临界值、$D_P$ 沿边界环的极值与角点值。每个区间上开弧数等于 $\delta$ 沿边界环穿越次数的一半（对任意拓扑都成立）；闭环需要内部极值，至多一个时（非退化极小或平板锥点），从它的值到子水平集首次触及 $\partial U_P$ 的环上极值之间恰有一个闭环（在小圆环上核验）。其余情形——$U_P$ 或其补集在格点上不连通、多个内部临界点、鞍点、折痕穿过 $U_P$、子水平集分量在边界上新生——抛 `TopologyEscape`，不猜。
- *核验。* `scripts/verify_dp_field_intervals.py` 在稠密网格上经 `evaluate_fields` 重算每个区间的计数：闭环取不触及边界的子/超水平集区域，开弧取沿追踪出的网格边界（经二分移到 $\partial U_P$ 上）的穿越次数（只用节点值不行：$D_P$ 离开 exit TIR 曲线按平方根下降）。五条 fixture 全部区间一致；实测值见英文版附录。

### 3.2 分层不变性：一个晕共享什么、变化什么

像素的纤维 $\{R : R\,\Phi_P(-R^{-1}\hat{\mathbf s}) = \mathbf d\}$ 只经由 $\Phi_P$ 依赖于光路。于是：

| 层 | 对象 | 由谁共享 |
|---|---|---|
| $\Phi$ | 场 $D_P$、它的等值线、$1/\lvert\nabla D_P\rvert$、对应关系 $\psi(\mathbf u,\alpha)$、临界点 | 整个 $\Phi$ 类，跨 PBD 类 |
| 成员 | $S^2$ 上的窗口场 $w_m = A_m T_m$，可加：$w_\Phi = \sum_m w_m$ | 每条面序列一份 |
| 对称 | 晶体对称元 $g$ 搬运这张图：$D_{gPg^{-1}}(\mathbf u) = D_P(g^{-1}\mathbf u)$，窗口同样搬运；在 $S^2$ 上整个 $D_{6h}$ 都起作用，镜面也在内（§3.3） | 只有 $\rho$ 能区分各成员 |
| $\rho$ | 纤维上的姿态密度 | 写作系列那张表的*列* |

一个 $\Phi$ 类就是一族等值线加一个有效窗口场；写作系列的核心表（行 = 光路类，列 = 姿态族）的骨架是：行 $= (D_P, w_\Phi)$，列 $= \rho$，格子 $=$ 线积分。代码里 $\Phi$ 层是 `path_class.phi_key`（key 相等即 $\Phi_P$ 精确相等，不做对称商）；写作系列更粗的类在 `lumice_integral.symmetry.signature` 里：一个 key $(M, a, \tilde a)$ 的 $D_{6h}$ 轨道是一个 canonical signature 类 $(M, \mathbf n_a, M^{-1}\mathbf n_b)$ 模 $D_{6h}$ 共轭——60° 与 90° 楔角恰好一个，平行（0°）轨道按 $M$ 的共轭类合并——而 `phi_class` 是这类轨道的并（framework 定理 5′：14 个 signature 类、6 个 $\Phi$ 类；测试 `test_path_class_phi_key.py::test_d6h_orbit_of_the_key_is_one_signature_class_and_refines_phi_class`）。

同样的陈述在 $\mathrm{SO}(3)$ 上也成立：对真旋转 $g$，$\mathrm{fiber}(gPg^{-1}) = \mathrm{fiber}(P)\,g^{-1}$ 精确成立，$\Phi$ 相同的成员共享一条曲线（任务 `path-class-rendering-unit` 在 row 651 对 `3-1-2-5` 测得「方向映射相同、权重不同」）。task 9 在 column 密度下的 `12x`，就是「对称」这一行配上一个在 $C_6$ 旋转和 $C_2'$ 下不变的 $\rho$；倾斜的 Parry 密度只靠 $\rho$ 就打破了它。

### 3.3 对称性是预计算，不产生新样本

代表光路在整个 $S^2$ 上的场，与全部成员在基本域 $F = S^2/G$ 上的场信息等价：代表的场在 $hF$ 这一块上，就是成员 $h^{-1}Ph$ 在 $F$ 上的场。「把一个事件搬运成 $|G|$ 个像」与「只算 $1/|G|$ 个球面」是同一件事。对称性省的是重复计算，不产生新样本；带求和的精度由落在带内、权重非零的*不同*预计算事件数决定（$K_{\mathrm{eff}}$ 数的是事件，不是搬运出来的像）。

镜面在 $S^2$ 上和旋转一样搬运：$w_{gPg^{-1}}(g\mathbf u) = w_P(\mathbf u)$，$\Phi_{gPg^{-1}}(g\mathbf u) = g\,\Phi_P(\mathbf u)$，有效域相同（对全部 24 个元素**实测**到 `1e-12`）。被搬运事件的姿态由 $(g\mathbf u, g\Phi, D)$ 通过两个正交标架重建，无论 $\det g$ 为何都是旋转；作用在代表姿态 $R$ 上就是 $L_g R g^{\mathsf T}$，其中 $L_g = I - (1-\det g)\,\mathbf m\mathbf m^{\mathsf T}$，$\mathbf m$ 是 $\hat{\mathbf s}$ 与像素所在平面的法向。「镜面成员需要自己的仓库」是 Phase I 的限制（$R g^{-1}$ 必须是 $\mathrm{SO}(3)$ 中的旋转），在 $S^2$ 上不存在；一个仓库服务整个类。

## 4. 求积 A：追踪等值线（M2，设计）

M2 scrum 依次构建：$D_P$ 在 $U_P$ 上的拓扑（从格点 seed 出发用 AD Newton 找内部临界点并按 Hessian 分类，$\partial U_P$ 上的受限临界点与角点，带分量计数的区间划分，并用稠密格点 marching 核对）；做成批量、可 `vmap` 模块的场层；给定 $\delta$ 的等值线提取（seed 取自事件仓库的带和格点 marching，Newton 细化到水平集，得到闭环和被 $\partial U_P$ 截断的弧，分量数对区间划分核验——这就是证书）；带常数的线求积，并与 Phase I 和带求和交叉验证；最后是第 10 章的数值裁定（§10）。

从 Phase I 成本画像继承的设计约束（[phase1_zh.md](phase1_zh.md) §5；证据 `scratchpad/task-pixel-cost-shape-stable-kernels/evidence/owner_cprofile_col126_rows300-340.prof`）：Phase I 点亮像素 `74 %` 的时间在 continuation 循环里，每步受 dispatch 限制，`home-wsl` 上 30 个 worker 已顶到物理核墙。Phase II 必须从一开始就设计成批量、无分支、可 `vmap`（$S^2$ 网格上的场求值、等值线提取和求积都写成数组程序），让一整张图成为一次场计算，GPU 也才用得上。逐像素的 Python 循环只会在新地方重现同一堵墙。

这一结构提示的 fixture：

- *Liljequist*（写作第 8 章）：`1-3-2` 与 `3-5-6-7-3` 的 $\Phi$ 相同（面 3/6 所在平面的镜面；三次在 ±60° 平面上的反射合成为一次），窗口不同。142° 锐边（$= 120° +$ 最小偏向 21.84°）是 $D_P$ 的一个临界值，与晶体形状无关；窄峰是 `3-5-6-7-3` 的窗口，随截面移动。一个球面上的两张图。*实测（任务 `dp-field-layer`）：* 两者都是 $M$ 相同的平板，场逐点相等，但 $U_P$ 不同，其上的临界值也不同：`1-3-2` 为 $\{0°, 115.607°\}$，`3-5-6-7-3` 为 $\{0°, 153.070°, 180°\}$（面 3 正入射在域内，即后向散射锥点）。按本仓库的面编号两者都没有 142°；人工核对见任务 `verify-liljequist-face-numbering`。
- *幻日环*：$D_P(\mathbf u) = \angle(M\mathbf u, \mathbf u)$ 只在 $\pm\mathbf n_M$ 处 $\nabla D_P = 0$（`3-1-6` 上两者都在 entry 大圆上、但在 $U_P$ 闭包之外，那里某次内反射 TIR 失败：根本没有内部临界点），所以环上**没有 fold**；环上的亮度变化全部来自窗口层。对板晶，环上方位角是晶体方位角的线性函数，所以剖面是同一个窗口函数的若干平移叠加（棱柱的三个镜面）：这是一个不受 Jacobian 干扰、检验窗口求和与搬运层的干净测试。
- *22° 晕*：fold；见 §10。

**为什么完整性证书很少需要鞍点分支。** 对六棱柱路径空间的系统搜索（entry 面取 `{1,3}`，
内反射最多 4 次，做过对称去重，并用下述折叠判别式过滤）在测到的 87 个非空候选里没有找到
一个内部鞍点：每一个的 $U_P$ 都是拓扑圆盘（自身与其在 $S^2$ 上的补集各自连通，两档格点
密度下均核对过），因此 Poincaré–Hopf 只要求内部临界点指数和为 $1$（单一非退化极小就已
满足），只有当 $U_P$ 不再单连通时鞍点才是拓扑上必需的。这并不证明六棱柱路径永远没有鞍点
（更长的路径只做过抽样，且每多一次内反射，非空候选的产出率就下降一个数量级），但它解释了
为什么本项目实际渲染的路径大概率用不到区间划分机制里的鞍点分支，也给 `task-dp-field-layer`
提供了一个默认的圆盘定义域假设，配一条明确、可核验的逃生舱口（在假定 Morse-Bott 简单性之前，
先核对 $U_P$ 自身与其补集的连通性）。实测记录：见附录「$D_P$ 场拓扑探针」。

## 5. 求积 B：带求和

来源：Ice Halo Simulation 仓库的 `doc/research/inverse-rendering.md`（中文 `inverse-rendering_zh.md`；「基于预计算标准事件的逆向渲染」，源自 Gislén 等 2004）。那篇笔记是设计草图；本节是本项目对这一思路的权威表述。笔记中的对象就是上面的场：

| 笔记 | 本文 |
|---|---|
| 标准事件 $(\hat a_0, \hat b_0)$ | 晶体系传播方向对 $(-\mathbf u, \Phi_P(-\mathbf u))$ |
| 散射角 $\omega$ | 偏折角场 $D_P(\mathbf u)$ |
| 事件权重 $w$ | 窗口场 $A_P T_P(\mathbf u)$ |
| eq. 19 的旋转 $U$ | 姿态 $R(\mathbf u, \psi(\mathbf u,\alpha))$ |
| $Q(U)$ | 相对 Haar 概率的 $\rho(R)$ |

**估计量。** 用 $N$ 个等面积点（每点 $4\pi/N$），像素的带取 $[\delta_{\mathrm{lo}}, \delta_{\mathrm{hi}}]$（其四角偏折角的最小/最大值），

$$
\hat I(\delta,\alpha)
= \frac{1}{2\pi N\,\Delta\delta\,\sin\delta}
  \sum_{i:\,D_P(\mathbf u_i)\in[\delta_{\mathrm{lo}},\delta_{\mathrm{hi}}]}
  A_P T_P(\mathbf u_i)\;\rho\big(R_i\big),
\qquad \Delta\delta = \delta_{\mathrm{hi}}-\delta_{\mathrm{lo}},
$$

它估计的是 $I\sin\delta'$ 的带平均再除以 $\sin\delta$。常数是推导出来的，从不拟合（**实测**：$N = 10^8$ 时对 Phase I 的中位比值 `1.0000`）。

**姿态用事件自己的偏折角。** $R_i$ 是唯一满足 $R_i\mathbf u_i = \hat{\mathbf s}$、且 $R_i\Phi_P(-\mathbf u_i)$ 位于偏折角 $D_P(\mathbf u_i)$ *和*像素方位角处的姿态：$R_i = [\hat{\mathbf s}, \mathbf e, \hat{\mathbf s}\times\mathbf e]\,[\mathbf u_i, \mathbf f_i, \mathbf u_i\times\mathbf f_i]^{\mathsf T}$，$\mathbf e$ 是像素绕 $\hat{\mathbf s}$ 的单位方位方向，$\mathbf f_i$ 是 $\Phi_P(-\mathbf u_i)$ 垂直于 $\mathbf u_i$ 的单位分量（`s2_store.event_rotations`）。这就是笔记 eq. 19 取 $\omega = D_P(\mathbf u_i)$ 的情形，与之相差 `7e-15`，且不需要其中的 $1/\sin^2\omega$。若像笔记第 3 步那样把像素自己的 $\delta$ 喂给 eq. 19，只要 $D_P(\mathbf u_i) \ne \delta$ 就会得到一个非正交矩阵（$\lVert U^{\mathsf T}U - I\rVert_F$ 中位 `5.5e-4`、最大 `1.4e-3`，与带宽同量级），$\rho$ 就是在一个非旋转上求值。确立带求和的探针是 `scripts/probe_band_sum.py`。

**对笔记的修正。**

1. *并非无噪声。* 带求和是确定性的，但有离散误差：散点约 $1/\sqrt{K_{\mathrm{eff}}}$，规则网格则会混叠。被窄带截出的 Fibonacci 格点通常表现得像散点，而且常常比 $1/\sqrt{K_{\mathrm{eff}}}$ 好 7-9 倍，但也会混叠（Parry 在 $10^7$ 处），所以仓库规模按最差关注像素 $K_{\mathrm{eff}} \ge 10^4$ 来定，而不按格点的典型增益来定。
2. *窄 $\rho$ 会逐像素地压低 $K_{\mathrm{eff}}/K$。* 带内的事件集只由 $\delta$ 决定，$\rho$ 从中选出有贡献的部分。**实测**：带内事件保留比例为 65%（太阳竖直面上的 column 密度）、17%（竖直面外）、3%（plate）、0.45%（Parry）、0.58%（Lowitz），但 $K_{\mathrm{eff}}$ 仍按 $N^{1.00}$ 增长，在点亮像素上从未塌缩；第 11 章五族在 $N = 10^8$ 下都可由一份均匀仓库服务，不需要 $\rho$ 感知的仓库。
3. *归一化要显式*：$I\sin\delta$、Haar $dA/4\pi\cdot d\psi/2\pi$、上面的常数。
4. *「焦散会自行调节亮度」* 是验收项，不是前提（§10）。
5. *按 $\Phi$ 组组织仓库*（§3.2），而不是一个按 $\omega$ 排序、混着光路 id 的大数组。

**像素模型。** 带求和的像素在 $\delta$ 方向是带平均、在 $\alpha$ 方向是点值；Phase I 和等值线法给的是点值。在陡边上两者的差别来自像素模型而不是采样误差（M1 回归里最差的像素在 22° 内缘焦散处，它们的带有一部分落在 $\min D_P$ 之下）。交叉验证要在同一口径下比较。

**分工。** 等值线法负责精度和完整性证书；带求和两者都没有，除采样外也没有收敛阶，但它无分支、可 `vmap`，一份仓库被所有像素、姿态密度和光源方向共享。它是快速渲染器、参数扫描工具和独立的交叉检查。生产模块：`lumice_integral.s2_store`（构建、缓存、provenance、带切片、$D_{6h}$ 搬运）与 `lumice_integral.band_sum`（估计量、光路与类渲染；CLI `scripts/render_band_sum.py`）。$N = 10^8$ 的 canonical 条带：Mac 4 个 worker `30.8 s`（§8 的 scatter 形态之前为 `169 s`），Phase I 在 `home-wsl` 30 个 worker 上 `34.7 min`（见附录）。

## 6. 一份预计算，三个用户

Phase I 也做预计算。`prescan.PrescanTable` 对固定太阳在 $\mathrm{SO}(3)$ 上抽 $4\times10^6$ 个 Haar 姿态，保留域内有效的并记下出射方向，用 k-d 树建索引；像素查询自己方向周围一个球冠内的样本，用作 Newton seed。

两者是同一个采样。一个 Haar 样本 $R$ 就是一对 $(\mathbf u, \psi)$：偏折角是 $D_P(\mathbf u)$，方位角由 $\psi$ 决定，而给定 $\mathbf u$ 和目标方位角时 $\psi$ 有闭式解。预扫表随机采 $\psi$，留下碰巧落在像素附近的；仓库把 $\psi$ 商掉，对像素带内的每个事件直接构造出**恰好**落在像素方位角上的姿态（`band_sum.band_poses`）。仓库是严格更强的对象：

| | Phase I 预扫表 | $S^2$ 事件仓库 |
|---|---|---|
| 样本 | $\mathrm{SO}(3)$ 上的 Haar 姿态，随机 | $S^2$ 上的点 $\mathbf u$，Fibonacci 格点 |
| 每像素用什么 | 像素方向周围球冠内的样本 | $\delta$ 带内的事件，各自恰在像素方位角处 |
| 候选的残差 | 二维（偏折角和方位角） | 一维（偏折角） |
| 索引 | 出射方向上的 k-d 树 | 按 $D$ 排一次序，`searchsorted` |
| 依赖 | 太阳、光路、折射率；不依赖晶体（无限棱柱；`entry_measure` 事后再判） | 晶体、光路、折射率；不依赖光源 |
| 保留 | 域内有效的样本 | $w > 0$ 的事件 |

一个带内事件能服务它所在环上的每个方位角，一个球冠样本只服务它自己的邻域；同样的代价下，仓库给每个像素的可用候选更多，seed 也离纤维更近。它有三个用户：

1. **带求和**（§5）：事件就是求积节点。
2. **等值线追踪**（§4）：带内事件离 $\{D_P = \delta\}$ 不超过半个带宽，作为 Newton 细化到等值线的 seed。
3. **Phase I seed**（M2 子任务 `phase1-seeds-from-store`，**设计**）：用带内姿态取代预扫候选；若 32 像素探针没有丢分量，就删掉 `PrescanTable`。同一批事件还给 Phase I 提供一个完整性交叉检查：每个带内事件都应落在某条已追纤维附近，离所有纤维都很远的事件就标记出一个漏掉的分量。这个检查是统计性的，但漏检概率有界——$N$ 乘以该分量在带内的测度——而支撑 `DEFAULT_SAMPLE_COUNT` 的那次密度调查给不出这个界。由于仓库不依赖光源，Phase I 也不必再为每个太阳高度重建预扫表。

## 7. 环不变性与各路线的成本

设 $Q$ 是绕 $\hat{\mathbf s}$ 的旋转。若 $R$ 把光线送到 $\mathbf d$，则 $QR$ 把它送到 $Q\mathbf d$，且 $\mathbf u = (QR)^{-1}\hat{\mathbf s} = R^{-1}\hat{\mathbf s}$ 不变。所以 $Q\mathbf d$ 的纤维就是 $\mathbf d$ 的纤维左乘 $Q$；同一个偏折角（绕光源的同一个环）上的所有像素共享 $S^2$ 上同一条水平集、沿它的同一个窗口和同一个 $1/\lvert\nabla D_P\rvert$，方位角只经由 $\rho$ 进入。Phase I 在每个方位角上重追同一条曲线，是因为它在 $\mathrm{SO}(3)$ 上逐像素地工作。

| 路线 | 一次性、与分辨率无关 | 每个 $\delta$ 环 | 每个像素 | 精度 |
|---|---|---|---|---|
| Phase I | 预扫表 | — | 发现、追踪、积分：`0.1-0.3 s`（实测） | 逐点，自适应误差估计 |
| 带求和 | $N$ 个事件的仓库（$10^8$ 时 `80 s`，实测） | — | 带内 $K$ 个事件：矩阵乘积后求 $\rho$，CPU `0.31 ms`（scatter，§8；gather `3.3 ms`，实测） | $\delta$ 方向带平均；约 $1/\sqrt{K_{\mathrm{eff}}}$ |
| 等值线（设计） | $D_P$ 场与临界点 | 提取并细化水平集 | 沿已存节点求 $\rho$ | 逐点、确定性、高阶 |

随分辨率的变化：

- Phase I：像素数 × 一个大常数。
- 带求和：像素数 × $K_{\mathrm{target}}$（小常数）。带宽跟着像素走，线分辨率翻倍时要保持 $K_{\mathrm{eff}}$ 就需要两倍的 $N$：仓库随分辨率线性增长，占的是磁盘（渲染器按需映射，§8）。
- 等值线：昂贵的部分 ∝ 环数 ∝ 线分辨率，而且是批量的场计算；只有最后的线积分 ∝ 像素数；没有采样噪声，也没有随分辨率增长的仓库（仓库只提供 seed）。每像素常数有多大，由 M2 实测。
- 若 $\rho$ 绕 $\hat{\mathbf s}$ 不变（random 密度），像素值只依赖 $\delta$：两条 Phase II 路线都只需每环一个数。第 11 章其余四族以 c 轴为参照，除非光源在天顶，否则绕 $\hat{\mathbf s}$ 不对称。

## 8. 按偏折角组织带求和

在任务 `band-sum-scatter-renderer` 之前，生产渲染器是 *gather* 形态：每个像素找自己的带、`searchsorted` 取事件、在像素方位角处重建姿态、求 $\rho$、求和。任务按整列分配，而一列跨越全部 $\delta$ 范围，所以每个 spawn 出的 worker 都把计划内所有仓库整份加载；schema 2 的仓库构建则把所有 $w > 0$ 的事件留在内存里，最后做一次 `argsort`。§7 的内存上限来自这种组织方式，而不是带求和本身。

把循环内外翻转（作者 2026-09-24 的提议，类比矩阵乘法里交换循环下标）求和不变、只改变工作的顺序：按偏折角顺序遍历事件，让每一块事件散射进它所碰到的每个像素带。gather（`band_sum.class_band_sum_pixel`）保留为测试 oracle，不再是第二条渲染路径。

**存储（任务 `s2-store-schema-3`）。** 每个数组一个 `.npy`（`.npz` 不能部分映射），provenance 记录各自的 SHA-256 与大小；`S2EventStore.load(..., mmap_mode="r")` 只读映射、只核对大小，`S2EventStore.verify` 按需做全量哈希，默认加载仍整读并校验哈希。构建时按 $D$ 把保留的事件分桶写盘（1024 个等宽桶，性能旋钮，不进缓存 key）、逐桶排序，结果与一次全局稳定排序逐位相同；`build_or_load` 直接写进缓存目录，构建期间整份仓库从不在内存里（$N = 10^8$ 时构建峰值减半，见英文版附录）。

**姿态拆成像素因子与事件因子。** §5 的姿态是 $R_i = W F_i^{\mathsf T}$，像素标架 $W = [\hat{\mathbf s}, \mathbf e, \hat{\mathbf s}\times\mathbf e]$，事件标架 $F_i = [\mathbf u_i, \mathbf f_i, \mathbf u_i\times\mathbf f_i]$（`s2_store.pixel_world_frame`、`s2_store.event_frames`）；$F_i$ 与像素、太阳都无关。五个姿态密度只通过体轴的天顶分量（$R_i$ 的第三行）看姿态（`ZenithGaussianPoseDensity` 看 c 轴，`ZenithRollGaussianPoseDensity` 看 c 轴与 roll $\operatorname{atan2}(-e_2, e_1)$，random 什么都不看）。因此

$$
\big(R_i\big)_{3j} = \hat{\mathbf z}\cdot R_i\mathbf e_j
= \big(W^{\mathsf T}\hat{\mathbf z}\big)\cdot\big(F_i^{\mathsf T}\big)_{\cdot j}
= \mathbf a_{\mathrm{pixel}}\cdot\mathbf b_{ij},
$$

即一个像素向量（$W[2, :]$）与一个事件向量（$F_i[j, :]$）的点积，不需要参考方位角，也不需要旋转 $Q_\alpha$。对 $M$ 个像素和 $K$ 个事件，每条被密度读取的体轴就是一次 $(M\times 3)(3\times K)$ 乘积。`pose_density` 以 `axis_zeniths`（该族读取的分量：`()`、`("e3",)` 或 `("e1", "e2", "e3")`）和 `evaluate_axis_zeniths(e1=, e2=, e3=)`（任意形状数组）提供这一接口；`evaluate_batch` 仍是定义性的 oracle（五族上两者一致到 `1e-13`，有测试）。§3.3 的 $D_{6h}$ 搬运 $L_g R g^{\mathsf T}$ 等于 $W (g F_i J)^{\mathsf T}$，$J = \operatorname{diag}(1, 1, \det g)$：只作用在事件侧的固定线性变换（`s2_store.transported_frames`），像素向量不变。

**渲染器**（`band_sum.render_band_sum_window`、`scatter_store`）。

1. worker 先算出每个像素的带（`pixel_band`，按列分任务；运算相同，所以 $\delta$、带与带宽与 gather 逐位相同）。
2. 父进程按带中心排序像素，切成每个 worker 一段、工作量（事件 × transport，由映射的 `D` 数组算出）大致相等。一段的事件在每个仓库里都是一段连续区间。
3. 每个 worker 依次只读映射计划内的每个仓库（`mmap_mode="r"`，一次一个 store group，处理完释放再映射下一个；内容由父进程统一哈希一次），以 1024 个事件为一块遍历自己的区间。每块的事件标架及其搬运只算一次；带与该块相交的像素按 32 个一组做上面的乘积、逐元素求 $\rho$、带掩膜、对事件求和。像素的带是下标区间 `searchsorted(D, [lo, hi])`，与 gather 同样左闭右开，所以 $K$ 完全相同。
4. 每个事件先把各 transport 的贡献加起来再平方（每个 transport 先算 $w\rho$ 再相加，与 gather 同序：接近下溢的乘积舍入方式也相同），所以 $K_{\rho>0}$ 与 $K_{\mathrm{eff}}$ 保持 §3.3 的按事件计语义。

数值上与 gather 的差别只在求和次序，以及 $\mathbf a\cdot\mathbf b$ 三个乘积的舍入方式。窄密度会放大后者（$d\log\rho = (\theta - \bar\theta)/\sigma^2\,d\theta$；Lowitz 在 c 轴近竖直时 roll 的 $d\psi \sim \epsilon / \sin\theta$）：像素值保持在 `1e-12` 以内；$K_{\mathrm{eff}}$ 接近 1 的像素（由一两个事件主导）误差不被平均，单测中最坏 `3e-12`。遍历里没有把「一个像素 = 一个 $\delta$ 带」写死：像素是有序仓库上的一个下标区间，§9 发散光的形式（$D \ge \theta$）只是另一种区间。

**实测**（英文版附录）。$N = 10^8$ 的 canonical 条带：Mac 4 个 worker `30.8 s`（gather `168.9 s`），单进程 `73.7 s`；worker RSS `441 MB`（所在段映射进来的那部分仓库，属于共享页缓存），gather 时每个 worker `1175 MB`；每个像素的 $K$、$K_{\rho>0}$ 相同，值的差 ≤ `7.9e-15`。多 store group 时峰值不再相加：12 个 $\Phi$ 组各一个仓库，物理 footprint 峰值 `119 MB`，只有一组时 `111 MB`，而按 gather 的方式全部加载需 `1285 MB`。稀疏像素集（单像素宽的剖面，各带不重叠）是 gather 的最佳情形，两者速度相当；scatter 的收益在像素共享事件的地方。GPU 后端不在范围内；CPU 上乘积（内维 3）相对逐元素的 $\rho$ 很便宜，进一步的加速要从 $\rho$ 本身找，而不是从乘积找。

## 9. 发散光

近处的点光源（位于 $L$ 的街灯）打破了一个假设，而这个假设不在晶体内核里。平行光下，同一条视线上的每个晶体看到同一对（入射，出射），所以像素是一个姿态积分；点光源下入射方向沿视线变化。场和仓库（§1.1）原样复用，只有成像方式变了。

**几何（推导；三个闭式已在 2000 组随机 $(\theta, t)$ 上数值核对，与有限差分一致到 `6e-6`）。** 设观察者在 $O$，$d = |OL|$，视线 $\mathbf x(t) = O + t\mathbf v$ 与灯方向夹角为 $\theta$，$\beta(t)$ 是三角形 $OL\mathbf x$ 在 $L$ 处的角。在 $\mathbf x$ 处散射回 $O$ 的光线的偏折角是外角 $\delta(t) = \theta + \beta(t)$，从 $\theta$（观察者身边的晶体）单调增到 $\pi$（远在后方，背散射）。由正弦定理，

$$
t(\delta) = \frac{d\,\sin(\delta-\theta)}{\sin\delta},\qquad
r(\delta) = |\mathbf x - L| = \frac{d\,\sin\theta}{\sin\delta},\qquad
\frac{dt}{d\delta} = \frac{d\,\sin\theta}{\sin^2\delta}.
$$

对强度为 $J$ 的各向同性光源、单次散射、无消光，像素亮度是 $\int n\,J/r^2\,I_{\hat{\mathbf s}(t)}(\delta(t),\alpha)\,dt$，其中 $n$ 是晶体数密度，$I_{\hat{\mathbf s}}$ 是光源位于局部指向灯的方向时的平行光值。换成变量 $\delta$ 后，因子 $(dt/d\delta)/r^2 = 1/(d\sin\theta)$ **沿整条视线是常数**：

$$
L(\theta,\alpha) = \frac{J}{d\,\sin\theta}\int_{\theta}^{\pi}
  n\big(\mathbf x(\delta)\big)\, I_{\hat{\mathbf s}(\delta)}(\delta,\alpha)\,d\delta ,
$$

其中 $d\sin\theta$ 是灯到视线的距离。散射平面（过 $O$、$L$ 和视线）对所有 $t$ 相同，所以方位标架沿视线固定；只有光源方向在这个平面内转动，而它只经由 $\rho$ 起作用。

**带求和（推导）。** 用 §2 的恒等式，$\delta$ 积分变成对**全部** $D_i \ge \theta$ 的事件求和，每个事件对应视线上唯一的点 $t_i = t(D_i)$，没有带宽，也不需要沿视线步进：

$$
\hat L(\theta,\alpha) = \frac{J}{d\,\sin\theta}\cdot\frac{1}{2\pi N}
  \sum_{i:\,D_i\ge\theta} \frac{n(\mathbf x(D_i))\, w_i\,\rho(R_i)}{\sin D_i},
$$

$R_i$ 按 §5 的方法构造，只是把 $\hat{\mathbf s}$ 换成从 $\mathbf x(D_i)$ 指向灯的方向。空间中 $D$ 取定值的轨迹是绕 $OL$ 轴的 Minnaert 纺锤面（「雪茄」）；笔记里 ray marching 与 Gislén 雪茄法的对比，在事件形式下不复存在。背散射附近 $\sin D_i \to 0$ 对应 $t \to \infty$，被有限范围的晶体云截断。

- random 密度加均匀晶体云：$L \propto \frac{1}{\sin\theta}\int_\theta^\pi I(\delta)\,d\delta$，是太阳晕径向剖面的累积积分；事件按 $D$ 排好序时就是一个后缀和，每像素一次查表。
- 每个像素用到的事件集从一个带变成全部 $D_i \ge \theta$；成本大约上升「偏折角范围 ÷ 带宽」倍（笔记估计 `10-100×`），这让 §8 更有价值。
- 等值线路线要对所有 $\delta \ge \theta$ 的水平集积分：coarea 反过来用，像素变成 $S^2$ 上 $\{D_P \ge \theta\}$ 区域的面积分，$1/\lvert\nabla D_P\rvert$ 和 fold 奇异性都消失；边缘来自积分下限 $\theta$。
- Phase I 能接受任意（入射，出射）对，但 $t$ 求积的每个节点都要一条纤维：只能作逐点参考，不能作渲染器。

**难点不在算法**：本项目还没有的场景模型（灯、观察者、晶体云 $n(\mathbf x)$ 及其范围、消光，以及相应的约定）；**没有 oracle**（Lumice 的 `light_source.type` 只有 `"sun"`；可用的检查只有灯推远的极限、共享内核的项目内粗暴 Monte Carlo，以及 Gislén 论文的定性形状）；有限光源尺寸（太阳的 0.5° 也一样）是天空上的卷积，是另一个维度。触发条件：Lumice 支持点光源，或写作系列需要街灯晕（已记入 backlog）。

## 10. 开放问题及其归属

- **22° 内缘**（写作第 10 章）。对随机取向，$D_P$ 在 $S^2$ 上的最小值是孤立、非退化的，而二维极小附近的 $\int d\ell/\lvert\nabla D\rvert$ 是有限的：内缘应是有限跳变，而 $I \sim 1/\sqrt{D - D_{\min}}$ 的剖面属于把 $\mathbf u$ 约束到一条曲线上的族（column、切弧）。第 10 章的表述应当是验收项，不是前提。M2 子任务 `ch10-numerical-verdicts`。
- **秩亏映射。** $W = 0$ 的类是光源方向上的点质量（任务 `path-class-rendering-unit`）；平行面类（$M \ne I$、$W = I$）的退化像来自 $\rho$ 对 $\mathbf u$ 的约束而非 $\Phi$，需要单独记账（「降维聚光」vs「Jacobian 聚光」，作为求解器的显式输出）。`ch10-numerical-verdicts`。
- **非均匀 $\rho$。** $\psi(\mathbf u,\alpha)$ 是单值的，所以 $\rho$ 逐点求值；只有第 11 章「天空上的卷积」这种读法需要均匀 $\rho$。
- **Jacobian 对齐。** 在纤维化的坐标变换下，$1/\lvert\nabla_{S^2} D_P\rvert$ 与 Phase I 的 $J_\perp$ 的对齐是交叉验证的接触点。`s2-contour-quadrature`。
- **与 Lumice 的绝对尺度**，在其投影面积修复（Ice Halo #597）之后：2026-09-24 完成（任务 `lumice-area-weighting-recheck`）：plate 与 Parry 族的带求和光路类渲染对 Lumice 浮点导出，$K_p = \bar y(550)\,\Omega_p/(S/2)$，不拟合（`docs/ch06-reference-fixture.md` 第 7 节 stage 4）。

| 部分 | 状态 | 在哪 |
|---|---|---|
| 带求和、事件仓库、$D_{6h}$ 搬运、$K_{\mathrm{eff}}$ | 实测，已投产 | 英文版附录；任务 13-19 |
| 仓库与光源无关；`.npy` + mmap；分桶构建 | 实测，已投产 | 英文版附录；任务 21 `s2-store-schema-3` |
| 按偏折角组织带求和（分段、逐类累加、GEMM） | 设计 | 任务 22 `band-sum-scatter-renderer` |
| 等值线求积、临界点、证书 | 设计 | scrum 24 `phase2-contour-quadrature` |
| 由仓库提供 Phase I seed 与交叉检查 | 设计 | scrum 24 子任务 5 |
| 第 10 章裁定 | 开放 | scrum 24 子任务 6 |
| 发散光 | 推导 | backlog |
