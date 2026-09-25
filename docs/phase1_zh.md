# Phase I：在 $\mathrm{SO}(3)$ 上追踪姿态纤维

English version: [phase1.md](phase1.md)。两份正文同步维护；若有出入，以英文版为准。实测记录只在英文版附录中保留，不翻译，以免两份数字各自改动后对不上。

Phase I 是对作者原始直接积分原型的重建：对每个像素，找出把入射光送到该像素的那一维晶体姿态集合，并沿它积分物理权重。它于 2026-09-23 收口，是项目的逐点参考；生产渲染器现在是 Phase II 的带求和（[phase2_zh.md](phase2_zh.md)）。规范性定义（坐标、测度、事件、接口、conformance C01-C14）在 [phase1-math-contract.md](phase1-math-contract.md)；第 6 章 fixture 及其验收阶段在 [ch06-reference-fixture.md](ch06-reference-fixture.md)。本文讲设计与来龙去脉。

## 1. 表述

固定晶体、光路 $P$、波长和入射传播方向 $\mathbf s$（从太阳射向晶体；指向太阳的方向是 $\hat{\mathbf s} = -\mathbf s$，见 [conventions.md](conventions.md)）。晶体姿态是一个旋转 $R \in \mathrm{SO}(3)$，光线传播定义了 halo map

$$
F_P : \mathrm{SO}(3) \longrightarrow S^2,
\qquad R \longmapsto \text{出射方向}.
$$

对像方向 $\mathbf d$，有贡献的姿态是原像 $X_{\mathbf d} = F_P^{-1}(\mathbf d)$。在正则值处，维数定理给出 $\dim X_{\mathbf d} = 3 - 2 = 1$：积分域是 $\mathrm{SO}(3)$ 中的若干条曲线，常常是闭环。姿态密度光滑时，一条光路的逐点贡献有 coarea 形式

$$
I_P(\mathbf d)=
\int_{X_{\mathbf d}}
\frac{\rho(R)\,A_P(R)\,T_P(R)}{J_{F_P}(R)}\,d\mathcal H^1(R),
$$

其中 $\rho$ 是相对 Haar 的姿态密度，$A_P$ 是入射测度（几何上可实现的入射点的投影面积，含有限晶体与遮挡），$T_P$ 是光学透过率（Fresnel 与全内反射），$J_{F_P}$ 是 halo map 的法向 Jacobian（代码里叫 $J_\perp$），$d\mathcal H^1$ 是沿纤维的弧长。每个因子单独输出；Haar 常数 $1/(8\pi^2)$ 与 $J_\perp$ 分开保存（契约 §7）。

「确定性」不等于精确：求根、continuation、自动微分和求积都是数值的，必须报告各自的误差。

## 2. 追踪一条纤维

对一个目标方向 $\mathbf d$：

1. 找一个满足 $F_P(R_0) = \mathbf d$ 的可行 seed $R_0$；
2. 在局部李代数坐标下用自动微分求微分；目标残差有两个分量，所以局部 Jacobian 是 $2\times 3$；
3. 它的一维零空间就是 $X_{\mathbf d}$ 的切向，逐步保持定向一致；
4. 用 $\mathrm{SO}(3)$ 指数映射步预测，再校正回约束上（带信任域的预测-校正）；
5. 一直走到分量闭合，或遇到有效光学域的真实边界；
6. 沿曲线求各因子，对乘积积分。

旋转以矩阵存储；四元数只作插值用的坐标卡，闭合判定考虑双覆盖 $q \sim -q$。技术栈是 Python 3.12 + JAX、float64（[ADR 0001](decisions/0001-phase-i-python-jax.md)）：定栈时的探针显示，continuation 是不规则的控制流，不适合做标量 GPU 负载，所以 GPU 只用于大批量。

第一个纵切片刻意做得很窄：六棱柱、光路 `3-5`、单波长、远处点光源、光滑姿态密度、一个已知纤维为闭环的像素。它的各项里程碑都已完成（无需手填姿态就找到 seed；残差有界地追完整个环；不提前闭合；在诊断坐标下画出环；四个因子分开输出；带收敛报告的积分），证据见英文版附录。

## 3. 从一条纤维到渲染器

渲染器要找到每个像素的*每个*连通分量，分清真闭合与近自接近，跨越坐标卡与四元数符号边界，在折射、TIR 和可行性边界处正确停下或继续，在焦散附近的秩亏处不崩溃，复用相邻像素而不悄悄丢失或合并分支，还要说清一个像素的含义（点值还是立体角平均）。生产流水线（`strip_pixel` / `strip_driver` / `strip_io`，CLI `scripts/render_ch06_strip.py`）对每个像素：

- **候选**来自场景级 seed 仓库（`s2_store.StoreSeeds`：该光路的 $S^2$ 事件仓库，$10^6$ 个点，像素偏折角带内的事件被闭式地摆到像素方位角上；2026-09-25 之前是一张 $4\times10^6$ 个 Haar 姿态、按出射方向建索引的预扫表，见 §5），而不是逐像素预扫；
- **Newton** 校正到纤维上，以上一行像素的分量作热启动（热启动永远不作为完整性的来源）；
- 每个不同候选做**一次生产追踪**；以事件终止的追踪从 seed 反向再追一次，两半拼成**开弧**（单条光路的解集常是开弧，六条光路的类和才是完整的晕）；
- 用新 seed 到已追曲线的 $\mathrm{SO}(3)$ 距离**去重**；
- **重采样后固定网格求积**：对曲线做样条重采样，批量回投、批量求因子，Simpson 以 $N$ 对 $N/2$ 逐 panel 比较作误差估计（§4「求积自身的误差」）；
- 逐像素的**过程性完整性**状态层，以及**点像素模型**（子像素模型已实现，成本 6-10 倍，只在 22° 内缘附近的焦散带有影响）。

## 4. 关键转折及其理由

**原型源码已遗失。** 只剩渲染数据和诊断图，所以 Phase I 是从数学和可观测结果重建方法：先写契约（scrum `phase1-reference-core`），并把第 6 章遗留产物按来源分级（explore `ch06-reference-fixture`）。

**v1 流水线（scrum `ch06-direct-integration`）与三个缺陷。** continuation 被编译成 JAX kernel（每条纤维快 55-61 倍），分量发现、因子、线求积和条带驱动都建了起来。owner 对前 58 列做了核账：每像素约 `18 s`，18% 的 unknown 像素吃掉 75% 的时间，点亮像素的大部分时间花在逐节点回投上，每个环走了三遍。目视检视发现三个缺陷：
(1) 闭合门的绝对最小弧长 $\pi$，让焦散附近短于 $\pi$ 的环要绕第二圈才闭合，使 rows 58-225 的值翻倍——这个阈值取自 canonical fixture 那条长 3.86 的环，恰恰在最亮的区域失效；而对历史图像 0.99 的 Spearman 相关没有察觉整段 ×2；
(2) 尾段衰减比历史 raw 快 3-4 倍；
(3) 贴着 TIR 边界时步长被钉死在地板上，rows 655-800 全部 unknown——之前一个 explore 把它归为「性能而非正确性」，而它的终端效果是四分之一张图没有值。
留下的教训：秩相关不是辐射度检验（要看对数刻度剖面和比值）；从单个 fixture 取来的阈值是代理量，必须在判据真正工作的区间里检验。

**v2 流水线（scrum `strip-pipeline-v2`）：按作者原型的顺序重排。** 每个场景预扫一次、查表取候选、一次激进追踪、用截面穿越判闭合（相对下限取代 $\pi$）、事件减速改为逼近速率限步（取代符号判据）、开弧作为一等分量、先重采样再积分。缺陷 1 和 3 关闭（201 051 个像素全部 complete，无台阶）；全图在 30 个 worker 上用了 `1.86 h`。

**逐像素成本：尺子错了。** v2 基准的 `0.067 s`/像素是单像素热缓存数字；稳态整列实测是 `0.29 s`，因为候选池与曲线的尺寸逐像素变化，每次变化都触发 XLA 重编译（60 个像素约 1100 次编译）。改用 numpy 批量距离并对 jit 做 2 的幂分桶之后，全图降到 `34.7 min`；`15 min` 的目标差 2.3 倍，已贴近热缓存下界和物理核数。从此以整列基准（`benchmarks/benchmark_column_steady_state.py`）为尺子。

**缺陷 2：先是约定，后是 oracle。** 一半出在晶体上：Lumice 的 `height 1.0` 是高除以*直径*，而 canonical 晶体用的是高除以边长，也就是只有一半高；改成 `h/a = 2` 就复现了历史平台段。与高度无关的尾段，由 Lumice 未经 tone mapping 的浮点导出裁定：本渲染器与 Lumice 在 Monte Carlo 噪声底内一致，离群的是历史 raw（它的尾段、偏离中心列的变窄、内缘偏移）。此前一张 8-bit tone-mapped 的 Lumice 图曾让人误以为它与历史 raw 一致。作者裁定：历史 raw 不再作为正确性参考。

**光路类与姿态族。** 一个晕是一个光路共轭类，而不是一条代表光路（一条代表可能只占其类的 `0.4 %`）：驱动把 signature 类展开成它的 PBD 轨道再求和；秩 0 类（$W = 0$）是太阳方向上的 Haar 平均点质量，从不追踪。$\Phi$ 相同的成员共享同一条纤维（`3-1-2-5` 走的是 `3-5` 的方向映射）。五个姿态密度族（random、plate、column、Parry、Lowitz）取代了单一的天顶高斯模型；只有被积函数改变。

**绝对尺度。** 不拟合任何参数，本渲染器与 Lumice 在匹配折射率下的亮带上一致到 `0.997-0.999`。最初推导换算因子时暴露出 Lumice 给每个姿态分配相等能量，而不是按晶体投影面积加权，于是它成了逐像素的 $K_p$；作者裁定这是 Lumice 的 bug 并在那边修复（Ice Halo #597）。对修复后的 Lumice，换算因子是 $K_p = N_{\mathrm{sym}}\,\bar y(550)\,\Omega_p/(S/2)$，$S$ 为晶体表面积：除像素立体角外是一个常数，已在 column 条带（`0.998`）与 plate 与 Parry 族（总通量 `0.9999` / `1.0001`）上重新核对（任务 `lumice-area-weighting-recheck`，fixture 规范第 7 节 stage 4）。

**求积自身的误差（2026-09-25，task `phase1-quadrature-start-and-speed`）。** Phase II 的等值线求积是一条独立链路，它暴露了重采样求积的两个缺陷，而求积自己的自洽检查看不到。(1) 弧长速度把相位条件 $\boldsymbol\nu\cdot\boldsymbol\delta = 0$ 求导成 $\boldsymbol\nu\cdot\boldsymbol\delta' = 0$，漏掉 $\boldsymbol\nu'\cdot\boldsymbol\delta$；$\boldsymbol\delta$ 由固定的预测样条决定，任何网格都消不掉这个偏差（canonical 像素 `5.6e-6`）。现在 $\boldsymbol\nu'$ 由样条二阶导数解析给出，Phase I 与 Phase II 相差 `3e-9`，此前停在 `5.6e-6`。(2) 像素值随追踪起点移动，最多 `1.6e-4`，而 $N$ 对 $N/2$ 的估计常常报得更小。原因不在速度：同一条 trace 只挪网格原点就能复现。被积函数有少数几个 `entry_measure` 拐点，每个拐点的 Simpson 误差的大小和符号随网格相位变化，全局的 $|I_N - I_{N/2}|$ 让它们相互抵消。误差估计改为逐 panel 求和 $\sum|S_h - S_{2h}|$，在检查过的每个网格相位上都界住了误差，数值是旧估计的 2-5 倍；网格随之加密（第 126 列节点数中位 257 → 513），整列成本没有可测的变化。

## 5. Phase I 的现状

Phase I 于 2026-09-23 收口：全部里程碑完成，条带渲染器在形状和绝对尺度上都与 Lumice 一致。它的成本画像（owner 对点亮像素做 cProfile：`74 %` 在 continuation 循环里，约 95 次校正尝试、每次约 `0.9 ms`，而每次的算术只是 3×3；成本在于逐步的 Python 编排和小 kernel dispatch）正是 Phase II 被要求批量化、无分支的原因，也是没有重写 Phase I continuation 的原因。

已记录、不阻塞的开放项：完整性证书（由 Phase II 的临界点提供，[phase2_zh.md](phase2_zh.md) §3.1）、有限日盘、焦散带的像素平均、契约 §12 的显式事件定位。

**seed 改由 $S^2$ 事件仓库提供（2026-09-25，task `phase1-seeds-from-store`）。** Haar 预扫表与事件仓库是同一个预采样（[phase2_zh.md](phase2_zh.md) §6）：一个 Haar 姿态就是一对 $(\mathbf u, \psi)$，预扫表留下碰巧落在像素附近的 $\psi$，仓库把 $\psi$ 商掉，把像素偏折角带内的每个事件恰好摆到像素方位角上。discovery 现在从仓库取候选（`s2_store.StoreSeeds`，`N = 1e6`，带半宽 `0.2 deg`），`prescan.PrescanTable` 已删除。32 像素探针上，从 `N = 1e5` / `0.02 deg` 起的每个仓库配置都找到了预扫表找到的全部分量（英文版附录 "Seeds from the store"）。同一批事件给出统计性的完整性交叉检查（`discovery.check_band_coverage`：带内每个事件都重访一遍；离所有已追纤维都远的可行纤维姿态记为疑似漏检；`exp(-k_min)` 界住「一个与已找到的最稀疏分量同样多带内事件的分量被整体漏掉」的概率）。一个路径类的所有成员经 `path_class.store_plan` 的 `D6h` 搬运共用一个仓库（与 band-sum 渲染器同一个 plan），且仓库与太阳方向无关。

取向分布：Phase I 假设 $\rho$ 是整个 $\mathrm{SO}(3)$ 上的普通（可能很窄的）密度。精确约束的取向族是低维集合上的奇异测度，需要不同的维数计数，不在 Phase I 范围内。
