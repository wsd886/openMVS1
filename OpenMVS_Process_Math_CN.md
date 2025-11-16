# OpenMVS 完整处理流程与数学原理详解

## 目录
1. [概述](#概述)
2. [第一步：密集点云重建 (DensifyPointCloud)](#第一步密集点云重建-densifypointcloud)
3. [第二步：网格重建 (ReconstructMesh)](#第二步网格重建-reconstructmesh)
4. [第三步：网格细化 (RefineMesh)](#第三步网格细化-refinemesh)
5. [第四步：纹理映射 (TextureMesh)](#第四步纹理映射-texturemesh)
6. [完整工作流程示例](#完整工作流程示例)

---

## 概述

OpenMVS (Open Multi-View Stereo) 是一个完整的多视图立体重建库，用于从相机姿态和稀疏点云生成完整的纹理化3D网格模型。它是摄影测量流水线的最后阶段。

**输入**:
- 相机姿态（内参和外参）
- 稀疏3D点云（来自SfM，如COLMAP、OpenMVG等）
- 原始图像

**输出**:
- 纹理化的3D网格模型

**核心流程**:
```
稀疏点云 → 密集点云 → 三角网格 → 细化网格 → 纹理化网格
```

---

## 第一步：密集点云重建 (DensifyPointCloud)

### 1.1 目标
从稀疏点云和多张图像中恢复场景的密集3D点云。

### 1.2 主要步骤

#### 步骤1：视图选择 (View Selection)
**代码位置**: `SceneDensify.cpp:143` - `SelectViews()`

**目的**: 为每个参考图像选择最佳的邻居视图用于立体匹配。

**数学原理**:
对于参考图像 `I_ref`，选择邻居图像 `I_i` 需要满足：

1. **基线角度约束**:
   ```
   θ_min < θ < θ_max

   其中：
   θ = arccos(r₁ · r₂)  // r₁, r₂ 是两个相机的视线方向
   θ_min = 3°  (默认)
   θ_max = 65° (默认)
   θ_optim = 12° (最优角度)
   ```

   - 角度太小：三角测量精度低（近乎平行）
   - 角度太大：外观变化过大，匹配困难

2. **共视区域约束**:
   ```
   overlap_ratio = Area(I_ref ∩ I_i) / Area(I_ref) > threshold
   threshold = 0.05 (默认5%重叠)
   ```

3. **分辨率约束**:
   ```
   0.2 < scale_i / scale_ref < 3.2
   ```

**评分函数**:
```
score = Σ(points_visible) × cos(θ - θ_optim) × overlap_ratio
```

#### 步骤2：深度图初始化
**代码位置**: `DepthMap.cpp:487` - `TriangulatePoints2DepthMap()`

**方法**: 使用CGAL的Delaunay三角剖分将稀疏点云投影到图像平面并插值。

**数学原理**:
1. **投影稀疏点到图像平面**:
   ```
   对于3D点 X_world:
   x_image = K × [R|t] × X_world

   其中：
   K = 相机内参矩阵 (3×3)
   [R|t] = 相机外参 (旋转和平移)
   x_image = (u, v, depth)
   ```

2. **Delaunay三角剖分**:
   在2D图像平面上对投影点进行Delaunay三角剖分，保证：
   - 任何三角形的外接圆内不包含其他点
   - 最大化最小角度（避免狭长三角形）

3. **深度插值**:
   对于三角形内的像素 `p`，使用重心坐标插值：
   ```
   p = α×v₁ + β×v₂ + γ×v₃
   其中 α + β + γ = 1

   depth(p) = α×d₁ + β×d₂ + γ×d₃
   ```

#### 步骤3：PatchMatch立体匹配
**代码位置**: `DepthMap.cpp:290` - `DepthEstimator`

这是密集重建的核心算法。

**算法概述**:
PatchMatch是一种高效的近似最近邻搜索算法，用于多视图立体匹配。

**数据结构**:
每个像素存储一个平面假设：
```
Hypothesis = (depth, normal)

平面方程: n·(X - X₀) = 0
其中：
- n = 法向量 (3D)
- X₀ = 参考点 = camera.C + depth × ray
- depth = 深度值
```

**算法流程**:

##### 3.1 初始化 (Initialization)
- **稀疏点初始化**: 使用三角剖分的深度和法向
- **随机初始化**: 其他像素随机生成深度和法向

```python
# 随机深度
depth = sqrt(uniform(d_min², d_max²))

# 随机法向（球面均匀分布）
θ₁ = uniform(0°, 180°)
θ₂ = uniform(90°, 180°)
normal = spherical_to_cartesian(θ₁, θ₂)

# 确保法向指向相机
if normal · view_ray > 0:
    normal = -normal
```

##### 3.2 迭代优化
对每个像素，重复以下步骤3-5次：

**a) 空间传播 (Spatial Propagation)**

检查邻居像素的假设：
```
candidates = {
    hypothesis[x-1, y],     // 左邻居
    hypothesis[x, y-1],     // 上邻居
    hypothesis[x+1, y],     // 右邻居 (反向传播)
    hypothesis[x, y+1]      // 下邻居 (反向传播)
}

对每个候选，计算其在当前像素的匹配代价
如果代价更低，则更新当前假设
```

**b) 随机搜索 (Random Search)**

在当前假设周围随机扰动：
```
for scale in [1/2, 1/4, 1/8, ...]:
    # 深度扰动
    depth' = depth × (1 + scale × uniform(-ratio, ratio))
    ratio = 0.003 (默认)

    # 法向扰动
    angle1' = angle1 + scale × uniform(-16°, 16°)
    angle2' = angle2 + scale × uniform(-10°, 10°)

    如果匹配代价降低，则更新假设
```

**c) 平滑性约束 (Smoothness)**

鼓励邻域内的深度和法向连续：
```
bonus = smoothBonus × similarity(current, neighbor)

其中：
depth_similarity = exp(-|d_cur - d_nei|² / (2σ_d²))
normal_similarity = exp(-angle(n_cur, n_nei)² / (2σ_n²))

σ_d = 0.02 (深度方差)
σ_n = 13° (法向方差)
smoothBonus = 0.93
```

##### 3.3 匹配代价计算

**核心：归一化互相关 (NCC - Normalized Cross-Correlation)**

**代码位置**: `DepthMap.cpp:496` - `ScorePixelImage()`

对于参考图像中的像素 `p` 和假设 `(depth, normal)`：

**步骤1**: 计算单应性矩阵

由于假设局部表面为平面，可以用单应变换：
```
H = K_i × (R_i + (t_i - t_ref) × n^T / (n·X₀)) × K_ref⁻¹

简化为：
H = K_i × (R_i×R_ref^T) × K_ref⁻¹ + K_i×R_i×(C_ref - C_i) × n^T / (n·X₀×depth)

其中：
- K_i, K_ref = 目标和参考相机内参
- R_i, R_ref = 旋转矩阵
- C_i, C_ref = 相机中心
- n = 法向量
- depth = 深度
```

**步骤2**: 采样图像块

在参考图像和目标图像上提取对应的9×9窗口（默认）：
```
参考图像: patch_ref[9×9] 中心在 (u, v)

目标图像: 对patch中每个点 (u+i, v+j):
    (u', v') = H × (u+i, v+j, 1)
    patch_tgt[i,j] = bilinear_interpolate(I_tgt, u', v')
```

**步骤3**: 计算加权NCC (DENSE_NCC_WEIGHTED)

```
# 1. 计算权重
对patch中每个像素 (i,j):
    w_color = exp(-|I_ref(i,j) - I_ref(center)|² / (2×0.1²))
    w_spatial = exp(-(i² + j²) / (2×(half_window-1)²))
    w[i,j] = w_color × w_spatial

# 2. 归一化
W = Σw[i,j]
w[i,j] = w[i,j] / W

# 3. 计算加权NCC
mean_ref = Σ(w[i,j] × patch_ref[i,j])
mean_tgt = Σ(w[i,j] × patch_tgt[i,j])

patch_ref' = patch_ref - mean_ref
patch_tgt' = patch_tgt - mean_tgt

numerator = Σ(w[i,j] × patch_ref'[i,j] × patch_tgt'[i,j])
denom_ref = sqrt(Σ(w[i,j] × patch_ref'[i,j]²))
denom_tgt = sqrt(Σ(w[i,j] × patch_tgt'[i,j]²))

NCC = numerator / (denom_ref × denom_tgt)
```

NCC范围：[-1, 1]，1表示完美匹配。

**步骤4**: 聚合多视图分数 (DENSE_AGGNCC_MINMEAN)

对所有邻居视图 `{I₁, I₂, ..., I_n}`：
```
scores = [NCC₁, NCC₂, ..., NCC_n]
scores = sort(scores, descending)

# 取最小值和平均值的混合
final_score = min(scores[0], mean(scores))

# 转换为代价（越小越好）
cost = 1 - final_score
```

##### 3.4 几何一致性检查

**代码位置**: `DepthMap.cpp` - Geometric Consistency

**目的**: 减少匹配错误。

**方法**: 左右一致性检查
```
1. 在参考图像A中估计深度图 depth_A
2. 在目标图像B中估计深度图 depth_B

3. 对A中每个像素 p:
   a) 获取深度 d_A = depth_A[p]
   b) 投影到B: p' = project(p, d_A, camera_A, camera_B)
   c) 获取B中深度 d_B = depth_B[p']
   d) 反投影到A: p'' = project(p', d_B, camera_B, camera_A)

   e) 检查一致性:
      if |p - p''| < threshold:  // 通常1-2像素
          depth_A[p] 有效
      else:
          depth_A[p] 无效（标记为0）
```

**几何代价权重**:
```
total_cost = photometric_cost + λ × geometric_cost
λ = 0.1 (默认)

geometric_cost = |p - p''|² / threshold²
```

#### 步骤4：深度图过滤与优化

##### 4.1 去除斑点 (Remove Speckles)
**代码位置**: `SceneDensify.cpp` - REMOVE_SPECKLES

连通域分析，移除小于阈值的孤立区域：
```
对每个连通的深度区域:
    if size < nSpeckleSize (默认100像素):
        depth[region] = 0
```

##### 4.2 填补空洞 (Fill Gaps)
**代码位置**: `SceneDensify.cpp` - FILL_GAPS

对小空洞进行插值：
```
对每个无效像素 (depth = 0):
    if gap_size < nIpolGapSize (默认7像素):
        depth = interpolate(left_depth, right_depth)
```

##### 4.3 置信度调整

基于以下因素计算置信度：
```
confidence = f(NCC_score, depth_variance, normal_variance, num_consistent_views)

其中：
- NCC_score: 匹配质量 (越接近1越好)
- depth_variance: 邻域深度方差 (越小越好)
- num_consistent_views: 一致视图数量 (越多越好)
```

#### 步骤5：深度图融合

**代码位置**: `SceneDensify.cpp` - `DepthMapsData::FuseDepthMaps()`

**目的**: 将所有深度图融合为统一的密集点云。

**算法**: 基于置信度的加权融合

```
对每个深度图中的像素 p:
    1. 反投影到3D空间:
       X = camera.C + depth[p] × ray[p]

    2. 投影到其他深度图，收集一致的深度估计:
       consistent_depths = []
       consistent_normals = []
       weights = []

       对每个其他深度图 j:
           p_j = project(X, camera_j)
           if depth_j[p_j] 存在:
               # 检查深度一致性
               depth_reprojected = |X - camera_j.C|
               if |depth_j[p_j] - depth_reprojected| / depth_reprojected < 0.01:
                   # 检查法向一致性
                   if angle(normal[p], normal_j[p_j]) < 25°:
                       consistent_depths.append(depth_j[p_j])
                       consistent_normals.append(normal_j[p_j])

                       # 计算权重 (基于NCC置信度)
                       conf = confidence_j[p_j]
                       w = 1 / (max(1-conf, 0.03) × depth²)
                       weights.append(w)

    3. 如果一致深度数量 >= nMinViewsFuse (默认2):
       # 加权平均融合
       X_fused = Σ(w_i × X_i) / Σw_i
       normal_fused = normalize(Σ(w_i × n_i))
       color_fused = average(colors from all views)

       添加到点云
```

**输出**: 密集点云 (PLY格式)，包含：
- 3D坐标
- 法向量
- 颜色
- 视图信息

---

## 第二步：网格重建 (ReconstructMesh)

### 2.1 目标
从密集点云重建三角网格表面。

### 2.2 算法：Delaunay四面体剖分 + Graph Cut

**代码位置**: `SceneReconstruct.cpp`

这是一个**全局优化**方法，将表面重建问题转化为图割问题。

### 2.3 详细步骤

#### 步骤1：3D Delaunay四面体剖分

**代码**: `SceneReconstruct.cpp:35` - 使用CGAL库

**输入**: 密集点云 `P = {p₁, p₂, ..., p_n}`

**输出**: 四面体网格 `T = {tet₁, tet₂, ..., tet_m}`

**性质**:
- 每个四面体的外接球内不包含其他点
- 完全填充点云的凸包

```
Delaunay四面体化:
空间被完全剖分为四面体单元
每个四面体有4个顶点、4个面、6条边
相邻四面体共享三角形面
```

#### 步骤2：构建能量函数

将表面重建转化为二元标签问题：

**标签**:
- `INSIDE = 0`: 四面体在表面内部（实心）
- `OUTSIDE = 1`: 四面体在表面外部（空心）

**目标**: 找到最优标签分配，使得标签边界形成物体表面。

**能量函数**:
```
E(L) = Σ D_i(l_i) + λ × Σ V_{ij}(l_i, l_j)
       i∈T           (i,j)∈N

其中:
L = 标签分配 {l₁, l₂, ..., l_m}
D_i = 数据项 (四面体i的代价)
V_{ij} = 平滑项 (相邻四面体i,j的边界代价)
λ = 权重系数
```

**数据项 D_i(l_i)**:

衡量将四面体i标记为内部或外部的代价：

```python
def data_cost(tet_i, label):
    # 计算四面体的空间支持
    # 空间支持 = 该四面体被多少相机"看见"

    visible_cameras = []
    for camera in cameras:
        if tet_i在camera的视锥内:
            # 检查是否被遮挡
            face = tet_i面向camera的面
            if not is_occluded(face, camera):
                visible_cameras.append(camera)

    support = len(visible_cameras)

    # 高支持 → 应该在表面上 → 边界四面体
    # 低支持 → 内部或外部

    if label == INSIDE:
        # 内部应该有高支持（被很多相机看见）
        cost = -support
    else:  # OUTSIDE
        # 外部应该有低支持
        cost = support

    return cost
```

更精确的公式：
```
D_i(INSIDE) = -Σ quality(tet_i, camera_j)
D_i(OUTSIDE) = +Σ quality(tet_i, camera_j)

quality(tet, cam) = {
    area(tet的投影) × cos(法向, 视线方向)  如果可见
    0                                    如果不可见或遮挡
}
```

**平滑项 V_{ij}(l_i, l_j)**:

衡量相邻四面体标签不同时的代价（即表面质量）：

```
V_{ij}(l_i, l_j) = {
    0                           如果 l_i == l_j (无边界)
    quality(共享面_{ij})         如果 l_i != l_j (有边界)
}

quality(face) = area(face) / avg_distance(face_vertices, point_cloud)

即：
- 面积越大 → 代价越高 (倾向于紧凑表面)
- 离点云越近 → 代价越低 (倾向于贴合点云)
```

#### 步骤3：Graph Cut优化

**图的构建**:

```
节点:
- 每个四面体对应一个节点
- 源节点 S (代表INSIDE)
- 汇节点 T (代表OUTSIDE)

边:
1. S-links: S → tet_i, 容量 = max(0, -D_i(INSIDE))
2. T-links: tet_i → T, 容量 = max(0, D_i(OUTSIDE))
3. N-links: tet_i ↔ tet_j (邻居), 容量 = V_{ij}
```

**最小割 = 最小能量**:

```
求解图的最小割 (S-T min-cut):
    使用最大流算法 (IBFS 或 Boykov-Kolmogorov)

割的含义:
    - S侧的四面体 → INSIDE
    - T侧的四面体 → OUTSIDE
    - 割边 = 重建的表面
```

**代码**: `SceneReconstruct.cpp:65` - MaxFlow class

**算法**: IBFS (Incremental Breadth-First Search) 或 BK算法
- 时间复杂度: O(n²m) 其中n=节点数, m=边数
- 实际运行很快

#### 步骤4：提取网格表面

```
表面 = 所有标签不同的相邻四面体的共享面

mesh.faces = []
for (tet_i, tet_j) in adjacent_tetrahedra:
    if label[tet_i] == INSIDE and label[tet_j] == OUTSIDE:
        face = shared_face(tet_i, tet_j)
        mesh.faces.append(face)
```

#### 步骤5：网格清理

**a) 移除孤立顶点和边**
```
移除不属于任何三角形的顶点
```

**b) 移除小连通组件**
```
对网格进行连通性分析
仅保留最大的连通组件（主物体）
```

**c) 填补小孔洞**
```
检测边界环 (boundary loops)
if loop.length < threshold:
    三角剖分填补
```

**d) 平滑**
```
Laplacian平滑 (可选):
    v_new = (1-λ) × v + λ × Σ(v_neighbor) / degree

    λ = 平滑系数 (0.5-0.8)
    保留特征边
```

### 2.4 输出
三角网格 (PLY/OBJ格式)：
- 顶点坐标
- 三角形面索引
- (可选) 顶点法向量

---

## 第三步：网格细化 (RefineMesh)

### 3.1 目标
通过优化顶点位置，使网格更精确地拟合图像观测，恢复精细几何细节。

### 3.2 算法：基于图像的梯度下降优化

**代码位置**: `SceneRefine.cpp`, `SceneRefineCUDA.cpp`

这是一个**非线性优化**问题。

### 3.3 数学模型

#### 问题定义

**优化变量**: 网格顶点位置 `V = {v₁, v₂, ..., v_n} ∈ ℝ^(3n)`

**目标函数**:
```
E(V) = E_photo(V) + λ_smooth × E_smooth(V)

其中:
- E_photo: 光度一致性能量（数据项）
- E_smooth: 正则化能量（平滑项）
- λ_smooth: 权重 (默认1.5)
```

#### 光度一致性能量 E_photo

**核心思想**: 同一3D点投影到不同图像，应该有相同的颜色。

```
E_photo = Σ  Σ  ρ(||I_i(π_i(v)) - I_j(π_j(v))||)
          v  (i,j)

其中:
- v: 顶点
- (i,j): 能看到v的图像对
- π_i(v): v投影到图像i的位置
- I_i(x): 图像i在位置x的颜色
- ρ(): 鲁棒损失函数
```

**投影函数**:
```
π_i(v) = K_i × [R_i | t_i] × v

详细:
v_cam = R_i × (v - C_i)  // 转到相机坐标系
v_proj = K_i × v_cam      // 投影到图像
u = v_proj[0] / v_proj[2]
v = v_proj[1] / v_proj[2]

π_i(v) = (u, v)
```

**鲁棒损失** (Huber loss):
```
ρ(x) = {
    x²            如果 |x| ≤ δ
    δ(2|x| - δ)   如果 |x| > δ
}

δ = 阈值，减少外点影响
```

**实际实现 - 面片光度误差**:

不直接优化顶点颜色，而是优化**面片**的光度一致性：

```python
def photometric_energy(mesh, images):
    E = 0

    for face in mesh.faces:
        # 面片的3个顶点
        v0, v1, v2 = face.vertices

        # 面法向
        normal = cross(v1-v0, v2-v0).normalize()

        # 找到能看见这个面的图像
        visible_images = []
        for img in images:
            view_dir = (face.center - img.camera.C).normalize()
            if dot(normal, view_dir) > 0:  # 背面剔除
                visible_images.append(img)

        if len(visible_images) < 2:
            continue

        # 参考图像（通常选择最正面观察的）
        ref_img = select_best_view(face, visible_images)

        # 在参考图像上采样面片
        samples = sample_on_triangle(v0, v1, v2, num_samples=100)

        for sample_point in samples:
            # 投影到参考图像
            uv_ref = project(sample_point, ref_img.camera)
            color_ref = ref_img.get_color(uv_ref)

            # 投影到其他图像并比较颜色
            for tgt_img in visible_images:
                if tgt_img == ref_img:
                    continue

                uv_tgt = project(sample_point, tgt_img.camera)

                # 检查深度一致性
                if not is_depth_similar(sample_point, uv_tgt, tgt_img.depthmap):
                    continue

                color_tgt = tgt_img.get_color(uv_tgt)

                # 颜色差异
                diff = ||color_ref - color_tgt||
                E += huber_loss(diff)

    return E
```

#### 正则化能量 E_smooth

防止网格过度变形，保持平滑性。

**a) 刚性能量 (Rigidity)**:

保持局部几何形状：
```
E_rigid = Σ  Σ  ||e_ij - e_ij^0||²
          v  j∈N(v)

其中:
- e_ij = v_i - v_j: 当前边向量
- e_ij^0: 初始边向量
- N(v): v的邻居顶点
```

**b) 弹性能量 (Elasticity)**:

保持表面光滑：
```
E_elastic = Σ ||L(v_i)||²
            v_i

其中 L 是拉普拉斯算子:
L(v_i) = v_i - (1/|N(i)|) × Σ v_j
                            j∈N(i)
```

**组合正则化**:
```
E_smooth = α × E_rigid + (1-α) × E_elastic

α = 0.8 (默认，刚性权重)
```

### 3.4 优化算法

#### 梯度下降

**目标**: 找到 `V*` 使得 `E(V*)` 最小

**迭代公式**:
```
V^(k+1) = V^k - η × ∇E(V^k)

其中:
- η: 学习率（步长）
- ∇E: 能量梯度
```

#### 梯度计算

**光度能量梯度**:

```
∂E_photo/∂v_i = Σ  ∂E_photo/∂I(u,v) × ∂I(u,v)/∂(u,v) × ∂(u,v)/∂v_i
                img

其中:
∂I/∂(u,v) = 图像梯度 (用Sobel算子计算)
∂(u,v)/∂v_i = 投影雅可比矩阵
```

**投影雅可比**:
```
π(v) = (u, v) = K × [R|t] × v

设 v_cam = R×(v - C) = (X, Y, Z)^T

u = f_x × X/Z + c_x
v = f_y × Y/Z + c_y

∂u/∂v = [f_x/Z × R[0], -f_x×X/Z² × R[2]]
∂v/∂v = [f_y/Z × R[1], -f_y×Y/Z² × R[2]]

J = [∂u/∂v; ∂v/∂v] ∈ ℝ^(2×3)
```

**代码**: `SceneRefine.cpp:146` - `ProjectVertex()`

**平滑能量梯度**:

```
∂E_rigid/∂v_i = Σ 2(e_ij - e_ij^0)
                j

∂E_elastic/∂v_i = 2L(v_i) × (|N(i)| + 1) - 2×Σ L(v_j)
                                              j∈N(i)
```

#### 多尺度优化

**代码**: `SceneRefine.cpp:129` - `nResolutionLevel`

```
从粗到精，逐步细化:

for scale in [0.5, 0.75, 1.0]:
    # 缩放图像
    images_scaled = resize(images, scale)

    # 优化
    for iter in range(max_iterations):
        # 渲染网格到所有图像
        render_mesh_to_images(mesh, images_scaled)

        # 计算梯度
        gradients = compute_gradients(mesh, images_scaled)

        # 更新顶点
        mesh.vertices -= learning_rate × gradients

        # 检查收敛
        if ||gradients|| < epsilon:
            break
```

**优点**:
- 粗尺度：快速收敛到大致形状
- 细尺度：恢复精细细节

#### 网格自适应细分

**代码**: `SceneRefine.cpp:141` - `SubdivideMesh()`

在优化过程中，动态细分面积过大的三角形：

```
while 存在大三角形:
    for face in mesh.faces:
        if area(face) > threshold:
            # 细分策略：在最长边中点插入新顶点
            longest_edge = find_longest_edge(face)
            midpoint = (edge.v0 + edge.v1) / 2

            # 将三角形分为两个
            subdivide_triangle(face, midpoint)

    # 网格清理
    remove_small_faces()
    improve_vertex_valence()  # 使顶点度数接近6
```

**阈值**:
```
threshold = max_face_area / 4

其中 max_face_area 根据图像分辨率自适应设置
```

### 3.5 GPU加速 (CUDA)

**代码**: `SceneRefineCUDA.cpp`

**并行化**:
- 每个线程处理一个顶点的梯度计算
- 使用共享内存缓存相机参数
- 纹理内存访问图像

**加速比**: 通常10-50倍

### 3.6 输出
细化后的高质量网格：
- 更准确的顶点位置
- 恢复了精细几何细节
- 更好地拟合原始图像

---

## 第四步：纹理映射 (TextureMesh)

### 4.1 目标
为网格生成高质量、无缝的纹理贴图。

### 4.2 算法流程

**代码位置**: `SceneTexture.cpp`

### 4.3 详细步骤

#### 步骤1：视图选择（每个面片）

**目标**: 为每个三角形面选择最佳的纹理源视图。

**质量评分**:
```python
def view_quality(face, image):
    # 1. 面法向与视线角度
    normal = face.normal
    view_dir = (face.center - image.camera.C).normalize()
    angle = arccos(dot(normal, view_dir))
    angle_quality = cos(angle)  # 越正面越好

    # 2. 分辨率
    # 面片投影到图像的面积
    vertices_2d = [project(v, image.camera) for v in face.vertices]
    area_2d = triangle_area(vertices_2d)
    resolution_quality = sqrt(area_2d)  # 投影越大，分辨率越高

    # 3. 遮挡检测
    if is_occluded(face, image, mesh):
        return 0

    # 4. 图像梯度（纹理清晰度）
    gradient = compute_gradient_on_face(face, image)
    sharpness = mean(gradient)

    # 综合评分
    quality = angle_quality × resolution_quality × sharpness

    return quality
```

**问题**: 如果每个面独立选择视图，会产生**接缝**（相邻面选择不同视图）。

#### 步骤2：全局视图分配优化

**建模为MRF（马尔可夫随机场）问题**:

**变量**: 每个面的视图标签 `L = {l₁, l₂, ..., l_n}`，其中 `l_i ∈ {0, 1, ..., num_views}`

**能量函数**:
```
E(L) = Σ D_i(l_i) + λ × Σ V_{ij}(l_i, l_j)
       i           (i,j)∈邻接

其中:
- D_i(l_i): 数据项，将视图l_i分配给面i的代价
- V_{ij}(l_i, l_j): 平滑项，相邻面选择不同视图的代价
```

**数据项**:
```
D_i(l) = -quality(face_i, view_l)

即质量越高，代价越低
```

**平滑项 - Potts模型**:
```
V_{ij}(l_i, l_j) = {
    0           如果 l_i == l_j (同一视图，无接缝)
    λ_seam      如果 l_i != l_j (不同视图，有接缝)
}

λ_seam = 常数惩罚
```

**优化算法**: LBP (Loopy Belief Propagation)

**代码**: `SceneTexture.cpp:69` - LBP inference

```python
# 信念传播
def LBP(faces, views, max_iterations=100):
    # 初始化消息
    messages = {}  # messages[i→j][label]

    for iter in range(max_iterations):
        for edge in mesh.edges:
            i, j = edge.faces

            # 从i传递到j的消息
            for label_j in labels:
                min_energy = infinity

                for label_i in labels:
                    # 数据项 + 平滑项 + 来自其他邻居的消息
                    energy = D[i][label_i] + V(label_i, label_j)

                    for k in neighbors(i) if k != j:
                        energy += messages[k→i][label_i]

                    min_energy = min(min_energy, energy)

                messages[i→j][label_j] = min_energy

        # 检查收敛
        if messages converged:
            break

    # 解码：选择最小边缘能量的标签
    for i in faces:
        belief[i][label] = D[i][label] + Σ messages[k→i][label]

        best_label[i] = argmin(belief[i])

    return best_label
```

#### 步骤3：纹理图集生成（UV展开）

**目标**: 将3D网格展开到2D纹理空间。

**a) 面片分组**

将选择相同视图的相邻面合并为**纹理块** (texture patch)：

```python
patches = []
visited = set()

for face in faces:
    if face in visited:
        continue

    # BFS/DFS 收集相邻的同视图面
    patch = []
    queue = [face]
    view = label[face]

    while queue:
        f = queue.pop()
        if f in visited or label[f] != view:
            continue

        visited.add(f)
        patch.append(f)

        # 添加同视图的邻居
        for neighbor in f.adjacent_faces:
            if label[neighbor] == view:
                queue.append(neighbor)

    patches.append(patch)
```

**b) UV坐标计算**

对每个纹理块，使用**直接投影法**：

```python
def compute_uv(patch, view):
    camera = views[view].camera

    # 计算包围盒
    uv_coords = []
    for face in patch:
        for vertex in face.vertices:
            # 投影到图像
            uv = project(vertex, camera)
            uv_coords.append(uv)

    # 归一化到[0,1]
    u_min, v_min = min(uv_coords)
    u_max, v_max = max(uv_coords)

    for face in patch:
        for i, vertex in enumerate(face.vertices):
            uv = project(vertex, camera)
            face.uv[i] = ((uv.u - u_min) / (u_max - u_min),
                          (uv.v - v_min) / (v_max - v_min))
```

**c) 矩形装箱 (Rectangle Packing)**

**代码**: `SceneTexture.cpp:34` - `RectsBinPack.h`

将所有纹理块高效地打包到一个或多个正方形纹理图集中：

```
算法: MaxRects (Maximal Rectangles)

1. 初始化: 空闲矩形列表 = [整个纹理]

2. 对每个纹理块 (按面积从大到小排序):
    a) 在所有空闲矩形中找最佳位置
       评分标准: 最小剩余面积、最短边优先等

    b) 放置纹理块

    c) 更新空闲矩形列表:
       - 移除被占用的矩形
       - 添加分割产生的新空闲矩形

    d) 合并重叠的空闲矩形
```

**纹理尺寸**: 通常2048×2048, 4096×4096或8192×8192

#### 步骤4：纹理采样与生成

```python
def generate_texture(mesh, patches, views, texture_size):
    texture = Image(texture_size, texture_size)

    for patch in patches:
        view = patch.view_index
        image = views[view].image

        for face in patch.faces:
            # 在纹理空间光栅化三角形
            uv0, uv1, uv2 = face.texture_coords

            # 转换到纹理像素坐标
            t0 = (uv0.u * texture_size, uv0.v * texture_size)
            t1 = (uv1.u * texture_size, uv1.v * texture_size)
            t2 = (uv2.u * texture_size, uv2.v * texture_size)

            # 光栅化三角形
            for pixel in rasterize_triangle(t0, t1, t2):
                # 计算重心坐标
                bary = barycentric(pixel, t0, t1, t2)

                # 在3D空间插值位置
                pos_3d = bary.x * face.v0 + bary.y * face.v1 + bary.z * face.v2

                # 投影到源图像
                uv_source = project(pos_3d, views[view].camera)

                # 采样颜色
                color = bilinear_sample(image, uv_source)

                texture[pixel] = color
```

#### 步骤5：接缝消除 (Seam Leveling)

即使视图分配已优化，接缝处仍可能有颜色不连续。

**全局接缝消除**:

**目标**: 最小化接缝处的颜色差异，同时保持纹理内部不变。

**建模为泊松融合问题**:

```
对纹理图像 T，求解:
ΔT = div(∇S)  在内部区域
T = observed    在边界

其中:
- ΔT: 拉普拉斯算子
- S: 源纹理
- div, ∇: 散度和梯度算子
```

**离散形式**:

对每个纹理像素 `(i,j)`:
```
4×T[i,j] - T[i-1,j] - T[i+1,j] - T[i,j-1] - T[i,j+1] =
    4×S[i,j] - S[i-1,j] - S[i+1,j] - S[i,j-1] - S[i,j+1]

边界: T[边界] = observed_color
```

**求解**: 稀疏线性系统 `A×x = b`

**代码**: `SceneTexture.cpp` - 使用Eigen库的SparseLU求解器

```python
def seam_leveling(texture, seams):
    height, width = texture.shape
    num_pixels = height * width

    # 构建稀疏矩阵
    A = sparse_matrix(num_pixels, num_pixels)
    b = vector(num_pixels)

    for i in range(height):
        for j in range(width):
            idx = i * width + j

            if is_boundary(i, j):
                # 边界约束
                A[idx, idx] = 1
                b[idx] = texture[i, j]
            else:
                # 拉普拉斯约束
                A[idx, idx] = 4
                A[idx, idx-1] = -1  # 左
                A[idx, idx+1] = -1  # 右
                A[idx, idx-width] = -1  # 上
                A[idx, idx+width] = -1  # 下

                # 右侧 = 梯度约束
                laplacian_S = (4*S[i,j] - S[i-1,j] - S[i+1,j]
                                - S[i,j-1] - S[i,j+1])
                b[idx] = laplacian_S

    # 求解
    T_seamless = solve_sparse(A, b)

    return T_seamless.reshape(height, width)
```

**局部接缝调整**:

在接缝附近的窄带区域进行颜色混合：

```python
def local_seam_blend(texture, seams, bandwidth=10):
    for seam_edge in seams:
        for pixel in seam_edge.neighborhood(bandwidth):
            # 计算到接缝的距离
            dist_to_seam = distance(pixel, seam_edge)

            # 混合权重（距离越近越平滑）
            if dist_to_seam < bandwidth:
                # 获取两侧视图的颜色
                color_view1 = sample_from_view1(pixel)
                color_view2 = sample_from_view2(pixel)

                # 基于距离的混合
                alpha = dist_to_seam / bandwidth
                texture[pixel] = alpha * color_view1 + (1-alpha) * color_view2
```

#### 步骤6：纹理优化（可选）

**a) 锐化**:
```
增强纹理细节
kernel = [[0, -1, 0],
          [-1, 5, -1],
          [0, -1, 0]]
```

**b) 颜色校正**:
```
统一不同视图的颜色分布
直方图匹配或色彩传递
```

**c) 填补空洞**:
```
对未被任何视图覆盖的区域（背面等）
使用周围颜色inpainting填补
```

### 4.4 输出

完整的纹理化模型：
- 网格文件 (OBJ/PLY): 包含顶点、面、UV坐标
- 纹理图像 (PNG/JPG): 一个或多个纹理贴图
- 材质文件 (MTL): 纹理映射关系

---

## 完整工作流程示例

### 5.1 输入准备

从SfM系统（如COLMAP）导出数据：

```bash
# 使用COLMAP接口转换
./InterfaceCOLMAP \
    -i /path/to/colmap/sparse \
    -o scene.mvs \
    --image-folder /path/to/images
```

**scene.mvs 包含**:
- 相机内参 (K矩阵)
- 相机外参 (R, t)
- 稀疏点云
- 图像路径

### 5.2 执行流程

```bash
# 步骤1: 密集重建
./DensifyPointCloud scene.mvs \
    --resolution-level 1 \
    --max-resolution 3200 \
    --number-views 8 \
    --min-views 3 \
    -w /path/to/workspace \
    -o dense.mvs

# 参数说明:
# --resolution-level: 图像降采样级别 (0=原始, 1=1/2, 2=1/4...)
# --max-resolution: 最大图像分辨率
# --number-views: 每个深度图使用的视图数
# --min-views: 融合时最少一致视图数
# 输出: dense.mvs + dense_ply, depth maps

# 步骤2: 网格重建
./ReconstructMesh dense.mvs \
    --thickness-factor 1.0 \
    --quality-factor 1.0 \
    -w /path/to/workspace \
    -o mesh.mvs

# 参数说明:
# --thickness-factor: 表面厚度因子 (控制表面平滑度)
# --quality-factor: 质量因子 (越高越精细，但更慢)
# 输出: mesh.mvs + mesh.ply

# 步骤3: 网格细化
./RefineMesh mesh.mvs \
    --resolution-level 1 \
    --min-resolution 640 \
    --max-views 8 \
    --max-face-area 32 \
    --scales 3 \
    --cuda-device 0 \
    -w /path/to/workspace \
    -o refined_mesh.mvs

# 参数说明:
# --resolution-level: 优化图像分辨率级别
# --max-views: 每个顶点考虑的最大视图数
# --max-face-area: 自适应细分阈值
# --scales: 多尺度级别数
# --cuda-device: GPU设备ID (-1表示CPU)
# 输出: refined_mesh.mvs + refined_mesh.ply

# 步骤4: 纹理映射
./TextureMesh refined_mesh.mvs \
    --resolution-level 0 \
    --texture-size-multiple 0 \
    --patch-packing-heuristic 3 \
    --cost-smoothness-ratio 1.0 \
    --empty-color 0 \
    -w /path/to/workspace \
    -o textured_mesh.mvs

# 参数说明:
# --resolution-level: 纹理源图像分辨率
# --texture-size-multiple: 纹理大小倍数 (0=自动)
# --patch-packing-heuristic: 装箱启发式 (0-7)
# --cost-smoothness-ratio: 视图选择平滑权重
# 输出: textured_mesh.mvs + textured_mesh.{obj,mtl,png}
```

### 5.3 Python API 示例

```python
import pyOpenMVS as omvs

# 创建场景
scene = omvs.Scene()

# 加载数据
scene.load("scene.mvs")

# 密集重建
print("密集重建中...")
scene.dense_reconstruction(
    resolution_level=1,
    max_resolution=3200,
    number_views=8
)
scene.save_pointcloud("dense.ply")

# 网格重建
print("网格重建中...")
scene.reconstruct_mesh(
    quality_factor=1.0
)
scene.save_mesh("mesh.ply")

# 网格细化
print("网格细化中...")
scene.refine_mesh(
    resolution_level=1,
    max_views=8,
    use_cuda=True
)
scene.save_mesh("refined_mesh.ply")

# 纹理映射
print("纹理映射中...")
scene.texture_mesh(
    resolution_level=0,
    cost_smoothness_ratio=1.0
)
scene.save_mesh("textured_mesh.obj")

print("完成!")
```

### 5.4 关键参数调优指南

#### DensifyPointCloud 参数

| 参数 | 默认值 | 建议范围 | 说明 |
|------|--------|----------|------|
| `--resolution-level` | 1 | 0-2 | 高分辨率图像用0，低端硬件用2 |
| `--number-views` | 5 | 4-12 | 更多视图→更准确但更慢 |
| `--min-views` | 3 | 2-5 | 最少一致视图，影响点云密度 |
| `--ncc-threshold-keep` | 0.55 | 0.5-0.9 | 更高→更少但更准确的点 |

**场景建议**:
- **建筑外观**: `--resolution-level 1`, `--number-views 8`
- **小物体**: `--resolution-level 0`, `--number-views 12`
- **大场景**: `--resolution-level 2`, `--number-views 5`

#### ReconstructMesh 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--thickness-factor` | 1.0 | 增加→更平滑表面 |
| `--quality-factor` | 1.0 | 增加→更精细网格 |

#### RefineMesh 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--scales` | 3 | 多尺度级别，3-5 |
| `--gradient-step` | 45.05 | 学习率，自动调整 |
| `--max-iterations` | 100 | 每尺度最大迭代次数 |

#### TextureMesh 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--cost-smoothness-ratio` | 1.0 | 增加→更少接缝但可能模糊 |
| `--texture-size-multiple` | 0 | 手动设置纹理大小 (2048, 4096...) |

### 5.5 性能与质量权衡

**快速预览** (低质量):
```bash
DensifyPointCloud --resolution-level 2 --number-views 4
ReconstructMesh --quality-factor 0.5
RefineMesh --scales 2 --max-iterations 50
TextureMesh --resolution-level 1
```

**生产质量** (高质量):
```bash
DensifyPointCloud --resolution-level 0 --number-views 12 --min-views 4
ReconstructMesh --quality-factor 2.0
RefineMesh --scales 5 --max-iterations 200 --cuda-device 0
TextureMesh --resolution-level 0 --texture-size-multiple 8192
```

---

## 总结

OpenMVS的4个步骤形成完整的MVS重建流水线：

1. **DensifyPointCloud**: PatchMatch立体匹配 → 密集点云
   - 核心: 基于平面假设的多视图NCC匹配
   - 数学: 单应变换、几何一致性、置信度融合

2. **ReconstructMesh**: Delaunay四面体 + Graph Cut → 三角网格
   - 核心: 全局能量优化，空间支持场
   - 数学: 最小割/最大流、能量最小化

3. **RefineMesh**: 基于图像的梯度下降 → 精细网格
   - 核心: 光度一致性优化
   - 数学: 非线性优化、投影几何、正则化

4. **TextureMesh**: 视图选择 + UV展开 → 纹理化模型
   - 核心: MRF标签分配、泊松融合
   - 数学: 信念传播、矩形装箱、梯度域融合

每一步都基于严格的数学模型，结合计算机视觉和图形学的经典算法，最终生成高质量的3D模型。
