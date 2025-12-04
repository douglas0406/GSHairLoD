# V3DG Build 参数说明

## 使用方法

```bash
python v3dg-build.py [参数]
```

---

## 📦 Asset 参数 (资源配置)

### `--ASSET_KIND`
- **类型**: `str`
- **默认值**: `"donut"`
- **说明**: 资源类型/种类

### `--ASSET_NAME`
- **类型**: `str`
- **默认值**: `"donut"`
- **说明**: 资源名称

### `--ASSET_GSPLY_FILENAME`
- **类型**: `str`
- **默认值**: `"point_cloud/iteration_30000/fused.ply"`
- **说明**: 3D Gaussian Splatting PLY 文件路径

---

## 🏗️ Build 参数 (构建配置)

### `--BUILD_APPROPRIATE_COUNT_OF_GAUSSIANS_IN_ONE_CLUSTER`
- **类型**: `int`
- **默认值**: `4096`
- **说明**: 每个 Cluster 中合适的高斯点数量 (n_g_in_c)

### `--BUILD_APPROPRIATE_COUNT_OF_CLUSTERS_IN_ONE_CLUSTER_GROUP`
- **类型**: `int`
- **默认值**: `2`
- **说明**: 每个 Cluster Group 中合适的 Cluster 数量 (n_c_in_cg)

### `--BUILD_MAX_COUNT_OF_CLUSTERS_IN_COARSEST_LOD_LAYER`
- **类型**: `int`
- **默认值**: `2`
- **说明**: 最粗糙 LOD 层的最大 Cluster 数量

---

## 🎯 Root Clustering 参数 (根聚类 - 新方法)

### `--ENABLE_ROOT_CLUSTERING`
- **类型**: `bool`
- **默认值**: `True`
- **说明**: 是否启用基于特征向量的根聚类方法
- **选项**: `true` / `false`

### `--ROOT_CLUSTERING_MIN_CLUSTERS`
- **类型**: `int`
- **默认值**: `2`
- **说明**: 最小聚类数量阈值，低于此值使用原始减半方法

### `--ROOT_CLUSTERING_INIT_STRATEGY`
- **类型**: `str`
- **默认值**: `"auto"`
- **说明**: 聚类初始化策略
- **选项**:
  - `"auto"`: 自动根据 `ROOT_CLUSTERING_REDUCTION_FACTOR` 计算
  - 数字字符串 (如 `"10"`): 指定目标聚类数量

### `--ROOT_CLUSTERING_REDUCTION_FACTOR`
- **类型**: `float`
- **默认值**: `0.5`
- **说明**: 聚类减少因子 (auto模式下使用)，决定目标聚类数 = hair_count × factor

---

## 📍 Spatial Prefiltering 参数 (空间预筛选)

### `--SPATIAL_THRESHOLD`
- **类型**: `float`
- **默认值**: `0.05`
- **说明**: 空间距离阈值，用于判断头发是否在同一空间区域

### `--SPATIAL_METHOD`
- **类型**: `str`
- **默认值**: `"grid"`
- **说明**: 空间预筛选方法
- **选项**:
  - `"grid"`: 网格划分方法 (快速，推荐)
  - `"knn"`: K近邻方法 (精确但较慢)

---

## 🎨 Feature Extraction 参数 (特征提取)

### `--FEATURE_SAMPLE_POINTS`
- **类型**: `int`
- **默认值**: `64`
- **说明**: 每根头发采样的点数

### `--FEATURE_SAMPLING_STRATEGY`
- **类型**: `str`
- **默认值**: `"uniform"`
- **说明**: 采样策略
- **选项**:
  - `"uniform"`: 均匀采样
  - `"adaptive"`: 自适应采样 (根部和尖端密集，中间稀疏)

### `--FEATURE_USE_PCA`
- **类型**: `bool`
- **默认值**: `False`
- **说明**: 是否使用 PCA 对特征向量降维
- **选项**: `true` / `false`

### `--FEATURE_PCA_COMPONENTS`
- **类型**: `int`
- **默认值**: `128`
- **说明**: PCA 降维后的维度 (仅当 `FEATURE_USE_PCA=true` 时生效)

### `--FEATURE_PCA_VARIANCE_RATIO`
- **类型**: `float`
- **默认值**: `0.95`
- **说明**: PCA 保留的方差比例 (仅当 `FEATURE_USE_PCA=true` 时生效)

---

## 🔧 Simplification 参数 (简化配置)

### `--SIMPLIFICATION_INITIALIZATION_DOWNSAMPLE_STRATEGY`
- **类型**: `str`
- **默认值**: `"voxels+osss23+s216"`
- **说明**: 简化初始化的下采样策略
- **选项**:
  - `"random+s213"`: 随机采样 + scale expansion 2^(1/3)
  - `"o+s213"`: 按 opacity 排序 + scale expansion 2^(1/3)
  - `"osss23+s216"`: 按 integral opacity 排序 + scale expansion 2^(1/6)
  - `"voxels+osss23+s216"`: 体素划分 + integral opacity 排序

### `--SIMPLIFICATION_INITIALIZATION_SCALE_EXPANSION`
- **类型**: `bool`
- **默认值**: `True`
- **说明**: 简化时是否扩展高斯点的 scale
- **选项**: `true` / `false`

### `--SIMPLIFICATION_ITERATION`
- **类型**: `int`
- **默认值**: `640`
- **说明**: 局部 splatting 优化的迭代次数 (0 = 不优化)

### `--SIMPLIFICATION_LOSS_LAMBDA_DSSIM`
- **类型**: `float`
- **默认值**: `0.2`
- **说明**: DSSIM 损失的权重系数 (loss = (1-λ)×L1 + λ×DSSIM)

---

## 🎓 Learning Rates 参数 (学习率)

### `--SIMPLIFICATION_LEARNING_RATE_POSITION`
- **类型**: `float`
- **默认值**: `0.0000160`
- **说明**: 位置参数的学习率

### `--SIMPLIFICATION_LEARNING_RATE_SCALE`
- **类型**: `float`
- **默认值**: `0.005`
- **说明**: 缩放参数的学习率

### `--SIMPLIFICATION_LEARNING_RATE_QUATERNION`
- **类型**: `float`
- **默认值**: `0.001`
- **说明**: 旋转四元数的学习率

### `--SIMPLIFICATION_LEARNING_RATE_OPACITY`
- **类型**: `float`
- **默认值**: `0.05`
- **说明**: 透明度的学习率

### `--SIMPLIFICATION_LEARNING_RATE_SH0`
- **类型**: `float`
- **默认值**: `0.0025`
- **说明**: 球谐系数 (颜色) 的学习率

---

## 🐛 Debug 参数

### `--SAVE_IMAGES_DURING_OPTIMIZATION`
- **类型**: `bool`
- **默认值**: `False`
- **说明**: 是否在优化过程中保存中间图像
- **选项**: `true` / `false`

---

## 💡 使用示例

### 基础用法 (使用默认参数)
```bash
python v3dg-build.py \
  --ASSET_KIND hair \
  --ASSET_NAME hair \
  --ASSET_GSPLY_FILENAME hair0619.ply
```

### 高级用法 (自定义聚类参数)
```bash
python v3dg-build.py \
  --ASSET_KIND hair \
  --ASSET_NAME hair \
  --ASSET_GSPLY_FILENAME hair0619.ply \
  --SIMPLIFICATION_ITERATION 16 \
  --FEATURE_SAMPLING_STRATEGY adaptive \
  --FEATURE_SAMPLE_POINTS 128 \
  --FEATURE_USE_PCA true \
  --FEATURE_PCA_COMPONENTS 64 \
  --SPATIAL_THRESHOLD 0.08 \
  --ROOT_CLUSTERING_REDUCTION_FACTOR 0.4
```

### 禁用根聚类 (使用旧方法)
```bash
python v3dg-build.py \
  --ASSET_KIND hair \
  --ASSET_NAME hair \
  --ASSET_GSPLY_FILENAME hair0619.ply \
  --ENABLE_ROOT_CLUSTERING false
```

---

## 📊 参数推荐配置

### 高质量构建 (速度慢)
```bash
--SIMPLIFICATION_ITERATION 640
--FEATURE_SAMPLE_POINTS 128
--FEATURE_USE_PCA true
--FEATURE_SAMPLING_STRATEGY adaptive
```

### 快速构建 (质量中等)
```bash
--SIMPLIFICATION_ITERATION 160
--FEATURE_SAMPLE_POINTS 64
--FEATURE_USE_PCA false
--FEATURE_SAMPLING_STRATEGY uniform
```

### 极速构建 (仅用于测试)
```bash
--SIMPLIFICATION_ITERATION 0
--FEATURE_SAMPLE_POINTS 32
--ENABLE_ROOT_CLUSTERING false
```

---

## 🔍 参数优化建议

1. **`FEATURE_SAMPLE_POINTS`**:
   - 值越大，特征表达越精确，但计算越慢
   - 推荐范围: 32-128

2. **`SPATIAL_THRESHOLD`**:
   - 值越小，分组越细，聚类越精确，但计算量越大
   - 推荐范围: 0.03-0.1

3. **`ROOT_CLUSTERING_REDUCTION_FACTOR`**:
   - 值越小，每层LOD减少的头发越多
   - 推荐范围: 0.3-0.6

4. **`FEATURE_USE_PCA`**:
   - 当 `FEATURE_SAMPLE_POINTS > 64` 时建议启用
   - 可以加速K-means聚类

---

**更新日期**: 2024-12-04
**版本**: V3DG with Feature-based Clustering
