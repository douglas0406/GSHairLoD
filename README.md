# GSHairLoD: Hair-Optimized Level-of-Detail System for 3D Gaussian Splatting

基于[**V3DG**](https://xijie-yang.github.io/V3DG/)的头发场景专用LOD系统。本项目针对头发这类具有线性结构特征的场景，在原V3DG基础上实现了基于特征向量的智能聚类算法。
## 头发LOD构建流程图
![PipeLine](_media/pipeline.png)
## 项目特点

相比原版V3DG，本项目的主要改进：

### 1. 基于特征向量的头发聚类

- **发根位置提取**：自动识别每根头发的根部位置
- **几何特征提取**：采样头发曲线上的点位置和切向量，构建高维特征向量
- **空间预筛选**：使用网格或KNN方法对空间接近的头发预分组，大幅减少计算量
- **K-means智能聚类**：基于头发相似度进行聚类，保留视觉相似的头发

### 2. 头发数据组织

- 支持 `group_id` 和 `strand_id` 标识头发
- 按头发进行分组和LOD简化
- 可视化输出按group_id着色的PLY文件

### 3. 丰富的参数控制

提供多达20+个参数用于精细控制聚类行为，包括：
- 特征采样点数和采样策略
- 空间预筛选阈值和方法
- 聚类减少因子和初始化策略
- PCA降维选项

详见 [PARAMETERS.md](./PARAMETERS.md)

## Get Started

### Environment Setup

All experiments were conducted on a Linux server running Ubuntu 22.04 with CUDA 11.8. Other versions may also be compatible.

```bash
conda create -y -n V3DG python=3.12
conda activate V3DG

pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install plyfile==1.1 einops==0.8.1 tqdm==4.67.1 scipy==1.15.2 gsplat==1.4.0 opencv-python==4.11.0.86
```
### 准备头发数据

将带有strands_id与group_id的头发ply放在 `./_3dgs-assets/hair/` 目录下。
提供两个测试文件：
[Download the data](https://drive.google.com/drive/folders/1BJGFEbNpATuU3RlMIsJMwNbwyoLtmApf?usp=sharing)


### 构建LOD层级

```bash
python v3dg-build.py \
  --ASSET_KIND hair \
  --ASSET_NAME hair \
  --ASSET_GSPLY_FILENAME hair0619.ply \
  --SIMPLIFICATION_ITERATION 16
```

### 核心代码 (`libraries/classes.py`)

主要修改的函数：

1. **`extract_strand_roots()`** 
   - 从Cluster中提取每根头发的根位置

2. **`extract_strand_features()`** 
   - 提取头发的位置+切向量特征

3. **`spatial_prefilter_by_roots()`**
   - 空间预筛选，减少计算量

4. **`feature_based_clustering()`** 
   - K-means聚类实现

5. **`Clusters.buildCoarserLodLayer()`**
   - 核心LOD构建流程，集成上述所有方法

6. **`Cluster.splitByGroupID()`** 
   - 按group_id拆分cluster

### 输出结果

构建完成后，会在 `./_v3dg-bundles/` 生成bundle文件，在 `./_output/` 生成：
- `plys/` - 各LOD层级的可视化PLY文件（按group_id着色）
- `images/` - 六方向渲染图
- `_records.py` - 构建耗时记录
- `build_log_*.txt` - 详细日志

## 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ENABLE_ROOT_CLUSTERING` | `true` | 是否启用基于特征的根聚类（新方法） |
| `FEATURE_SAMPLE_POINTS` | `64` | 每根头发采样的点数，影响特征精度 |
| `FEATURE_SAMPLING_STRATEGY` | `uniform` | `uniform`均匀采样 或 `adaptive`自适应采样 |
| `SPATIAL_THRESHOLD` | `0.05` | 空间距离阈值，值越小分组越细 |
| `ROOT_CLUSTERING_REDUCTION_FACTOR` | `0.5` | 每层LOD保留的头发比例 |
| `SIMPLIFICATION_ITERATION` | `640` | 局部优化迭代次数，0表示不优化 |

完整参数列表请查看 [PARAMETERS.md](./PARAMETERS.md)

## License

与原V3DG项目保持一致

## 联系方式

如有问题或建议，欢迎提issue或PR。
