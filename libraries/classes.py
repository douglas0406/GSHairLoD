# https://stackoverflow.com/questions/33533148/how-do-i-type-hint-a-method-with-the-type-of-the-enclosing-class
from __future__ import annotations

import pathlib, math, time, functools, datetime, zoneinfo

import numpy as np
import hashlib
import plyfile
import torch
import einops

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
import scipy.spatial.transform

# https://tqdm.github.io/docs/tqdm/#tqdm-objects
import tqdm

# https://docs.gsplat.studio/main/index.html
import gsplat

from libraries.utilities import ExLog, ExTimer, UTILITY
from libraries.cliconfigs import VGBuildConfig
def extract_strand_roots(cluster: Cluster) -> torch.Tensor:
    """
    从 Cluster 中按 group_id 分组，取每组 strand_id 最小的点作为该根头发的根位置。
    返回 Tensor roots，shape = (hairCount, 3)。
    对每个 gid，mask 选出所有属于该头发的 Gaussians，torch.argmin(strand_ids) 找到序号最小（最接近头皮）的那个点，作为根。
    """
    # 找到所有不同的 group_id
    gids = torch.unique(cluster.group_id)
    hair_count = gids.numel()
    # 准备输出
    roots = torch.zeros((hair_count, 3),
                        dtype=cluster.positions.dtype,
                        device=cluster.positions.device)
    # 对每个 group_id 找到最小 strand_id 的位置
    for i, gid in enumerate(gids):
        mask = (cluster.group_id == gid)
        # 该组内部所有点的 strand_id
        strand_ids = cluster.strand_id[mask]
        # 找到最小的索引位置
        min_idx_in_mask = torch.argmin(strand_ids)
        # 提取对应的 3D 坐标
        roots[i] = cluster.positions[mask][min_idx_in_mask]
    return roots


def extract_strand_features(
    cluster: 'Cluster',
    max_sample_points: int = 64,
    sampling_strategy: str = 'uniform'
) -> torch.Tensor:
    """
    提取每根头发的特征向量（位置 + 切线方向）

    参数:
        cluster: 包含多根头发的Cluster
        max_sample_points: 每根头发最多采样的点数（默认64）
        sampling_strategy: 采样策略 'uniform' 或 'adaptive'

    返回:
        torch.Tensor: shape=(hair_count, feature_dim)
        feature_dim = max_sample_points * 6 (每个采样点6维: xyz + 切向量txyz)
    """
    unique_gids = torch.unique(cluster.group_id)
    hair_count = unique_gids.numel()
    feature_dim = max_sample_points * 6  # 位置xyz + 切向量txyz

    # 准备输出特征矩阵
    features = torch.zeros((hair_count, feature_dim),
                          dtype=cluster.positions.dtype,
                          device=cluster.positions.device)

    # 记录每根头发的实际长度，用于后续分析
    strand_lengths = []
    sampled_lengths = []

    for i, gid in enumerate(unique_gids):
        # 获取当前头发的所有点
        mask = (cluster.group_id == gid)
        group_positions = cluster.positions[mask]  # (N, 3)
        group_strand_ids = cluster.strand_id[mask]  # (N,)

        # 按strand_id排序（从根到梢）
        sorted_indices = torch.argsort(group_strand_ids)
        sorted_positions = group_positions[sorted_indices]

        actual_length = sorted_positions.shape[0]
        strand_lengths.append(actual_length)

        # 采样策略
        if sampling_strategy == 'uniform':
            # 均匀采样
            if actual_length <= max_sample_points:
                sampled_positions = sorted_positions
                sample_count = actual_length
            else:
                # 均匀间隔采样
                indices = torch.linspace(0, actual_length - 1, max_sample_points,
                                       dtype=torch.long, device=cluster.positions.device)
                sampled_positions = sorted_positions[indices]
                sample_count = max_sample_points

        elif sampling_strategy == 'adaptive':
            # 自适应采样：根部、尖端密集，中间稀疏
            if actual_length <= max_sample_points:
                sampled_positions = sorted_positions
                sample_count = actual_length
            else:
                root_count = max_sample_points // 4
                tip_count = max_sample_points // 4
                middle_count = max_sample_points - root_count - tip_count

                # 根部采样
                root_indices = torch.linspace(0, actual_length // 4, root_count,
                                            dtype=torch.long, device=cluster.positions.device)
                # 中部采样
                start_idx = actual_length // 4
                end_idx = actual_length * 3 // 4
                middle_indices = torch.linspace(start_idx, end_idx, middle_count,
                                              dtype=torch.long, device=cluster.positions.device)
                # 尖端采样
                tip_start = actual_length * 3 // 4
                tip_indices = torch.linspace(tip_start, actual_length - 1, tip_count,
                                           dtype=torch.long, device=cluster.positions.device)

                all_indices = torch.cat([root_indices, middle_indices, tip_indices])
                sampled_positions = sorted_positions[all_indices]
                sample_count = max_sample_points
        else:
            raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")

        sampled_lengths.append(sample_count)

        # 计算切向量（相邻点的方向）
        if sample_count > 1:
            tangents = sampled_positions[1:] - sampled_positions[:-1]  # (sample_count-1, 3)
            # 归一化切向量
            tangent_norms = torch.norm(tangents, dim=1, keepdim=True) + 1e-8
            tangents = tangents / tangent_norms
            # 最后一个点的切向量重复前一个
            tangents = torch.cat([tangents, tangents[-1:]], dim=0)  # (sample_count, 3)
        else:
            tangents = torch.zeros_like(sampled_positions)

        # 组合特征：[pos0, tangent0, pos1, tangent1, ...]
        strand_features = torch.cat([sampled_positions, tangents], dim=1)  # (sample_count, 6)
        strand_features = strand_features.flatten()  # (sample_count * 6,)

        # 填充到固定维度
        base_idx = i * feature_dim
        valid_dim = min(sample_count * 6, feature_dim)
        features[i, :valid_dim] = strand_features[:valid_dim]

    # 可选：记录采样信息
    sampling_info = {
        'hair_count': hair_count,
        'avg_strand_length': sum(strand_lengths) / len(strand_lengths),
        'avg_sampled_length': sum(sampled_lengths) / len(sampled_lengths),
    }
    ExLog(f"特征提取: {hair_count}根头发, 平均长度{sampling_info['avg_strand_length']:.1f}, "
          f"采样{sampling_info['avg_sampled_length']:.1f}点", "DEBUG")

    return features


def spatial_prefilter_by_roots(
    roots: torch.Tensor,
    group_ids: torch.Tensor,
    spatial_threshold: float = 0.05,
    method: str = 'grid'
) -> list[list[int]]:
    """
    根据发根的空间位置进行预筛选，将空间上接近的头发分组

    参数:
        roots: (hair_count, 3) 发根位置
        group_ids: (hair_count,) 每根头发的group_id
        spatial_threshold: 空间距离阈值
        method: 'grid' 或 'knn'

    返回:
        list[list[int]]: 每个子列表是一组空间上接近的group_id
    """
    hair_count = roots.shape[0]
    roots_np = roots.cpu().numpy()

    if method == 'grid':
        # 网格方法：将空间划分为网格，同一网格及相邻网格的头发分为一组
        from collections import defaultdict

        grid_size = spatial_threshold
        min_coords = roots_np.min(axis=0)

        # 计算每个点的网格坐标
        grid_coords = ((roots_np - min_coords) / grid_size).astype(np.int32)

        # 构建网格字典
        grid_dict = defaultdict(list)
        for i, grid_coord in enumerate(grid_coords):
            gid = int(group_ids[i].item())
            grid_key = tuple(grid_coord)
            grid_dict[grid_key].append(gid)

        # 合并相邻网格
        local_groups = []
        processed_grids = set()

        for grid_key, gids_in_grid in grid_dict.items():
            if grid_key in processed_grids:
                continue

            # 创建一个新的局部组，包括当前网格和相邻网格
            local_group = set(gids_in_grid)
            processed_grids.add(grid_key)

            # 检查27个相邻网格（3x3x3）
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        neighbor_key = (grid_key[0] + dx, grid_key[1] + dy, grid_key[2] + dz)
                        if neighbor_key in grid_dict:
                            local_group.update(grid_dict[neighbor_key])
                            processed_grids.add(neighbor_key)

            local_groups.append(list(local_group))

    elif method == 'knn':
        # KNN方法：使用最近邻构建连通图
        from sklearn.neighbors import NearestNeighbors

        k = min(10, hair_count)  # 每个点最多找10个邻居
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(roots_np)
        distances, indices = nbrs.kneighbors(roots_np)

        # 构建连通图（使用并查集或DFS）
        adj = [set() for _ in range(hair_count)]
        for i in range(hair_count):
            for j, dist in zip(indices[i], distances[i]):
                if dist < spatial_threshold and i != j:
                    adj[i].add(j)
                    adj[j].add(i)

        # DFS找连通分量
        visited = [False] * hair_count
        local_groups = []

        def dfs(node, component):
            visited[node] = True
            component.append(int(group_ids[node].item()))
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    dfs(neighbor, component)

        for i in range(hair_count):
            if not visited[i]:
                component = []
                dfs(i, component)
                local_groups.append(component)

    else:
        raise ValueError(f"Unknown method: {method}")

    # 过滤掉太小的组（可选）
    local_groups = [g for g in local_groups if len(g) > 0]

    # 统计信息
    total_pairs_before = hair_count * (hair_count - 1) // 2
    total_pairs_after = sum(len(g) * (len(g) - 1) // 2 for g in local_groups)
    reduction_ratio = 1.0 - total_pairs_after / max(total_pairs_before, 1)

    ExLog(f"空间预筛选: {hair_count}根头发 → {len(local_groups)}个局部组, "
          f"比较次数减少{reduction_ratio*100:.1f}%", "DEBUG")

    return local_groups


def feature_based_clustering(
    features: torch.Tensor,
    target_cluster_count: int,
    similarity_metric: str = 'euclidean',
    vg_build_config: 'VGBuildConfig' = None
) -> list[int]:
    """
    基于特征向量的K-means聚类

    参数:
        features: (hair_count, feature_dim) 特征矩阵
        target_cluster_count: 目标聚类数量
        similarity_metric: 相似度度量 'euclidean' 或 'cosine'
        vg_build_config: 配置对象（用于PCA等高级选项）

    返回:
        list[int]: 长度为hair_count的标签列表
    """
    try:
        import sklearn.cluster
        from sklearn.preprocessing import normalize, StandardScaler
        from sklearn.decomposition import PCA
    except ImportError:
        ExLog("需要安装scikit-learn: pip install scikit-learn", "ERROR")
        raise

    import time

    hair_count, original_dim = features.shape
    features_np = features.cpu().numpy()

    ExLog(f"特征聚类: {hair_count}根头发, 特征维度{original_dim}, 目标{target_cluster_count}类", "DEBUG")

    # 预处理
    start_time = time.time()

    # 标准化
    scaler = StandardScaler()
    features_np = scaler.fit_transform(features_np)

    # 可选：降维（如果维度太高）
    pca_applied = False
    if original_dim > 128 and vg_build_config is not None:
        if hasattr(vg_build_config, 'FEATURE_USE_PCA') and vg_build_config.FEATURE_USE_PCA:
            n_components = min(128, hair_count - 1)
            try:
                pca = PCA(n_components=n_components)
                features_pca = pca.fit_transform(features_np)
                explained_variance_ratio = pca.explained_variance_ratio_.sum()
                ExLog(f"PCA降维: {original_dim} → {n_components}, 保留{explained_variance_ratio*100:.1f}%方差", "DEBUG")
                features_np = features_pca
                pca_applied = True
            except Exception as e:
                ExLog(f"PCA失败: {e}, 使用原始特征", "WARNING")

    final_dim = features_np.shape[1]

    # 归一化（如果使用余弦相似度）
    if similarity_metric == 'cosine':
        features_np = normalize(features_np, norm='l2')

    preprocessing_time = time.time() - start_time

    # K-means聚类
    clustering_start = time.time()

    kmeans = sklearn.cluster.MiniBatchKMeans(
        n_clusters=target_cluster_count,
        init='k-means++',
        n_init='auto',
        max_iter=300,
        batch_size=min(1024, hair_count),
        random_state=0,
    )

    labels = kmeans.fit_predict(features_np)

    clustering_time = time.time() - clustering_start

    # 统计信息
    inertia = kmeans.inertia_
    n_iter = kmeans.n_iter_
    unique_labels = len(set(labels))

    ExLog(f"K-means完成: {n_iter}次迭代, inertia={inertia:.2f}, "
          f"实际{unique_labels}类, 耗时{clustering_time:.2f}s", "DEBUG")

    # 检查聚类平衡性
    from collections import Counter
    cluster_sizes = Counter(labels)
    for i, (cluster_id, group_size) in enumerate(cluster_sizes.most_common()):
        if i < 5:  # 只显示前5个最大的类
            ExLog(f"  聚类{cluster_id}: {group_size}根头发", "DEBUG")

    return labels.tolist()


# ========== 旧的三角化相关函数已删除 ==========
# compute_uv() 和 region_growing_clustering() 已被新的特征向量方法替代
# 参见: extract_strand_features(), spatial_prefilter_by_roots(), feature_based_clustering()



class Camera:
    def DeriveRandomPoseLookingAtOrigin(
        center: torch.Tensor,
        radius: float,
        asset_name: str = "donut",
    ) -> Camera:
        """
        center: (3,)
        radius: pass in four times of the furthest distance in a cluster / cluster group
        """
        camera_position_z = -radius + 2.0 * radius * torch.rand(1)
        camera_position_xy_radius = torch.sqrt(radius**2 - camera_position_z**2)
        camera_position_xy_theta = torch.rand(1) * 2.0 * torch.pi
        camera_position_x = camera_position_xy_radius * torch.cos(
            camera_position_xy_theta
        )
        camera_position_y = camera_position_xy_radius * torch.sin(
            camera_position_xy_theta
        )
        camera_position_relative = torch.tensor(
            [camera_position_x, camera_position_y, camera_position_z]
        )
        camera_position_absolute = center + camera_position_relative

        camera_lookat = (
            -camera_position_relative / camera_position_relative.pow(2).sum().sqrt()
        )
        camera_upward = torch.tensor([camera_lookat[1], -camera_lookat[0], 0])
        camera_upward = camera_upward / camera_upward.pow(2).sum().sqrt()
        camera_cross = torch.cross(camera_lookat, camera_upward, dim=0)

        # use row vector
        T = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [
                    -camera_position_absolute[0],
                    -camera_position_absolute[1],
                    -camera_position_absolute[2],
                    1,
                ],
            ]
        )
        R = torch.tensor(
            [
                [camera_cross[0], -camera_upward[0], camera_lookat[0], 0],
                [camera_cross[1], -camera_upward[1], camera_lookat[1], 0],
                [camera_cross[2], -camera_upward[2], camera_lookat[2], 0],
                [0, 0, 0, 1],
            ]
        )

        if asset_name == "donut":
            original_image_width = 800
            original_image_height = 800
            original_focal_x = 1111.1111350971692
            original_focal_y = 1111.1111350971692
        else:
            raise NotImplementedError

        # TODO This can be modified.
        image_width = 64
        image_height = 64
        # TODO keep fov unchanged?
        focal_x = original_focal_x * image_width / original_image_width
        focal_y = original_focal_y * image_height / original_image_height

        camera_temp = Camera(
            image_width=image_width,
            image_height=image_height,
            focal_x=focal_x,
            focal_y=focal_y,
            # don't use the initialization of view_matrix
            R=torch.eye(3),
            t=torch.tensor(
                [0.0, 0.0, 0.0],
                dtype=torch.float,
            ),
        )
        camera_temp.view_matrix = T @ R

        return camera_temp

    def DeriveSixDirections(
        center: torch.Tensor,
        distance: float,
        asset_name: str = "donut",
    ) -> list[Camera]:
        # TODO explore this! how to give the parameters
        if asset_name == "donut":
            original_image_width = 800
            original_image_height = 800
            original_focal_x = 1111.1111350971692
            original_focal_y = 1111.1111350971692
        else:
            raise NotImplementedError

        # (1, 3) -> (3,)
        center = center[0]

        # TODO This can be modified.
        image_width = 64
        image_height = 64
        # TODO keep fov unchanged?
        focal_x = original_focal_x * image_width / original_image_width
        focal_y = original_focal_y * image_height / original_image_height

        # front, left, right, back, up, down
        R_front = torch.tensor(
            scipy.spatial.transform.Rotation.from_rotvec(
                [0, 0, 180], degrees=True
            ).as_matrix(),
            dtype=torch.float,
        ) @ torch.tensor(
            scipy.spatial.transform.Rotation.from_rotvec(
                [-90, 0, 0], degrees=True
            ).as_matrix(),
            dtype=torch.float,
        )
        cameras: list[Camera] = [
            Camera(
                image_width=image_width,
                image_height=image_height,
                focal_x=focal_x,
                focal_y=focal_y,
                R=R_front,
                t=torch.tensor(
                    [center[0], center[1] + distance, center[2]], dtype=torch.float
                ),
            ),
            Camera(
                image_width=image_width,
                image_height=image_height,
                focal_x=focal_x,
                focal_y=focal_y,
                R=torch.tensor(
                    scipy.spatial.transform.Rotation.from_rotvec(
                        [0, 0, -90], degrees=True
                    ).as_matrix(),
                    dtype=torch.float,
                )
                @ R_front,
                t=torch.tensor(
                    [center[0] + distance, center[1], center[2]], dtype=torch.float
                ),
            ),
            Camera(
                image_width=image_width,
                image_height=image_height,
                focal_x=focal_x,
                focal_y=focal_y,
                R=torch.tensor(
                    scipy.spatial.transform.Rotation.from_rotvec(
                        [0, 0, 90], degrees=True
                    ).as_matrix(),
                    dtype=torch.float,
                )
                @ R_front,
                t=torch.tensor(
                    [center[0] - distance, center[1], center[2]], dtype=torch.float
                ),
            ),
            Camera(
                image_width=image_width,
                image_height=image_height,
                focal_x=focal_x,
                focal_y=focal_y,
                R=torch.tensor(
                    scipy.spatial.transform.Rotation.from_rotvec(
                        [0, 0, 180], degrees=True
                    ).as_matrix(),
                    dtype=torch.float,
                )
                @ R_front,
                t=torch.tensor(
                    [center[0], center[1] - distance, center[2]], dtype=torch.float
                ),
            ),
            Camera(
                image_width=image_width,
                image_height=image_height,
                focal_x=focal_x,
                focal_y=focal_y,
                R=torch.tensor(
                    scipy.spatial.transform.Rotation.from_rotvec(
                        [90, 0, 0], degrees=True
                    ).as_matrix(),
                    dtype=torch.float,
                )
                @ R_front,
                t=torch.tensor(
                    [center[0], center[1], center[2] + distance], dtype=torch.float
                ),
            ),
            Camera(
                image_width=image_width,
                image_height=image_height,
                focal_x=focal_x,
                focal_y=focal_y,
                R=torch.tensor(
                    scipy.spatial.transform.Rotation.from_rotvec(
                        [-90, 0, 0], degrees=True
                    ).as_matrix(),
                    dtype=torch.float,
                )
                @ R_front,
                t=torch.tensor(
                    [center[0], center[1], center[2] - distance], dtype=torch.float
                ),
            ),
        ]

        return cameras

    def FocalToFov(focal, pixels):
        return 2 * math.atan(pixels / (2 * focal))

    def GetProjectionMatrix(z_near, z_far, fov_x, fov_y):
        tan_half_fov_x = math.tan((fov_x / 2))
        tan_half_fov_y = math.tan((fov_y / 2))

        top = tan_half_fov_y * z_near
        bottom = -top
        right = tan_half_fov_x * z_near
        left = -right

        P = torch.zeros(4, 4)

        z_sign = 1.0

        P[0, 0] = 2.0 * z_near / (right - left)
        P[1, 1] = 2.0 * z_near / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * z_far / (z_far - z_near)
        P[2, 3] = -(z_far * z_near) / (z_far - z_near)
        return P

    def __init__(self, image_width, image_height, focal_x, focal_y, R, t) -> None:
        self.image_width = image_width
        self.image_height = image_height
        self.focal_x = focal_x
        self.focal_y = focal_y

        self.fov_x = Camera.FocalToFov(focal=self.focal_x, pixels=self.image_width)
        self.fov_y = Camera.FocalToFov(focal=self.focal_y, pixels=self.image_height)

        # [view matrix]

        Rt = torch.zeros((4, 4), dtype=torch.float)
        Rt[:3, :3] = R
        Rt[:3, 3] = t
        Rt[3, 3] = 1.0
        Rt_inv: torch.Tensor = torch.linalg.inv(Rt)
        # homogenous coordinate, row vector
        self.view_matrix = Rt_inv.transpose(0, 1)

        # [projection matrix]

        z_far = 100.0  # no use
        z_near = 0.01  # no use
        # homogenous coordinate, row vector
        self.projection_matrix = Camera.GetProjectionMatrix(
            z_near=z_near, z_far=z_far, fov_x=self.fov_x, fov_y=self.fov_y
        ).transpose(0, 1)


class LearnableGaussians(torch.nn.Module):
    
    def ActivationScales(x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x)

    def InverseActivationScales(y: torch.Tensor) -> torch.Tensor:
        return torch.log(y)

    def ActivationQauternions(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(x)

    def ActivationOpacities(x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)

    def InverseActivationOpacities(y: torch.Tensor) -> torch.Tensor:
        return torch.log(y / (1 - y))

    def ActivationSh0(x: torch.Tensor) -> torch.Tensor:
        SH_C0 = 0.28209479177387814
        return torch.clip(x * SH_C0 + 0.5, min=0.0, max=None)

    def InverseActivationSh0(y: torch.Tensor) -> torch.Tensor:
        SH_C0 = 0.28209479177387814
        return (y - 0.5) / SH_C0

    def __init__(
        self,
        vg_build_config: VGBuildConfig,
        original_gaussians: Cluster,
        gaussians: Cluster,
        center: torch.Tensor,
        distance: float,
    ) -> None:
        """
        pass in initialized 3D Gaussians. LearnableGaussians is only responsible for the optimization task. The count of 3D Gaussians won't change during optimization.
        """
        super().__init__()
        self.group_id = gaussians.group_id.clone()
        self.strand_id = gaussians.strand_id.clone()

        self.vg_build_config = vg_build_config

        self.count = gaussians.count

        self.center = center
        self.distance = distance

        self.original_gaussians = original_gaussians

        if gaussians.positions.isnan().any():
            ExLog("LearnableGaussian.__init__() has nan!!!", "ERROR")
            exit(-1)

        with torch.no_grad():
            self.parameters_positions: torch.Tensor = torch.nn.Parameter(
                gaussians.positions.clone(), requires_grad=True
            )
            self.parameters_scales: torch.Tensor = torch.nn.Parameter(
                LearnableGaussians.InverseActivationScales(gaussians.scales.clone()),
                requires_grad=True,
            )
            self.parameters_quaternions: torch.Tensor = torch.nn.Parameter(
                gaussians.quaternions.clone(), requires_grad=True
            )
            self.parameters_opacities: torch.Tensor = torch.nn.Parameter(
                LearnableGaussians.InverseActivationOpacities(
                    gaussians.opacities.clone()
                ),
                requires_grad=True,
            )
            self.parameters_sh0: torch.Tensor = torch.nn.Parameter(
                LearnableGaussians.InverseActivationSh0(gaussians.rgbs.clone()),
                requires_grad=True,
            )

    @property
    def positions(self) -> torch.Tensor:
        return self.parameters_positions

    @property
    def scales(self) -> torch.Tensor:
        return LearnableGaussians.ActivationScales(self.parameters_scales)

    @property
    def quaternions(self) -> torch.Tensor:
        return LearnableGaussians.ActivationQauternions(self.parameters_quaternions)

    @property
    def opacities(self) -> torch.Tensor:
        return LearnableGaussians.ActivationOpacities(self.parameters_opacities)

    @property
    def rgbs(self) -> torch.Tensor:
        return LearnableGaussians.ActivationSh0(self.parameters_sh0)

    # NOTICE: Directly call functions in class Cluster.

    def render(self, *args, **kwargs):
        return Cluster.render(self, *args, **kwargs)

    def renderReturnCountAndDuration(self, *args, **kwargs):
        return Cluster.renderReturnCountAndDuration(self, *args, **kwargs)

    def renderFullImageConsolidatingSixDirections(self, *args, **kwargs):
        return Cluster.renderFullImageConsolidatingSixDirections(self, *args, **kwargs)

    def train(self) -> None:
        # https://pytorch.org/docs/stable/optim.html
        parameters = [
            {
                "params": [self.parameters_positions],
                "lr": self.vg_build_config.SIMPLIFICATION_LEARNING_RATE_POSITION,
            },
            {
                "params": [self.parameters_scales],
                "lr": self.vg_build_config.SIMPLIFICATION_LEARNING_RATE_SCALE,
            },
            {
                "params": [self.parameters_quaternions],
                "lr": self.vg_build_config.SIMPLIFICATION_LEARNING_RATE_QUATERNION,
            },
            {
                "params": [self.parameters_opacities],
                "lr": self.vg_build_config.SIMPLIFICATION_LEARNING_RATE_OPACITY,
            },
            {
                "params": [self.parameters_sh0],
                "lr": self.vg_build_config.SIMPLIFICATION_LEARNING_RATE_SH0,
            },
        ]
        optimizer = torch.optim.Adam(parameters, lr=0.0, eps=1e-15)

        # DEBUG save intermediate results
        current_time_str = datetime.datetime.now(
            tz=zoneinfo.ZoneInfo("Asia/Shanghai")
        ).strftime("%y%m%d-%H%M%S")
        if self.vg_build_config.SAVE_IMAGES_DURING_OPTIMIZATION:
            UTILITY.SaveImage(
                self.original_gaussians.renderFullImageConsolidatingSixDirections(
                    center=self.center, distance=self.distance
                ),
                self.vg_build_config.OUTPUT_FOLDER_PATH
                / f"images/{current_time_str}-original.png",
            )
        for iter in range(self.vg_build_config.SIMPLIFICATION_ITERATION + 1):
            # [calculate loss]

            random_camera_looking_at_center = Camera.DeriveRandomPoseLookingAtOrigin(
                center=self.center[0], radius=self.distance
            )

            # (4, h, w)
            image_gt: torch.Tensor = self.original_gaussians.render(
                camera=random_camera_looking_at_center
            )
            image_render: torch.Tensor = self.render(
                camera=random_camera_looking_at_center
            )

            if self.vg_build_config.SAVE_IMAGES_DURING_OPTIMIZATION:
                if iter % 160 == 0:
                    UTILITY.SaveImage(
                        self.renderFullImageConsolidatingSixDirections(
                            center=self.center, distance=self.distance
                        ),
                        self.vg_build_config.OUTPUT_FOLDER_PATH
                        / f"images/{current_time_str}-iter{iter}.png",
                    )

            # add black background
            loss_l1: torch.Tensor = UTILITY.L1Loss(
                image=image_render[:3] * image_render[3],
                target=image_gt[:3] * image_gt[3],
            )
            ssim: torch.Tensor = UTILITY.Ssim(
                image=image_render[:3] * image_render[3],
                target=image_gt[:3] * image_gt[3],
            )

            loss_dssim: torch.Tensor = 1.0 - ssim
            # add alpha channel supervision here
            loss: torch.Tensor = (
                (1.0 - self.vg_build_config.SIMPLIFICATION_LOSS_LAMBDA_DSSIM) * loss_l1
                + self.vg_build_config.SIMPLIFICATION_LOSS_LAMBDA_DSSIM * loss_dssim
                + 0.1 * torch.abs((image_render[3] - image_gt[3])).mean()
            )

            # [backward]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def toGaussians(self, lod_level: int) -> Cluster:
        return Cluster(
            vg_build_config=self.vg_build_config,
            count=self.count,
            positions=self.positions.clone().detach().requires_grad_(False),
            scales=self.scales.clone().detach().requires_grad_(False),
            quaternions=self.quaternions.clone().detach().requires_grad_(False),
            opacities=self.opacities.clone().detach().requires_grad_(False),
            rgbs=self.rgbs.clone().detach().requires_grad_(False),
            group_id=self.group_id.clone().detach().requires_grad_(False),
            strand_id=self.strand_id.clone().detach().requires_grad_(False),
            lod_level=lod_level,
        )


class ClustersList:
    def __init__(
        self,
        vg_build_config: VGBuildConfig,
        clusters_list: list[Clusters],
    ) -> None:
        self.vg_build_config = vg_build_config

        self.clusters_list: list[Clusters] = clusters_list
        self.count: int = len(self.clusters_list)

    def append(self, clusters: Clusters) -> None:
        self.clusters_list.append(clusters)
        self.count = len(self.clusters_list)

    def extend(self, clusters_list: list[Clusters]) -> None:
        self.clusters_list.extend(clusters_list)
        self.count = len(self.clusters_list)

    def consolidateIntoClusters(self) -> Clusters:
        clusters: list[Cluster] = functools.reduce(
            lambda a, b: a + b.clusters, self.clusters_list, []
        )
        return Clusters(
            vg_build_config=self.vg_build_config, clusters=clusters, lod_level=None
        )

    def savePlyWithDifferentColors(self, path: pathlib.Path) -> None:
        color_choices = np.random.randint(
            low=0, high=255, size=(self.count, 3), dtype=np.uint8
        )

        ply_points = np.concatenate(
            [
                np.concatenate(
                    [
                        clusters.consolidateIntoASingleCluster()
                        .positions.cpu()
                        .numpy(),
                        np.zeros(
                            clusters.consolidateIntoASingleCluster().positions.shape,
                            dtype=np.uint8,
                        )
                        + color_choices[i],
                    ],
                    axis=1,
                )
                for i, clusters in enumerate(self.clusters_list)
            ],
            axis=0,
        )
        ply_properties = [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
        ] + [
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]

        UTILITY.SavePlyUsingPlyfilePackage(
            path=path,
            points=ply_points,
            properties=ply_properties,
        )

        ExLog(f"Save {self.count} cluster groups at {path}.")


    def saveBundle(self) -> None:
        # [save clusters.npz]

        clusters_count = 0
        for clusters in self.clusters_list:
            clusters_count += clusters.count
            ExLog(
                f"LOD{clusters.lod_level}, {clusters.count} clusters, gaussians in each cluster {[cluster.count for cluster in clusters.clusters]}",
                "DEBUG",
            )

        lod_levels = np.zeros((clusters_count, 1), dtype=np.int32)
        start_indices = np.zeros((clusters_count, 1), dtype=np.int32)
        counts = np.zeros((clusters_count, 1), dtype=np.int32)
        child_centers = np.zeros((clusters_count, 3), dtype=np.float32)
        parent_centers = np.zeros((clusters_count, 3), dtype=np.float32)
        child_radii = np.zeros((clusters_count, 1), dtype=np.float32)
        parent_radii = np.zeros((clusters_count, 1), dtype=np.float32)

        start_index = 0
        i_cluster = 0
        for clusters in self.clusters_list:
            for cluster in clusters.clusters:
                lod_levels[i_cluster] = cluster.lod_level
                start_indices[i_cluster] = start_index
                counts[i_cluster] = cluster.count
                child_centers[i_cluster] = cluster.child_center_in_cluster_group
                parent_centers[i_cluster] = cluster.parent_center_in_cluster_group
                child_radii[i_cluster] = cluster.child_radius_in_cluster_group
                parent_radii[i_cluster] = cluster.parent_radius_in_cluster_group

                start_index += cluster.count
                i_cluster += 1

        np.savez(
            self.vg_build_config.BUNDLE_CLUSTERS_NPZ_PATH,
            lod_levels=lod_levels,
            start_indices=start_indices,
            counts=counts,
            child_centers=child_centers,
            parent_centers=parent_centers,
            child_radii=child_radii,
            parent_radii=parent_radii,
        )

        # [save gaussians.npz]

        np.savez(
            self.vg_build_config.BUNDLE_GAUSSIANS_NPZ_PATH,
            positions=np.concatenate(
                [
                    cluster.positions.cpu().numpy()
                    for cluster in self.consolidateIntoClusters().clusters
                ],
                axis=0,
            ),
            scales=np.concatenate(
                [
                    cluster.scales.cpu().numpy()
                    for cluster in self.consolidateIntoClusters().clusters
                ],
                axis=0,
            ),
            quaternions=np.concatenate(
                [
                    cluster.quaternions.cpu().numpy()
                    for cluster in self.consolidateIntoClusters().clusters
                ],
                axis=0,
            ),
            opacities=np.concatenate(
                [
                    cluster.opacities.cpu().numpy()
                    for cluster in self.consolidateIntoClusters().clusters
                ],
                axis=0,
            ),
            rgbs=np.concatenate(
                [
                    cluster.rgbs.cpu().numpy()
                    for cluster in self.consolidateIntoClusters().clusters
                ],
                axis=0,
            ),
        )


class Clusters:
    def __init__(
        self,
        clusters: list[Cluster],
        vg_build_config: VGBuildConfig = None,
        lod_level: int | None = None,
    ) -> None:
        """
        Assign `lod_level` an int if all clusters are at the same lod level.
        """
        self.vg_build_config = vg_build_config

        self.clusters = clusters
        self.count = len(self.clusters)
        self.lod_level = lod_level

    
    def savePlyByGroupID(self, path: pathlib.Path) -> None:
        # 1. 先收集所有子聚类对应的 group_id（它们内部是一致的）
        cluster_ids = [int(cluster.group_id[0].item()) for cluster in self.clusters]
        unique_ids = sorted(set(cluster_ids))
        # 2. 为每个唯一的 group_id 生成一个"确定性"的 RGB
        #    这里用 MD5 把 id hash 成 3 个字节
        color_map = {}
        for gid in unique_ids:
            h = hashlib.md5(str(gid).encode()).digest()
            color_map[gid] = np.frombuffer(h[:3], dtype=np.uint8)
        # 3. 按顺序把每个子聚类的点和对应颜色堆起来
        ply_points = []
        for cluster, gid in zip(self.clusters, cluster_ids):
            pts = cluster.positions.cpu().numpy()                            # (N,3)
            cols = np.tile(color_map[gid], (cluster.count, 1))               # (N,3)
            ply_points.append(np.hstack([pts, cols]))                       # (N,6)
        all_points = np.concatenate(ply_points, axis=0)
        # 4. 写 PLY
        props = [("x","f4"),("y","f4"),("z","f4"),
                 ("red","u1"),("green","u1"),("blue","u1")]
        UTILITY.SavePlyUsingPlyfilePackage(
            path=path,
            points=all_points,
            properties=props,
        )
        ExLog(f"按 group_id 着色并保存 {self.count} 个子聚类到 {path}.")


    def updatePositionsOfAllClusters(self) -> None:
        # 241105 change property of positions to instance variable
        positions_of_all_clusters = torch.zeros((self.count, 3), dtype=torch.float32)
        for i_cluster in range(self.count):
            positions_of_all_clusters[i_cluster] = self.clusters[i_cluster].getCenter()
        self.positions = positions_of_all_clusters

    def append(self, cluster: Cluster) -> None:
        self.clusters.append(cluster)
        self.count = len(self.clusters)

    def extend(self, clusters: list[Cluster]) -> None:
        self.clusters.extend(clusters)
        self.count = len(self.clusters)

    def consolidateIntoASingleCluster(self) -> Cluster:
        count_simplified = sum([c.count for c in self.clusters])
        positions_simplified = torch.cat([c.positions for c in self.clusters], dim=0)
        scales_simplified = torch.cat([c.scales for c in self.clusters], dim=0)
        quaternions_simplified = torch.cat(
            [c.quaternions for c in self.clusters], dim=0
        )
        opacities_simplified = torch.cat([c.opacities for c in self.clusters], dim=0)
        rgbs_simplified = torch.cat([c.rgbs for c in self.clusters], dim=0)
        group_ids = torch.cat([c.group_id for c in self.clusters], dim=0)
        strand_ids = torch.cat([c.strand_id for c in self.clusters], dim=0)
        return Cluster(
            vg_build_config=self.vg_build_config,
            count=count_simplified,
            positions=positions_simplified,
            scales=scales_simplified,
            quaternions=quaternions_simplified,
            opacities=opacities_simplified,
            rgbs=rgbs_simplified,
            group_id=group_ids,
            strand_id=strand_ids,
            lod_level=self.lod_level,
        )

    def splitIntoClusterGroups(self) -> ClustersList:
        # # [clusters -> cluster groups]

        # count_cluster_groups = int(
        #     self.count
        #     / self.vg_build_config.BUILD_APPROPRIATE_COUNT_OF_CLUSTERS_IN_ONE_CLUSTER_GROUP
        # )

        # if count_cluster_groups >= 2:
        #     cluster_centers = np.zeros((self.count, 3))
        #     for i_cluster_group in range(self.count):
        #         cluster_centers[i_cluster_group] = (
        #             self.clusters[i_cluster_group].positions.mean(dim=0).cpu()
        #         )

        #     with ExTimer("kmeans"):
        #         kmeans = sklearn.cluster.MiniBatchKMeans(
        #             n_clusters=count_cluster_groups,
        #             init="k-means++",
        #             n_init="auto",
        #             random_state=0,
        #         ).fit(cluster_centers)

        #     labels = torch.from_numpy(kmeans.labels_)

        #     cluster_groups: ClustersList = ClustersList(
        #         vg_build_config=self.vg_build_config, clusters_list=[]
        #     )
        #     with ExTimer("form CG"):
        #         # find clusters in current cluster group
        #         for i_cluster_group in range(count_cluster_groups):
        #             cluster_group: Clusters = Clusters(
        #                 vg_build_config=self.vg_build_config,
        #                 clusters=[
        #                     self.clusters[c]
        #                     for c in torch.where(labels == i_cluster_group)[0]
        #                 ],
        #                 lod_level=self.lod_level,
        #             )
        #             ExLog(f"{i_cluster_group=} {cluster_group.count=} clusters.counts={[cluster.count for cluster in cluster_group.clusters]}", "DEBUG")
        #             if cluster_group.count >= 2:
        #                 cluster_groups.append(cluster_group)
        # else:
        #     # only one cluster group
        #     cluster_groups: ClustersList = ClustersList(
        #         vg_build_config=self.vg_build_config, clusters_list=[self]
        #     )

        # cluster_groups.savePlyWithDifferentColors(
        #     path=self.vg_build_config.OUTPUT_FOLDER_PATH
        #     / f"plys/lod{self.lod_level}-to-lod{self.lod_level+1}-cluster-groups.ply"
        # )

        # [new version: replace kmeans with median split]

        if (
            self.count
            > self.vg_build_config.BUILD_APPROPRIATE_COUNT_OF_CLUSTERS_IN_ONE_CLUSTER_GROUP
        ):
            all_complete_cluster_groups: list[Clusters] = []
            all_incomplete_cluster_groups: list[Clusters] = [self]

            while len(all_incomplete_cluster_groups) != 0:
                current_incomplete_cluster_groups: list[Clusters] = []
                for incomplete_cluster_group in all_incomplete_cluster_groups:
                    lengths = torch.tensor(
                        [
                            incomplete_cluster_group.positions[:, 0].max().item()
                            - incomplete_cluster_group.positions[:, 0].min().item(),
                            incomplete_cluster_group.positions[:, 1].max().item()
                            - incomplete_cluster_group.positions[:, 1].min().item(),
                            incomplete_cluster_group.positions[:, 2].max().item()
                            - incomplete_cluster_group.positions[:, 2].min().item(),
                        ]
                    )
                    axis_to_split = lengths.argmax().item()
                    axis_median = (
                        incomplete_cluster_group.positions[:, axis_to_split]
                        .median()
                        .item()
                    )

                    cluster_group_left: Clusters = Clusters(
                        clusters=[],
                        vg_build_config=self.vg_build_config,
                        lod_level=self.lod_level,
                    )
                    cluster_group_right: Clusters = Clusters(
                        clusters=[],
                        vg_build_config=self.vg_build_config,
                        lod_level=self.lod_level,
                    )
                    for i_cluster in range(incomplete_cluster_group.count):
                        if (
                            incomplete_cluster_group.positions[i_cluster, axis_to_split]
                            <= axis_median
                        ):
                            cluster_group_left.append(
                                incomplete_cluster_group.clusters[i_cluster]
                            )
                        else:
                            cluster_group_right.append(
                                incomplete_cluster_group.clusters[i_cluster]
                            )

                    # update positions
                    cluster_group_left.positions = incomplete_cluster_group.positions[
                        incomplete_cluster_group.positions[:, axis_to_split]
                        <= axis_median
                    ]
                    cluster_group_right.positions = incomplete_cluster_group.positions[
                        incomplete_cluster_group.positions[:, axis_to_split]
                        > axis_median
                    ]
                    # ExLog(
                    #     f"{cluster_group_left.positions.shape=} {cluster_group_right.positions.shape=}",
                    #     "DEBUG",
                    # )

                    if (
                        cluster_group_left.count
                        <= self.vg_build_config.BUILD_APPROPRIATE_COUNT_OF_CLUSTERS_IN_ONE_CLUSTER_GROUP
                    ):
                        all_complete_cluster_groups.append(cluster_group_left)
                    else:
                        current_incomplete_cluster_groups.append(cluster_group_left)

                    if (
                        cluster_group_right.count
                        <= self.vg_build_config.BUILD_APPROPRIATE_COUNT_OF_CLUSTERS_IN_ONE_CLUSTER_GROUP
                    ):
                        all_complete_cluster_groups.append(cluster_group_right)
                    else:
                        current_incomplete_cluster_groups.append(cluster_group_right)

                    # ExLog(
                    #     f"{len(all_complete_clusters)=} {len(all_incomplete_clusters)=} {len(current_incomplete_clusters)=} {incomplete_cluster.count=} {cluster_left.count=} {cluster_right.count=}"
                    # )

                all_incomplete_cluster_groups = current_incomplete_cluster_groups

            cluster_groups: ClustersList = ClustersList(
                vg_build_config=self.vg_build_config,
                clusters_list=all_complete_cluster_groups,
            )
        else:
            # only one cluster group
            cluster_groups: ClustersList = ClustersList(
                vg_build_config=self.vg_build_config,
                clusters_list=[self],
            )

        cluster_groups.savePlyWithDifferentColors(
            path=self.vg_build_config.OUTPUT_FOLDER_PATH
            / f"plys/lod{self.lod_level}-to-lod{self.lod_level+1}-cluster-groups.ply"
        )

        # ExLog(f"There are {cluster_groups.count} cluster groups.", "DEBUG")
        # for i_cluster_group in range(cluster_groups.count):
        #     current_cluster_group = cluster_groups.clusters_list[i_cluster_group]
        # ExLog(
        #     f"CG_{i_cluster_group}: {current_cluster_group.count} clusters; gaussians in clusters={[cluster.count for cluster in current_cluster_group.clusters]}",
        #     "DEBUG",
        # )

        return cluster_groups

    def setParentCenterAndRadiusValueForFinerLodLayerInClusterGroup(
        self, center: list[float], radius_value: float
    ) -> None:
        for cluster in self.clusters:
            cluster.parent_radius_in_cluster_group = radius_value
            cluster.parent_center_in_cluster_group = center

    def setChildCenterAndRadiusValueForCoarserLodLayerInClusterGroup(
        self, center: list[float], radius_value: float
    ) -> None:
        for cluster in self.clusters:
            cluster.child_radius_in_cluster_group = radius_value
            cluster.child_center_in_cluster_group = center

    def buildCoarserLodLayer(self) -> Clusters:
        """
        首先对所有根进行聚类以减少根的数量，然后对每个聚类中的头发进行减半处理，
        最终构建下一层LOD。
        """
        # 新建一个空的下一层 LOD
        coarser_lod = Clusters(
            vg_build_config=self.vg_build_config,
            clusters=[],
            lod_level=self.lod_level + 1,
        )

        # 检查是否启用根聚类功能
        if not self.vg_build_config.ENABLE_ROOT_CLUSTERING:
            ExLog("根聚类功能已禁用，使用原始减半方法", "DEBUG")
            for cluster in tqdm.tqdm(self.clusters,
                                desc=f"LOD{self.lod_level}→{self.lod_level+1}(原始方法)",
                                unit="cluster"):
                simplified = cluster.downSampleHalf()
                if self.vg_build_config.SIMPLIFICATION_ITERATION != 0:
                    simplified = simplified.optimizeUsingLocalSplatting(cluster)
                coarser_lod.append(simplified)
            coarser_lod.updatePositionsOfAllClusters()
            return coarser_lod

        # Step 1: 合并所有clusters为一个大cluster以便提取发根信息
        consolidated_cluster = self.consolidateIntoASingleCluster()
        
        # Step 2: 提取发根并进行聚类
        roots = extract_strand_roots(consolidated_cluster)
        hair_count = roots.shape[0]
        
        # 检查发根坐标是否有效
        if torch.isnan(roots).any() or torch.isinf(roots).any():
            ExLog(f"发根坐标包含NaN或无穷大值，使用原始减半方法", "WARNING")
            for cluster in tqdm.tqdm(self.clusters,
                                desc=f"LOD{self.lod_level}→{self.lod_level+1}(坐标异常)",
                                unit="cluster"):
                simplified = cluster.downSampleHalf()
                if self.vg_build_config.SIMPLIFICATION_ITERATION != 0:
                    simplified = simplified.optimizeUsingLocalSplatting(cluster)
                coarser_lod.append(simplified)
            coarser_lod.updatePositionsOfAllClusters()
            return coarser_lod
        
        # 如果发根数量很少，直接按原方式处理
        if hair_count <= self.vg_build_config.ROOT_CLUSTERING_MIN_CLUSTERS:
            ExLog(f"发根数量过少({hair_count}≤{self.vg_build_config.ROOT_CLUSTERING_MIN_CLUSTERS})，使用原始减半方法", "DEBUG")
            for cluster in tqdm.tqdm(self.clusters,
                                desc=f"LOD{self.lod_level}→{self.lod_level+1}(发根过少)",
                                unit="cluster"):
                simplified = cluster.downSampleHalf()
                if self.vg_build_config.SIMPLIFICATION_ITERATION != 0:
                    simplified = simplified.optimizeUsingLocalSplatting(cluster)
                coarser_lod.append(simplified)
            coarser_lod.updatePositionsOfAllClusters()
            return coarser_lod

        # Step 3: 🆕 空间预筛选 - 按发根空间位置预分组
        unique_group_ids = torch.unique(consolidated_cluster.group_id)

        # 获取空间阈值参数
        spatial_threshold = getattr(self.vg_build_config, 'SPATIAL_THRESHOLD', 0.05)
        spatial_method = getattr(self.vg_build_config, 'SPATIAL_METHOD', 'grid')

        ExLog(f"开始空间预筛选: spatial_threshold={spatial_threshold}, method={spatial_method}", "DEBUG")

        local_groups = spatial_prefilter_by_roots(
            roots=roots,
            group_ids=unique_group_ids,
            spatial_threshold=spatial_threshold,
            method=spatial_method
        )

        ExLog(f"空间预筛选完成: {hair_count}根头发 → {len(local_groups)}个空间分组", "INFO")

        # Step 4-6: 🆕 对每个空间分组进行特征提取和K-means聚类
        all_labels = {}  # group_id -> cluster_label 的映射

        for group_idx, local_group in enumerate(local_groups):
            ExLog(f"处理空间分组 {group_idx+1}/{len(local_groups)}, 包含{len(local_group)}根头发", "DEBUG")

            # 创建该空间分组的局部cluster
            local_mask = torch.zeros(consolidated_cluster.count, dtype=torch.bool,
                                    device=consolidated_cluster.positions.device)
            for gid in local_group:
                local_mask |= (consolidated_cluster.group_id == gid)

            if local_mask.sum() == 0:
                continue

            # 创建局部cluster用于特征提取
            local_cluster = Cluster(
                vg_build_config=self.vg_build_config,
                count=local_mask.sum().item(),
                positions=consolidated_cluster.positions[local_mask],
                scales=consolidated_cluster.scales[local_mask],
                quaternions=consolidated_cluster.quaternions[local_mask],
                opacities=consolidated_cluster.opacities[local_mask],
                rgbs=consolidated_cluster.rgbs[local_mask],
                group_id=consolidated_cluster.group_id[local_mask],
                strand_id=consolidated_cluster.strand_id[local_mask],
                lod_level=self.lod_level,
            )

            # Step 4: 🆕 提取特征向量（64点采样 + 切向量）
            max_sample_points = getattr(self.vg_build_config, 'FEATURE_SAMPLE_POINTS', 64)
            sampling_strategy = getattr(self.vg_build_config, 'FEATURE_SAMPLING_STRATEGY', 'uniform')

            features = extract_strand_features(
                cluster=local_cluster,
                max_sample_points=max_sample_points,
                sampling_strategy=sampling_strategy
            )

            # Step 5: 确定该空间分组的目标聚类数量
            local_hair_count = len(local_group)

            if self.vg_build_config.ROOT_CLUSTERING_INIT_STRATEGY == "auto":
                local_target_clusters = max(
                    1,  # 至少1个聚类
                    int(local_hair_count * self.vg_build_config.ROOT_CLUSTERING_REDUCTION_FACTOR)
                )
            else:
                try:
                    # 按比例分配目标聚类数
                    global_target = int(self.vg_build_config.ROOT_CLUSTERING_INIT_STRATEGY)
                    local_target_clusters = max(
                        1,
                        int(global_target * local_hair_count / hair_count)
                    )
                except ValueError:
                    ExLog(f"无效的ROOT_CLUSTERING_INIT_STRATEGY，使用auto模式", "WARNING")
                    local_target_clusters = max(1, int(local_hair_count * 0.5))

            ExLog(f"  特征聚类: {local_hair_count}根头发 → 目标{local_target_clusters}类", "DEBUG")

            # Step 6: 🆕 K-means聚类
            try:
                local_labels = feature_based_clustering(
                    features=features,
                    target_cluster_count=local_target_clusters,
                    similarity_metric='euclidean',
                    vg_build_config=self.vg_build_config
                )

                # 将局部标签映射到全局（添加偏移避免冲突）
                base_label = sum(len(set(all_labels.values())) for _ in range(1)) if all_labels else 0
                local_unique_gids = torch.unique(local_cluster.group_id).cpu().numpy()

                for i, gid in enumerate(local_unique_gids):
                    all_labels[int(gid)] = local_labels[i] + group_idx * 1000  # 使用组索引避免标签冲突

            except Exception as e:
                ExLog(f"  特征聚类失败: {e}, 跳过该分组", "ERROR")
                # 失败时每根头发单独成类
                local_unique_gids = torch.unique(local_cluster.group_id).cpu().numpy()
                for i, gid in enumerate(local_unique_gids):
                    all_labels[int(gid)] = group_idx * 1000 + i

        # 将字典转换为列表（保持与原代码一致）
        labels = [all_labels.get(int(gid), 0) for gid in unique_group_ids.cpu().numpy()]
        actual_cluster_count = len(set(labels))

        ExLog(f"聚类完成: {hair_count}根头发 → {actual_cluster_count}个聚类", "INFO")
        
        # Step 7: 根据聚类结果重新组织头发数据
        unique_labels = sorted(set(labels))
        label_to_group_ids = {}
        
        # 构建从聚类标签到group_id的映射
        unique_group_ids = torch.unique(consolidated_cluster.group_id).cpu().numpy()
        for i, gid in enumerate(unique_group_ids):
            cluster_label = labels[i]
            if cluster_label not in label_to_group_ids:
                label_to_group_ids[cluster_label] = []
            label_to_group_ids[cluster_label].append(gid)
        
        # Step 8: 为每个聚类创建新的cluster（单GPU串行处理）
        for cluster_label in tqdm.tqdm(unique_labels,
                                    desc=f"构建根聚类LOD{self.lod_level}→{self.lod_level+1}",
                                    unit="cluster"):
            # 获取当前聚类包含的所有group_id
            group_ids_in_cluster = label_to_group_ids[cluster_label]
            
            # 收集所有属于这些group_id的Gaussians
            mask = torch.zeros(consolidated_cluster.count, dtype=torch.bool)
            for gid in group_ids_in_cluster:
                mask |= (consolidated_cluster.group_id == gid)
            
            if mask.sum() == 0:
                continue
                
            # 创建包含多根头发的cluster，并重新分配group_id
            # 将聚类后的所有头发统一分配一个新的group_id
            new_group_id = torch.full((mask.sum().item(),), cluster_label, 
                                    dtype=consolidated_cluster.group_id.dtype,
                                    device=consolidated_cluster.group_id.device)
            
            merged_cluster = Cluster(
                vg_build_config=self.vg_build_config,
                count=mask.sum().item(),
                positions=consolidated_cluster.positions[mask],
                scales=consolidated_cluster.scales[mask],
                quaternions=consolidated_cluster.quaternions[mask],
                opacities=consolidated_cluster.opacities[mask],
                rgbs=consolidated_cluster.rgbs[mask],
                group_id=new_group_id,  # 使用新的统一group_id
                strand_id=consolidated_cluster.strand_id[mask],  # 保留原始strand_id
                lod_level=self.lod_level,
            )
            
            # Step 9: 对合并后的cluster进行减半处理
            simplified = merged_cluster.downSampleHalf()
            
            # Step 10: 可选的本地优化
            if self.vg_build_config.SIMPLIFICATION_ITERATION != 0:
                simplified = simplified.optimizeUsingLocalSplatting(merged_cluster)
            
            # Step 11: 添加到下一层LOD
            coarser_lod.append(simplified)

        # 更新下一层每个 cluster 的中心位置索引
        coarser_lod.updatePositionsOfAllClusters()
        
        ExLog(f"最终LOD层: {len(unique_labels)} 个聚类，总计 {sum(c.count for c in coarser_lod.clusters)} 个Gaussians")
        
        return coarser_lod




    def saveUsefulImages(self) -> None:
        consolited_cluster_of_current_layer = self.consolidateIntoASingleCluster()
        center = consolited_cluster_of_current_layer.getCenter()
        radius = consolited_cluster_of_current_layer.getFarthestDistanceInClusterGroup()
        lod0_image_full = consolited_cluster_of_current_layer.renderFullImageConsolidatingSixDirections(
            center=center,
            distance=radius * 4,
        )
        UTILITY.SaveImage(
            image=lod0_image_full,
            path=self.vg_build_config.OUTPUT_FOLDER_PATH
            / f"images/6directions-LOD{self.lod_level if self.lod_level != None else 's'}.png",
        )


class Cluster:
    def __init__(
        self,
        # [set when instantialized]
        count: int,
        positions: torch.Tensor,
        scales: torch.Tensor,
        quaternions: torch.Tensor,
        opacities: torch.Tensor,
        rgbs: torch.Tensor,
        group_id: torch.Tensor = None,  # (count,)
        strand_id: torch.Tensor = None,  # (count,)
        # Assign an int if all 3D Gaussians in the cluster are at the same level.
        lod_level: int | None = None,
        # [set after instantialized]
        # for selection
        child_center_in_cluster_group: list[float] = [0.0, 0.0, 0.0],
        parent_center_in_cluster_group: list[float] = [math.inf, math.inf, math.inf],
        child_radius_in_cluster_group: float = -1.0,
        parent_radius_in_cluster_group: float = math.inf,
        # [config]
        vg_build_config: VGBuildConfig = None,
    ) -> None:
        """
        This function is the only entrance for creating Cluster instance.
        All other setup methods should call this function and return a Cluster.
        """

        self.vg_build_config = vg_build_config

        # count of 3D Gaussians
        self.count: int = count

        self.positions: torch.Tensor = positions
        self.scales: torch.Tensor = scales
        self.quaternions: torch.Tensor = quaternions
        self.opacities: torch.Tensor = opacities
        # only rgb (sh0) is considered in this project
        self.rgbs: torch.Tensor = rgbs

        self.lod_level: int | None = lod_level

        if group_id is not None:
            self.group_id = group_id.squeeze()
            # 确保即使只有一个元素也保持1维
            if self.group_id.dim() == 0:
                self.group_id = self.group_id.unsqueeze(0)
        if strand_id is not None:
            self.strand_id = strand_id.squeeze()
            # 确保即使只有一个元素也保持1维
            if self.strand_id.dim() == 0:
                self.strand_id = self.strand_id.unsqueeze(0)

        self.child_center_in_cluster_group: list[float] = child_center_in_cluster_group
        self.parent_center_in_cluster_group: list[float] = (
            parent_center_in_cluster_group
        )
        self.child_radius_in_cluster_group: float = child_radius_in_cluster_group
        self.parent_radius_in_cluster_group: float = parent_radius_in_cluster_group

    def splitByGroupID(self) -> Clusters:
        """
        按照每个 Gaussian 的 group_id 字段，将它们分到不同的 Cluster 里。
        """
        unique_ids = torch.unique(self.group_id)
        clusters = []
        for gid in unique_ids:
            mask = (self.group_id == gid).squeeze()
            clusters.append(
                Cluster(
                    vg_build_config=self.vg_build_config,
                    count=mask.sum().item(),
                    positions=self.positions[mask],
                    scales=self.scales[mask],
                    quaternions=self.quaternions[mask],
                    opacities=self.opacities[mask],
                    rgbs=self.rgbs[mask],
                    group_id=self.group_id[mask],
                    strand_id=self.strand_id[mask],
                    lod_level=self.lod_level,
                )
            )
        return Clusters(vg_build_config=self.vg_build_config, clusters=clusters, lod_level=self.lod_level)

    def splitIntoClusters(self) -> Clusters:
        if (
            self.count
            > self.vg_build_config.BUILD_APPROPRIATE_COUNT_OF_GAUSSIANS_IN_ONE_CLUSTER
        ):
            all_complete_clusters: list[Cluster] = []
            all_incomplete_clusters: list[Cluster] = [self]

            while len(all_incomplete_clusters) != 0:
                current_incomplete_clusters: list[Cluster] = []
                for incomplete_cluster in all_incomplete_clusters:
                    lengths = torch.tensor(
                        [
                            incomplete_cluster.positions[:, 0].max().item()
                            - incomplete_cluster.positions[:, 0].min().item(),
                            incomplete_cluster.positions[:, 1].max().item()
                            - incomplete_cluster.positions[:, 1].min().item(),
                            incomplete_cluster.positions[:, 2].max().item()
                            - incomplete_cluster.positions[:, 2].min().item(),
                        ]
                    )
                    axis_to_split = lengths.argmax().item()
                    axis_median = (
                        incomplete_cluster.positions[:, axis_to_split].median().item()
                    )

                    indices_left = (
                        incomplete_cluster.positions[:, axis_to_split] <= axis_median
                    )
                    indices_right = (
                        incomplete_cluster.positions[:, axis_to_split] > axis_median
                    )

                    cluster_left = Cluster(
                        vg_build_config=self.vg_build_config,
                        count=indices_left.sum().item(),
                        positions=incomplete_cluster.positions[indices_left],
                        scales=incomplete_cluster.scales[indices_left],
                        quaternions=incomplete_cluster.quaternions[indices_left],
                        opacities=incomplete_cluster.opacities[indices_left],
                        rgbs=incomplete_cluster.rgbs[indices_left],

                        group_id=incomplete_cluster.group_id[indices_left],
                        strand_id=incomplete_cluster.strand_id[indices_left],

                        lod_level=incomplete_cluster.lod_level,
                    )
                    cluster_right = Cluster(
                        vg_build_config=self.vg_build_config,
                        count=indices_right.sum().item(),
                        positions=incomplete_cluster.positions[indices_right],
                        scales=incomplete_cluster.scales[indices_right],
                        quaternions=incomplete_cluster.quaternions[indices_right],
                        opacities=incomplete_cluster.opacities[indices_right],
                        rgbs=incomplete_cluster.rgbs[indices_right],

                        group_id=incomplete_cluster.group_id[indices_right],
                        strand_id=incomplete_cluster.strand_id[indices_right],

                        lod_level=incomplete_cluster.lod_level,
                    )
                    if (
                        cluster_left.count
                        <= self.vg_build_config.BUILD_APPROPRIATE_COUNT_OF_GAUSSIANS_IN_ONE_CLUSTER
                    ):
                        all_complete_clusters.append(cluster_left)
                    else:
                        current_incomplete_clusters.append(cluster_left)

                    if (
                        cluster_right.count
                        <= self.vg_build_config.BUILD_APPROPRIATE_COUNT_OF_GAUSSIANS_IN_ONE_CLUSTER
                    ):
                        all_complete_clusters.append(cluster_right)
                    else:
                        current_incomplete_clusters.append(cluster_right)

                    # ExLog(
                    #     f"{len(all_complete_clusters)=} {len(all_incomplete_clusters)=} {len(current_incomplete_clusters)=} {incomplete_cluster.count=} {cluster_left.count=} {cluster_right.count=}"
                    # )

                all_incomplete_clusters = current_incomplete_clusters

            return Clusters(
                vg_build_config=self.vg_build_config,
                clusters=all_complete_clusters,
                lod_level=self.lod_level,
            )
        else:
            clusters: list[Cluster] = [self]
            return Clusters(
                vg_build_config=self.vg_build_config,
                clusters=clusters,
                lod_level=self.lod_level,
            )

    def buildAllLodLayers(self) -> ClustersList:
        """
        input: primitives in gsply of the 3DGS asset
        output: all lod layers for this asset

        This function should be called only once on primitives read from gsply / LOD0.
        """

        ExLog(f"input gaussians -> all lod layers...")

        # [primitives -> LOD0 clusters]

        ExLog(f"input gaussians -> LOD0 clusters...")

        with ExTimer("splitIntoClusters()"):
            # lod0: Clusters = self.splitIntoClusters()
            lod0: Clusters = self.splitByGroupID()
            lod0.updatePositionsOfAllClusters()
        ExLog(
            f"split primitives into LOD0 clusters: {self.count} gaussians -> {lod0.count} clusters.",
        )

        # lod0.savePlyWithDifferentColors(
        #     path=self.vg_build_config.OUTPUT_FOLDER_PATH
        #     / f"plys/lod{lod0.lod_level}-clusters.ply"
        # )

        lod0.savePlyByGroupID(
            path=self.vg_build_config.OUTPUT_FOLDER_PATH
            / f"plys/lod{lod0.lod_level}-clusters.ply"
        )

        lod0.saveUsefulImages()

        # [LODx -> LODx+1]

        lod_layers: ClustersList = ClustersList(
            vg_build_config=self.vg_build_config, clusters_list=[lod0]
        )

        current_lod_layer: Clusters = lod0
        while (
            current_lod_layer.count
            > self.vg_build_config.BUILD_MAX_COUNT_OF_CLUSTERS_IN_COARSEST_LOD_LAYER
        ):
            print()
            ExLog(
                f"LOD{current_lod_layer.lod_level} -> LOD{current_lod_layer.lod_level + 1}...",
            )
            coarser_lod_layer = current_lod_layer.buildCoarserLodLayer()
            coarser_lod_layer.updatePositionsOfAllClusters()
            ExLog(
                f"Simplification: {current_lod_layer.count} clusters -> {coarser_lod_layer.count} clusters.",
            )
            total_before = sum(c.count for c in current_lod_layer.clusters)
            total_after  = sum(c.count for c in coarser_lod_layer.clusters)
            ExLog(
                f"Simplification: {total_before} gaussians in {current_lod_layer.count} clusters"
                f" -> {total_after} gaussians in {coarser_lod_layer.count} clusters."
            )

            # [save useful things]

            # coarser_lod_layer.savePlyWithDifferentColors(
            #     path=self.vg_build_config.OUTPUT_FOLDER_PATH
            #     / f"plys/lod{coarser_lod_layer.lod_level}-clusters.ply"
            # )

            coarser_lod_layer.savePlyByGroupID(
                path=self.vg_build_config.OUTPUT_FOLDER_PATH
                / f"plys/lod{coarser_lod_layer.lod_level}-clusters.ply"
            )
            coarser_lod_layer.saveUsefulImages()

            # [save useful things - ends]

            lod_layers.append(coarser_lod_layer)
            current_lod_layer = coarser_lod_layer

        return lod_layers

    def downSampleHalf(self) -> Cluster:
        if (
            self.vg_build_config.SIMPLIFICATION_INITIALIZATION_DOWNSAMPLE_STRATEGY
            == "random+s213"
        ):
             # step 1: 找到所有 group_id
            uids = torch.unique(self.group_id)
            selected_indices = []
            # 确保所有操作都在同一设备上
            device = self.positions.device
            for gid in uids:
                # 把同一个 gid 的全部索引挑出来
                idx = (self.group_id == gid).nonzero(as_tuple=False).view(-1)
                # 打乱 - 确保randperm在正确的设备上
                perm = idx[torch.randperm(idx.shape[0], device=device)]
                # 每组取一半
                half = perm[: idx.shape[0] // 2]
                selected_indices.append(half)
            # 拼成最终要保留的索引
            important_indices = torch.cat(selected_indices, dim=0)
            return Cluster(
                vg_build_config=self.vg_build_config,
                count=important_indices.numel(),
                positions=self.positions[important_indices],
                scales=(
                    self.scales[important_indices] * (2.0 ** (1 / 3))
                    if self.vg_build_config.SIMPLIFICATION_INITIALIZATION_SCALE_EXPANSION
                    else self.scales[important_indices]
                ),
                quaternions=self.quaternions[important_indices],
                opacities=self.opacities[important_indices],
                rgbs=self.rgbs[important_indices],
                group_id=self.group_id[important_indices],
                strand_id=self.strand_id[important_indices],
                lod_level=self.lod_level + 1,
            )
        elif (
            self.vg_build_config.SIMPLIFICATION_INITIALIZATION_DOWNSAMPLE_STRATEGY
            == "o+s213"
        ):
            # [only keeps 3D Gaussians with large opacities]
            # 确保所有操作都在同一设备上
            device = self.positions.device
            integral_opacities = self.opacities

            descending_indices = integral_opacities.argsort(dim=0, descending=True)[
                :, 0
            ]
            important_indices = descending_indices[: self.count // 2]

            return Cluster(
                vg_build_config=self.vg_build_config,
                count=important_indices.numel(),
                positions=self.positions[important_indices],
                scales=(
                    self.scales[important_indices] * (2.0 ** (1 / 3))
                    if self.vg_build_config.SIMPLIFICATION_INITIALIZATION_SCALE_EXPANSION
                    else self.scales[important_indices]
                ),
                quaternions=self.quaternions[important_indices],
                opacities=self.opacities[important_indices],
                rgbs=self.rgbs[important_indices],
                group_id=self.group_id[important_indices],
                strand_id=self.strand_id[important_indices],
                lod_level=self.lod_level + 1,
            )
        elif (
            self.vg_build_config.SIMPLIFICATION_INITIALIZATION_DOWNSAMPLE_STRATEGY
            == "osss23+s216"
        ):
            # [integral opacity: only keep gaussians with large scales and opacity]
            # 确保所有操作都在同一设备上
            device = self.positions.device
            areas = torch.prod(self.scales, dim=1, keepdim=True) ** (2 / 3)
            integral_opacities = self.opacities * areas

            descending_indices = integral_opacities.argsort(dim=0, descending=True)[
                :, 0
            ]
            important_indices = descending_indices[: self.count // 2]

            return Cluster(
                vg_build_config=self.vg_build_config,
                count=important_indices.numel(),
                positions=self.positions[important_indices],
                scales=(
                    self.scales[important_indices] * (2.0 ** (1 / 6))
                    if self.vg_build_config.SIMPLIFICATION_INITIALIZATION_SCALE_EXPANSION
                    else self.scales[important_indices]
                ),
                quaternions=self.quaternions[important_indices],
                opacities=self.opacities[important_indices],
                rgbs=self.rgbs[important_indices],
                group_id=self.group_id[important_indices],
                strand_id=self.strand_id[important_indices],
                lod_level=self.lod_level + 1,
            )
        elif (
            self.vg_build_config.SIMPLIFICATION_INITIALIZATION_DOWNSAMPLE_STRATEGY
            == "voxels+osss23+s216"
        ):
            # 确保所有操作都在同一设备上
            device = self.positions.device
            median_x = self.positions[:, 0].median()
            median_y = self.positions[:, 1].median()
            median_z = self.positions[:, 2].median()

            xp, xn = self.positions[:, 0] >= median_x, self.positions[:, 0] < median_x
            yp, yn = self.positions[:, 1] >= median_y, self.positions[:, 1] < median_y
            zp, zn = self.positions[:, 2] >= median_z, self.positions[:, 2] < median_z

            eight_voxels_indices = [
                torch.where(xp & yp & zp)[0].to(device),
                torch.where(xp & yp & zn)[0].to(device),
                torch.where(xp & yn & zp)[0].to(device),
                torch.where(xp & yn & zn)[0].to(device),
                torch.where(xn & yp & zp)[0].to(device),
                torch.where(xn & yp & zn)[0].to(device),
                torch.where(xn & yn & zp)[0].to(device),
                torch.where(xn & yn & zn)[0].to(device),
            ]

            eight_important_indices = []

            for i in range(8):
                indices_to_select = eight_voxels_indices[i]

                # [only keeps 3D Gaussians with large opacities]
                areas = torch.prod(
                    self.scales[indices_to_select], dim=1, keepdim=True
                ) ** (2 / 3)
                integral_opacities = self.opacities[indices_to_select] * areas

                descending_indices = integral_opacities.argsort(dim=0, descending=True)[
                    :, 0
                ]
                important_indices = descending_indices[
                    : indices_to_select.numel() // 2 + 1
                ]

                eight_important_indices.append(indices_to_select[important_indices])

            important_indices = torch.cat(eight_important_indices)

            return Cluster(
                vg_build_config=self.vg_build_config,
                count=important_indices.numel(),
                positions=self.positions[important_indices],
                scales=(
                    self.scales[important_indices] * (2.0 ** (1 / 6))
                    if self.vg_build_config.SIMPLIFICATION_INITIALIZATION_SCALE_EXPANSION
                    else self.scales[important_indices]
                ),
                quaternions=self.quaternions[important_indices],
                opacities=self.opacities[important_indices],
                rgbs=self.rgbs[important_indices],
                group_id=self.group_id[important_indices],
                strand_id=self.strand_id[important_indices],
                lod_level=self.lod_level + 1,
            )
        else:
            ExLog("no strategy for downsampling", "ERROR")
            exit(-1)

    def optimizeUsingLocalSplatting(self, cluster_original: Cluster) -> Cluster:
        """
        For an original cluster group, with `self.count` 3D Gaussians, first down sample half and increase their scales to derive the initial simplified cluster group. Then use `LearnableGaussians` to optimize properties of newly created 3D Gaussians to distill the appearance of the original cluster group.
        """
        cluster_original_center = cluster_original.getCenter()
        cluster_original_cluster_radius = (
            cluster_original.getFarthestDistanceInClusterGroup()
        )
        cluster_original_cluster_mean_scale = (
            cluster_original.getMeanScaleInClusterGroup()
        )
        # ExLog(
        #     "DEBUG",
        #     f"{cluster_original_center=} {cluster_original_cluster_radius=:.4f} {cluster_original_cluster_mean_scale=:.4f}",
        # )

        # [cluster_simplified_init]

        cluster_simplified_init = self

        # [optimize to simplify cluster_original and derive cluster_simplified]

        learnable_cluster = LearnableGaussians(
            vg_build_config=self.vg_build_config,
            original_gaussians=cluster_original,
            gaussians=cluster_simplified_init,
            center=cluster_original_center,
            distance=cluster_original_cluster_radius * 4.0,
        )
        learnable_cluster.train()
        cluster_simplified = learnable_cluster.toGaussians(lod_level=self.lod_level)

        return cluster_simplified

    def render(self, camera: Camera) -> torch.Tensor:
        # (b, h, w, 3), (b, h, w, 1)
        image_rgb_premultiplied, gsplat_image_a, statistics = gsplat.rasterization(
            means=self.positions,
            quats=self.quaternions,
            scales=self.scales,
            opacities=self.opacities[:, 0],
            sh_degree=None,
            colors=self.rgbs,
            viewmats=camera.view_matrix.T[
                None, :, :
            ],  # Camera class uses row vectors, while gsplat uses colume vectors.
            Ks=torch.tensor(
                [
                    [camera.focal_x, 0.0, camera.image_width / 2],
                    [0.0, camera.focal_y, camera.image_height / 2],
                    [0.0, 0.0, 1.0],
                ]
            )[None, :, :],
            width=camera.image_width,
            height=camera.image_height,
            eps2d=0.0,
        )

        # only render one image
        # (h, w, 3), (h, w, 1)
        image_rgb_premultiplied = image_rgb_premultiplied[0, ...]
        gsplat_image_a = gsplat_image_a[0, ...]

        # premultiplies rgb -> not premultiplied rgb
        mask_alpha_none_zero = (gsplat_image_a != 0.0)[..., 0]
        image_rgb_premultiplied[mask_alpha_none_zero] = image_rgb_premultiplied[
            mask_alpha_none_zero
        ] / gsplat_image_a[mask_alpha_none_zero].clip(max=1.0)

        # move color channel to the first position
        # (3, h, w), (1, h, w,)
        image_rgb_premultiplied = einops.rearrange(
            image_rgb_premultiplied, "h w c -> c h w"
        )
        gsplat_image_a = einops.rearrange(gsplat_image_a, "h w c -> c h w")

        # concatenate together
        # (4, h, w)
        image_rgba = torch.concat([image_rgb_premultiplied, gsplat_image_a], dim=0)

        return image_rgba

    def renderReturnCountAndDuration(
        self,
        camera: Camera,
        gsplat_radius_clip: float = 0.0,
    ) -> tuple[torch.Tensor, int, float]:
        """
        return:
            torch.Tensor: image (4 w h)
            int: number of gaussians in the frustum
            float: duration to rasterize
        """

        # NOTICE only use for vg-select fps metrics, should remove when do vg-build
        for i in range(5):
            if i == 4:
                time_start_gsplat: float = time.perf_counter()
            # (b, h, w, 3), (b, h, w, 1)
            image_rba_premultiplied, gsplat_image_a, statistics = gsplat.rasterization(
                means=self.positions,
                quats=self.quaternions,
                scales=self.scales,
                opacities=self.opacities[:, 0],
                sh_degree=None,
                colors=self.rgbs,
                viewmats=camera.view_matrix.T[
                    None, :, :
                ],  # Camera class uses row vectors, while gsplat uses colume vectors.
                Ks=torch.tensor(
                    [
                        [camera.focal_x, 0.0, camera.image_width / 2],
                        [0.0, camera.focal_y, camera.image_height / 2],
                        [0.0, 0.0, 1.0],
                    ]
                )[None, :, :],
                width=camera.image_width,
                height=camera.image_height,
                eps2d=0.0,
                radius_clip=gsplat_radius_clip,
            )
            if i == 4:
                time_end_gsplat: float = time.perf_counter()
                time_duration_gsplat: float = time_end_gsplat - time_start_gsplat

        number_of_gaussians_in_frustum: int = statistics["gaussian_ids"].shape[0]
        number_of_duplicated_gaussians: int = statistics["flatten_ids"].shape[0]

        # only render one image
        # (h, w, 3), (h, w, 1)
        image_rba_premultiplied = image_rba_premultiplied[0, ...]
        gsplat_image_a = gsplat_image_a[0, ...]

        # premultiplies rgb -> not premultiplied rgb
        mask_alpha_none_zero = (gsplat_image_a != 0.0)[..., 0]
        image_rba_premultiplied[mask_alpha_none_zero] = image_rba_premultiplied[
            mask_alpha_none_zero
        ] / gsplat_image_a[mask_alpha_none_zero].clip(max=1.0)

        # move color channel to the first position
        # (3, h, w), (1, h, w,)
        image_rba_premultiplied = einops.rearrange(
            image_rba_premultiplied, "h w c -> c h w"
        )
        gsplat_image_a = einops.rearrange(gsplat_image_a, "h w c -> c h w")

        # concatenate together
        # (4, h, w)
        image_rgba = torch.concat([image_rba_premultiplied, gsplat_image_a], dim=0)

        return (
            image_rgba,
            number_of_gaussians_in_frustum,
            number_of_duplicated_gaussians,
            time_duration_gsplat,
        )

    def renderFullImageConsolidatingSixDirections(
        self, center: torch.Tensor, distance: float
    ) -> torch.Tensor:
        cameras: list[Camera] = Camera.DeriveSixDirections(
            center=center, distance=distance
        )

        image_rgb_full = torch.zeros(
            (4, cameras[0].image_height * 3, cameras[0].image_width * 4),
            dtype=torch.float,
        )
        for i, camera in enumerate(cameras):
            image_rgb = self.render(camera=camera)
            if i == 0:
                image_rgb_full[
                    :,
                    1 * camera.image_height : 2 * camera.image_height,
                    1 * camera.image_width : 2 * camera.image_width,
                ] = image_rgb
            elif i == 1:
                image_rgb_full[
                    :,
                    1 * camera.image_height : 2 * camera.image_height,
                    0 * camera.image_width : 1 * camera.image_width,
                ] = image_rgb
            elif i == 2:
                image_rgb_full[
                    :,
                    1 * camera.image_height : 2 * camera.image_height,
                    2 * camera.image_width : 3 * camera.image_width,
                ] = image_rgb
            elif i == 3:
                image_rgb_full[
                    :,
                    1 * camera.image_height : 2 * camera.image_height,
                    3 * camera.image_width : 4 * camera.image_width,
                ] = image_rgb
            elif i == 4:
                image_rgb_full[
                    :,
                    0 * camera.image_height : 1 * camera.image_height,
                    1 * camera.image_width : 2 * camera.image_width,
                ] = image_rgb
            elif i == 5:
                image_rgb_full[
                    :,
                    2 * camera.image_height : 3 * camera.image_height,
                    1 * camera.image_width : 2 * camera.image_width,
                ] = image_rgb
        return image_rgb_full

    def getCenter(self) -> torch.Tensor:
        """
        return (1,3)
        """
        return self.positions.mean(dim=0, keepdim=True)

    def getFarthestDistanceInClusterGroup(self) -> float:
        distance: float = (
            torch.sqrt(
                torch.pow(
                    (self.positions - self.getCenter()),
                    2,
                ).sum(dim=1)
            )
            .max()
            .item()
        )
        return distance

    def getMeanScaleInClusterGroup(self) -> float:
        return self.scales.mean(dim=0).mean().item()

    def getRadiusValueInClusterGroupForSelection(self) -> float:
        """
        Use a bounding sphere to put all gaussians in self(cluster) inside. Return the value of the radius of the bounding sphere.

        Should use after simplification in a cluster group.
        """
        distance: float = self.getFarthestDistanceInClusterGroup()
        # TODO not aligns with paper
        mean_scale: float = self.getMeanScaleInClusterGroup()
        # 240806-1337 remove scale
        radius = distance  # + 3.0 * mean_scale  # 99%
        return radius

    def saveActivatedPly(self, path: pathlib.Path) -> None:
        """
        Different from vanilla 3DGS, we activate all the learnable parameters and save the physical properties of 3D Gaussians in ply files.
        """

        # TODO add lod level, frequency
        ply_points = np.concatenate(
            [
                self.positions.cpu().numpy(),
                self.scales.cpu().numpy(),
                self.quaternions.cpu().numpy(),
                self.opacities.cpu().numpy(),
                # Save using the original precision. To visualize the color, convert this value to uint8/u1.
                self.rgbs.cpu().numpy(),
            ],
            axis=1,
        )
        ply_properties = (
            [
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
            ]
            + [
                ("scales_0", "f4"),
                ("scales_1", "f4"),
                ("scales_2", "f4"),
            ]
            + [
                ("quaternions_0", "f4"),
                ("quaternions_1", "f4"),
                ("quaternions_2", "f4"),
                ("quaternions_3", "f4"),
            ]
            + [("opacities", "f4")]
            + [
                ("rgbs_0", "f4"),
                ("rgbs_1", "f4"),
                ("rgbs_2", "f4"),
            ]
        )

        UTILITY.SavePlyUsingPlyfilePackage(
            path=path,
            points=ply_points,
            properties=ply_properties,
        )
        ExLog(f"Saved {self.count} points to {path}.")

    def saveOriginalPly(self, path: pathlib.Path) -> None:
        ply_points = np.concatenate(
            [
                self.positions.cpu().numpy(),
                LearnableGaussians.InverseActivationScales(self.scales).cpu().numpy(),
                self.quaternions.cpu().numpy(),
                LearnableGaussians.InverseActivationOpacities(self.opacities)
                .cpu()
                .numpy(),
                # Save using the original precision. To visualize the color, convert this value to uint8/u1.
                LearnableGaussians.InverseActivationSh0(self.rgbs).cpu().numpy(),
            ],
            axis=1,
        )
        ply_properties = (
            [
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
            ]
            + [
                ("scale_0", "f4"),
                ("scale_1", "f4"),
                ("scale_2", "f4"),
            ]
            + [
                ("rot_0", "f4"),
                ("rot_1", "f4"),
                ("rot_2", "f4"),
                ("rot_3", "f4"),
            ]
            + [("opacity", "f4")]
            + [
                ("f_dc_0", "f4"),
                ("f_dc_1", "f4"),
                ("f_dc_2", "f4"),
            ]
        )

        UTILITY.SavePlyUsingPlyfilePackage(
            path=path,
            points=ply_points,
            properties=ply_properties,
        )
        ExLog(f"Saved {self.count} points to {path}.")

    def savePlyWithClusterColors(
        self,
        path: pathlib.Path,
        k: int,
        labels: np.ndarray,
    ) -> None:
        color_choices = np.random.randint(low=0, high=255, size=(k, 3), dtype=np.uint8)
        colors = np.zeros((self.count, 3), dtype=np.int8)
        colors = color_choices[labels]

        ply_points = np.concatenate(
            [
                self.positions.cpu().numpy(),
                colors,
            ],
            axis=1,
        )
        ply_properties = [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
        ] + [
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]

        UTILITY.SavePlyUsingPlyfilePackage(
            path=path,
            points=ply_points,
            properties=ply_properties,
        )

    # [select]

    def transformed(
        self,
        scale_factor: float,
        R: torch.Tensor,
        t: torch.Tensor,
    ) -> Cluster:
        Rt_matrices_original = torch.zeros((self.count, 4, 4), dtype=torch.float32)
        Rt_matrices_original[:, 3, 3] = 1.0
        Rt_matrices_original[:, :3, 3] = self.positions
        # NOTICE: r xyz -> xyz w
        Rt_matrices_original[:, :3, :3] = torch.from_numpy(
            scipy.spatial.transform.Rotation.from_quat(
                self.quaternions.cpu().numpy(),
                scalar_first=True,
            ).as_matrix()
        )

        # random scale (notice that we only use uniform scaling on xyz the same time)
        S1 = torch.eye(4, dtype=torch.float32)
        S1[:3] *= scale_factor
        new_scales = self.scales * scale_factor

        # random rotation
        R1 = torch.eye(4, dtype=torch.float32)
        R1[:3, :3] = R

        # layout tables
        T1 = torch.eye(4, dtype=torch.float32)
        T1[:3, 3] = t

        # column vector, from right to left
        # the order of R1 and S1 can be changed, but T1 must be after R1
        Rt_full = S1 @ R1

        Rt_matrices_new = T1 @ Rt_full @ Rt_matrices_original
        new_positions = Rt_matrices_new[:, :3, 3]
        # NOTICE: xyz w -> r xyz
        new_quaternions = torch.from_numpy(
            scipy.spatial.transform.Rotation.from_matrix(
                (R1 @ Rt_matrices_original)[:, :3, :3].cpu().numpy()
            ).as_quat(scalar_first=True),
        ).to(device="cuda", dtype=torch.float32)

        return Cluster(
            vg_build_config=self.vg_build_config,
            count=self.count,
            positions=new_positions,
            scales=new_scales,
            quaternions=new_quaternions,
            opacities=self.opacities.clone(),
            rgbs=self.rgbs.clone(),
            group_id=self.group_id.clone(),
            strand_id=self.strand_id.clone(),
        )

    def selectedInFrustum(self, camera: Camera) -> Cluster:
        # (Gaussians_Count, 4)
        Gaussians_Positions_Homogeneous = torch.zeros(
            (self.count, 4),
            dtype=torch.float32,
        )
        Gaussians_Positions_Homogeneous[:, :3] = self.positions
        Gaussians_Positions_Homogeneous[:, 3] = 1.0

        # (Gaussians_Count, 4)
        Gaussians_Positions_Viewed = (
            Gaussians_Positions_Homogeneous @ camera.view_matrix
        )

        # (Gaussians_Count, 4)
        Gaussians_Positions_Projected = (
            Gaussians_Positions_Viewed @ camera.projection_matrix
        )
        Gaussians_Positions_Projected = Gaussians_Positions_Projected * (
            1.0 / (Gaussians_Positions_Projected[:, -1:] + 0.0000001)
        )

        selected_mask = (
            (Gaussians_Positions_Projected[:, 0] > -1.1)
            & (Gaussians_Positions_Projected[:, 0] < 1.1)
            & (Gaussians_Positions_Projected[:, 1] > -1.1)
            & (Gaussians_Positions_Projected[:, 1] < 1.1)
            & (Gaussians_Positions_Viewed[:, 2] > 0.0)
        )
        ExLog(
            f"{Gaussians_Positions_Projected.shape=} {selected_mask.shape=} {selected_mask.sum().item()=}"
        )

        return Cluster(
            count=selected_mask.sum().item(),
            positions=self.positions[selected_mask, :],
            scales=self.scales[selected_mask, :],
            quaternions=self.quaternions[selected_mask, :],
            opacities=self.opacities[selected_mask, :],
            rgbs=self.rgbs[selected_mask, :],
            group_id=self.group_id[selected_mask],
            strand_id=self.strand_id[selected_mask],
        )


class GsPly:
    def Normalize(x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).sum(dim=1).sqrt()[:, None]
        return x / norm

    def Sigmoid(x: torch.Tensor) -> torch.Tensor:
        return 1.0 / (1.0 + torch.exp(-x))

    def Sh0ToRgb(sh0: torch.Tensor) -> torch.Tensor:
        SH_C0 = 0.28209479177387814
        return torch.clip(sh0 * SH_C0 + 0.5, min=0.0, max=None)

    def __init__(
        self,
        vg_build_config: VGBuildConfig,
    ) -> None:
        self.vg_build_config = vg_build_config

    def read(self) -> Cluster:
        """
        Read activated values / physical properties from gsply.
        """
        points: plyfile.PlyElement = plyfile.PlyData.read(
            self.vg_build_config.ASSET_GSPLY_PATH
        )["vertex"]
        ExLog(
            f"Read {points.count} points from {self.vg_build_config.ASSET_GSPLY_PATH}."
        )

        gsply_positions: np.ndarray = np.column_stack(
            (
                points["x"],
                points["y"],
                points["z"],
            )
        )
        gsply_scales: np.ndarray = np.column_stack(
            (
                points["scale_0"],
                points["scale_1"],
                points["scale_2"],
            )
        )
        gsply_quaternions: np.ndarray = np.column_stack(
            (
                points["rot_0"],
                points["rot_1"],
                points["rot_2"],
                points["rot_3"],
            )
        )
        gsply_opacities: np.ndarray = np.column_stack((points["opacity"],)).astype(
            np.float32
        )
        gsply_sh0s: np.ndarray = np.column_stack(
            (points["f_dc_0"], points["f_dc_1"], points["f_dc_2"])
        )

        gsply_group_id = np.column_stack((points["group_id"]))  # 读取group_id
        gsply_strand_id = np.column_stack((points["strand_id"]))  # 读取strand_id

        return Cluster(
            vg_build_config=self.vg_build_config,
            count=points.count,
            positions=torch.tensor(gsply_positions, dtype=torch.float32),
            scales=torch.exp(torch.tensor(gsply_scales, dtype=torch.float32)),
            quaternions=GsPly.Normalize(
                torch.tensor(gsply_quaternions, dtype=torch.float32)
            ),
            opacities=GsPly.Sigmoid(torch.tensor(gsply_opacities, dtype=torch.float32)),
            rgbs=GsPly.Sh0ToRgb(torch.tensor(gsply_sh0s, dtype=torch.float32)),
            group_id=torch.tensor(gsply_group_id, dtype=torch.int32),  # 添加
            strand_id=torch.tensor(gsply_strand_id, dtype=torch.int32),  # 添加
            lod_level=0,
        )
