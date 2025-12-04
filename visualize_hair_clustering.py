#!/usr/bin/env python3
"""
头发聚类效果可视化工具

功能：
1. 可视化不同LOD层级的聚类结果
2. 对比不同聚类方法的效果
3. 展示聚类统计信息
4. 生成聚类质量分析报告
5. 支持交互式3D查看器（可选）

使用方法：
python visualize_hair_clustering.py --output-folder ./_output/_build/YOUR_BUILD_FOLDER/ --mode all
"""

import argparse
import pathlib
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Tuple, Optional
import json
import warnings
warnings.filterwarnings('ignore')

# 确保可以导入项目的库
import sys
sys.path.append('.')
try:
    from libraries.utilities import UTILITY, ExLog
    from libraries.classes import *
except ImportError as e:
    print(f"警告：无法导入项目库 {e}，将使用简化版本")
    def ExLog(msg, level="INFO"):
        print(f"[{level}] {msg}")

class HairClusteringVisualizer:
    """头发聚类可视化器"""
    
    def __init__(self, output_folder: pathlib.Path):
        self.output_folder = pathlib.Path(output_folder)
        self.visualization_folder = self.output_folder / "visualizations"
        self.visualization_folder.mkdir(exist_ok=True)
        
        # 存储加载的数据
        self.ply_files = {}
        self.clustering_stats = {}
        
        ExLog(f"初始化可视化器，输出文件夹: {self.output_folder}")
        
    def scan_available_files(self):
        """扫描可用的PLY文件"""
        ExLog("扫描可用的可视化文件...")
        
        # 寻找各种PLY文件
        patterns = {
            'roots_debug': 'roots_debug.ply',
            'roots_clustered': 'roots_clustered_debug.ply',
            'lod_clusters': 'plys/lod*-clusters.ply',
            'feature_clustered': 'roots_feature_clustered_lod*.ply'
        }
        
        for pattern_name, pattern in patterns.items():
            if '*' in pattern:
                # 使用glob匹配
                found_files = list(self.output_folder.glob(pattern))
                self.ply_files[pattern_name] = found_files
                ExLog(f"找到 {pattern_name}: {len(found_files)} 个文件")
                for f in found_files:
                    ExLog(f"  - {f.name}")
            else:
                # 直接查找文件
                file_path = self.output_folder / pattern
                if file_path.exists():
                    self.ply_files[pattern_name] = [file_path]
                    ExLog(f"找到 {pattern_name}: {file_path.name}")
                else:
                    self.ply_files[pattern_name] = []
                    ExLog(f"未找到 {pattern_name}")
    
    def load_ply_data(self, ply_path: pathlib.Path) -> Dict:
        """加载PLY文件数据"""
        try:
            from plyfile import PlyData
            ply_data = PlyData.read(str(ply_path))
            vertex_data = ply_data['vertex']
            
            # 提取坐标
            points = np.column_stack([
                vertex_data['x'], 
                vertex_data['y'], 
                vertex_data['z']
            ])
            
            # 尝试提取颜色信息
            colors = None
            if 'red' in vertex_data.dtype.names:
                colors = np.column_stack([
                    vertex_data['red'], 
                    vertex_data['green'], 
                    vertex_data['blue']
                ]) / 255.0
            
            return {
                'points': points,
                'colors': colors,
                'count': len(points)
            }
        except Exception as e:
            ExLog(f"加载PLY文件失败 {ply_path}: {e}", "ERROR")
            return None
    
    def visualize_clustering_comparison(self):
        """可视化聚类对比效果"""
        ExLog("生成聚类对比可视化...")
        
        # 创建大图包含多个子图
        fig = plt.figure(figsize=(20, 12))
        
        # 1. 原始头发根分布
        if self.ply_files['roots_debug']:
            data = self.load_ply_data(self.ply_files['roots_debug'][0])
            if data:
                ax1 = fig.add_subplot(2, 3, 1, projection='3d')
                ax1.scatter(data['points'][:, 0], data['points'][:, 1], data['points'][:, 2], 
                           s=1, alpha=0.6, c='blue')
                ax1.set_title(f'原始头发根分布\n({data["count"]} 根头发)')
                ax1.set_xlabel('X')
                ax1.set_ylabel('Y')
                ax1.set_zlabel('Z')
        
        # 2. 区域生长聚类结果
        if self.ply_files['roots_clustered']:
            data = self.load_ply_data(self.ply_files['roots_clustered'][0])
            if data:
                ax2 = fig.add_subplot(2, 3, 2, projection='3d')
                if data['colors'] is not None:
                    ax2.scatter(data['points'][:, 0], data['points'][:, 1], data['points'][:, 2], 
                               s=2, alpha=0.8, c=data['colors'])
                else:
                    ax2.scatter(data['points'][:, 0], data['points'][:, 1], data['points'][:, 2], 
                               s=2, alpha=0.8)
                ax2.set_title(f'区域生长聚类结果\n({data["count"]} 根头发)')
                ax2.set_xlabel('X')
                ax2.set_ylabel('Y')
                ax2.set_zlabel('Z')
        
        # 3. 特征聚类结果（如果有）
        if self.ply_files['feature_clustered']:
            # 选择最新的特征聚类文件
            latest_feature_file = max(self.ply_files['feature_clustered'], 
                                    key=lambda x: x.stat().st_mtime)
            data = self.load_ply_data(latest_feature_file)
            if data:
                ax3 = fig.add_subplot(2, 3, 3, projection='3d')
                if data['colors'] is not None:
                    ax3.scatter(data['points'][:, 0], data['points'][:, 1], data['points'][:, 2], 
                               s=2, alpha=0.8, c=data['colors'])
                else:
                    ax3.scatter(data['points'][:, 0], data['points'][:, 1], data['points'][:, 2], 
                               s=2, alpha=0.8)
                ax3.set_title(f'特征聚类结果\n({data["count"]} 根头发)')
                ax3.set_xlabel('X')
                ax3.set_ylabel('Y')
                ax3.set_zlabel('Z')
        
        # 4-6. LOD层级可视化
        if self.ply_files['lod_clusters']:
            for i, lod_file in enumerate(sorted(self.ply_files['lod_clusters'])[:3]):
                data = self.load_ply_data(lod_file)
                if data:
                    ax = fig.add_subplot(2, 3, 4+i, projection='3d')
                    if data['colors'] is not None:
                        ax.scatter(data['points'][:, 0], data['points'][:, 1], data['points'][:, 2], 
                                  s=1, alpha=0.7, c=data['colors'])
                    else:
                        ax.scatter(data['points'][:, 0], data['points'][:, 1], data['points'][:, 2], 
                                  s=1, alpha=0.7)
                    
                    # 从文件名提取LOD级别
                    lod_level = self.extract_lod_level(lod_file.name)
                    ax.set_title(f'LOD{lod_level} 聚类\n({data["count"]} 个点)')
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
        
        plt.tight_layout()
        
        # 保存图像
        output_path = self.visualization_folder / "clustering_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        ExLog(f"聚类对比图已保存到: {output_path}")
        plt.close()
    
    def visualize_clustering_statistics(self):
        """可视化聚类统计信息"""
        ExLog("生成聚类统计可视化...")
        
        # 收集统计信息
        stats = {}
        
        # 原始头发数量
        if self.ply_files['roots_debug']:
            data = self.load_ply_data(self.ply_files['roots_debug'][0])
            if data:
                stats['原始头发数'] = data['count']
        
        # LOD层级统计
        lod_stats = {}
        if self.ply_files['lod_clusters']:
            for lod_file in self.ply_files['lod_clusters']:
                data = self.load_ply_data(lod_file)
                if data:
                    lod_level = self.extract_lod_level(lod_file.name)
                    lod_stats[f'LOD{lod_level}'] = data['count']
        
        # 创建统计图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. LOD层级数量变化
        if lod_stats:
            lod_levels = sorted(lod_stats.keys(), key=lambda x: int(x.replace('LOD', '')))
            counts = [lod_stats[level] for level in lod_levels]
            
            ax1.plot(lod_levels, counts, 'bo-', linewidth=2, markersize=8)
            ax1.set_title('不同LOD层级的点数变化', fontsize=14)
            ax1.set_xlabel('LOD层级')
            ax1.set_ylabel('点数')
            ax1.grid(True, alpha=0.3)
            
            # 添加数值标签
            for i, (level, count) in enumerate(zip(lod_levels, counts)):
                ax1.annotate(f'{count}', (i, count), textcoords="offset points", 
                            xytext=(0,10), ha='center')
        
        # 2. 压缩比例
        if lod_stats and len(lod_stats) > 1:
            lod_levels = sorted(lod_stats.keys(), key=lambda x: int(x.replace('LOD', '')))
            counts = [lod_stats[level] for level in lod_levels]
            ratios = []
            for i in range(1, len(counts)):
                ratio = counts[i] / counts[i-1]
                ratios.append(ratio)
            
            transitions = [f"{lod_levels[i]}→{lod_levels[i+1]}" for i in range(len(ratios))]
            
            bars = ax2.bar(transitions, ratios, color='skyblue', alpha=0.7)
            ax2.set_title('LOD层级间压缩比例', fontsize=14)
            ax2.set_ylabel('压缩比例')
            ax2.tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for bar, ratio in zip(bars, ratios):
                height = bar.get_height()
                ax2.annotate(f'{ratio:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        
        # 3. 聚类效果饼图（如果有聚类信息）
        if self.ply_files['roots_clustered']:
            data = self.load_ply_data(self.ply_files['roots_clustered'][0])
            if data and data['colors'] is not None:
                # 分析聚类颜色分布
                unique_colors, counts = self.analyze_cluster_colors(data['colors'])
                
                # 只显示前10个最大的聚类
                top_clusters = sorted(zip(counts, unique_colors), reverse=True)[:10]
                cluster_counts = [count for count, _ in top_clusters]
                cluster_colors = [color for _, color in top_clusters]
                
                ax3.pie(cluster_counts, colors=cluster_colors, autopct='%1.1f%%', startangle=90)
                ax3.set_title(f'前10大聚类分布\n(总共{len(unique_colors)}个聚类)', fontsize=14)
        
        # 4. 数据摘要表格
        ax4.axis('tight')
        ax4.axis('off')
        
        summary_data = []
        if 'original_count' in stats:
            summary_data.append(['原始头发数', f"{stats['original_count']:,}"])
        
        if lod_stats:
            for level in sorted(lod_stats.keys(), key=lambda x: int(x.replace('LOD', ''))):
                summary_data.append([f'{level} 点数', f"{lod_stats[level]:,}"])
                if level != 'LOD0':
                    # 计算相对于原始数量的压缩比
                    if lod_stats.get('LOD0'):
                        compression = lod_stats[level] / lod_stats['LOD0']
                        summary_data.append([f'{level} 压缩比', f"{compression:.3f}"])
        
        if summary_data:
            table = ax4.table(cellText=summary_data, 
                             colLabels=['指标', '数值'],
                             cellLoc='center',
                             loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 1.5)
            ax4.set_title('聚类统计摘要', fontsize=14, pad=20)
        
        plt.tight_layout()
        
        # 保存图像
        output_path = self.visualization_folder / "clustering_statistics.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        ExLog(f"聚类统计图已保存到: {output_path}")
        plt.close()
        
        return stats, lod_stats
    
    def visualize_lod_progression(self):
        """可视化LOD层级演进"""
        ExLog("生成LOD层级演进可视化...")
        
        if not self.ply_files['lod_clusters']:
            ExLog("未找到LOD聚类文件，跳过LOD演进可视化", "WARNING")
            return
        
        # 按LOD级别排序文件
        lod_files = sorted(self.ply_files['lod_clusters'], 
                          key=lambda x: self.extract_lod_level(x.name))
        
        n_files = len(lod_files)
        cols = min(4, n_files)
        rows = (n_files + cols - 1) // cols
        
        fig = plt.figure(figsize=(5*cols, 4*rows))
        
        for i, lod_file in enumerate(lod_files):
            data = self.load_ply_data(lod_file)
            if data:
                ax = fig.add_subplot(rows, cols, i+1, projection='3d')
                
                if data['colors'] is not None:
                    ax.scatter(data['points'][:, 0], data['points'][:, 1], data['points'][:, 2], 
                              s=2, alpha=0.7, c=data['colors'])
                else:
                    ax.scatter(data['points'][:, 0], data['points'][:, 1], data['points'][:, 2], 
                              s=2, alpha=0.7)
                
                lod_level = self.extract_lod_level(lod_file.name)
                ax.set_title(f'LOD{lod_level}\n{data["count"]:,} 个点')
                ax.set_xlabel('X')
                ax.set_ylabel('Y') 
                ax.set_zlabel('Z')
                
                # 设置相同的视角以便对比
                ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        
        # 保存图像
        output_path = self.visualization_folder / "lod_progression.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        ExLog(f"LOD演进图已保存到: {output_path}")
        plt.close()
    
    def generate_clustering_report(self):
        """生成聚类分析报告"""
        ExLog("生成聚类分析报告...")
        
        report = {
            "扫描时间": str(datetime.datetime.now()),
            "输出文件夹": str(self.output_folder),
            "可视化文件": {}
        }
        
        # 收集文件信息
        for file_type, files in self.ply_files.items():
            report["可视化文件"][file_type] = []
            for file_path in files:
                if file_path.exists():
                    data = self.load_ply_data(file_path)
                    file_info = {
                        "文件名": file_path.name,
                        "文件大小": f"{file_path.stat().st_size / (1024*1024):.2f} MB",
                        "点数": data['count'] if data else 0,
                        "有颜色": data['colors'] is not None if data else False
                    }
                    report["可视化文件"][file_type].append(file_info)
        
        # 保存报告
        report_path = self.visualization_folder / "clustering_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        ExLog(f"聚类分析报告已保存到: {report_path}")
        return report
    
    def analyze_cluster_colors(self, colors: np.ndarray) -> Tuple[List, List]:
        """分析聚类颜色分布"""
        # 将RGB颜色转换为唯一标识
        color_ids = []
        for color in colors:
            color_id = tuple(np.round(color, 3))  # 四舍五入避免浮点误差
            color_ids.append(color_id)
        
        # 统计每种颜色的出现次数
        from collections import Counter
        color_counter = Counter(color_ids)
        
        unique_colors = list(color_counter.keys())
        counts = list(color_counter.values())
        
        return unique_colors, counts
    
    def extract_lod_level(self, filename: str) -> int:
        """从文件名提取LOD级别"""
        import re
        match = re.search(r'lod(\d+)', filename.lower())
        return int(match.group(1)) if match else 0
    
    def run_all_visualizations(self):
        """运行所有可视化"""
        ExLog("开始运行所有可视化...")
        
        # 1. 扫描文件
        self.scan_available_files()
        
        # 2. 生成各种可视化
        self.visualize_clustering_comparison()
        self.visualize_clustering_statistics()
        self.visualize_lod_progression()
        
        # 3. 生成报告
        report = self.generate_clustering_report()
        
        ExLog(f"所有可视化已完成！结果保存在: {self.visualization_folder}")
        return report

def main():
    parser = argparse.ArgumentParser(description="头发聚类效果可视化工具")
    parser.add_argument("--output-folder", "-o", required=True, 
                       help="构建输出文件夹路径 (例如: ./_output/_build/YOUR_BUILD_FOLDER/)")
    parser.add_argument("--mode", "-m", choices=['all', 'comparison', 'statistics', 'lod', 'report'], 
                       default='all', help="可视化模式")
    
    args = parser.parse_args()
    
    output_folder = pathlib.Path(args.output_folder)
    if not output_folder.exists():
        print(f"错误：输出文件夹不存在: {output_folder}")
        return
    
    # 创建可视化器
    visualizer = HairClusteringVisualizer(output_folder)
    
    # 扫描文件
    visualizer.scan_available_files()
    
    # 根据模式运行不同的可视化
    if args.mode == 'all':
        visualizer.run_all_visualizations()
    elif args.mode == 'comparison':
        visualizer.visualize_clustering_comparison()
    elif args.mode == 'statistics':
        visualizer.visualize_clustering_statistics()
    elif args.mode == 'lod':
        visualizer.visualize_lod_progression()
    elif args.mode == 'report':
        visualizer.generate_clustering_report()
    
    print(f"\n✅ 可视化完成！结果保存在: {visualizer.visualization_folder}")
    print(f"📊 可以查看以下文件：")
    for viz_file in visualizer.visualization_folder.glob("*.png"):
        print(f"   - {viz_file.name}")
    
    if (visualizer.visualization_folder / "clustering_report.json").exists():
        print(f"   - clustering_report.json (详细报告)")

if __name__ == "__main__":
    # 添加必要的导入
    import datetime
    main() 