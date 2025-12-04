import argparse, time, datetime

from libraries.classes import GsPly, Cluster
from libraries.utilities import ExLog, setup_torch_and_random
from libraries.cliconfigs import VGBuildConfig


def VgBuild(vg_build_config: VGBuildConfig):
    # load 3DGS asset
    gsply: GsPly = GsPly(vg_build_config=vg_build_config)
    primitives: Cluster = gsply.read()

    # iteratively build all levels of detail
    ExLog(f"Start building LODs...")
    lods = primitives.buildAllLodLayers()
    ExLog(f"Finish building LODs.")

    # save V3DG bundle
    ExLog(f"Start saving LODs...")
    lods.saveBundle()
    ExLog(f"Finish saving LODs...")


if __name__ == "__main__":
    ExLog("START")
    print()

    setup_torch_and_random()

    parser = argparse.ArgumentParser()
    vg_build_config = VGBuildConfig(parser=parser)

    args = parser.parse_args()
    vg_build_config.extract(args=args)
    vg_build_config.process()
    
    # 设置日志文件保存到输出文件夹中
    log_filename = vg_build_config.OUTPUT_FOLDER_PATH / f"build_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    ExLog.set_log_file(str(log_filename))
    ExLog(f"日志将保存到: {log_filename}", "INFO")
    
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    cluster = GsPly(vg_build_config).read()
    ExLog(f"Sample group_id: {cluster.group_id[:5]}")  # 检查前5个group_id
    ExLog(f"Sample strand_id: {cluster.strand_id[:5]}")  # 检查前5个strand_id

    time_start = time.perf_counter()
    VgBuild(vg_build_config=vg_build_config)
    time_end = time.perf_counter()
    duration = time_end - time_start
    with open(vg_build_config.OUTPUT_FOLDER_PATH / "_records.py", "w") as f:
        f.write(f"{duration=}\n")

    ExLog(f"构建总耗时: {duration:.2f} 秒", "INFO")
    print()
    ExLog("END")
