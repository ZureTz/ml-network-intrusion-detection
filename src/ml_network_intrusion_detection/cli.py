"""命令行入口；解析后才导入数值库以设置线程环境。"""

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    """执行离线命令或常驻模拟服务；用户输入错误返回非零退出码。"""
    parser = argparse.ArgumentParser(description="网络流量入侵检测（UNSW-NB15 二分类）")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="启动持续学习模拟与本地 Web Dashboard")
    serve.add_argument("--config", required=True, type=Path)
    service = sub.add_parser("service", help="持续回放与定时训练；支持 macOS/Linux")
    service.add_argument("--config", required=True, type=Path)
    for name in ("prepare", "train"):
        command = sub.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path, help="新目录，拒绝覆盖")
        if name == "train":
            command.add_argument("--data", required=True, type=Path)
            command.add_argument("--device", help="覆盖配置: auto/cpu/mps/cuda/cuda:N")
    for name in ("evaluate", "predict"):
        command = sub.add_parser(name)
        command.add_argument("--model", required=True, type=Path, help="仅加载可信模型目录")
        command.add_argument("--input", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path)
        command.add_argument("--batch-size", type=int, default=10000)
        command.add_argument("--device", default="auto", help="auto/cpu/mps/cuda/cuda:N")
    args = parser.parse_args()
    try:
        from ml_network_intrusion_detection.common import load_config, read_json

        if args.command == "serve":
            from ml_network_intrusion_detection.simulation.web import serve

            serve(args.config)
            return
        if args.command == "service":
            from ml_network_intrusion_detection.inference.service import (
                load_service_config,
                run_service,
            )

            run_service(load_service_config(args.config))
            return
        if args.command in ("prepare", "train"):
            config = load_config(args.config)
            if args.command == "train" and args.device is not None:
                config["device"] = args.device
            threads = config["threads"]
        else:
            threads = read_json(args.model / "metadata.json")["config"]["threads"]
            if args.batch_size < 1:
                raise ValueError("batch-size 必须为正整数")
        for variable in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
            os.environ[variable] = str(threads)
        if args.command == "prepare":
            from ml_network_intrusion_detection.data.prepare import prepare

            prepare(config, args.output)
        elif args.command == "train":
            from ml_network_intrusion_detection.training.train import train

            train(config, args.data, args.output)
        elif args.command == "evaluate":
            from ml_network_intrusion_detection.training.evaluate import evaluate

            evaluate(args.model, args.input, args.output, args.batch_size, args.device)
        else:
            from ml_network_intrusion_detection.inference.predict import predict

            predict(args.model, args.input, args.output, args.batch_size, args.device)
        print(f"{args.command} 完成: {args.output}")
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        print(f"错误: {error}", file=sys.stderr)
        sys.exit(1)
