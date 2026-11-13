"""用 Pandoc 生成独立 LaTeX 源文件，再用 XeLaTeX 编译公开版最终报告。"""

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """依赖外部 Pandoc、XeLaTeX；中间文件仅写入被忽略的 outputs。"""
    for command in ("pandoc", "xelatex"):
        if shutil.which(command) is None:
            raise SystemExit(f"缺少 {command}；请安装 Pandoc 和含 ctex/Fandol 的 TeX Live。")
    report = ROOT / "reports"
    output = ROOT / "outputs/report-build"
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "pandoc",
            "README.md",
            "--from=markdown",
            "--to=latex",
            "--standalone",
            "--top-level-division=section",
            "--number-sections",
            "--toc",
            "--template=final/template.tex",
            "--lua-filter=final/report-filter.lua",
            "--syntax-highlighting=none",
            "-o",
            "final/report.tex",
        ],
        cwd=report,
        check=True,
    )
    # 三遍稳定目录、交叉引用与长表列宽；失败日志保留在 outputs。
    for _ in range(3):
        result = subprocess.run(
            [
                "xelatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-output-directory={output}",
                "final/report.tex",
            ],
            cwd=report,
            capture_output=True,
            text=True,
        )
        (output / "compiler-stdout.txt").write_text(result.stdout)
        if result.returncode:
            raise SystemExit(f"LaTeX 编译失败，请查看 {output / 'compiler-stdout.txt'}")
    shutil.copy2(output / "report.pdf", report / "final/report.pdf")
    print("Generated reports/final/report.tex and reports/final/report.pdf")


if __name__ == "__main__":
    main()
