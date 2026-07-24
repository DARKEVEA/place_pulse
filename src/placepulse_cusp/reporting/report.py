from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from placepulse_cusp.provenance import metadata, write_json


def write_verdict(
    config: dict[str, Any],
    verdict: str,
    *,
    reasons: list[str] | None = None,
    gates: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> Path:
    target = Path(config["reporting"]["artifacts_dir"]) / "report" / "verdict.json"
    write_json(
        target,
        {
            "verdict": verdict,
            "reasons": reasons or [],
            "gates": gates or {},
            "metrics": metrics or {},
            "provenance": metadata(config),
        },
    )
    return target


def build_report(config: dict[str, Any]) -> Path:
    root = Path(config["reporting"]["artifacts_dir"])
    verdict_path = root / "report" / "verdict.json"
    verdict = json.loads(verdict_path.read_text("utf-8")) if verdict_path.exists() else {
        "verdict": "DATA_INSUFFICIENT",
        "reasons": ["verdict_not_generated"],
        "gates": {},
        "metrics": {},
    }
    validation_path = Path(config["data"]["processed_dir"]) / "data_validation.json"
    validation = (
        json.loads(validation_path.read_text("utf-8"))
        if validation_path.exists()
        else {"status": "not_run"}
    )
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    html_path = report_dir / "experiment_report.html"
    payload = html.escape(json.dumps(verdict, ensure_ascii=False, indent=2))
    validation_payload = html.escape(json.dumps(validation, ensure_ascii=False, indent=2))
    html_path.write_text(
        f"""<!doctype html>
<html lang="zh"><meta charset="utf-8"><title>Place Pulse CUSP experiment</title>
<style>body{{font:16px/1.55 system-ui;max-width:1000px;margin:40px auto;padding:0 20px}}
pre{{background:#f5f5f5;padding:18px;overflow:auto}}.verdict{{font-size:1.5rem}}</style>
<h1>Place Pulse 标量充分性与 CUSP 实验</h1>
<p class="verdict">最终判定：<strong>{html.escape(verdict["verdict"])}</strong></p>
<h2>判定记录</h2><pre>{payload}</pre>
<h2>数据预检</h2><pre>{validation_payload}</pre>
</html>""",
        "utf-8",
    )
    _write_manuscript_sections(report_dir, verdict)
    return html_path


def _write_manuscript_sections(report_dir: Path, verdict: dict[str, Any]) -> None:
    name = verdict["verdict"]
    result_text = {
        "DATA_INSUFFICIENT": "原始逐次投票数据未满足预注册的数据充分性条件，因此未进行确认性模型比较。",
        "SCALAR_NOT_REJECTED": "异质性模型未在留出数据上稳定优于共享标量模型，现有数据不足以否定单一排序的经验充分性。",
        "SCALAR_REJECTED_CONTINUOUS": "连续偏好模型在留出数据上优于共享标量模型，但未恢复稳定的离散知觉类别。",
        "SCALAR_REJECTED_MIXTURE": "潜在类别模型恢复了稳定、非微小且具有排序反转的知觉机制，但条件双峰或 CUSP 门控未通过。",
        "BIMODAL_NON_CUSP": "条件双峰得到支持，但普通混合专家模型足以解释该结构，未获得 CUSP 特异性证据。",
        "CUSP_COMPATIBLE": "随机 CUSP 密度在留出数据上优于预注册替代模型；结果仅支持横截面 CUSP 兼容性。",
    }[name]
    (report_dir / "nature_results.md").write_text(
        "# Results\n\n" + result_text + "\n", "utf-8"
    )
    (report_dir / "methods.md").write_text(
        "# Methods\n\n"
        "原始 left/right/equal 比较使用 Davidson 三分类似然建模。共享标量、连续偏好和"
        "潜在类别模型在固定外层留出上比较；只有异质性与条件双峰门控通过后才比较 stochastic "
        "CUSP、线性高斯、样条高斯和混合专家密度。\n",
        "utf-8",
    )
    (report_dir / "limitations.md").write_text(
        "# Limitations\n\n"
        "Place Pulse 是横截面选择数据。即使 CUSP 模型获胜，也不能据此宣称观察到滞后、"
        "个体状态跳变或真实时间动力学；本实验也不检验完全未见图像的视觉泛化。\n",
        "utf-8",
    )

