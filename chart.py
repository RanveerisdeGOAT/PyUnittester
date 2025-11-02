import matplotlib.pyplot as plt
import numpy as np
from pyunittester import *

def visualize_unit_test_report(report: UnitTestReport, save_image: bool = False) -> None:
    """Generate a matplotlib visual summary of a UnitTestReport with summary text."""
    # --- Basic stats ---
    total_passed = len(report.passed)
    total_failed = len(report.failed)
    total = total_passed + total_failed
    total_duration = sum(result.duration for _, result in report.passed + report.failed)
    avg_duration = total_duration / total if total > 0 else 0.0
    success_rate = (total_passed / total * 100) if total > 0 else 0.0

    # --- Figure setup ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    fig.suptitle("Unit Test Report", fontsize=18, fontweight="bold", y=0.95)

    # --- Pie chart: pass/fail ---
    axes[0].set_title("Test Results Summary")
    axes[0].pie(
        [total_passed, total_failed],
        labels=["Passed", "Failed"],
        autopct="%1.1f%%",
        colors=["#4CAF50", "#F44336"],
        startangle=140,
        textprops={"fontsize": 10},
    )

    # --- Duration chart ---
    all_tests = report.passed + report.failed
    if all_tests:
        labels = [
            f"{tc.func.__name__}({', '.join(map(str, tc.args))})"
            for tc, _ in all_tests
        ]
        durations = [result.duration for _, result in all_tests]
        colors = [
            "#4CAF50" if (tc, result) in report.passed else "#F44336"
            for tc, result in all_tests
        ]

        axes[1].set_title("Test Durations")
        y_pos = np.arange(len(labels))
        axes[1].barh(y_pos, durations, color=colors)
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(labels, fontsize=8)
        axes[1].set_xlabel("Seconds")

    # --- Add summary text box ---
    summary_text = (
        f"Total tests: {total}\n"
        f"Passed: {total_passed}\n"
        f"Failed: {total_failed}\n"
        f"Success rate: {success_rate:.2f}%\n"
        f"Avg test duration: {avg_duration:.4f}s"
    )

    # Add summary text below both plots
    fig.text(
        0.5, -0.05, summary_text,
        ha="center", va="top",
        fontsize=11,
        fontfamily="monospace",
        bbox=dict(facecolor="#f5f5f5", edgecolor="#cccccc", boxstyle="round,pad=0.6")
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.92])

    # --- Save or display ---
    if save_image:
        plt.savefig("unit_test_report.png", dpi=300, bbox_inches="tight")
    plt.show()
