"""
2.	请利用AHP给出自己一个合理的毕业选择建议。

实例（毕业去向选择）
一、毕业去向的方案：国内深造，留学深造，自主创业，直接就业

毕业去向的约束：
二、发展前景，地理位置，工作环境，个人兴趣，家长期望。

三、总目标：合理的毕业选择建议

程序描述：
    1. 构建三层递阶结构：总目标 → 约束层（5 个）→ 方案层（4 个）。
    2. 用户通过字母菜单（A-I，J 为自定义）完成 40 次成对比较，实时提示剩余题数。
    3. 用和积法求各矩阵特征向量与最大特征根，由 CR = CI/RI 做一致性检验。
    4. 层次总排序：方案得分 = Σ(约束权重 × 方案在该约束下的权重)。
    5. 输出报告，自动记录姓名与日期时间，并作为文件名（姓名在日期之前）。
"""
import numpy as np
from datetime import datetime
from itertools import combinations

# ============ 一、评价框架 ============
GOAL = "合理的毕业选择建议"
CRITERIA = ["发展前景", "地理位置", "工作环境", "个人兴趣", "家长期望"]
CHOICES = ["国内深造", "留学深造", "自主创业", "直接就业"]

# 1-9 标度含义
SCALE = {
    1: "同等重要", 2: "介于1和3之间", 3: "稍微重要", 4: "介于3和5之间",
    5: "明显重要", 6: "介于5和7之间", 7: "强烈重要", 8: "介于7和9之间",
    9: "极端重要",
}

# 随机一致性指标 RI
RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12,
      6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}


# ============ 二、和积法计算权重 ============
def ahp_weights(A):
    """和积法：返回 (归一化特征向量W, 最大特征根lambda_max, CR)"""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]

    W = (A / A.sum(axis=0)).sum(axis=1)
    W = W / W.sum()

    lambda_max = float(np.mean((A @ W) / W))
    CI = (lambda_max - n) / (n - 1) if n > 2 else 0.0
    CR = CI / RI[n] if RI.get(n, 0) > 0 else 0.0
    return W, lambda_max, CR


# ============ 三、用户交互比较  ============
# 选项菜单：字母键 + 文字描述 + 方括号内标度值。
# 前半段(A-D)表示前者更重要，中间(E)同等，后半段(F-J)表示后者更重要。
OPTIONS = [
    ("A", "前者 极端 重要", 9, "9"),
    ("B", "前者 强烈 重要", 7, "7"),
    ("C", "前者 明显 重要", 5, "5"),
    ("D", "前者 稍微 重要", 3, "3"),
    ("E", "两者 同等 重要", 1, "1"),
    ("F", "后者 稍微 重要", 1 / 3, "1/3"),
    ("G", "后者 明显 重要", 1 / 5, "1/5"),
    ("H", "后者 强烈 重要", 1 / 7, "1/7"),
    ("I", "后者 极端 重要", 1 / 9, "1/9"),
]
CUSTOM_KEY = "J"
_OPT_BY_KEY = {k: v for k, _, v, _ in OPTIONS}


def choose_option(left, right):
    """展示选项菜单，返回选中的标度值"""
    print(f"      「{left}」 vs 「{right}」：")
    for key, text, _, shown in OPTIONS:
        print(f"        ({key}) {text}  [{shown}]")
    print(f"        ({CUSTOM_KEY}) 自己输入分值")

    while True:
        raw = input("      请选择 → ").strip().upper()

        if raw == CUSTOM_KEY:
            return ask_custom(left, right)
        if raw in _OPT_BY_KEY:
            return _OPT_BY_KEY[raw]
        print(f"      × 请输入选项字母 A-{CUSTOM_KEY}。")


def parse_value(raw):
    """解析 '5'、'1/3'、'0.2' 三种写法，失败返回 None"""
    raw = raw.strip()
    try:
        if "/" in raw:
            a, b = raw.split("/", 1)
            return float(a) / float(b)
        return float(raw)
    except (ValueError, ZeroDivisionError):
        return None


def ask_custom(left, right):
    """自定义分值：可输入 1-9 或 1/3、1/5 等分数"""
    while True:
        raw = input("      请输入重要程度（1-9，或 1/3、1/5 等分数）: ").strip()
        v = parse_value(raw)
        if v is None or not 0 < v <= 9:
            print("      × 请输入 0-9 之间的数值，或 1/3、1/5 这样的分数。")
            continue
        break

    if abs(v - 1) < 1e-9:
        return 1.0
    if v < 1:          # 直接输入了倒数（如 1/3），视为后者更重要
        return v

    d = input(f"      「{left}」更重要输入 1，「{right}」更重要输入 2: ").strip()
    while d not in ("1", "2"):
        d = input("      × 请输入 1 或 2: ").strip()
    return v if d == "1" else 1 / v


def pairwise_matrix(labels, title, counter):
    """对 labels 做两两比较，返回判断矩阵。counter 为 [已完成, 总数]"""
    n = len(labels)
    A = np.ones((n, n))
    print(f"\n{'=' * 56}\n{title}\n{'=' * 56}")

    for i, j in combinations(range(n), 2):
        left, right = labels[i], labels[j]
        print(f"\n  [{counter[0] + 1}/{counter[1]}] 「{left}」相比「{right}」")
        A[i, j] = choose_option(left, right)
        A[j, i] = 1 / A[i, j]
        counter[0] += 1
        print(f"      → 记录 {left}/{right} = {A[i, j]:.4f}  (已完成 {counter[0]}/{counter[1]})")

    return A


def run_all_comparisons():
    """完成所有成对比较，返回 (约束矩阵, {约束: 方案矩阵})"""
    n_total = len(list(combinations(range(len(CRITERIA)), 2))) \
        + len(CRITERIA) * len(list(combinations(range(len(CHOICES)), 2)))
    counter = [0, n_total]
    print(f"\n共需回答 {n_total} 个成对比较问题（约束间 10 个 + 每个约束下方案 30 个）。")

    A_crit = pairwise_matrix(CRITERIA, f"第一层：各约束对总目标「{GOAL}」的重要程度", counter)

    A_alt = {}
    for c in CRITERIA:
        A_alt[c] = pairwise_matrix(CHOICES, f"第二层：在「{c}」方面，各毕业去向的优劣", counter)

    return A_crit, A_alt


# ============ 四、合成与报告 ============
def synthesize(A_crit, A_alt):
    """层次总排序：返回 (综合得分, 约束权重, 明细)"""
    w_crit, lam_crit, cr_crit = ahp_weights(A_crit)
    detail = {}
    for c in CRITERIA:
        w_alt, lam, cr = ahp_weights(A_alt[c])
        detail[c] = (w_alt, lam, cr)
    # 方案综合得分 = 各约束下方案权重 按约束权重加权平均
    score = sum(w_crit[k] * detail[c][0] for k, c in enumerate(CRITERIA))
    return score, w_crit, (lam_crit, cr_crit), detail


def build_report(name, score, w_crit, crit_stat, detail):
    """生成文本报告"""
    lines = []
    lines.append("=" * 60)
    lines.append("毕业去向选择 —— AHP 层次分析报告".center(50))
    lines.append("=" * 60)
    lines.append(f"姓名：{name}")
    lines.append(f"总目标：{GOAL}")
    lines.append(f"约束条件：{', '.join(CRITERIA)}")
    lines.append(f"备选方案：{', '.join(CHOICES)}")

    lines.append("\n" + "-" * 60)
    lines.append("【第一层】约束对总目标的权重")
    lines.append("-" * 60)
    for c, w in sorted(zip(CRITERIA, w_crit), key=lambda x: -x[1]):
        lines.append(f"  {c:<8} {w:.4f}")
    lam, cr = crit_stat
    lines.append(f"  λmax = {lam:.4f}, CR = {cr:.4f} → {'通过' if cr < 0.1 else '未通过'}一致性检验")

    lines.append("\n" + "-" * 60)
    lines.append("【第二层】各约束下方案的权重")
    lines.append("-" * 60)
    for c in CRITERIA:
        w_alt, lam, cr = detail[c]
        lines.append(f"\n  {c}（λmax={lam:.4f}, CR={cr:.4f}, {'通过' if cr < 0.1 else '未通过'}）")
        for ch, w in sorted(zip(CHOICES, w_alt), key=lambda x: -x[1]):
            lines.append(f"      {ch:<8} {w:.4f}")

    lines.append("\n" + "-" * 60)
    lines.append("【层次总排序】各毕业去向综合得分")
    lines.append("-" * 60)
    for ch, s in sorted(zip(CHOICES, score), key=lambda x: -x[1]):
        lines.append(f"  {ch:<8} {s:.4f}  {'█' * int(round(s * 50))}")

    best = CHOICES[int(np.argmax(score))]
    lines.append("\n" + "=" * 60)
    lines.append(f"结论：综合得分最高的是「{best}」，建议优先考虑。")
    lines.append("=" * 60)
    return "\n".join(lines)


def ask_name():
    """读取姓名（用于报告标题与文件名），带简单类型检查"""
    while True:
        raw = input("请输入你的姓名: ").strip()
        if not raw:
            print("  × 姓名不能为空。")
            continue
        return raw


def safe_name(name):
    """过滤文件名中的非法字符"""
    for ch in '\\/:*?"<>|':
        name = name.replace(ch, "_")
    return name.strip() or "匿名"


def main():
    print("=" * 56)
    print("毕业去向选择 —— AHP 层次分析".center(48))
    print("=" * 56)
    name = ask_name()

    A_crit, A_alt = run_all_comparisons()
    score, w_crit, crit_stat, detail = synthesize(A_crit, A_alt)

    report = build_report(name, score, w_crit, crit_stat, detail)
    print("\n\n" + report)

    now = datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S")
    filename = f"AHP毕业去向报告_{safe_name(name)}_{stamp}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"姓名：{name}\n生成时间：{now:%Y-%m-%d %H:%M:%S}\n\n")
        f.write(report)
    print(f"\n报告已保存：{filename}")


if __name__ == "__main__":
    main()