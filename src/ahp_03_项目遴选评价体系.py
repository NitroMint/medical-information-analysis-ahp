"""
层次分析法（递阶层次构建）—— 创新创业项目遴选评价体系

创新项目遴选指标的约束：

(1)创新型：
    1.选题:体现学术/应用价值;国内外情况的把握； 
    2.方法:科学，新颖。

(2)可行性： 
    1.研究基础：学生团队，指导老师；
    2.研究条件：经费预算，投入时间，是否具备所需软硬件环境。


(3)表达性：
    1.内容充实，思路清晰，表达简洁。


程序描述：
    1. 构建三层递阶结构：总目标 → 准则层（创新型/可行性/表达性）→ 指标层（各 2 个指标）。
    2. 评委通过字母菜单（A-I，J 为自定义）完成 6 次成对比较，每题可附一句判断依据。
    3. 用和积法（列归一化→行求和→归一化）求各矩阵的特征向量与最大特征根。
    4. 由 CR = CI/RI 做一致性检验，指标组合权重 = 准则权重 × 准则内权重。
    5. 输出报告，自动记录评委姓名与日期时间，并作为文件名（姓名在日期之前）。
"""
import numpy as np
from datetime import datetime
from itertools import combinations

# ============ 一、评价框架（三层递阶） ============
GOAL = "遴选优秀创新创业项目"

# 准则层
CRITERIA = ["创新型", "可行性", "表达性"]

# 指标层：每个准则下的具体指标
SUB_CRITERIA = {
    "创新型": ["选题价值", "方法新颖"],
    "可行性": ["研究基础", "研究条件"],
    "表达性": ["内容思路", "表达效果"],
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


# ============ 三、面向评委的交互式成对比较 ============
# 选项菜单：字母键 + 文字描述 + 方括号内标度值。
# 前半段(A-D)表示前者更重要，中间(E)同等，后半段(F-I)表示后者更重要。
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


def choose_option(left, right, question):
    """展示题目与选项菜单，返回选中的标度值"""
    print(f"      {question}")
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


def ask_desc(a, b):
    """读取评委对某个比较的文字理由（可留空跳过）"""
    return input(f"      可选：说明「{a}」与「{b}」的判断理由（回车跳过）: ").strip()


def pairwise_matrix(labels, title, counter, collect_reasons=False):
    """对 labels 做两两比较，返回 (判断矩阵, 理由列表)"""
    n = len(labels)
    A = np.ones((n, n))
    reasons = []
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")

    for i, j in combinations(range(n), 2):
        left, right = labels[i], labels[j]
        print(f"\n  [{counter[0] + 1}/{counter[1]}] 「{left}」相比「{right}」哪个更重要、重要多少？")

        A[i, j] = choose_option(left, right, f"「{left}」 vs 「{right}」")
        A[j, i] = 1 / A[i, j]
        counter[0] += 1
        print(f"      → 记录 {left}/{right} = {A[i, j]:.4f}  (已完成 {counter[0]}/{counter[1]})")

        if collect_reasons:
            r = ask_desc(left, right)
            if r:
                reasons.append(f"{left} vs {right}：{r}")

    return A, reasons


def run_all_comparisons(collect_reasons=False):
    """完成所有成对比较，返回 (准则矩阵, {准则: 指标矩阵}, 理由)"""
    n_crit = len(list(combinations(range(len(CRITERIA)), 2)))
    n_sub = sum(len(list(combinations(range(len(v)), 2))) for v in SUB_CRITERIA.values())
    n_total = n_crit + n_sub
    counter = [0, n_total]

    print(f"\n共需回答 {n_total} 个成对比较问题（准则 {n_crit} + 指标 {n_sub}）。")

    reasons = []

    # 第一层：准则对总目标
    A_crit, r = pairwise_matrix(
        CRITERIA, f"第一层：各准则对总目标「{GOAL}」的重要程度",
        counter, collect_reasons=collect_reasons,
    )
    reasons += r

    # 第二层：指标对所属准则
    A_sub = {}
    for c in CRITERIA:
        A_sub[c], r = pairwise_matrix(
            SUB_CRITERIA[c], f"第二层：在「{c}」准则下，各指标的重要程度",
            counter, collect_reasons=collect_reasons,
        )
        reasons += r

    return A_crit, A_sub, reasons


# ============ 四、合成与报告 ============
def synthesize(A_crit, A_sub):
    """
    层次总排序：
        指标组合权重 = 准则权重 × 指标在准则内权重
    返回 (准则权重, 指标组合权重, 一致性明细)
    """
    w_crit, lam_crit, cr_crit = ahp_weights(A_crit)

    w_sub_global = {}  # 指标的组合权重
    consist = {}       # {名称: (n, lam, cr)}

    consist["准则层"] = (len(CRITERIA), lam_crit, cr_crit)

    for k, c in enumerate(CRITERIA):
        w, lam, cr = ahp_weights(A_sub[c])
        consist[f"指标层 · {c}"] = (len(SUB_CRITERIA[c]), lam, cr)
        for si, s in enumerate(SUB_CRITERIA[c]):
            w_sub_global[s] = w_crit[k] * w[si]

    return w_crit, w_sub_global, consist


def build_report(name, w_crit, w_sub_global, consist, reasons):
    """生成文本报告"""
    L = []
    L.append("=" * 64)
    L.append("创新创业项目遴选 —— AHP 评价体系构建报告".center(56))
    L.append("=" * 64)
    L.append(f"评委：{name}")
    L.append(f"总目标：{GOAL}")
    L.append(f"准则层：{', '.join(CRITERIA)}")
    for c in CRITERIA:
        L.append(f"    · {c} → {', '.join(SUB_CRITERIA[c])}")

    L.append("\n" + "-" * 64)
    L.append("【一】准则层权重（对总目标）")
    L.append("-" * 64)
    for c, w in sorted(zip(CRITERIA, w_crit), key=lambda x: -x[1]):
        L.append(f"  {c:<6} {w:.4f}")

    L.append("\n" + "-" * 64)
    L.append("【二】指标层组合权重（准则权重 × 准则内权重，合计为 1）")
    L.append("-" * 64)
    for s, w in sorted(w_sub_global.items(), key=lambda x: -x[1]):
        L.append(f"  {s:<8} {w:.4f}  {'█' * int(round(w * 60))}")

    L.append("\n" + "-" * 64)
    L.append("【三】一致性检验")
    L.append("-" * 64)
    all_ok = True
    for label, (n, lam, cr) in consist.items():
        ok = cr < 0.1
        if n > 2:          # n=2 的矩阵恒为一致，不参与结论判断
            all_ok &= ok
        note = "恒一致(无需检验)" if n <= 2 else ("通过" if ok else "未通过")
        L.append(f"  {label:<16} n={n}  λmax={lam:.4f}  CR={cr:.4f}  {note}")
    L.append(f"  总体结论：{'全部通过一致性检验，权重可用。' if all_ok else '存在未通过的矩阵，建议重新调整判断。'}")

    if reasons:
        L.append("\n" + "-" * 64)
        L.append("【四】评委判断理由记录")
        L.append("-" * 64)
        for r in reasons:
            L.append(f"  · {r}")

    top_c = max(zip(CRITERIA, w_crit), key=lambda x: x[1])[0]
    top_s = max(w_sub_global.items(), key=lambda x: x[1])[0]
    L.append("\n" + "=" * 64)
    L.append(f"结论：准则层中「{top_c}」权重最高，指标层中「{top_s}」组合权重最高，"
             f"建议评审时重点关注。")
    L.append("=" * 64)
    return "\n".join(L)


def ask_name():
    """读取评委姓名（用于报告标题与文件名），带简单类型检查"""
    while True:
        raw = input("请输入评委姓名: ").strip()
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
    print("=" * 60)
    print("创新创业项目遴选 —— 面向评委的 AHP 评价体系".center(52))
    print("=" * 60)
    name = ask_name()

    collect = input("是否记录每个判断的理由？(y/N): ").strip().lower() == "y"

    A_crit, A_sub, reasons = run_all_comparisons(collect_reasons=collect)
    w_crit, w_sub_global, consist = synthesize(A_crit, A_sub)

    report = build_report(name, w_crit, w_sub_global, consist, reasons)
    print("\n\n" + report)

    now = datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S")
    filename = f"AHP项目遴选评价体系_{safe_name(name)}_{stamp}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"评委：{name}\n生成时间：{now:%Y-%m-%d %H:%M:%S}\n\n")
        f.write(report)
    print(f"\n报告已保存：{filename}")


if __name__ == "__main__":
    main()