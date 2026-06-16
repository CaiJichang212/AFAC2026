"""Insurance domain profile data.

Populated from the 16 PDFs in ``data/public_dataset_upload/raw/insurance/``
and cross-referenced with the A-split questions file.

Product list (doc_id -> canonical name, verified via PDF first-page titles):

===== ============================================================
Doc   Canonical product name (from PDF title page)
===== ============================================================
1     平安智盈金生专属商业养老保险
2     国寿增益宝终身寿险（万能型）（2025版）
3     众安个人急性白血病复发医疗保险（互联网2026版A款）
4     平安安佑福重大疾病保险
5     平安e生保住院7.0医疗保险A款
6     太保团体百万医疗保险（2022版）
7     平安产险预防接种意外伤害保险（E款）（互联网版）
8     众安营运交通工具团体意外伤害保险（互联网版2025A款）
9     平安特种车商业保险示范条款（2020版）
10    众安特种车商业保险示范条款（2020版）
11    平安产险家庭财产保险（家庭版）（2025版）
12    众安家庭财产综合保险（互联网2023版）
13    众安食品安全责任保险（互联网2026版）
14    平安产险食品安全责任保险（2021版）
15    国寿鑫享添盈养老年金保险（互联网专属）
16    平安富鸿金生（悦享版）养老年金保险（分红型）
===== ============================================================

This module only exports *raw data* (dicts / lists).  The ``DomainProfile``
instance is built in ``agent.domain_profiles.__init__`` to avoid circular
imports.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Product aliases: short name / question-stem name -> canonical full name
# ---------------------------------------------------------------------------
# The canonical names were read from the first pages of each PDF.
# Aliases include the short forms used in the A-split questions so that the
# question parser (Task 7) can normalise product references.

PRODUCT_ALIASES: dict[str, str] = {
    # ---- Doc 1: 平安智盈金生专属商业养老保险 ----
    "平安智盈金生专属商业养老保险": "平安智盈金生专属商业养老保险",
    "平安智盈金生": "平安智盈金生专属商业养老保险",
    "智盈金生": "平安智盈金生专属商业养老保险",
    # ---- Doc 2: 国寿增益宝终身寿险（万能型）（2025版） ----
    "国寿增益宝终身寿险（万能型）（2025版）": "国寿增益宝终身寿险（万能型）（2025版）",
    "国寿增益宝终身寿险": "国寿增益宝终身寿险（万能型）（2025版）",
    "国寿增益宝": "国寿增益宝终身寿险（万能型）（2025版）",
    "增益宝": "国寿增益宝终身寿险（万能型）（2025版）",
    # ---- Doc 3: 众安个人急性白血病复发医疗保险（互联网2026版A款） ----
    "众安个人急性白血病复发医疗保险（互联网2026版A款）": "众安个人急性白血病复发医疗保险（互联网2026版A款）",
    "众安个人急性白血病复发医疗保险": "众安个人急性白血病复发医疗保险（互联网2026版A款）",
    "众安白血病医疗险": "众安个人急性白血病复发医疗保险（互联网2026版A款）",
    "白血病医疗险": "众安个人急性白血病复发医疗保险（互联网2026版A款）",
    # ---- Doc 4: 平安安佑福重大疾病保险 ----
    "平安安佑福重大疾病保险": "平安安佑福重大疾病保险",
    "平安安佑福重疾险": "平安安佑福重大疾病保险",
    "平安安佑福": "平安安佑福重大疾病保险",
    "安佑福": "平安安佑福重大疾病保险",
    "安佑福重疾险": "平安安佑福重大疾病保险",
    # ---- Doc 5: 平安e生保住院7.0医疗保险A款 ----
    "平安e生保住院7.0医疗保险A款": "平安e生保住院7.0医疗保险A款",
    "平安e生保住院医疗保险A款": "平安e生保住院7.0医疗保险A款",
    "平安e生保": "平安e生保住院7.0医疗保险A款",
    "e生保": "平安e生保住院7.0医疗保险A款",
    # ---- Doc 6: 太保团体百万医疗保险（2022版） ----
    "太保团体百万医疗保险（2022版）": "太保团体百万医疗保险（2022版）",
    "太保团体百万医疗": "太保团体百万医疗保险（2022版）",
    "团体百万医疗": "太保团体百万医疗保险（2022版）",
    "太保团体百万医疗保险": "太保团体百万医疗保险（2022版）",
    # ---- Doc 7: 平安产险预防接种意外伤害保险（E款）（互联网版） ----
    "平安产险预防接种意外伤害保险（E款）（互联网版）": "平安产险预防接种意外伤害保险（E款）（互联网版）",
    "平安预防接种意外险": "平安产险预防接种意外伤害保险（E款）（互联网版）",
    "预防接种意外险": "平安产险预防接种意外伤害保险（E款）（互联网版）",
    "平安产险预防接种意外伤害保险": "平安产险预防接种意外伤害保险（E款）（互联网版）",
    # ---- Doc 8: 众安营运交通工具团体意外伤害保险（互联网版2025A款） ----
    "众安营运交通工具团体意外伤害保险（互联网版2025A款）": "众安营运交通工具团体意外伤害保险（互联网版2025A款）",
    "众安营运交通意外险": "众安营运交通工具团体意外伤害保险（互联网版2025A款）",
    "营运交通意外险": "众安营运交通工具团体意外伤害保险（互联网版2025A款）",
    "众安营运交通工具团体意外伤害保险": "众安营运交通工具团体意外伤害保险（互联网版2025A款）",
    # ---- Doc 9: 平安特种车商业保险示范条款（2020版） ----
    "平安特种车商业保险示范条款（2020版）": "平安特种车商业保险示范条款（2020版）",
    "平安特种车险": "平安特种车商业保险示范条款（2020版）",
    "平安特种车商业保险": "平安特种车商业保险示范条款（2020版）",
    # ---- Doc 10: 众安特种车商业保险示范条款（2020版） ----
    "众安特种车商业保险示范条款（2020版）": "众安特种车商业保险示范条款（2020版）",
    "众安特种车险": "众安特种车商业保险示范条款（2020版）",
    "众安特种车商业保险": "众安特种车商业保险示范条款（2020版）",
    # ---- Doc 11: 平安产险家庭财产保险（家庭版）（2025版） ----
    "平安产险家庭财产保险（家庭版）（2025版）": "平安产险家庭财产保险（家庭版）（2025版）",
    "平安家财险": "平安产险家庭财产保险（家庭版）（2025版）",
    "平安产险家庭财产保险": "平安产险家庭财产保险（家庭版）（2025版）",
    # ---- Doc 12: 众安家庭财产综合保险（互联网2023版） ----
    "众安家庭财产综合保险（互联网2023版）": "众安家庭财产综合保险（互联网2023版）",
    "众安家财险": "众安家庭财产综合保险（互联网2023版）",
    "众安家庭财产综合保险": "众安家庭财产综合保险（互联网2023版）",
    # ---- Doc 13: 众安食品安全责任保险（互联网2026版） ----
    "众安食品安全责任保险（互联网2026版）": "众安食品安全责任保险（互联网2026版）",
    "众安食责险": "众安食品安全责任保险（互联网2026版）",
    "众安食品安全责任险": "众安食品安全责任保险（互联网2026版）",
    "众安食品安全责任保险": "众安食品安全责任保险（互联网2026版）",
    # ---- Doc 14: 平安产险食品安全责任保险（2021版） ----
    "平安产险食品安全责任保险（2021版）": "平安产险食品安全责任保险（2021版）",
    "平安食品安全责任险": "平安产险食品安全责任保险（2021版）",
    "平安食责险": "平安产险食品安全责任保险（2021版）",
    "平安产险食品安全责任保险": "平安产险食品安全责任保险（2021版）",
    # ---- Doc 15: 国寿鑫享添盈养老年金保险（互联网专属） ----
    "国寿鑫享添盈养老年金保险（互联网专属）": "国寿鑫享添盈养老年金保险（互联网专属）",
    "国寿鑫享添盈": "国寿鑫享添盈养老年金保险（互联网专属）",
    "鑫享添盈": "国寿鑫享添盈养老年金保险（互联网专属）",
    "国寿鑫享添盈养老年金保险": "国寿鑫享添盈养老年金保险（互联网专属）",
    # ---- Doc 16: 平安富鸿金生（悦享版）养老年金保险（分红型） ----
    "平安富鸿金生（悦享版）养老年金保险（分红型）": "平安富鸿金生（悦享版）养老年金保险（分红型）",
    "平安富鸿金生": "平安富鸿金生（悦享版）养老年金保险（分红型）",
    "富鸿金生": "平安富鸿金生（悦享版）养老年金保险（分红型）",
    "平安富鸿金生养老年金保险": "平安富鸿金生（悦享版）养老年金保险（分红型）",
}

# ---------------------------------------------------------------------------
# Insurer aliases
# ---------------------------------------------------------------------------

INSURER_ALIASES: dict[str, str] = {
    "国寿": "中国人寿保险股份有限公司",
    "中国人寿": "中国人寿保险股份有限公司",
    "平安": "中国平安保险（集团）股份有限公司",
    "中国平安": "中国平安保险（集团）股份有限公司",
    "平安产险": "中国平安财产保险股份有限公司",
    "平安健康": "平安健康保险股份有限公司",
    "平安养老": "平安养老保险股份有限公司",
    "太保": "中国太平洋保险（集团）股份有限公司",
    "太平洋保险": "中国太平洋保险（集团）股份有限公司",
    "太平洋健康": "太平洋健康保险股份有限公司",
    "众安": "众安在线财产保险股份有限公司",
    "众安在线": "众安在线财产保险股份有限公司",
}

# ---------------------------------------------------------------------------
# Keywords: broad insurance clause vocabulary for title-recovery & node matching
# ---------------------------------------------------------------------------

KEYWORDS: list[str] = [
    # Structural / clause markers
    "第X条",
    "第X章",
    "第X部分",
    "阅读指引",
    "条款目录",
    "总则",
    "保险责任",
    "责任免除",
    "释义",
    "附录",
    "附表",
    "脚注",
    # Core insurance concepts
    "犹豫期",
    "宽限期",
    "等待期",
    "保险期间",
    "保单年度",
    "保险合同",
    "保险金额",
    "基本保额",
    "基本保险金额",
    "保险费",
    "投保人",
    "被保险人",
    "受益人",
    "保险人",
    "保险金",
    "保险单",
    "保险凭证",
    "保险费率",
    "续保",
    "保证续保",
    # Financial / value terms
    "现金价值",
    "保单账户价值",
    "个人账户价值",
    "投资组合账户",
    "退保费用",
    "初始费用",
    "保单贷款",
    "减额交清",
    "保险费自动垫交",
    "部分领取",
    "红利",
    "分红",
    "利息",
    "投资收益",
    # Benefit types
    "身故保险金",
    "养老保险金",
    "养老年金",
    "满期保险金",
    "满期生存保险金",
    "重大疾病保险金",
    "医疗保险金",
    "医疗费用补偿",
    "住院医疗保险金",
    "特定药品费用",
    "意外伤残保险金",
    "意外身故保险金",
    "失能护理",
    "失能失智护理",
    "长寿金",
    "康复津贴",
    # Deductible / reimbursement concepts
    "免赔额",
    "给付比例",
    "赔偿限额",
    "最高限额",
    "补偿原则",
    "年度免赔额",
    "家庭共享免赔额",
    "免赔额抵扣",
    "医保",
    "基本医疗保险",
    "公费医疗",
    "自费",
    # Product types / categories
    "专属商业养老保险",
    "终身寿险",
    "万能型",
    "重大疾病保险",
    "医疗保险",
    "意外伤害保险",
    "家庭财产保险",
    "食品安全责任保险",
    "特种车商业保险",
    "分红型",
    "养老保险",
    "年金保险",
    # Operational / procedural
    "合同成立",
    "合同生效",
    "合同解除",
    "合同终止",
    "合同效力中止",
    "合同效力恢复",
    "复效",
    "如实告知",
    "保险事故通知",
    "保险金申请",
    "保险金给付",
    "索赔",
    "理赔",
    "核保",
    "承保",
    # Special concepts from the 16 docs
    "特定药品处方审核",
    "指定药店",
    "CAR-T细胞免疫疗法",
    "形态学复发",
    "完全缓解",
    "急性白血病",
    "预防接种",
    "异常反应",
    "偶合症",
    "营运交通工具",
    "车上人员",
    "第三者责任",
    "特种车",
    "食品安全事故",
    "食源性疾患",
    "追溯期",
    "施救费用",
    "个人养老金制度",
    "养老年金开始领取日",
    "领取年龄",
    "领取方式",
]

# ---------------------------------------------------------------------------
# Liability terms: standardised categories of insurance coverage
# ---------------------------------------------------------------------------

LIABILITY_TERMS: list[str] = [
    "身故保险金",
    "养老保险金",
    "养老年金",
    "满期生存保险金",
    "满期保险金",
    "重大疾病保险金",
    "医疗费用补偿",
    "住院医疗保险金",
    "特定药品费用医疗保险金",
    "意外伤残保险金",
    "意外身故保险金",
    "意外伤害医疗保险金",
    "失能失智护理保险金",
    "急性白血病复发医疗保险金",
    "CAR-T细胞免疫疗法康复津贴",
    "长寿金",
    "家庭财产损失赔偿",
    "食品安全责任赔偿",
    "特种车损失赔偿",
    "第三者责任赔偿",
    "车上人员责任赔偿",
]

# ---------------------------------------------------------------------------
# Calculation patterns: the types of calculations the engine must handle
# ---------------------------------------------------------------------------

CALCULATION_PATTERNS: list[dict] = [
    {
        "id": "death_benefit_comparison",
        "label": "身故保险金比较",
        "description": "Compare death-benefit amounts across multiple products given parameters like premiums paid, cash value, policy account value, basic sum insured, and annuity already received.",
        "typical_formula": "max(已交保费, 现金价值, 保单账户价值, 基本保额*系数, 已交保费-已领年金, ...)",
    },
    {
        "id": "surrender_value",
        "label": "退保所得 / 现金价值",
        "description": "Calculate the amount the policyholder receives upon surrender, typically = account value - surrender charge, or = cash value per table.",
        "typical_formula": "退保所得 = 个人账户价值*(1-退保费用率)  or  退保所得 = 保单账户价值*比例",
    },
    {
        "id": "medical_fee_deduction",
        "label": "医疗费用扣减医保+免赔额",
        "description": "Medical reimbursement after deducting social-insurance reimbursement and the policy deductible, capped at the policy limit.",
        "typical_formula": "赔付 = min(总费用-医保报销-免赔额, 赔付上限) * 给付比例",
    },
    {
        "id": "payout_ratio_and_cap",
        "label": "给付比例与最高限额",
        "description": "Apply a payout ratio (e.g. 100%, 80%, 60%) to eligible expenses and cap at the per-policy or per-category maximum.",
        "typical_formula": "赔付 = min(可报销费用*给付比例, 最高限额)",
    },
    {
        "id": "ranking_sorting",
        "label": "排序比较",
        "description": "Sort multiple product results by a numeric metric (e.g. benefit amount, surrender value) in ascending or descending order.",
        "typical_formula": "sorted(results, key=lambda x: x.amount, reverse=True)",
    },
    {
        "id": "annuity_offset",
        "label": "已领养老年金扣减",
        "description": "When calculating death benefit for an annuity product, subtract the cumulative annuity already received from premiums paid.",
        "typical_formula": "身故保险金 = max(已交保费-已领养老年金, 现金价值)",
    },
    {
        "id": "family_shared_deductible",
        "label": "家庭共享免赔额",
        "description": "When a family shares a deductible, aggregate all family members' eligible expenses before applying the shared deductible once.",
        "typical_formula": "家庭总赔付 = max(0, sum(各成员可报销费用) - 共享免赔额)",
    },
    {
        "id": "multi_policy_coordination",
        "label": "多保单协调赔付",
        "description": "When multiple policies cover the same loss, determine coordination order (primary/secondary) and avoid double recovery beyond actual loss.",
        "typical_formula": "总赔付 <= 实际损失; 各保单按顺序或比例分摊",
    },
]

# ---------------------------------------------------------------------------
# Quality thresholds
# ---------------------------------------------------------------------------
# Reasonable defaults; these will be refined during Tasks 3 (preprocess) and
# 5 (node spans & index quality).

QUALITY_THRESHOLDS: dict = {
    "min_title_count": 3,
    "max_empty_title_ratio": 0.3,
    "min_page_mapping_coverage": 0.95,
    "min_avg_chars_per_page": 100,
    "max_toc_page_residual": 5,
    "min_liability_term_recall": 0.8,
}
