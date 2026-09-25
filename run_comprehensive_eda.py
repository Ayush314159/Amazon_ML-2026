"""
Amazon ML Challenge - Comprehensive Exploratory Data Analysis (EDA)
Covers full analysis across df (Ground Truth), df1 (Source 1), df2 (Source 2), and df3 (Source 3).
"""

import sys
import io
import re
import unicodedata
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Ensure UTF-8 output encoding for console prints
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ==========================================
# 1. SETUP & VISUALIZATION STYLES
# ==========================================
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
pd.set_option('display.max_columns', 15)
pd.set_option('display.max_colwidth', 100)


# ==========================================
# 2. DATA INGESTION
# ==========================================
def load_data(sample_rows=None):
    """
    Loads df (ground truth), df1 (source1), df2 (source2), df3 (source3).
    Pass sample_rows=None to load the complete ~12.5M rows across all datasets.
    """
    print("[+] Loading datasets...")
    df_gt = pd.read_csv("train_ground_truth.tsv", sep="\t", nrows=sample_rows)
    df1 = pd.read_csv("train_source1.tsv", sep="\t", nrows=sample_rows)
    df2 = pd.read_csv("train_source2.tsv", sep="\t", nrows=sample_rows)
    df3 = pd.read_csv("train_source3.tsv", sep="\t", nrows=sample_rows)

    print(f"[*] df (Ground Truth) Shape : {df_gt.shape}")
    print(f"[*] df1 (Source 1) Shape    : {df1.shape}")
    print(f"[*] df2 (Source 2) Shape    : {df2.shape}")
    print(f"[*] df3 (Source 3) Shape    : {df3.shape}")
    return df_gt, df1, df2, df3


# ==========================================
# 3. DATA HYGIENE & NULL AUDIT
# ==========================================
def audit_data_hygiene(dfs_dict):
    """
    Checks true NaNs, empty strings, whitespace-only, and placeholder strings.
    """
    print("\n" + "=" * 70)
    print("[*] 1. DATA HYGIENE & NULL AUDIT")
    print("=" * 70)

    placeholders = {'nan', 'none', 'null', 'n/a', 'na', 'unknown', 'missing', ''}
    records = []

    for name, df in dfs_dict.items():
        total = len(df)
        for col in df.columns:
            null_count = df[col].isnull().sum()
            str_col = df[col].astype(str).str.strip()
            empty_count = (str_col == '').sum()
            placeholder_count = str_col.str.lower().isin(placeholders).sum()

            records.append({
                'Dataset': name,
                'Column': col,
                'Total Rows': total,
                'True NaN Count': null_count,
                'True NaN %': round((null_count / total) * 100, 4),
                'Empty/Whitespace Count': empty_count,
                'Placeholder/Invalid Count': placeholder_count,
                'Unique Values': df[col].nunique()
            })

    audit_df = pd.DataFrame(records)
    print(audit_df.to_string(index=False))
    return audit_df


# ==========================================
# 4. PRIMARY KEY & ENTITY ID ANALYSIS
# ==========================================
def analyze_entity_ids(df_gt, df1, df2, df3):
    """
    Validates regex format, uniqueness, and cross-source collisions.
    """
    print("\n" + "=" * 70)
    print("[*] 2. PRIMARY KEY & ENTITY ID VALIDATION")
    print("=" * 70)

    id_sources = {
        'df1': (df1['entity_id'], r'^S1-\d+$'),
        'df2': (df2['entity_id'], r'^S2-\d+$'),
        'df3': (df3['entity_id'], r'^S3-\d+$'),
        'df_gt (source1)': (df_gt['source1_entity_id'], r'^S1-\d+$')
    }

    for name, (id_series, pattern) in id_sources.items():
        total = len(id_series)
        unique = id_series.nunique()
        matches_pattern = id_series.astype(str).str.match(pattern).sum()

        print(f"Dataset: {name}")
        print(f"  Total IDs: {total:,} | Unique: {unique:,} | Duplicates: {total - unique:,}")
        print(f"  Valid Prefix Pattern ({pattern}): {matches_pattern:,} ({(matches_pattern/total)*100:.2f}%)")

    # Cross-source collision test
    s1_ids = set(df1['entity_id'])
    s2_ids = set(df2['entity_id'])
    s3_ids = set(df3['entity_id'])
    print("\nCross-Source ID Overlaps:")
    print(f"  S1 ∩ S2: {len(s1_ids.intersection(s2_ids))}")
    print(f"  S1 ∩ S3: {len(s1_ids.intersection(s3_ids))}")
    print(f"  S2 ∩ S3: {len(s2_ids.intersection(s3_ids))}")


# ==========================================
# 5. GEOGRAPHIC / COUNTRY DISTRIBUTION
# ==========================================
def analyze_country_distribution(df1, df2, df3):
    """
    Compares geographic distribution across df1, df2, and df3.
    """
    print("\n" + "=" * 70)
    print("[*] 3. GEOGRAPHIC / COUNTRY DISTRIBUTION")
    print("=" * 70)

    country_summary = pd.DataFrame({
        'df1_count': df1['country'].value_counts(dropna=False),
        'df1_%': (df1['country'].value_counts(normalize=True, dropna=False) * 100).round(2),
        'df2_count': df2['country'].value_counts(dropna=False),
        'df2_%': (df2['country'].value_counts(normalize=True, dropna=False) * 100).round(2),
        'df3_count': df3['country'].value_counts(dropna=False),
        'df3_%': (df3['country'].value_counts(normalize=True, dropna=False) * 100).round(2),
    }).fillna(0)

    print(country_summary)
    return country_summary


# ==========================================
# 6. TEXT & NLP ANALYSIS (business_name)
# ==========================================
def analyze_business_names(dfs_dict):
    """
    Performs character, token, script (Indic/Devanagari vs Latin),
    legal suffix, and casing analysis on business_name.
    """
    print("\n" + "=" * 70)
    print("[*] 4. BUSINESS NAME NLP & LINGUISTIC PROFILE")
    print("=" * 70)

    legal_suffixes = [
        'llc', 'inc', 'incorporated', 'corp', 'corporation', 'ltd', 'limited',
        'pvt ltd', 'private limited', 'llp', 'co', 'company', 'enterprises',
        'services', 'solutions', 'associates', 'group', 'holdings', 'industries',
        'trust', 'bank', 'center', 'centre', 'mart', 'store', 'hospital', 'clinic'
    ]

    stats = []
    for name, df in dfs_dict.items():
        if 'business_name' not in df.columns:
            continue
        col = df['business_name'].astype(str)
        char_lens = col.apply(len)
        word_counts = col.apply(lambda x: len(x.split()))
        has_non_ascii = col.apply(lambda x: bool(re.search(r'[^\x00-\x7F]', x)))
        has_url = col.apply(lambda x: bool(re.search(r'(www\.|http|\.com|\.in|\.org|\.net|\.co)', x.lower())))
        is_upper = col.apply(lambda x: x.isupper())
        is_lower = col.apply(lambda x: x.islower())
        is_title = col.apply(lambda x: x.istitle())

        stats.append({
            'Dataset': name,
            'Char Len (Mean)': round(char_lens.mean(), 2),
            'Char Len (Median)': int(char_lens.median()),
            'Char Len (Max)': int(char_lens.max()),
            'Word Count (Mean)': round(word_counts.mean(), 2),
            'Word Count (Median)': int(word_counts.median()),
            'Non-ASCII / Indic %': round(has_non_ascii.mean() * 100, 2),
            'Contains URL %': round(has_url.mean() * 100, 2),
            'UPPERCASE %': round(is_upper.mean() * 100, 2),
            'Title Case %': round(is_title.mean() * 100, 2),
        })

    print(pd.DataFrame(stats).to_string(index=False))

    # Top entity suffixes in Source 1
    print("\nTop Legal/Corporate Entity Suffixes Detected (Source 1 Sample):")
    all_names = df1['business_name'].astype(str).str.lower()
    suffix_counts = {suffix: all_names.str.contains(r'\b' + re.escape(suffix) + r'\b', regex=True).sum()
                     for suffix in legal_suffixes}
    sorted_suffixes = sorted(suffix_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for s, count in sorted_suffixes:
        print(f"  '{s.upper()}': {count:,} records ({(count/len(df1))*100:.2f}%)")


# ==========================================
# 7. STRUCTURAL ANALYSIS (business_address)
# ==========================================
def analyze_business_addresses(dfs_dict):
    """
    Analyzes address length, delimiters, PO Box presence, and postal code structure.
    """
    print("\n" + "=" * 70)
    print("[*] 5. BUSINESS ADDRESS STRUCTURAL PROFILE")
    print("=" * 70)

    stats = []
    for name, df in dfs_dict.items():
        if 'business_address' not in df.columns:
            continue
        col = df['business_address'].astype(str)
        char_lens = col.apply(len)
        comma_counts = col.apply(lambda x: x.count(','))
        has_po_box = col.apply(lambda x: bool(re.search(r'\b(p\.?\s*o\.?\s*box|box\s+\d+)\b', x, re.I)))
        has_unit_apt = col.apply(lambda x: bool(re.search(r'\b(unit|apt|apartment|ste|suite|fl|floor|tower|block)\b', x, re.I)))
        has_zip_pin = col.apply(lambda x: bool(re.search(r'\b\d{5,6}\b', x)))

        stats.append({
            'Dataset': name,
            'Char Len (Mean)': round(char_lens.mean(), 2),
            'Char Len (Median)': int(char_lens.median()),
            'Avg Delimiter Commas': round(comma_counts.mean(), 2),
            'PO Box %': round(has_po_box.mean() * 100, 2),
            'Unit/Apt/Suite %': round(has_unit_apt.mean() * 100, 2),
            'ZIP / PIN Code Present %': round(has_zip_pin.mean() * 100, 2),
        })

    print(pd.DataFrame(stats).to_string(index=False))


# ==========================================
# 8. GROUND TRUTH (df) MATCH GRAPH & CARDINALITY
# ==========================================
def analyze_ground_truth(df_gt, df1, df2, df3):
    """
    Evaluates match counts, S2 vs S3 match ratios, target entity re-use, and referential integrity.
    """
    print("\n" + "=" * 70)
    print("[*] 6. GROUND TRUTH (df) MATCH GRAPH & CARDINALITY")
    print("=" * 70)

    def parse_matches(matched_str):
        if pd.isna(matched_str) or not str(matched_str).strip():
            return 0, 0, 0
        tokens = [t.strip() for t in str(matched_str).split(',') if t.strip()]
        s2 = sum(1 for t in tokens if t.startswith('S2-'))
        s3 = sum(1 for t in tokens if t.startswith('S3-'))
        return len(tokens), s2, s3

    parsed = [parse_matches(m) for m in df_gt['matched_entity_ids']]
    df_gt['total_matches'] = [p[0] for p in parsed]
    df_gt['s2_matches'] = [p[1] for p in parsed]
    df_gt['s3_matches'] = [p[2] for p in parsed]

    print("Match Count Distribution per S1 Entity:")
    print(df_gt[['total_matches', 's2_matches', 's3_matches']].describe())

    # Multi-source breakdown
    only_s2 = ((df_gt['s2_matches'] > 0) & (df_gt['s3_matches'] == 0)).sum()
    only_s3 = ((df_gt['s3_matches'] > 0) & (df_gt['s2_matches'] == 0)).sum()
    both_s2_s3 = ((df_gt['s2_matches'] > 0) & (df_gt['s3_matches'] > 0)).sum()
    zero_matches = (df_gt['total_matches'] == 0).sum()
    total_queries = len(df_gt)

    print("\nMatching Modality Breakdown:")
    print(f"  Only Source 2 Matches : {only_s2:,} ({only_s2/total_queries*100:.2f}%)")
    print(f"  Only Source 3 Matches : {only_s3:,} ({only_s3/total_queries*100:.2f}%)")
    print(f"  Both S2 and S3 Matches: {both_s2_s3:,} ({both_s2_s3/total_queries*100:.2f}%)")
    print(f"  Zero Matches Recorded : {zero_matches:,} ({zero_matches/total_queries*100:.2f}%)")

    # Target entity re-use (Many-to-Many vs One-to-Many)
    all_matched_ids = [m.strip() for row in df_gt['matched_entity_ids'].dropna() for m in str(row).split(',') if m.strip()]
    target_counts = Counter(all_matched_ids)
    reused_targets = sum(1 for _, count in target_counts.items() if count > 1)
    max_reuse = max(target_counts.values()) if target_counts else 0

    print(f"\nTarget Entity Reuse (Many-to-Many Check):")
    print(f"  Total Target Links     : {len(all_matched_ids):,}")
    print(f"  Unique Target Entities : {len(target_counts):,}")
    print(f"  Re-used Targets (>1 S1): {reused_targets:,} ({reused_targets/len(target_counts)*100:.2f}%)")
    print(f"  Max S1 links to single target: {max_reuse}")


# ==========================================
# 9. TRUE MATCH SIMILARITY & NOISE PROFILE
# ==========================================
def analyze_pair_similarity(df_gt, df1, df2, df3, sample_n=5000):
    """
    Computes exact match, token Jaccard similarity, and address variation across true positive pairs.
    """
    print("\n" + "=" * 70)
    print("[*] 7. TRUE MATCH PAIR SIMILARITY & VARIATION AUDIT")
    print("=" * 70)

    # Build lookups
    s1_map = df1.set_index('entity_id')[['business_name', 'business_address', 'country']].to_dict('index')
    s2_map = df2.set_index('entity_id')[['business_name', 'business_address', 'country']].to_dict('index')
    s3_map = df3.set_index('entity_id')[['business_name', 'business_address', 'country']].to_dict('index')

    def jaccard_sim(str1, str2):
        s1 = set(re.findall(r'\w+', str(str1).lower()))
        s2 = set(re.findall(r'\w+', str(str2).lower()))
        if not s1 or not s2:
            return 0.0
        return len(s1.intersection(s2)) / len(s1.union(s2))

    name_exact, name_case_exact, name_jaccard = [], [], []
    addr_exact, addr_case_exact, addr_jaccard = [], [], []
    country_exact = []

    sampled_gt = df_gt.dropna(subset=['matched_entity_ids']).head(sample_n)

    for _, row in sampled_gt.iterrows():
        s1_id = row['source1_entity_id']
        if s1_id not in s1_map:
            continue
        s1_item = s1_map[s1_id]

        targets = [t.strip() for t in row['matched_entity_ids'].split(',') if t.strip()]
        for t_id in targets:
            target_map = s2_map if t_id.startswith('S2-') else s3_map
            if t_id not in target_map:
                continue
            t_item = target_map[t_id]

            # Name similarity
            n1 = str(s1_item['business_name'])
            n2 = str(t_item['business_name'])
            name_exact.append(n1 == n2)
            name_case_exact.append(n1.lower() == n2.lower())
            name_jaccard.append(jaccard_sim(n1, n2))

            # Address similarity
            a1 = str(s1_item['business_address'])
            a2 = str(t_item['business_address'])
            addr_exact.append(a1 == a2)
            addr_case_exact.append(a1.lower() == a2.lower())
            addr_jaccard.append(jaccard_sim(a1, a2))

            # Country consistency
            country_exact.append(str(s1_item['country']).strip() == str(t_item['country']).strip())

    if name_exact:
        print(f"Sampled Pairs Evaluated: {len(name_exact):,}")
        print(f"  Business Name Exact Match (Case-Sensitive)  : {np.mean(name_exact)*100:.2f}%")
        print(f"  Business Name Exact Match (Case-Insensitive): {np.mean(name_case_exact)*100:.2f}%")
        print(f"  Business Name Mean Token Jaccard Similarity : {np.mean(name_jaccard):.4f}")
        print(f"  Business Address Exact Match                : {np.mean(addr_exact)*100:.2f}%")
        print(f"  Business Address Case-Insensitive Match     : {np.mean(addr_case_exact)*100:.2f}%")
        print(f"  Business Address Mean Token Jaccard         : {np.mean(addr_jaccard):.4f}")
        print(f"  Country Match Consistency                   : {np.mean(country_exact)*100:.2f}%")


# ==========================================
# 10. MAIN RUNNER
# ==========================================
if __name__ == "__main__":
    df_gt, df1, df2, df3 = load_data(sample_rows=50000)

    dfs = {'df1 (Source 1)': df1, 'df2 (Source 2)': df2, 'df3 (Source 3)': df3, 'df_gt (Ground Truth)': df_gt}

    audit_data_hygiene(dfs)
    analyze_entity_ids(df_gt, df1, df2, df3)
    analyze_country_distribution(df1, df2, df3)
    analyze_business_names(dfs)
    analyze_business_addresses(dfs)
    analyze_ground_truth(df_gt, df1, df2, df3)
    analyze_pair_similarity(df_gt, df1, df2, df3, sample_n=5000)

    print("\n[+] Complete EDA process executed successfully!")
