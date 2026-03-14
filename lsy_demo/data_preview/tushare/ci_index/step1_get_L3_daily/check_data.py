import pandas as pd

def check_parquet_completeness(parquet_path='ci_l3_daily.parquet', dict_path='citic_l3_dict.csv'):
    print(f"正在加载数据，请稍候...\n")
    try:
        df = pd.read_parquet(parquet_path)
        l3_dict = pd.read_csv(dict_path)
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    # 1. 提取基础信息
    all_expected_codes = set(l3_dict['l3_code'].dropna().unique())
    present_codes = set(df.index.get_level_values('ts_code').unique())
    
    global_dates = df.index.get_level_values('trade_date').unique().sort_values()
    global_max_date = global_dates.max()
    global_min_date = global_dates.min()

    print("========== 1. 概览信息 ==========")
    print(f"全局最早交易日: {global_min_date.strftime('%Y-%m-%d')}")
    print(f"全局最晚交易日: {global_max_date.strftime('%Y-%m-%d')}")
    print(f"全局总交易天数: {len(global_dates)} 天")
    print(f"字典中应有指数: {len(all_expected_codes)} 个")
    print(f"实际包含的指数: {len(present_codes)} 个\n")

    # 2. 检查完全缺失的指数
    print("========== 2. 完全缺失检查 ==========")
    completely_missing = all_expected_codes - present_codes
    if completely_missing:
        print(f"❌ 警告：共有 {len(completely_missing)} 个指数完全没有数据！")
        for code in completely_missing:
            name = l3_dict[l3_dict['l3_code'] == code]['l3_name'].iloc[0]
            print(f"   - {code} ({name})")
    else:
        print("✅ 完美：所有字典中的指数均在 Parquet 中有数据记录。")
    print()

    # 3. 检查内部空洞与异常截断
    print("========== 3. 内部空洞与异常截断检查 ==========")
    
    missing_holes = {}
    truncated_indices = []
    
    for code in present_codes:
        # 获取该指数的所有交易日期
        idx_dates = df.xs(code, level='ts_code').index.sort_values()
        if len(idx_dates) == 0:
            continue
            
        min_dt = idx_dates.min()
        max_dt = idx_dates.max()
        
        # --- 检查内部空洞 ---
        # 该指数生命周期内的标准交易日
        expected_dates = global_dates[(global_dates >= min_dt) & (global_dates <= max_dt)]
        missing_dts = expected_dates.difference(idx_dates)
        
        if len(missing_dts) > 0:
            missing_holes[code] = missing_dts
            
        # --- 检查异常截断 (尾部断档) ---
        # 如果某个指数最后一天比全局最后一天早超过 10 个交易日，极有可能是因为分段下载失败
        # （部分指数可能真实退市，但行业指数极少出现此情况，值得重点排查）
        if max_dt < global_dates[-10]:
            truncated_indices.append({
                'code': code, 
                'max_dt': max_dt.strftime('%Y-%m-%d'),
                'miss_days_approx': len(global_dates[global_dates > max_dt])
            })

    # 输出内部空洞结果
    if missing_holes:
        print(f"❌ 发现 {len(missing_holes)} 个指数存在中间漏数据的情况：")
        for code, dts in missing_holes.items():
            print(f"   - {code} 缺失 {len(dts)} 天。示例日期: {dts[0].strftime('%Y-%m-%d')} ...")
    else:
        print("✅ 完美：所有已存在数据的指数，在其生命周期内没有任何中间漏数据！")
        
    print()

    # 输出异常截断结果
    if truncated_indices:
        print(f"⚠️ 发现 {len(truncated_indices)} 个指数疑似【尾部数据截断】(未更新至最新日期)：")
        print("   这通常是由于网络断线导致某个年代的分段 Chunk 没拉下来。")
        for item in truncated_indices:
            name = l3_dict[l3_dict['l3_code'] == item['code']]['l3_name'].iloc[0]
            print(f"   - {item['code']} ({name}) 数据停留在 {item['max_dt']}，落后全局约 {item['miss_days_approx']} 个交易日")
    else:
        print("✅ 完美：所有指数的数据均整齐地更新到了最后一个交易日（无意外截断）！")

    print("\n========== 检查完毕 ==========")

if __name__ == "__main__":
    check_parquet_completeness()