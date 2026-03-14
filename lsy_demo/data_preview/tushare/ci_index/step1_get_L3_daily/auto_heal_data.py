import tushare as ts
import pandas as pd
import time
from datetime import timedelta

import tushare as ts
#tushare版本 1.4.24
token = "1650cd309ee490e5d83523e004de5e142461bf2602ae208bebef7dd35b47"

pro = ts.pro_api(token)

pro._DataApi__token = token # 保证有这个代码，不然不可以获取
pro._DataApi__http_url = 'http://lianghua.nanyangqiankun.top'  # 保证有这个代码，不然不可以获取

def auto_heal_parquet(parquet_path='ci_l3_daily.parquet', 
                      dict_path='citic_l3_dict.csv', 
                      history_path='citic_members_history_full.csv'):
    
    print("========== 1. 加载数据与构建高速字典 ==========")
    try:
        df_main = pd.read_parquet(parquet_path)
        l3_dict = pd.read_csv(dict_path)
        df_history = pd.read_csv(history_path).rename(columns={'ts_code': 'con_code'})
    except Exception as e:
        print(f"读取基础文件失败: {e}")
        return

    # 构建成分股历史匹配字典 (消除 pandas 警告的写法)
    df_history['in_date'] = df_history['in_date'].astype(int)
    df_history['out_date'] = df_history['out_date'].fillna('20991231').astype(int)
    history_dict = df_history.groupby('l3_code').apply(
        lambda x: x[['con_code', 'in_date', 'out_date']].to_dict('records'),
        include_groups=False
    ).to_dict()

    print("========== 2. 开始诊断缺失情况 ==========")
    all_expected_codes = set(l3_dict['l3_code'].dropna().unique())
    present_codes = set(df_main.index.get_level_values('ts_code').unique())
    
    # 提取全市场真实的交易日历作为基准
    global_dates = df_main.index.get_level_values('trade_date').unique().sort_values()
    global_max_date = global_dates.max()
    
    # 待拉取任务列表
    fetch_tasks = []

    # 1. 查找完全缺失的指数
    completely_missing = all_expected_codes - present_codes
    for code in completely_missing:
        print(f"🔍 发现完全缺失指数: {code}，将全量拉取。")
        fetch_tasks.append({'code': code, 'start': '20010101', 'end': '20101231'})
        fetch_tasks.append({'code': code, 'start': '20110101', 'end': '20201231'})
        fetch_tasks.append({'code': code, 'start': '20210101', 'end': '20251231'})

    # 2. 查找尾部截断的指数 (最后一天比全局最后一天早超过 10 天)
    for code in present_codes:
        idx_dates = df_main.xs(code, level='ts_code').index.sort_values()
        max_dt = idx_dates.max()
        
        if max_dt < global_dates[-10]:
            # 【核心改进】从真实的交易日历中，寻找 max_dt 之后的第一个交易日
            future_dates = global_dates[global_dates > max_dt]
            
            if len(future_dates) > 0:
                next_trade_dt = future_dates[0] # 真实的下一个交易日
                start_patch_dt = next_trade_dt.strftime('%Y%m%d')
                print(f"⚠️ 发现截断指数: {code} (停在 {max_dt.strftime('%Y-%m-%d')}) -> 智能对齐下个交易日: {start_patch_dt}")
                fetch_tasks.append({'code': code, 'start': start_patch_dt, 'end': '20251231'})
            else:
                # 极端情况兜底
                start_patch_dt = (max_dt + timedelta(days=1)).strftime('%Y%m%d')
                fetch_tasks.append({'code': code, 'start': start_patch_dt, 'end': '20251231'})

    if not fetch_tasks:
        print("\n✅ 诊断完毕：数据非常完整，没有发现需要修复的缺失！")
        return

    print(f"\n========== 3. 开始自动修复 (共 {len(fetch_tasks)} 个任务) ==========")
    patched_dfs = []
    
    for i, task in enumerate(fetch_tasks):
        code, start, end = task['code'], task['start'], task['end']
        try:
            df_patch = pro.ci_daily(
                ts_code=code, start_date=start, end_date=end,
                fields='ts_code,trade_date,open,low,high,close,pre_close,change,pct_change,vol,amount'
            )
            if df_patch is not None and not df_patch.empty:
                patched_dfs.append(df_patch)
                print(f"[{i+1}/{len(fetch_tasks)}] 🟢 成功拉取 {code} [{start}-{end}]，共 {len(df_patch)} 条。")
            else:
                # 增加更清晰的提示
                print(f"[{i+1}/{len(fetch_tasks)}] ⚪ 提示: {code} [{start}-{end}] 无数据 (可能已退役停更)。")
        except Exception as e:
            print(f"[{i+1}/{len(fetch_tasks)}] 🔴 拉取 {code} 报错: {e}")
        
        time.sleep(0.4) 

    if not patched_dfs:
        print("\nℹ️ 修复结束：未拉取到新数据。这说明之前停在 2019 年的指数确实已经被中信退役，属于正常现象。")
        return

    print("\n========== 4. 处理补丁数据格式并合并 ==========")
    df_new = pd.concat(patched_dfs, ignore_index=True)
    df_new['trade_date_int'] = df_new['trade_date'].astype(int)

    # 匹配成分股
    def get_daily_con_codes(row):
        l3_idx = row['ts_code']
        dt_int = row['trade_date_int']
        if l3_idx not in history_dict: return ""
        valid_stocks = [
            item['con_code'] for item in history_dict[l3_idx] 
            if item['in_date'] <= dt_int <= item['out_date']
        ]
        return ",".join(valid_stocks)

    print("正在匹配成分股...")
    df_new['con_codes'] = df_new.apply(get_daily_con_codes, axis=1)
    df_new = pd.merge(df_new, l3_dict[['l3_code', 'l3_name']], left_on='ts_code', right_on='l3_code', how='left')
    
    # 格式规范化
    df_new['trade_date'] = pd.to_datetime(df_new['trade_date'])
    float_cols = ['open', 'low', 'high', 'close', 'pre_close', 'change', 'pct_change', 'vol', 'amount']
    for col in float_cols:
        df_new[col] = df_new[col].astype(float)
        
    df_new.set_index(['trade_date', 'ts_code'], inplace=True)
    df_new = df_new[['open', 'low', 'high', 'close', 'pre_close', 'change', 'pct_change', 'vol', 'amount', 'con_codes', 'l3_name']]

    # 无缝合并与去重
    print("正在合并并覆写 Parquet 文件...")
    df_final = pd.concat([df_main, df_new])
    df_final = df_final[~df_final.index.duplicated(keep='last')].sort_index()
    
    df_final.to_parquet(parquet_path)
    print("\n========== 大功告成 ==========")
    print(f"修复完成！成功补充了 {len(df_new)} 条数据。")
    print(f"当前 Parquet 总条数: {len(df_final)}")

if __name__ == "__main__":
    auto_heal_parquet()