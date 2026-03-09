import tushare as ts
import pandas as pd

import tushare as ts
#tushare版本 1.4.24
token = "1650cd309ee490e5d83523e004de5e142461bf2602ae208bebef7dd35b47"

pro = ts.pro_api(token)

pro._DataApi__token = token # 保证有这个代码，不然不可以获取
pro._DataApi__http_url = 'http://lianghua.nanyangqiankun.top'  # 保证有这个代码，不然不可以获取

def patch_missing_data():
    print("1. 读取已有的 Parquet 文件...")
    df_main = pd.read_parquet('ci_l3_daily.parquet')
    
    # 重新加载基础字典用于匹配成分股
    df_history = pd.read_csv('citic_members_history_full.csv').rename(columns={'ts_code': 'con_code'})
    df_l3_dict = pd.read_csv('citic_l3_dict.csv')
    
    df_history['in_date'] = df_history['in_date'].astype(int)
    df_history['out_date'] = df_history['out_date'].fillna('20991231').astype(int)
    # 消除警告的写法
    history_dict = df_history.groupby('l3_code').apply(
        lambda x: x[['con_code', 'in_date', 'out_date']].to_dict('records'),
        include_groups=False
    ).to_dict()

    print("2. 正在单独拉取 CI005549.CI [20210101-20251231] 的缺失数据...")
    df_patch = pro.ci_daily(
        ts_code='CI005549.CI', 
        start_date='20210101', 
        end_date='20251231',
        fields='ts_code,trade_date,open,low,high,close,pre_close,change,pct_change,vol,amount'
    )
    
    if df_patch is None or df_patch.empty:
        print("未拉取到数据，请检查网络/代理是否正常。")
        return
        
    print(f"成功拉取补丁数据 {len(df_patch)} 条，正在处理格式...")
    df_patch['trade_date_int'] = df_patch['trade_date'].astype(int)

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

    df_patch['con_codes'] = df_patch.apply(get_daily_con_codes, axis=1)
    df_patch = pd.merge(df_patch, df_l3_dict[['l3_code', 'l3_name']], left_on='ts_code', right_on='l3_code', how='left')
    
    # 类型转换与 MultiIndex 设置
    df_patch['trade_date'] = pd.to_datetime(df_patch['trade_date'])
    float_cols = ['open', 'low', 'high', 'close', 'pre_close', 'change', 'pct_change', 'vol', 'amount']
    for col in float_cols:
        df_patch[col] = df_patch[col].astype(float)
        
    df_patch.set_index(['trade_date', 'ts_code'], inplace=True)
    final_columns = ['open', 'low', 'high', 'close', 'pre_close', 'change', 'pct_change', 'vol', 'amount', 'con_codes', 'l3_name']
    df_patch = df_patch[final_columns]
    
    print("3. 正在合并数据...")
    # 合并、去除可能存在的重复行，并按索引重新排序
    df_final = pd.concat([df_main, df_patch])
    df_final = df_final[~df_final.index.duplicated(keep='last')].sort_index()
    
    # 覆盖保存
    df_final.to_parquet('ci_l3_daily.parquet')
    print("========== 修复完成 ==========")
    print(f"当前数据总条数: {len(df_final)}")

if __name__ == "__main__":
    patch_missing_data()