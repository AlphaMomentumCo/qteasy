import tushare as ts
import pandas as pd
import time

import tushare as ts
#tushare版本 1.4.24
token = "1650cd309ee490e5d83523e004de5e142461bf2602ae208bebef7dd35b47"

pro = ts.pro_api(token)

pro._DataApi__token = token # 保证有这个代码，不然不可以获取
pro._DataApi__http_url = 'http://lianghua.nanyangqiankun.top'  # 保证有这个代码，不然不可以获取

def download_and_format_l3_daily():
    # ==========================================
    # 步骤 1：读取前期准备的基础数据
    # ==========================================
    print("步骤 1：读取历史成分股明细和 L3 字典...")
    try:
        df_history = pd.read_csv('/home/lsy/data/project/Quant/AlphaData/data_preview/tushare/ci_index/step0_get_L3_index/citic_members_history_full.csv')
        df_l3_dict = pd.read_csv('/home/lsy/data/project/Quant/AlphaData/data_preview/tushare/ci_index/step0_get_L3_index/citic_l3_dict.csv')
        # 重命名成分股的 ts_code 为 con_code，避免和指数的 ts_code 冲突
        df_history = df_history.rename(columns={'ts_code': 'con_code'})
    except Exception as e:
        print(f"读取基础文件失败，请确保目录下有 citic_members_history_full.csv 和 citic_l3_dict.csv\n错误：{e}")
        return

    l3_codes = df_l3_dict['l3_code'].dropna().unique().tolist()
    print(f"共加载 {len(l3_codes)} 个 L3 行业代码。")

    # ==========================================
    # 步骤 2：预处理成分股历史变动逻辑，极大提升匹配速度
    # ==========================================
    print("步骤 2：构建成分股历史变动的高速查询字典...")
    # 将日期转为整数以便快速比较大小，如 20200101
    df_history['in_date'] = df_history['in_date'].astype(int)
    df_history['out_date'] = df_history['out_date'].fillna('20991231').astype(int)
    
    # 构造成分股字典: { l3_code: [{'con_code': '000001.SZ', 'in_date': 20000101, 'out_date': 20991231}, ...] }
    history_dict = df_history.groupby('l3_code').apply(
        lambda x: x[['con_code', 'in_date', 'out_date']].to_dict('records')
    ).to_dict()

    # ==========================================
    # 步骤 3：分段拉取指数行情数据
    # ==========================================
    print("步骤 3：开始分段拉取从 2001 到 2025 的 L3 行业日线行情...")
    # 为了突破 4000 条限制，将 25 年分成 3 段
    date_chunks = [
        ('20010101', '20101231'),
        ('20110101', '20201231'),
        ('20210101', '20251231')
    ]
    
    all_daily_quotes = []
    
    for i, code in enumerate(l3_codes):
        for start_dt, end_dt in date_chunks:
            try:
                df_daily = pro.ci_daily(
                    ts_code=code, 
                    start_date=start_dt, 
                    end_date=end_dt,
                    fields='ts_code,trade_date,open,low,high,close,pre_close,change,pct_change,vol,amount'
                )
                if df_daily is not None and not df_daily.empty:
                    all_daily_quotes.append(df_daily)
            except Exception as e:
                print(f"拉取 {code} [{start_dt}-{end_dt}] 行情报错: {e}")
            time.sleep(0.3) # 严格控制频率防封锁
            
        if (i + 1) % 10 == 0:
            print(f"已完成 {i + 1} / {len(l3_codes)} 个指数的数据拉取...")

    print("行情拉取完毕，正在合并数据...")
    df_all_quotes = pd.concat(all_daily_quotes, ignore_index=True)

    # ==========================================
    # 步骤 4：匹配每日成分股并规范化数据格式
    # ==========================================
    print("步骤 4：正在匹配每日实时成分股并构建 MultiIndex (可能需要几十秒)...")
    
    # 保留一个整数型的 trade_date 用于比较
    df_all_quotes['trade_date_int'] = df_all_quotes['trade_date'].astype(int)

    # 核心匹配函数：匹配当前交易日属于该指数的成分股
    def get_daily_con_codes(row):
        l3_idx = row['ts_code']
        dt_int = row['trade_date_int']
        
        if l3_idx not in history_dict:
            return ""
            
        # 筛选逻辑：纳入日期 <= 当前日期 且 剔除日期 >= 当前日期
        valid_stocks = [
            item['con_code'] 
            for item in history_dict[l3_idx] 
            if item['in_date'] <= dt_int <= item['out_date']
        ]
        # 用逗号拼接成字符串保存
        return ",".join(valid_stocks)

    # 应用匹配逻辑生成 con_codes 列
    df_all_quotes['con_codes'] = df_all_quotes.apply(get_daily_con_codes, axis=1)
    
    # 关联 L3 行业名称 (l3_name)
    df_all_quotes = pd.merge(df_all_quotes, df_l3_dict[['l3_code', 'l3_name']], left_on='ts_code', right_on='l3_code', how='left')

    # 将 trade_date 转换为 Timestamp 格式
    df_all_quotes['trade_date'] = pd.to_datetime(df_all_quotes['trade_date'])

    # 规范化列的数据类型为 float64
    float_cols = ['open', 'low', 'high', 'close', 'pre_close', 'change', 'pct_change', 'vol', 'amount']
    for col in float_cols:
        df_all_quotes[col] = df_all_quotes[col].astype(float)

    # 设置 MultiIndex: (trade_date, ts_code)
    df_all_quotes.set_index(['trade_date', 'ts_code'], inplace=True)

    # 保留最终需要的列，并确保顺序一致
    final_columns = ['open', 'low', 'high', 'close', 'pre_close', 'change', 'pct_change', 'vol', 'amount', 'con_codes', 'l3_name']
    df_final = df_all_quotes[final_columns].sort_index()

    # ==========================================
    # 步骤 5：保存为 Parquet 文件
    # ==========================================
    output_path = 'ci_l3_daily.parquet'
    df_final.to_parquet(output_path)
    print(f"\n========== 大功告成 ==========")
    print(f"数据已成功处理并保存为: {output_path}")
    print("\n最终 DataFrame 结构预览:")
    print(df_final.info())

if __name__ == "__main__":
    download_and_format_l3_daily()