import tushare as ts
import pandas as pd
import time

import tushare as ts
#tushare版本 1.4.24
token = "1650cd309ee490e5d83523e004de5e142461bf2602ae208bebef7dd35b47"

pro = ts.pro_api(token)

pro._DataApi__token = token # 保证有这个代码，不然不可以获取
pro._DataApi__http_url = 'http://lianghua.nanyangqiankun.top'  # 保证有这个代码，不然不可以获取


def extract_all_citic_history_and_levels():
    # 1. 构造中信一级行业代码探测池 (CI005001.CI - CI005035.CI)
    l1_codes = [f"CI00{i}.CI" for i in range(5001, 5036)]
    print(f"步骤 1：准备探测 {len(l1_codes)} 个潜在的一级行业，提取全量历史...")

    all_members = []
    # 必须同时获取 Y(最新) 和 N(历史剔除)，以满足回测无未来函数的要求
    status_list = ['Y', 'N'] 
    
    # 2. 开始拉取数据
    for code in l1_codes:
        code_has_data = False
        for status in status_list:
            try:
                df = pro.ci_index_member(
                    l1_code=code, 
                    is_new=status, 
                    fields='l1_code,l1_name,l2_code,l2_name,l3_code,l3_name,ts_code,name,in_date,out_date,is_new'
                )
                
                if df is not None and not df.empty:
                    all_members.append(df)
                    code_has_data = True
                    
            except Exception as e:
                print(f"提取 {code} (状态:{status}) 数据时发生错误: {e}")
                
            # 休眠防限流，保证稳定性
            time.sleep(0.3)
            
        if code_has_data:
            print(f"成功获取一级行业 {code} 的全量历史数据。")
            
    print("\n========== 数据拉取完成，开始处理和清洗 ==========")

    if not all_members:
        print("未能提取到任何数据，请检查网络或Token权限。")
        return

    # 3. 合并全量明细数据
    df_all = pd.concat(all_members, ignore_index=True)
    
    # 填补最新成分股的 out_date，方便历史截面切片
    df_all['out_date'] = df_all['out_date'].fillna('20991231').replace('', '20991231')
    
    # 4. 从全量数据中分离出 L1, L2, L3 独立字典并去重
    # 提取一级行业字典
    df_l1 = df_all[['l1_code', 'l1_name']].dropna().drop_duplicates().sort_values('l1_code').reset_index(drop=True)
    # 提取二级行业字典
    df_l2 = df_all[['l2_code', 'l2_name']].dropna().drop_duplicates().sort_values('l2_code').reset_index(drop=True)
    # 提取三级行业字典
    df_l3 = df_all[['l3_code', 'l3_name']].dropna().drop_duplicates().sort_values('l3_code').reset_index(drop=True)

    print(f"全市场中信成分股历史记录总计: {len(df_all)} 条")
    print(f"剥离出 一级(L1)行业: {len(df_l1)} 个")
    print(f"剥离出 二级(L2)行业: {len(df_l2)} 个")
    print(f"剥离出 三级(L3)行业: {len(df_l3)} 个")

    # 5. 分别保存为 4 个独立的文件
    df_all.to_csv('citic_members_history_full.csv', index=False, encoding='utf_8_sig')
    df_l1.to_csv('citic_l1_dict.csv', index=False, encoding='utf_8_sig')
    df_l2.to_csv('citic_l2_dict.csv', index=False, encoding='utf_8_sig')
    df_l3.to_csv('citic_l3_dict.csv', index=False, encoding='utf_8_sig')
    
    print("\n所有文件已成功保存至本地！")

if __name__ == "__main__":
    extract_all_citic_history_and_levels()