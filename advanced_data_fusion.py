import pandas as pd
import os
import glob
import re
from datetime import datetime, date, time, timedelta
from tqdm import tqdm
import numpy as np

# 配置参数
CONFIG = {
    "USER_QA_DATA": "./问答-买卖价差",
    "CSMAR_SPREAD_DATA": "./csmar买卖价差数据",
    "TURNOVER_DATA": "./个股换手率表(日)",
    "AMIHUD_DATA": "./个股Amihud指标表(日)2018到2023年",
    "OUTPUT_DIR": "./数据融合结果_New",
    "YEARS_TO_PROCESS": [2018, 2019, 2020, 2021, 2022, 2023],  # 处理所有年份的数据
    "TRADING_END_TIME": time(15, 0)  # 15:00为交易结束时间
}

# 确保输出目录存在
os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)


def normalize_stock_code(stock_code):
    """标准化股票代码，去除前缀/后缀，保留纯数字部分，并确保是6位数"""
    if pd.isna(stock_code) or stock_code is None:
        return None
    try:
        # 提取纯数字部分
        digits = re.sub(r'\D', '', str(stock_code))
        # 确保股票代码是6位数
        if len(digits) < 6:
            digits = digits.zfill(6)
        elif len(digits) > 6:
            digits = digits[-6:]
        return digits
    except Exception:
        return None


def normalize_date_format(date_str):
    """标准化日期格式，将斜杠分隔转换为连字符分隔"""
    if pd.isna(date_str) or date_str is None:
        return None
    try:
        if isinstance(date_str, str):
            if '/' in date_str:
                parts = date_str.split('/')
                if len(parts) == 3:
                    year, month, day = parts
                    return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        return date_str
    except Exception:
        return date_str


def read_csv_with_encoding(file_path, **kwargs):
    """尝试多种编码读取CSV文件"""
    encodings = ['utf-8', 'gbk', 'gb18030', 'utf-16']
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding, **kwargs)
        except UnicodeDecodeError:
            continue
    raise Exception(f"无法读取文件 {file_path}，所有编码尝试失败")


def build_trading_calendar():
    """构建全量交易日历和下一个交易日映射"""
    print("正在构建交易日历...")
    
    # 收集所有交易日期
    trading_dates_set = set()
    
    # 辅助函数：从文件中提取日期
    def extract_dates_from_file(file_path):
        """从文件中提取日期信息"""
        try:
            df = None
            if file_path.endswith('.csv'):
                df = read_csv_with_encoding(file_path, usecols=['Trddt'], dtype={'Trddt': str})
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path, engine='openpyxl', usecols=['Trddt'])
            elif file_path.endswith('.xls'):
                df = pd.read_excel(file_path, engine='xlrd', usecols=['Trddt'])
            else:
                return []
            
            normalized_dates = df['Trddt'].apply(normalize_date_format).dropna().unique()
            return normalized_dates
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {str(e)}")
            return []
    
    # 从csmar买卖价差数据中提取日期
    spread_folders = glob.glob(os.path.join(CONFIG["CSMAR_SPREAD_DATA"], "*"))
    for folder in spread_folders:
        # 获取所有支持的文件类型
        files = glob.glob(os.path.join(folder, "*.*"))
        supported_files = [f for f in files if f.endswith((".csv", ".xlsx", ".xls")) and not f.startswith('~')]
        
        for file_path in tqdm(supported_files, desc=f"处理 {os.path.basename(folder)} 文件夹"):
            dates = extract_dates_from_file(file_path)
            trading_dates_set.update(dates)
    
    # 从Amihud数据中补充日期
    amihud_files = glob.glob(os.path.join(CONFIG["AMIHUD_DATA"], "*.*"))
    supported_amihud_files = [f for f in amihud_files if f.endswith((".csv", ".xlsx", ".xls")) and not f.startswith('~')]
    
    for file_path in tqdm(supported_amihud_files, desc="处理Amihud数据"):
        dates = extract_dates_from_file(file_path)
        trading_dates_set.update(dates)
    
    # 从换手率数据中补充日期
    turnover_files = glob.glob(os.path.join(CONFIG["TURNOVER_DATA"], "*.*"))
    supported_turnover_files = [f for f in turnover_files if f.endswith((".csv", ".xlsx", ".xls")) and not f.startswith('~')]
    
    for file_path in tqdm(supported_turnover_files, desc="处理换手率数据"):
        dates = extract_dates_from_file(file_path)
        trading_dates_set.update(dates)
    
    # 转换为datetime.date对象并排序
    ALL_TRADING_DATES = []
    for date_str in trading_dates_set:
        try:
            if isinstance(date_str, str):
                dt = datetime.strptime(date_str, '%Y-%m-%d').date()
                ALL_TRADING_DATES.append(dt)
        except Exception:
            continue
    
    ALL_TRADING_DATES.sort()
    
    # 过滤出2018-2023年的日期
    start_date = date(2018, 1, 1)
    end_date = date(2023, 12, 31)
    ALL_TRADING_DATES = [dt for dt in ALL_TRADING_DATES if start_date <= dt <= end_date]
    
    # 创建下一个交易日映射
    next_trading_day_map = {}
    # 生成2018-2023年的所有自然日
    current_date = start_date
    while current_date <= end_date:
        # 找到下一个交易日
        next_trading = None
        for trading_dt in ALL_TRADING_DATES:
            if trading_dt > current_date:
                next_trading = trading_dt
                break
        next_trading_day_map[current_date] = next_trading
        current_date += timedelta(days=1)
    
    print(f"交易日历构建完成，共 {len(ALL_TRADING_DATES)} 个交易日")
    return ALL_TRADING_DATES, next_trading_day_map


def get_market_reaction_date(event_datetime, all_trading_dates, next_trading_day_map):
    """计算市场反应日"""
    if pd.isna(event_datetime) or event_datetime is None or event_datetime == "":
        return None
    
    try:
        # 转换为datetime对象
        if isinstance(event_datetime, str):
            event_datetime = pd.to_datetime(event_datetime, errors='coerce')
        if pd.isna(event_datetime):
            return None
        
        event_date = event_datetime.date()
        event_time = event_datetime.time()
        
        # 判断是否为交易日
        if event_date in all_trading_dates:
            # 交易日判断时间
            if event_time > CONFIG["TRADING_END_TIME"]:
                # 超过15:00，返回下一个交易日
                return next_trading_day_map.get(event_date)
            else:
                # 15:00及之前，返回当天
                return event_date
        else:
            # 非交易日，返回下一个交易日
            return next_trading_day_map.get(event_date)
    except Exception:
        return None


def process_turnover_data():
    """处理个股换手率数据（简化版）"""
    print(f"\n[{datetime.now()}] 开始处理个股换手率数据...")
    
    # 直接处理每个换手率文件
    turnover_files = ['LIQ_TOVER_D0.xlsx', 'LIQ_TOVER_D1.xlsx', 'LIQ_TOVER_D2.xlsx', 'LIQ_TOVER_D3.xlsx']
    
    data_by_year = {year: [] for year in CONFIG["YEARS_TO_PROCESS"]}
    
    for i, filename in enumerate(turnover_files, 1):
        file_path = os.path.join(CONFIG["TURNOVER_DATA"], filename)
        if not os.path.exists(file_path):
            print(f"[{datetime.now()}] 文件 {filename} 不存在，跳过")
            continue
        
        print(f"\n[{datetime.now()}] [{i}/{len(turnover_files)}] 正在处理文件: {filename}")
        try:
            # 使用更简单的方式读取数据，只跳过前2行
            print(f"[{datetime.now()}]   正在使用openpyxl读取.xlsx文件...")
            df = pd.read_excel(file_path, engine='openpyxl', skiprows=2)
            print(f"[{datetime.now()}]   文件读取完成，共 {len(df)} 条记录")
            
            # 直接使用默认列名
            print(f"[{datetime.now()}]   列名: {list(df.columns)}")
            
            # 只保留需要的列，假设前两列是Stkcd和Trddt，第三列是换手率
            if len(df.columns) >= 3:
                df = df.iloc[:, :3]
                df.columns = ['Stkcd', 'Trddt', 'Turnover']
                print(f"[{datetime.now()}]   简化后列名: {list(df.columns)}")
            else:
                print(f"[{datetime.now()}]   警告: 文件列数不足，跳过")
                continue
            
            # 标准化股票代码
            print(f"[{datetime.now()}]   正在标准化股票代码...")
            df['Stkcd'] = df['Stkcd'].apply(normalize_stock_code)
            
            # 标准化日期格式
            print(f"[{datetime.now()}]   正在标准化日期格式...")
            df['Trddt'] = df['Trddt'].apply(normalize_date_format)
            
            # 提取年份
            print(f"[{datetime.now()}]   正在提取年份...")
            df['temp_year'] = pd.to_datetime(df['Trddt'], errors='coerce').dt.year
            
            # 打印年份分布
            year_counts = df['temp_year'].value_counts().sort_index()
            print(f"[{datetime.now()}]   年份分布: {year_counts.to_dict()}")
            
            # 按年份分发数据
            print(f"[{datetime.now()}]   正在按年份分发数据...")
            for year in CONFIG["YEARS_TO_PROCESS"]:
                year_slice = df[df['temp_year'] == year].copy()
                if not year_slice.empty:
                    year_slice = year_slice[['Stkcd', 'Trddt', 'Turnover']]
                    data_by_year[year].append(year_slice)
                    print(f"[{datetime.now()}]   -> 年份 {year}: 新增 {len(year_slice)} 条记录")
                    
        except Exception as e:
            print(f"[{datetime.now()}]   处理文件 {filename} 失败: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # 合并保存
    print(f"\n[{datetime.now()}] 正在按年份合并换手率数据...")
    for year, df_list in data_by_year.items():
        if df_list:
            year_df = pd.concat(df_list, ignore_index=True)
            year_df = year_df.drop_duplicates(subset=['Stkcd', 'Trddt'])
            output_file = os.path.join(CONFIG["OUTPUT_DIR"], f"temp_turnover_{year}.pkl")
            year_df.to_pickle(output_file)
            print(f"[{datetime.now()}] -> {year} 年数据保存成功: {len(year_df)} 条")
        else:
            print(f"[{datetime.now()}] 警告: {year} 年无换手率数据")
    
    print(f"[{datetime.now()}] 换手率数据处理完成")


def process_amihud_data():
    """处理个股Amihud数据（修正版：增强列名识别，支持多种文件格式）"""
    print("\n开始处理个股Amihud数据...")
    
    # 获取所有支持的文件类型
    amihud_files = glob.glob(os.path.join(CONFIG["AMIHUD_DATA"], "*.*"))
    supported_files = [f for f in amihud_files if f.endswith((".csv", ".xlsx", ".xls")) and not f.startswith('~')]
    
    data_by_year = {year: [] for year in CONFIG["YEARS_TO_PROCESS"]}
    
    for file_path in tqdm(supported_files, desc="读取Amihud文件"):
        filename = os.path.basename(file_path)
        try:
            print(f"\n  正在处理文件: {filename}")
            
            # 统一文件读取逻辑
            df = None
            if file_path.endswith('.csv'):
                df = read_csv_with_encoding(file_path)
            elif file_path.endswith('.xlsx'):
                # 强制使用 openpyxl，移除固定的 skiprows 参数
                df = pd.read_excel(file_path, engine='openpyxl')
            elif file_path.endswith('.xls'):
                # 移除固定的 skiprows 参数
                df = pd.read_excel(file_path, engine='xlrd')
            else:
                print(f"  警告: 文件类型不支持，跳过")
                continue
            
            print(f"  原始列名: {list(df.columns)}")
            
            # --- 修改点：列名清洗 --- 
            df.columns = [c.strip() for c in df.columns]
            
            # 寻找 Amihud 列
            amihud_col = None
            for col in df.columns:
                # 匹配 Amihud, amihud, Illiq 等常见变体
                if col.lower() in ['amihud', 'illiq', 'amihudmeasure', 'illiqmeasure', 'illiqty']:
                    amihud_col = col
                    break
            
            if 'Stkcd' not in df.columns or 'Trddt' not in df.columns or amihud_col is None:
                print(f"  警告: 文件 {filename} 缺少关键列。现有列名: {list(df.columns)}")
                continue
            
            print(f"  选择 {amihud_col} 作为Amihud列")
            
            print("  正在标准化股票代码...")
            df['Stkcd'] = df['Stkcd'].apply(normalize_stock_code)
            
            print("  正在标准化日期格式...")
            df['Trddt'] = df['Trddt'].apply(normalize_date_format)
            
            print("  正在提取年份...")
            df['temp_year'] = pd.to_datetime(df['Trddt'], errors='coerce').dt.year
            
            # 统一重命名
            df = df.rename(columns={amihud_col: 'Amihud'})
            
            print("  正在按年份分发数据...")
            for year in CONFIG["YEARS_TO_PROCESS"]:
                year_slice = df[df['temp_year'] == year].copy()
                if not year_slice.empty:
                    year_slice = year_slice[['Stkcd', 'Trddt', 'Amihud']]
                    data_by_year[year].append(year_slice)
                    print(f"  -> 年份 {year}: 新增 {len(year_slice)} 条记录")
                    
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n正在按年份合并Amihud数据...")
    for year, df_list in data_by_year.items():
        if df_list:
            print(f"正在合并 {year} 年 Amihud 数据...")
            year_df = pd.concat(df_list, ignore_index=True)
            year_df = year_df.drop_duplicates(subset=['Stkcd', 'Trddt'])
            output_file = os.path.join(CONFIG["OUTPUT_DIR"], f"temp_amihud_{year}.pkl")
            year_df.to_pickle(output_file)
            print(f"-> 保存完成: {output_file}, 共 {len(year_df)} 条")
        else:
            print(f"警告: {year} 年没有找到任何 Amihud 数据")


def process_qa_data(year, all_trading_dates, next_trading_day_map):
    """处理某一年的问答数据"""
    print(f"\n{'='*80}")
    print(f"开始处理 {year} 年问答数据")
    print(f"{'='*80}")
    
    # 读取问答数据 - 匹配中文括号和英文括号
    print(f"[{datetime.now()}] 开始寻找 {year} 年问答数据文件...")
    print(f"[{datetime.now()}] 问答数据目录: {CONFIG['USER_QA_DATA']}")
    
    # 列出目录中的所有文件
    all_files = os.listdir(CONFIG["USER_QA_DATA"])
    print(f"[{datetime.now()}] 目录中的所有文件: {all_files}")
    
    # 直接遍历所有文件，进行更灵活的匹配
    qa_files = []
    for file in all_files:
        if str(year) in file and file.endswith('.csv'):
            qa_files.append(os.path.join(CONFIG["USER_QA_DATA"], file))
    
    if not qa_files:
        print(f"[{datetime.now()}] 未找到 {year} 年问答数据文件")
        return
    
    print(f"[{datetime.now()}] 找到 {len(qa_files)} 个 {year} 年问答数据文件: {qa_files}")
    
    qa_data = []
    for file_path in tqdm(qa_files, desc=f"读取 {year} 年问答文件"):
        try:
            df = read_csv_with_encoding(file_path)
            qa_data.append(df)
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {str(e)}")
            continue
    
    if not qa_data:
        print(f"无法读取 {year} 年问答数据")
        return
    
    qa_df = pd.concat(qa_data, ignore_index=True)
    print(f"问答数据读取完成，共 {len(qa_df)} 条记录")
    
    # 标准化股票代码
    qa_df['Scode'] = qa_df['Scode'].apply(normalize_stock_code)
    
    # 计算目标交易日
    print("\n开始计算目标交易日...")
    
    # 计算提问目标日
    if 'Qtm' in qa_df.columns:
        tqdm.pandas(desc="计算提问目标日")
        qa_df['TargetDate_Ask'] = qa_df['Qtm'].progress_apply(
            lambda x: get_market_reaction_date(x, all_trading_dates, next_trading_day_map)
        )
    
    # 计算回复目标日
    if 'Recvtm' in qa_df.columns:
        tqdm.pandas(desc="计算回复目标日")
        qa_df['TargetDate_Reply'] = qa_df['Recvtm'].progress_apply(
            lambda x: get_market_reaction_date(x, all_trading_dates, next_trading_day_map)
        )
    
    # 转换日期为字符串格式，确保类型一致性
    if 'TargetDate_Ask' in qa_df.columns:
        qa_df['TargetDate_Ask'] = qa_df['TargetDate_Ask'].apply(
            lambda x: x.strftime('%Y-%m-%d') if x else None
        )
    if 'TargetDate_Reply' in qa_df.columns:
        qa_df['TargetDate_Reply'] = qa_df['TargetDate_Reply'].apply(
            lambda x: x.strftime('%Y-%m-%d') if x else None
        )
    
    # 加载辅助数据
    print("\n开始加载辅助数据...")
    
    # 加载当年和下一年的价差数据（解决跨年效应）
    spread_data = []
    spread_folders = glob.glob(os.path.join(CONFIG["CSMAR_SPREAD_DATA"], "*"))
    for folder in spread_folders:
        csv_files = glob.glob(os.path.join(folder, "*.csv"))
        for csv_file in tqdm(csv_files, desc=f"读取 {os.path.basename(folder)} 价差数据"):
            try:
                df = read_csv_with_encoding(csv_file, dtype={'Stkcd': str, 'Trddt': str})
                # 标准化处理
                df['Stkcd'] = df['Stkcd'].apply(normalize_stock_code)
                df['Trddt'] = df['Trddt'].apply(normalize_date_format)
                # 过滤年份：加载当年和下一年的数据
                df['year'] = pd.to_datetime(df['Trddt'], errors='coerce').dt.year
                df = df[(df['year'] == year) | (df['year'] == year + 1)].drop(columns=['year'])
                # 去重
                df = df.drop_duplicates(subset=['Stkcd', 'Trddt'])
                spread_data.append(df)
            except Exception as e:
                print(f"读取文件 {csv_file} 时出错: {str(e)}")
                continue
    
    spread_df = None
    if spread_data:
        spread_df = pd.concat(spread_data, ignore_index=True)
        spread_df = spread_df.drop_duplicates(subset=['Stkcd', 'Trddt'])
        # 确保类型一致
        spread_df['Stkcd'] = spread_df['Stkcd'].astype(str)
        spread_df['Trddt'] = spread_df['Trddt'].astype(str)
        print(f"价差数据加载完成，共 {len(spread_df)} 条记录")
    
    # 加载当年和下一年的换手率数据（解决跨年效应）
    turnover_df = None
    turnover_dfs = []
    for y in [year, year + 1]:
        turnover_file = os.path.join(CONFIG["OUTPUT_DIR"], f"temp_turnover_{y}.pkl")
        if os.path.exists(turnover_file):
            df = pd.read_pickle(turnover_file)
            # 确保类型一致
            df['Stkcd'] = df['Stkcd'].astype(str)
            df['Trddt'] = df['Trddt'].astype(str)
            turnover_dfs.append(df)
            print(f"加载 {y} 年换手率数据完成，共 {len(df)} 条记录")
    
    if turnover_dfs:
        turnover_df = pd.concat(turnover_dfs, ignore_index=True)
        turnover_df = turnover_df.drop_duplicates(subset=['Stkcd', 'Trddt'])
        print(f"合并后换手率数据共 {len(turnover_df)} 条记录")
    
    # 加载当年和下一年的Amihud数据（解决跨年效应）
    amihud_df = None
    amihud_dfs = []
    for y in [year, year + 1]:
        amihud_file = os.path.join(CONFIG["OUTPUT_DIR"], f"temp_amihud_{y}.pkl")
        if os.path.exists(amihud_file):
            df = pd.read_pickle(amihud_file)
            # 确保类型一致
            df['Stkcd'] = df['Stkcd'].astype(str)
            df['Trddt'] = df['Trddt'].astype(str)
            amihud_dfs.append(df)
            print(f"加载 {y} 年Amihud数据完成，共 {len(df)} 条记录")
    
    if amihud_dfs:
        amihud_df = pd.concat(amihud_dfs, ignore_index=True)
        amihud_df = amihud_df.drop_duplicates(subset=['Stkcd', 'Trddt'])
        print(f"合并后Amihud数据共 {len(amihud_df)} 条记录")
    
    # 开始融合数据
    print("\n开始融合数据...")
    
    # 确保qa_df的关联键类型一致
    qa_df['Scode'] = qa_df['Scode'].astype(str)
    if 'TargetDate_Ask' in qa_df.columns:
        qa_df['TargetDate_Ask'] = qa_df['TargetDate_Ask'].astype(str)
    if 'TargetDate_Reply' in qa_df.columns:
        qa_df['TargetDate_Reply'] = qa_df['TargetDate_Reply'].astype(str)
    
    # 融合提问相关指标
    if spread_df is not None and 'TargetDate_Ask' in qa_df.columns:
        print("\n融合提问相关价差数据...")
        qa_df = qa_df.merge(
            spread_df[['Stkcd', 'Trddt', 'Esp_Amount', 'Esp_Volume', 'Esp_time']],
            left_on=['Scode', 'TargetDate_Ask'],
            right_on=['Stkcd', 'Trddt'],
            how='left',
            suffixes=('', '_Ask')
        )
        # 重命名列
        qa_df = qa_df.rename(columns={
            'Esp_Amount': 'Spread_Ask',
            'Esp_Volume': 'Spread_Ask_Volume',
            'Esp_time': 'Spread_Ask_time'
        })
        qa_df = qa_df.drop(columns=['Stkcd', 'Trddt'])
    
    if turnover_df is not None and 'TargetDate_Ask' in qa_df.columns:
        print("融合提问相关换手率数据...")
        qa_df = qa_df.merge(
            turnover_df[['Stkcd', 'Trddt', 'Turnover']],
            left_on=['Scode', 'TargetDate_Ask'],
            right_on=['Stkcd', 'Trddt'],
            how='left',
            suffixes=('', '_Ask')
        )
        qa_df = qa_df.rename(columns={'Turnover': 'Turnover_Ask'})
        qa_df = qa_df.drop(columns=['Stkcd', 'Trddt'])
    
    if amihud_df is not None and 'TargetDate_Ask' in qa_df.columns:
        print("融合提问相关Amihud数据...")
        qa_df = qa_df.merge(
            amihud_df[['Stkcd', 'Trddt', 'Amihud']],
            left_on=['Scode', 'TargetDate_Ask'],
            right_on=['Stkcd', 'Trddt'],
            how='left',
            suffixes=('', '_Ask')
        )
        qa_df = qa_df.rename(columns={'Amihud': 'Amihud_Ask'})
        qa_df = qa_df.drop(columns=['Stkcd', 'Trddt'])
    
    # 融合回复相关指标
    if spread_df is not None and 'TargetDate_Reply' in qa_df.columns:
        print("\n融合回复相关价差数据...")
        qa_df = qa_df.merge(
            spread_df[['Stkcd', 'Trddt', 'Esp_Amount', 'Esp_Volume', 'Esp_time']],
            left_on=['Scode', 'TargetDate_Reply'],
            right_on=['Stkcd', 'Trddt'],
            how='left',
            suffixes=('', '_Reply')
        )
        qa_df = qa_df.rename(columns={
            'Esp_Amount': 'Spread_Reply',
            'Esp_Volume': 'Spread_Reply_Volume',
            'Esp_time': 'Spread_Reply_time'
        })
        qa_df = qa_df.drop(columns=['Stkcd', 'Trddt'])
    
    if turnover_df is not None and 'TargetDate_Reply' in qa_df.columns:
        print("融合回复相关换手率数据...")
        qa_df = qa_df.merge(
            turnover_df[['Stkcd', 'Trddt', 'Turnover']],
            left_on=['Scode', 'TargetDate_Reply'],
            right_on=['Stkcd', 'Trddt'],
            how='left',
            suffixes=('', '_Reply')
        )
        qa_df = qa_df.rename(columns={'Turnover': 'Turnover_Reply'})
        qa_df = qa_df.drop(columns=['Stkcd', 'Trddt'])
    
    if amihud_df is not None and 'TargetDate_Reply' in qa_df.columns:
        print("融合回复相关Amihud数据...")
        qa_df = qa_df.merge(
            amihud_df[['Stkcd', 'Trddt', 'Amihud']],
            left_on=['Scode', 'TargetDate_Reply'],
            right_on=['Stkcd', 'Trddt'],
            how='left',
            suffixes=('', '_Reply')
        )
        qa_df = qa_df.rename(columns={'Amihud': 'Amihud_Reply'})
        qa_df = qa_df.drop(columns=['Stkcd', 'Trddt'])
    
    # 填充缺失值
    print("\n开始填充缺失值...")
    # 定义需要填充的列，区分数值型和字符串型
    numeric_columns = [
        'Spread_Ask', 'Spread_Ask_Volume',
        'Turnover_Ask', 'Amihud_Ask',
        'Spread_Reply', 'Spread_Reply_Volume',
        'Turnover_Reply', 'Amihud_Reply'
    ]
    string_columns = [
        'Spread_Ask_time',
        'Spread_Reply_time'
    ]
    
    # 数值型列使用NaN填充（保持数值类型）
    for col in numeric_columns:
        if col in qa_df.columns:
            qa_df[col] = qa_df[col].fillna(float('nan'))
    
    # 字符串型列使用空字符串填充
    for col in string_columns:
        if col in qa_df.columns:
            qa_df[col] = qa_df[col].fillna('')
    
    # 保存结果
    output_file = os.path.join(CONFIG["OUTPUT_DIR"], f"问答_全指标融合数据集_{year}.csv")
    qa_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n{year} 年数据融合完成，保存到 {output_file}")
    print(f"融合结果共 {len(qa_df)} 条记录")


def main():
    """主函数"""
    print("="*100)
    print("高级数据融合脚本开始执行")
    print("="*100)
    print(f"当前时间: {datetime.now()}")
    
    try:
        # 第一阶段：构建交易日历
        print(f"\n[{datetime.now()}] 开始构建交易日历...")
        ALL_TRADING_DATES, next_trading_day_map = build_trading_calendar()
        print(f"[{datetime.now()}] 交易日历构建完成，共 {len(ALL_TRADING_DATES)} 个交易日")
        
        # 第二阶段：处理辅助数据
        # 暂时跳过换手率数据处理，只处理Amihud数据
        # print(f"\n[{datetime.now()}] 开始处理换手率数据...")
        # process_turnover_data()
        # print(f"[{datetime.now()}] 换手率数据处理完成")
        
        print(f"\n[{datetime.now()}] 开始处理Amihud数据...")
        process_amihud_data()
        print(f"[{datetime.now()}] Amihud数据处理完成")
        
        # 第三阶段：按年份融合数据
        print(f"\n[{datetime.now()}] 开始按年份融合数据...")
        for year in CONFIG["YEARS_TO_PROCESS"]:
            print(f"\n[{datetime.now()}] 开始处理 {year} 年数据...")
            process_qa_data(year, ALL_TRADING_DATES, next_trading_day_map)
            print(f"[{datetime.now()}] {year} 年数据处理完成")
        
        print(f"\n[{datetime.now()}] 所有数据融合完成！")
        print("="*100)
    except Exception as e:
        print(f"\n[{datetime.now()}] 脚本执行失败: {str(e)}")
        import traceback
        traceback.print_exc()
        print("="*100)


if __name__ == "__main__":
    main()
