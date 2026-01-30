import pandas as pd
import os
import glob
import re
from datetime import datetime, time, timedelta
from tqdm import tqdm

# 配置参数
CONFIG = {
    "USER_QA_DATA": "./用户与公司问答",
    "CSMAR_SPREAD_DATA": "./csmar买卖价差数据",
    "TURNOVER_DATA": "./个股换手率表(日)_CSV_2018-2023",
    "AMIHUD_DATA": "./个股Amihud指标表(日)2018到2023年",
    "OUTPUT_DIR": "./数据融合结果_New",
    "YEARS_TO_PROCESS": [2018, 2019, 2020, 2021, 2022, 2023],
    "CLOSING_TIME": time(15, 0),  # 15:00为交易结束时间
    "SPREAD_FIELDS": ['Esp_Amount', 'Esp_Volume', 'Esp_time']
}

# 确保输出目录存在
print(f"创建输出目录: {CONFIG['OUTPUT_DIR']}")
os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
print(f"输出目录创建成功: {os.path.exists(CONFIG['OUTPUT_DIR'])}")


# 工具函数
def read_csv_with_encoding(file_path, **kwargs):
    """尝试多种编码读取CSV文件"""
    encodings = ['utf-8', 'gbk', 'gb18030']
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding, **kwargs)
        except UnicodeDecodeError:
            continue
    raise Exception(f"无法读取文件 {file_path}，所有编码尝试失败")


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


def parse_datetime_with_fallback(date_str):
    """稳健的日期解析，尝试多种格式"""
    if pd.isna(date_str) or date_str is None:
        return None
    
    # 尝试多种日期格式
    formats = ['%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S',
               '%Y-%m-%d %H:%M', '%Y/%m/%d %H:%M',
               '%Y-%m-%d', '%Y/%m/%d']
    
    for fmt in formats:
        try:
            return datetime.strptime(str(date_str), fmt)
        except ValueError:
            continue
    
    # 最后尝试pandas的to_datetime
    try:
        return pd.to_datetime(date_str, errors='coerce')
    except:
        return None


# 交易日历构建
def build_trading_calendar():
    """构建交易日历和下一个交易日映射"""
    print("=" * 80)
    print("开始构建交易日历...")
    print("=" * 80)
    
    trade_dates_set = set()
    spread_files = []
    
    # 遍历所有价差数据文件
    spread_sub_folders = ['2018-2020', '2021-2023']
    for sub_folder in tqdm(spread_sub_folders, desc="处理价差数据文件夹"):
        folder_path = os.path.join(CONFIG["CSMAR_SPREAD_DATA"], sub_folder)
        # 查找所有HF_Spread开头的CSV文件
        csv_files = glob.glob(os.path.join(folder_path, 'HF_Spread*.csv'))
        spread_files.extend(csv_files)
    
    # 提取所有交易日期
    for csv_file in tqdm(spread_files, desc="提取交易日期"):
        try:
            # 只读取日期列
            for chunk in pd.read_csv(csv_file, dtype={'Trddt': str}, usecols=['Trddt'], chunksize=100000):
                # 标准化日期格式
                normalized_dates = chunk['Trddt'].apply(normalize_date_format).unique()
                trade_dates_set.update(normalized_dates)
        except Exception as e:
            print(f"读取文件 {csv_file} 时出错: {str(e)}")
            continue
    
    # 生成排序后的交易日期列表
    trade_dates = sorted([date for date in trade_dates_set if date])
    trade_dates_datetime = [datetime.strptime(date, '%Y-%m-%d').date() for date in trade_dates]
    
    print(f"共找到 {len(trade_dates)} 个交易日")
    
    # 构建下一个交易日映射
    next_trading_day_map = {}
    for i, date in enumerate(trade_dates_datetime):
        # 对于最后一个交易日，下一个交易日设为None
        if i < len(trade_dates_datetime) - 1:
            next_trading_day_map[date] = trade_dates_datetime[i + 1].strftime('%Y-%m-%d')
        else:
            next_trading_day_map[date] = None
    
    return trade_dates, next_trading_day_map


def get_target_trading_date(dt, trade_dates, next_trading_day_map):
    """计算目标交易日
    
    Args:
        dt: datetime对象，提问时间或回答时间
        trade_dates: 交易日期列表
        next_trading_day_map: 下一个交易日映射字典
        
    Returns:
        str: 目标交易日，格式为YYYY-MM-DD，若无法计算则返回None
    """
    if pd.isna(dt):
        return None
    
    # 获取日期部分
    date_part = dt.date()
    time_part = dt.time()
    date_str = date_part.strftime('%Y-%m-%d')
    
    # 检查是否为交易日
    if date_str in trade_dates:
        # 情况A：在交易日且时间 <= 15:00 -> 当天
        if time_part <= CONFIG["CLOSING_TIME"]:
            return date_str
        # 情况B：在交易日且时间 > 15:00 -> 下一个交易日
        else:
            return next_trading_day_map.get(date_part, None)
    else:
        # 情况C：非交易日 -> 查找下一个交易日
        # 从trade_dates中找到第一个大于date_part的日期
        for trade_date_str in trade_dates:
            trade_date = datetime.strptime(trade_date_str, '%Y-%m-%d').date()
            if trade_date > date_part:
                return trade_date_str
        return None


# 辅助数据加载
def load_turnover_data():
    """加载换手率数据，按年份缓存"""
    print(f"[{datetime.now()}] 开始加载换手率数据...")
    
    # 按年份缓存数据
    turnover_data_by_year = {}
    
    # 遍历所有换手率文件
    turnover_files = glob.glob(os.path.join(CONFIG["TURNOVER_DATA"], "*.csv"))
    
    for file_path in tqdm(turnover_files, desc="加载换手率文件"):
        try:
            df = read_csv_with_encoding(file_path)
            
            # 确保关键列存在
            if 'Stkcd' not in df.columns or 'Trddt' not in df.columns or 'ToverOs' not in df.columns:
                print(f"警告: 文件 {os.path.basename(file_path)} 缺少关键列")
                continue
            
            # 标准化股票代码
            df['Stkcd'] = df['Stkcd'].apply(normalize_stock_code)
            
            # 标准化日期格式
            df['Trddt'] = df['Trddt'].apply(normalize_date_format)
            
            # 提取年份
            df['Year'] = pd.to_datetime(df['Trddt'], errors='coerce').dt.year
            
            # 按年份缓存
            for year in df['Year'].dropna().unique():
                year = int(year)
                if year not in turnover_data_by_year:
                    turnover_data_by_year[year] = []
                
                year_data = df[df['Year'] == year].copy()
                # 只保留需要的列
                year_data = year_data[['Stkcd', 'Trddt', 'ToverOs']]
                # 去重
                year_data = year_data.drop_duplicates(subset=['Stkcd', 'Trddt'])
                turnover_data_by_year[year].append(year_data)
                
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {str(e)}")
            continue
    
    # 合并每年的数据
    for year, data_list in turnover_data_by_year.items():
        if data_list:
            turnover_data_by_year[year] = pd.concat(data_list, ignore_index=True)
            turnover_data_by_year[year] = turnover_data_by_year[year].drop_duplicates(subset=['Stkcd', 'Trddt'])
            # 确保类型正确
            turnover_data_by_year[year]['Stkcd'] = turnover_data_by_year[year]['Stkcd'].astype(str)
            turnover_data_by_year[year]['Trddt'] = turnover_data_by_year[year]['Trddt'].astype(str)
            print(f"加载 {year} 年换手率数据完成，共 {len(turnover_data_by_year[year])} 条记录")
    
    print(f"[{datetime.now()}] 换手率数据加载完成")
    return turnover_data_by_year


def load_amihud_data():
    """加载Amihud数据，按年份缓存"""
    print(f"[{datetime.now()}] 开始加载Amihud数据...")
    
    # 按年份缓存数据
    amihud_data_by_year = {}
    
    # 遍历所有Amihud文件
    amihud_files = glob.glob(os.path.join(CONFIG["AMIHUD_DATA"], "*.csv"))
    
    for file_path in tqdm(amihud_files, desc="加载Amihud文件"):
        try:
            df = read_csv_with_encoding(file_path)
            
            # 确保Stkcd和Trddt列存在
            if 'Stkcd' not in df.columns or 'Trddt' not in df.columns:
                print(f"警告: 文件 {os.path.basename(file_path)} 缺少关键列")
                continue
            
            # 寻找Amihud列
            amihud_col = None
            for col in df.columns:
                if col.lower() in ['amihud', 'illiq']:
                    amihud_col = col
                    break
            
            if amihud_col is None:
                print(f"警告: 文件 {os.path.basename(file_path)} 缺少Amihud列")
                continue
            
            # 标准化股票代码
            df['Stkcd'] = df['Stkcd'].apply(normalize_stock_code)
            
            # 标准化日期格式
            df['Trddt'] = df['Trddt'].apply(normalize_date_format)
            
            # 提取年份
            df['Year'] = pd.to_datetime(df['Trddt'], errors='coerce').dt.year
            
            # 重命名Amihud列
            df = df.rename(columns={amihud_col: 'Amihud'})
            
            # 按年份缓存
            for year in df['Year'].dropna().unique():
                year = int(year)
                if year not in amihud_data_by_year:
                    amihud_data_by_year[year] = []
                
                year_data = df[df['Year'] == year].copy()
                # 只保留需要的列
                year_data = year_data[['Stkcd', 'Trddt', 'Amihud']]
                # 去重
                year_data = year_data.drop_duplicates(subset=['Stkcd', 'Trddt'])
                amihud_data_by_year[year].append(year_data)
                
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {str(e)}")
            continue
    
    # 合并每年的数据
    for year, data_list in amihud_data_by_year.items():
        if data_list:
            amihud_data_by_year[year] = pd.concat(data_list, ignore_index=True)
            amihud_data_by_year[year] = amihud_data_by_year[year].drop_duplicates(subset=['Stkcd', 'Trddt'])
            # 确保类型正确
            amihud_data_by_year[year]['Stkcd'] = amihud_data_by_year[year]['Stkcd'].astype(str)
            amihud_data_by_year[year]['Trddt'] = amihud_data_by_year[year]['Trddt'].astype(str)
            print(f"加载 {year} 年Amihud数据完成，共 {len(amihud_data_by_year[year])} 条记录")
    
    print(f"[{datetime.now()}] Amihud数据加载完成")
    return amihud_data_by_year


def load_spread_data_for_year(year, stock_codes, target_dates):
    """加载指定年份及相关的价差数据"""
    print(f"开始加载 {year} 年相关的价差数据...")
    
    all_spread_chunks = []
    
    # 遍历所有价差文件
    spread_sub_folders = ['2018-2020', '2021-2023']
    spread_files = []
    for sub_folder in spread_sub_folders:
        folder_path = os.path.join(CONFIG["CSMAR_SPREAD_DATA"], sub_folder)
        csv_files = glob.glob(os.path.join(folder_path, 'HF_Spread*.csv'))
        spread_files.extend(csv_files)
    
    for csv_file in tqdm(spread_files, desc="处理价差文件"):
        try:
            # 分批读取
            for chunk in pd.read_csv(csv_file, dtype={'Stkcd': str, 'Trddt': str}, chunksize=100000):
                # 标准化日期和股票代码
                chunk['Trddt_normalized'] = chunk['Trddt'].apply(normalize_date_format)
                chunk['Stkcd_normalized'] = chunk['Stkcd'].apply(normalize_stock_code)
                
                # 过滤需要的数据
                filtered_chunk = chunk[
                    (chunk['Stkcd_normalized'].isin(stock_codes)) & 
                    (chunk['Trddt_normalized'].isin(target_dates))
                ]
                
                if not filtered_chunk.empty:
                    all_spread_chunks.append(filtered_chunk)
        except Exception as e:
            print(f"读取文件 {csv_file} 时出错: {str(e)}")
            continue
    
    if not all_spread_chunks:
        print(f"未找到匹配的价差数据")
        return None
    
    # 合并所有chunk
    spread_df = pd.concat(all_spread_chunks, ignore_index=True)
    print(f"价差数据加载完成，共 {len(spread_df)} 条记录")
    
    return spread_df


# 数据融合
def merge_spread_data(qa_df, spread_df, merge_on_date, suffix):
    """合并价差数据"""
    if spread_df is None:
        # 如果没有价差数据，添加空的价差字段
        for field in CONFIG["SPREAD_FIELDS"]:
            qa_df[f'{field}{suffix}'] = 'NA'
        return qa_df
    
    # 准备价差数据，只保留需要的字段
    spread_data = spread_df[['Stkcd_normalized', 'Trddt_normalized'] + CONFIG["SPREAD_FIELDS"]].copy()
    
    # 重命名价差字段，添加后缀
    rename_map = {}
    for field in CONFIG["SPREAD_FIELDS"]:
        rename_map[field] = f'{field}{suffix}'
    spread_data = spread_data.rename(columns=rename_map)
    
    # 左连接
    merged_df = pd.merge(
        qa_df, 
        spread_data, 
        left_on=['Scode_normalized', merge_on_date], 
        right_on=['Stkcd_normalized', 'Trddt_normalized'], 
        how='left'
    )
    
    # 删除临时列
    merged_df = merged_df.drop(columns=['Stkcd_normalized_y', 'Trddt_normalized'], errors='ignore')
    
    # 重命名股票代码列
    if 'Stkcd_normalized_x' in merged_df.columns:
        merged_df = merged_df.rename(columns={'Stkcd_normalized_x': 'Stkcd_normalized'})
    
    # 填充缺失值
    for field in CONFIG["SPREAD_FIELDS"]:
        merged_df[f'{field}{suffix}'] = merged_df[f'{field}{suffix}'].fillna('NA')
    
    return merged_df


def merge_turnover_data(qa_df, turnover_df, merge_on_date, suffix):
    """合并换手率数据"""
    if turnover_df is None:
        qa_df[f'Turnover{suffix}'] = float('nan')
        return qa_df
    
    # 左连接
    merged_df = pd.merge(
        qa_df, 
        turnover_df[['Stkcd', 'Trddt', 'ToverOs']], 
        left_on=['Scode_normalized', merge_on_date], 
        right_on=['Stkcd', 'Trddt'], 
        how='left'
    )
    
    # 重命名列
    merged_df = merged_df.rename(columns={'ToverOs': f'Turnover{suffix}'})
    # 删除临时列
    merged_df = merged_df.drop(columns=['Stkcd', 'Trddt'], errors='ignore')
    # 填充缺失值
    merged_df[f'Turnover{suffix}'] = merged_df[f'Turnover{suffix}'].fillna(float('nan'))
    
    return merged_df


def merge_amihud_data(qa_df, amihud_df, merge_on_date, suffix):
    """合并Amihud数据"""
    if amihud_df is None:
        qa_df[f'ILLIQ{suffix}'] = float('nan')
        return qa_df
    
    # 左连接
    merged_df = pd.merge(
        qa_df, 
        amihud_df[['Stkcd', 'Trddt', 'Amihud']], 
        left_on=['Scode_normalized', merge_on_date], 
        right_on=['Stkcd', 'Trddt'], 
        how='left'
    )
    
    # 重命名列
    merged_df = merged_df.rename(columns={'Amihud': f'ILLIQ{suffix}'})
    # 删除临时列
    merged_df = merged_df.drop(columns=['Stkcd', 'Trddt'], errors='ignore')
    # 填充缺失值
    merged_df[f'ILLIQ{suffix}'] = merged_df[f'ILLIQ{suffix}'].fillna(float('nan'))
    
    return merged_df


def merge_all_indicators(qa_df, spread_df, turnover_df, amihud_df):
    """合并所有指标"""
    # 合并价差数据
    if 'TargetDate_Ask' in qa_df.columns:
        qa_df = merge_spread_data(qa_df, spread_df, 'TargetDate_Ask', '_Ask')
    if 'TargetDate_Reply' in qa_df.columns:
        qa_df = merge_spread_data(qa_df, spread_df, 'TargetDate_Reply', '_Reply')
    
    # 合并换手率数据
    if 'TargetDate_Ask' in qa_df.columns:
        qa_df = merge_turnover_data(qa_df, turnover_df, 'TargetDate_Ask', '_Ask')
    if 'TargetDate_Reply' in qa_df.columns:
        qa_df = merge_turnover_data(qa_df, turnover_df, 'TargetDate_Reply', '_Reply')
    
    # 合并Amihud数据
    if 'TargetDate_Ask' in qa_df.columns:
        qa_df = merge_amihud_data(qa_df, amihud_df, 'TargetDate_Ask', '_Ask')
    if 'TargetDate_Reply' in qa_df.columns:
        qa_df = merge_amihud_data(qa_df, amihud_df, 'TargetDate_Reply', '_Reply')
    
    return qa_df


def filter_output_fields(qa_df):
    """过滤输出字段，只保留需要的字段"""
    # 定义输出字段白名单
    output_fields = [
        # 原始问答数据字段
        'Scode', 'Sname', 'AskId', 'AskTid', 'Qtm', 'Recvtm', 'Qusername', 'Qcontent', 'Areply',
        # 标准化字段
        'Scode_normalized',
        # 目标交易日
        'TargetDate_Ask', 'TargetDate_Reply',
        # 价差指标
        'Esp_Amount_Ask', 'Esp_Volume_Ask', 'Esp_time_Ask',
        'Esp_Amount_Reply', 'Esp_Volume_Reply', 'Esp_time_Reply',
        # 换手率指标
        'Turnover_Ask', 'Turnover_Reply',
        # Amihud指标
        'ILLIQ_Ask', 'ILLIQ_Reply'
    ]
    
    # 只保留存在的字段
    existing_fields = [field for field in output_fields if field in qa_df.columns]
    return qa_df[existing_fields]


# 主函数
def process_qa_data(year, trade_dates, next_trading_day_map, turnover_data_by_year, amihud_data_by_year):
    """处理某一年的问答数据"""
    print(f"\n{'='*80}")
    print(f"开始处理 {year} 年问答数据")
    print(f"{'='*80}")
    
    # 找到当年的问答数据文件
    qa_files = []
    for file in glob.glob(os.path.join(CONFIG["USER_QA_DATA"], f"*{year}*.csv")):
        qa_files.append(file)
    
    if not qa_files:
        print(f"未找到 {year} 年的问答数据文件")
        return
    
    print(f"找到 {len(qa_files)} 个 {year} 年的问答数据文件")
    
    for qa_file in qa_files:
        print(f"\n处理文件: {os.path.basename(qa_file)}")
        
        try:
            # 读取问答数据
            print(f"[{datetime.now()}] 读取问答数据...")
            qa_df = read_csv_with_encoding(qa_file)
            print(f"[{datetime.now()}] 读取完成，共 {len(qa_df)} 条记录")
            
            # 标准化股票代码
            qa_df['Scode_normalized'] = qa_df['Scode'].apply(normalize_stock_code)
            
            # 计算目标交易日
            print("\n开始计算目标交易日...")
            
            # 计算提问目标日
            if 'Qtm' in qa_df.columns:
                print("计算提问目标日...")
                qa_df['Qtm'] = qa_df['Qtm'].apply(parse_datetime_with_fallback)
                print(f"Qtm解析完成，非空值数量: {qa_df['Qtm'].notna().sum()}")
                qa_df['TargetDate_Ask'] = qa_df['Qtm'].apply(
                    lambda x: get_target_trading_date(x, trade_dates, next_trading_day_map)
                )
                print(f"TargetDate_Ask计算完成，非空值数量: {qa_df['TargetDate_Ask'].notna().sum()}")
            
            # 计算回复目标日
            if 'Recvtm' in qa_df.columns:
                print("计算回复目标日...")
                qa_df['Recvtm'] = qa_df['Recvtm'].apply(parse_datetime_with_fallback)
                print(f"Recvtm解析完成，非空值数量: {qa_df['Recvtm'].notna().sum()}")
                qa_df['TargetDate_Reply'] = qa_df['Recvtm'].apply(
                    lambda x: get_target_trading_date(x, trade_dates, next_trading_day_map)
                )
                print(f"TargetDate_Reply计算完成，非空值数量: {qa_df['TargetDate_Reply'].notna().sum()}")
            
            # 收集需要匹配的股票代码和交易日
            print("收集需要匹配的股票代码和交易日...")
            stock_codes = set(qa_df['Scode_normalized'].dropna())
            print(f"股票代码数量: {len(stock_codes)}")
            target_dates = set()
            if 'TargetDate_Ask' in qa_df.columns:
                target_dates.update(qa_df['TargetDate_Ask'].dropna())
            if 'TargetDate_Reply' in qa_df.columns:
                target_dates.update(qa_df['TargetDate_Reply'].dropna())
            print(f"交易日数量: {len(target_dates)}")
            
            if not stock_codes or not target_dates:
                print("没有需要匹配的股票代码或交易日")
                continue
            
            # 加载价差数据
            spread_df = load_spread_data_for_year(year, stock_codes, target_dates)
            
            # 加载换手率数据
            turnover_dfs = []
            for y in [year, year + 1]:
                if y in turnover_data_by_year:
                    turnover_dfs.append(turnover_data_by_year[y])
                    print(f"加载 {y} 年换手率数据，共 {len(turnover_data_by_year[y])} 条记录")
            
            turnover_df = pd.concat(turnover_dfs, ignore_index=True) if turnover_dfs else None
            
            # 加载Amihud数据
            amihud_dfs = []
            for y in [year, year + 1]:
                if y in amihud_data_by_year:
                    amihud_dfs.append(amihud_data_by_year[y])
                    print(f"加载 {y} 年Amihud数据，共 {len(amihud_data_by_year[y])} 条记录")
            
            amihud_df = pd.concat(amihud_dfs, ignore_index=True) if amihud_dfs else None
            
            # 融合数据
            print("\n开始融合数据...")
            qa_df = merge_all_indicators(qa_df, spread_df, turnover_df, amihud_df)
            
            # 过滤输出字段
            print("\n过滤输出字段...")
            qa_df = filter_output_fields(qa_df)
            
            # 保存结果
            print("\n保存结果...")
            print(f"原始文件路径: {qa_file}")
            base_name = os.path.basename(qa_file)
            print(f"文件名: {base_name}")
            name_without_ext = os.path.splitext(base_name)[0]
            print(f"无扩展名文件名: {name_without_ext}")
            new_file_name = f"{name_without_ext}_with_indicators.csv"
            print(f"新文件名: {new_file_name}")
            new_file_path = os.path.join(CONFIG["OUTPUT_DIR"], new_file_name)
            print(f"新文件路径: {new_file_path}")
            print(f"输出目录是否存在: {os.path.exists(CONFIG['OUTPUT_DIR'])}")
            print(f"数据框列数: {len(qa_df.columns)}")
            print(f"数据框行数: {len(qa_df)}")
            
            try:
                qa_df.to_csv(new_file_path, index=False, encoding='utf-8-sig')
                print(f"保存完成，文件: {new_file_name}")
                print(f"文件是否存在: {os.path.exists(new_file_path)}")
                print(f"文件大小: {os.path.getsize(new_file_path) if os.path.exists(new_file_path) else '文件不存在'}")
                print(f"融合结果共 {len(qa_df)} 条记录")
            except Exception as e:
                print(f"保存文件时出错: {str(e)}")
                import traceback
                traceback.print_exc()
            
            # 统计匹配情况
            print("\n匹配情况统计:")
            indicator_columns = [
                'Esp_Amount_Ask', 'Turnover_Ask', 'ILLIQ_Ask',
                'Esp_Amount_Reply', 'Turnover_Reply', 'ILLIQ_Reply'
            ]
            for col in indicator_columns:
                if col in qa_df.columns:
                    if col.startswith('Esp_'):
                        not_na_count = (qa_df[col] != 'NA').sum()
                    else:
                        not_na_count = qa_df[col].notna().sum()
                    not_na_ratio = (not_na_count / len(qa_df)) * 100
                    print(f"{col}: {not_na_count}/{len(qa_df)} ({not_na_ratio:.2f}%)")
                    
        except Exception as e:
            print(f"处理文件 {qa_file} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            continue


def main():
    """主函数"""
    print("="*100)
    print("最终统一数据融合脚本开始执行")
    print("="*100)
    print(f"当前时间: {datetime.now()}")
    
    try:
        # 构建交易日历
        trade_dates, next_trading_day_map = build_trading_calendar()
        
        # 加载辅助数据
        turnover_data_by_year = load_turnover_data()
        amihud_data_by_year = load_amihud_data()
        
        # 按年份处理数据
        print(f"\n[{datetime.now()}] 开始按年份处理数据...")
        
        for year in CONFIG["YEARS_TO_PROCESS"]:
            print(f"\n[{datetime.now()}] 开始处理 {year} 年数据...")
            process_qa_data(year, trade_dates, next_trading_day_map, turnover_data_by_year, amihud_data_by_year)
            print(f"[{datetime.now()}] {year} 年数据处理完成")
        
        print(f"\n[{datetime.now()}] 所有数据处理完成！")
        print("="*100)
        
    except Exception as e:
        print(f"\n[{datetime.now()}] 脚本执行失败: {str(e)}")
        import traceback
        traceback.print_exc()
        print("="*100)


if __name__ == "__main__":
    main()
