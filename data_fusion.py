import pandas as pd
import os
import glob
import re
from datetime import datetime, time, timedelta
from tqdm import tqdm

# 数据路径配置
ROOT_DIR = r'D:\桌面\trae工作区\1.26数据匹配'
SPREAD_DATA_ROOT = os.path.join(ROOT_DIR, 'csmar买卖价差数据')
QA_DATA_ROOT = os.path.join(ROOT_DIR, '问答-买卖价差')
OUTPUT_DIR = os.path.join(ROOT_DIR, '数据融合结果_New')

# 子文件夹列表
SPREAD_SUB_FOLDERS = ['2018-2020', '2021-2023', '2024-2025']
QA_YEAR_FOLDERS = ['2018', '2019', '2020', '2021', '2022', '2023']

# 收盘时间定义
CLOSING_TIME = time(15, 0, 0)  # 15:00:00

# 编码尝试列表
ENCODINGS = ['utf-8', 'gbk', 'gb18030']

# 价差数据字段名
SPREAD_FIELDS = ['Esp_Amount', 'Esp_Volume', 'Esp_time']


def normalize_stock_code(stock_code):
    """标准化股票代码，确保是6位数"""
    if pd.isna(stock_code):
        return None
    # 提取纯数字部分
    digits = re.sub(r'\D', '', str(stock_code))
    # 确保6位数
    if len(digits) < 6:
        digits = digits.zfill(6)
    elif len(digits) > 6:
        digits = digits[-6:]
    return digits


def normalize_date_format(date_str):
    """标准化日期格式，将斜杠分隔转换为连字符分隔"""
    if not date_str:
        return None
    try:
        if '/' in date_str:
            parts = date_str.split('/')
            if len(parts) == 3:
                year, month, day = parts
                return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        return date_str
    except:
        return date_str


def build_trading_calendar():
    """构建交易日历和下一个交易日映射"""
    print("=" * 80)
    print("开始构建交易日历...")
    print("=" * 80)
    
    trade_dates_set = set()
    spread_files = []
    
    # 遍历所有价差数据文件
    for sub_folder in tqdm(SPREAD_SUB_FOLDERS, desc="处理价差数据文件夹"):
        folder_path = os.path.join(SPREAD_DATA_ROOT, sub_folder)
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
        if time_part <= CLOSING_TIME:
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


def process_qa_data(year, trade_dates, next_trading_day_map):
    """处理指定年份的问答数据"""
    print(f"\n开始处理 {year} 年的问答数据...")
    
    # 查找该年份的CSV文件，确保只匹配完整年份（如"2018"而不是"2018-2023"）
    qa_files = []
    for file in glob.glob(os.path.join(QA_DATA_ROOT, '*.csv')):
        basename = os.path.basename(file)
        # 检查文件名中是否包含单独的年份，而不是年份范围
        if f"（{year}）" in basename or f"({year})" in basename:
            qa_files.append(file)
    if not qa_files:
        print(f"未找到 {year} 年的问答数据文件")
        return None, 0
    
    print(f"找到 {len(qa_files)} 个 {year} 年的问答数据文件")
    
    all_qa_chunks = []
    total_records = 0
    
    for qa_file in tqdm(qa_files, desc="处理问答文件"):
        # 跳过临时文件
        if '~$' in qa_file:
            continue
        
        # 尝试不同编码
        encoding_success = False
        for encoding in ENCODINGS:
            try:
                print(f"尝试使用 {encoding} 编码读取 {qa_file}...")
                
                print(f"开始分批读取文件 {qa_file}...")
                chunk_count = 0
                # 分批读取
                for chunk in pd.read_csv(qa_file, encoding=encoding, chunksize=100000):
                    chunk_count += 1
                    chunk_records = len(chunk)
                    total_records += chunk_records
                    
                    print(f"  批次 {chunk_count}: 读取 {chunk_records} 条记录，累计 {total_records} 条记录")
                    print(f"  当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    # 1. 删除旧的日期相关字段和旧的Esp字段
                    old_date_columns = ['提问日期', '回答日期', '目标交易日_提问', '目标交易日_回答']
                    old_esp_columns = []
                    # 找出所有以Esp_开头的旧字段（不包括新的带后缀的字段）
                    for col in chunk.columns:
                        if col.startswith('Esp_') and not (col.endswith('_Ask') or col.endswith('_Reply')):
                            old_esp_columns.append(col)
                    # 删除所有旧字段
                    chunk = chunk.drop(columns=[col for col in old_date_columns + old_esp_columns if col in chunk.columns])
                    print(f"  已删除旧字段")
                    
                    # 2. 股票代码标准化
                    if 'Scode' in chunk.columns:
                        chunk['Scode_normalized'] = chunk['Scode'].apply(normalize_stock_code)
                    print(f"  已完成股票代码标准化")
                    
                    # 3. 转换时间字段为datetime类型
                    if 'Qtm' in chunk.columns:
                        chunk['Qtm'] = pd.to_datetime(chunk['Qtm'], errors='coerce')
                        # 4. 计算目标交易日_提问
                        chunk['TargetDate_Ask'] = chunk['Qtm'].apply(
                            lambda x: get_target_trading_date(x, trade_dates, next_trading_day_map)
                        )
                    print(f"  已完成提问时间处理")
                    
                    if 'Recvtm' in chunk.columns:
                        chunk['Recvtm'] = pd.to_datetime(chunk['Recvtm'], errors='coerce')
                        # 4. 计算目标交易日_回答
                        chunk['TargetDate_Reply'] = chunk['Recvtm'].apply(
                            lambda x: get_target_trading_date(x, trade_dates, next_trading_day_map)
                        )
                    print(f"  已完成回答时间处理")
                    
                    all_qa_chunks.append(chunk)
                    print(f"  已将该批次数据添加到列表")
                
                encoding_success = True
                print(f"使用 {encoding} 编码读取成功")
                break
            except UnicodeDecodeError:
                print(f"使用 {encoding} 编码读取失败，尝试下一种编码...")
                continue
            except Exception as e:
                print(f"读取文件 {qa_file} 时出错: {str(e)}")
                break
        
        if not encoding_success:
            print(f"所有编码都尝试失败，跳过文件 {qa_file}")
    
    if not all_qa_chunks:
        print(f"未成功读取任何 {year} 年的问答数据")
        return None, 0
    
    # 合并所有chunk
    qa_df = pd.concat(all_qa_chunks, ignore_index=True)
    print(f"{year} 年问答数据处理完成，共 {len(qa_df)} 条记录")
    
    return qa_df, total_records


def load_spread_data_for_year(year, spread_files, stock_codes, target_dates):
    """加载指定年份及相关的价差数据"""
    print(f"开始加载 {year} 年相关的价差数据...")
    
    all_spread_chunks = []
    
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


def merge_spread_data(qa_df, spread_df, merge_on_date, suffix):
    """合并价差数据
    
    Args:
        qa_df: 问答数据
        spread_df: 价差数据
        merge_on_date: 合并使用的日期字段名
        suffix: 字段后缀，如"_Ask"或"_Reply"
        
    Returns:
        pd.DataFrame: 合并后的数据
    """
    if spread_df is None:
        # 如果没有价差数据，添加空的价差字段
        for field in SPREAD_FIELDS:
            qa_df[f'{field}{suffix}'] = 'NA'
        return qa_df
    
    # 准备价差数据，只保留需要的字段
    spread_data = spread_df[['Stkcd_normalized', 'Trddt_normalized'] + SPREAD_FIELDS].copy()
    
    # 重命名价差字段，添加后缀
    rename_map = {}
    for field in SPREAD_FIELDS:
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
    for field in SPREAD_FIELDS:
        merged_df[f'{field}{suffix}'] = merged_df[f'{field}{suffix}'].fillna('NA')
    
    return merged_df


def main():
    """主函数"""
    # 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"创建输出目录: {OUTPUT_DIR}")
    
    print("=" * 100)
    print("开始执行数据融合任务")
    print("=" * 100)
    
    # 1. 构建交易日历
    trade_dates, next_trading_day_map = build_trading_calendar()
    
    # 2. 遍历年份处理数据
    for i, year in enumerate(QA_YEAR_FOLDERS):
        print(f"\n{'=' * 100}")
        print(f"[{i+1}/{len(QA_YEAR_FOLDERS)}] 开始处理 {year} 年数据")
        print(f"{'=' * 100}")
        print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # 3. 处理问答数据
            qa_df, total_records = process_qa_data(year, trade_dates, next_trading_day_map)
            if qa_df is None:
                continue
            
            # 4. 收集需要匹配的股票代码和交易日
            if 'Scode_normalized' in qa_df.columns:
                stock_codes = set(qa_df['Scode_normalized'].dropna())
            else:
                print("问答数据中没有股票代码字段，无法匹配价差数据")
                continue
            
            target_dates = set()
            if 'TargetDate_Ask' in qa_df.columns:
                target_dates.update(qa_df['TargetDate_Ask'].dropna())
            if 'TargetDate_Reply' in qa_df.columns:
                target_dates.update(qa_df['TargetDate_Reply'].dropna())
            
            if not stock_codes or not target_dates:
                print("没有需要匹配的股票代码或交易日")
                continue
            
            # 5. 加载价差数据
            # 先找到所有价差文件
            spread_files = []
            for sub_folder in SPREAD_SUB_FOLDERS:
                folder_path = os.path.join(SPREAD_DATA_ROOT, sub_folder)
                csv_files = glob.glob(os.path.join(folder_path, 'HF_Spread*.csv'))
                spread_files.extend(csv_files)
            
            spread_df = load_spread_data_for_year(year, spread_files, stock_codes, target_dates)
            
            # 6. 合并价差数据
            print("开始合并价差数据...")
            
            # 合并提问时间的价差
            if 'TargetDate_Ask' in qa_df.columns:
                qa_df = merge_spread_data(qa_df, spread_df, 'TargetDate_Ask', '_Ask')
            
            # 合并回答时间的价差
            if 'TargetDate_Reply' in qa_df.columns:
                qa_df = merge_spread_data(qa_df, spread_df, 'TargetDate_Reply', '_Reply')
            
            print("价差数据合并完成")
            
            # 7. 输出结果
            print(f"开始输出 {year} 年结果...")
            
            # 7.1 输出CSV文件
            csv_output_file = os.path.join(OUTPUT_DIR, f'问答-买卖价差融合数据集（{year}）.csv')
            print(f"输出CSV文件: {csv_output_file}")
            
            # 分批写入
            batch_size = 100000
            for i in tqdm(range(0, len(qa_df), batch_size), desc="写入CSV文件"):
                batch_end = min(i + batch_size, len(qa_df))
                batch_df = qa_df.iloc[i:batch_end]
                mode = 'w' if i == 0 else 'a'
                header = i == 0
                batch_df.to_csv(csv_output_file, mode=mode, header=header, index=False, encoding='utf-8-sig')
            
            # 7.2 生成数据处理报告
            report_file = os.path.join(OUTPUT_DIR, f'数据处理报告（{year}）.txt')
            print(f"生成数据处理报告: {report_file}")
            
            # 计算缺失比例
            missing_ratios = {}
            spread_columns = []
            for suffix in ['_Ask', '_Reply']:
                spread_columns.extend([f'{field}{suffix}' for field in SPREAD_FIELDS])
            
            for col in spread_columns:
                if col in qa_df.columns:
                    missing_count = (qa_df[col] == 'NA').sum()
                    missing_ratio = (missing_count / len(qa_df)) * 100
                    missing_ratios[col] = missing_ratio
            
            # 生成报告内容
            report_content = f"""{year} 年数据处理报告

1. 数据读取情况
- 问答记录总数: {len(qa_df)}
- 匹配的股票代码数量: {len(stock_codes)}
- 匹配的交易日数量: {len(target_dates)}

2. 价差字段缺失比例
"""
            
            for col, ratio in missing_ratios.items():
                report_content += f"- {col}: {ratio:.2f}%\n"
            
            # 写入报告
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            print(f"{year} 年数据处理完成！")
            
        except Exception as e:
            print(f"处理 {year} 年数据时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            print(f"跳过 {year} 年的数据处理")
            continue
    
    print("\n所有年份的数据处理完成！")


if __name__ == "__main__":
    main()
