import jieba
import re
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from gensim.corpora import Dictionary
from gensim.similarities import SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex, SoftCosineSimilarity
from tqdm import tqdm

# 检查并安装依赖
import subprocess
import sys

def install_dependencies():
    dependencies = ['jieba', 'tqdm', 'pandas', 'numpy', 'gensim', 'openpyxl']
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✓ {dep} 已安装")
        except ImportError:
            print(f"✗ {dep} 未安装，正在安装...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', dep])
            print(f"✓ {dep} 安装成功")

# 安装依赖
install_dependencies()


# 文本预处理

def preprocess(text, stopwords=None):
    if not isinstance(text, str):
        return []
    text = re.sub(r'[^\u4e00-\u9fa5]+', '', text)  # 保留中文
    words = jieba.lcut(text)
    if stopwords:
        words = [w for w in words if w not in stopwords and len(w.strip()) > 0]
    return words


# 优化的批量软余弦相似度计算
def compute_soft_cosine_batch(q_list_words, a_list_words, model, batch_size=1000):
    """
    批量计算软余弦相似度，避免重复创建字典和相似度矩阵
    :param q_list_words: 提问词列表的列表，如[[word1, word2], [word3, word4], ...]
    :param a_list_words: 回复词列表的列表，如[[word1, word2], [word3, word4], ...]
    :param model: 预训练词向量模型
    :param batch_size: 内部批次大小，进一步优化内存使用
    :return: 相似度分数列表
    """
    if not q_list_words or not a_list_words:
        return [np.nan] * len(q_list_words)
    
    # 创建全局字典，包含所有问答对中的单词
    all_words = q_list_words + a_list_words
    dictionary = Dictionary(all_words)
    
    # 创建相似度索引和矩阵（只创建一次）
    similarity_index = WordEmbeddingSimilarityIndex(model)
    similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dictionary)
    
    # 将所有问答转换为词袋模型
    print("转换为词袋模型...")
    q_bows = [dictionary.doc2bow(q_words) for q_words in tqdm(q_list_words, desc="提问词袋转换")]
    a_bows = [dictionary.doc2bow(a_words) for a_words in tqdm(a_list_words, desc="回复词袋转换")]
    
    # 计算相似度：进一步分小批次处理，减少内存占用
    scores = [np.nan] * len(q_bows)
    
    for i in tqdm(range(0, len(q_bows), batch_size), desc="相似度计算批次"):
        end_idx = min(i + batch_size, len(q_bows))
        q_batch_bows = q_bows[i:end_idx]
        a_batch_bows = a_bows[i:end_idx]
        
        for j in range(len(q_batch_bows)):
            q_bow = q_batch_bows[j]
            a_bow = a_batch_bows[j]
            
            if not q_bow or not a_bow:
                continue
            
            try:
                # 每次只需要创建一次SoftCosineSimilarity实例
                soft_cosine = SoftCosineSimilarity([a_bow], similarity_matrix)
                score = soft_cosine[q_bow][0]
                scores[i + j] = float(score)
            except Exception as e:
                print(f"计算相似度时出错：{e}")
                scores[i + j] = np.nan
    
    return scores

# 保留原函数用于兼容
def compute_soft_cosine(q_words, a_words, model):
    return compute_soft_cosine_batch([q_words], [a_words], model)[0]


# 优化的批量计算互动质量
def compute_interaction_quality_from_excel(
        excel_path,
        model_path,
        stopwords_path,
        output_path=None,
        batch_size=10000,  # 批次大小，可根据内存调整
        sample_size=None  # 可选：只处理前N条数据用于测试
):
    # 自动生成输出路径，避免覆盖原始文件
    if output_path is None:
        import os
        base_name, ext = os.path.splitext(excel_path)
        output_path = f"{base_name}_带相似度{ext}"
    
    # 读取问答数据
    print("正在读取Excel数据...")
    df = pd.read_excel(excel_path, header=1)
    print(f"✓ 成功读取数据，共{len(df)}条记录")
    
    # 如果指定了样本大小，只处理前N条数据
    if sample_size:
        df = df.head(sample_size)
        print(f"✓ 已选择前{sample_size}条数据进行处理")

    # 停用词加载
    stopwords = set()
    if stopwords_path:
        with open(stopwords_path, "r", encoding="utf-8") as f:
            stopwords = set([w.strip() for w in f.readlines()])
        print(f"成功加载停用词，共{len(stopwords)}个")

    # 加载词向量模型（添加进度显示）
    print("正在加载词向量模型...这可能需要几分钟")
    
    # 尝试使用更高效的方式加载模型
    import os
    model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    print(f"模型大小: {model_size:.2f} MB")
    
    # 加载模型并显示进度
    try:
        model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        print(f"✓ 词向量模型加载完成！共包含 {len(model.key_to_index)} 个词向量")
    except Exception as e:
        print(f"✗ 加载模型失败: {e}")
        raise

    # 文本预处理：使用apply替代iterrows，提高效率
    print("开始文本预处理...")
    df["q_text"] = df["提问内容"].astype(str).str.strip()
    df["a_text"] = df["回复内容"].astype(str).str.strip().fillna("")
    
    # 使用向量化操作进行文本预处理
    def preprocess_vectorized(text):
        if not isinstance(text, str) or text.strip() == "":
            return []
        text = re.sub(r'[^\u4e00-\u9fa5]+', '', text)
        words = jieba.lcut(text)
        if stopwords:
            words = [w for w in words if w not in stopwords and len(w.strip()) > 0]
        return words
    
    # 导入tqdm，确保在非notebook环境下也能正常工作
    try:
        from tqdm.notebook import tqdm as tqdm_notebook
        from tqdm import tqdm
    except ImportError:
        from tqdm import tqdm
        tqdm_notebook = tqdm
    
    # 使用apply进行预处理
    df["q_words"] = df["q_text"].apply(preprocess_vectorized)
    df["a_words"] = df["a_text"].apply(preprocess_vectorized)
    
    # 筛选有效数据
    valid_mask = (df["q_words"].str.len() > 0) & (df["a_words"].str.len() > 0) & (df["a_text"] != "")
    print(f"有效问答数据：{valid_mask.sum()}条，无效数据：{(~valid_mask).sum()}条")
    
    # 计算相似度：分批次处理，减少内存占用
    print("开始计算相似度...")
    scores = [np.nan] * len(df)
    
    if valid_mask.sum() > 0:
        # 获取有效数据
        valid_indices = df[valid_mask].index
        q_valid_words = df.loc[valid_indices, "q_words"].tolist()
        a_valid_words = df.loc[valid_indices, "a_words"].tolist()
        
        # 分批次计算
        if batch_size is None or batch_size >= len(valid_indices):
            # 单次计算
            valid_scores = compute_soft_cosine_batch(q_valid_words, a_valid_words, model)
        else:
            # 分批次计算
            valid_scores = []
            for i in tqdm(range(0, len(valid_indices), batch_size), desc="批次处理"):
                end_idx = min(i + batch_size, len(valid_indices))
                q_batch = q_valid_words[i:end_idx]
                a_batch = a_valid_words[i:end_idx]
                batch_scores = compute_soft_cosine_batch(q_batch, a_batch, model)
                valid_scores.extend(batch_scores)
        
        # 将计算结果赋值回原列表
        for idx, score in zip(valid_indices, valid_scores):
            scores[idx] = score
    
    # 写入结果列
    df["SoftCosineSimilarity"] = scores
    
    # 清理临时列
    df = df.drop(["q_text", "a_text", "q_words", "a_words"], axis=1)
    
    # 保存结果
    print("保存结果中...")
    df.to_excel(output_path, index=False, engine='openpyxl')
    print(f"计算完成！结果已保存到：{output_path}")
    print(f"相似度统计：平均值={df['SoftCosineSimilarity'].mean():.4f}, 中位数={df['SoftCosineSimilarity'].median():.4f}, 最大值={df['SoftCosineSimilarity'].max():.4f}, 最小值={df['SoftCosineSimilarity'].min():.4f}")

    return df


# 调用
if __name__ == "__main__":
    # 使用原始字符串避免路径解析错误
    excel_path = r"d:\桌面\trae工作区\互动文本\用户与公司问答\2022\用户与公司问答-2022_1.xlsx"
    model_path = r"d:\桌面\trae工作区\互动文本\用户与公司问答\light_Tencent_AILab_ChineseEmbedding.bin"  # 替换为你的中文词向量模型路径
    stopwords_path = r"d:\桌面\trae工作区\互动文本\用户与公司问答\Chinese_stopwords.txt"  # 可选

    result_df = compute_interaction_quality_from_excel(excel_path, model_path, stopwords_path)
