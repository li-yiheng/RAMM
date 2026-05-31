import os
import torch
import numpy as np
from PIL import Image
import json
from tqdm import tqdm
import copy

            similar_ref_samples_info.append({
                "index": original_ref_index,
                "id": ref_item_id,
                "score": float(score_val)
            })

        results.append({
            "query_sample": {
                "index": original_query_index,

from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline


print("正在使用 ModelScope 初始化 pipeline...")
pipeline = pipeline(task=Tasks.multi_modal_embedding,
    model='iic/multi-modal_clip-vit-large-patch14_336_zh', model_revision='v1.0.1')

print("ModelScope pipeline 初始化完成。")



def get_embeddings(data_list, desc=""):
    embeddings = []
    image_base_path = "/mnt/workspace/"
    #image_base_path = "/root/autodl-tmp/"

    for item in tqdm(data_list, desc=desc):
        text = item["text"]
        image_filename = item["image"]
        image_path = os.path.join(image_base_path, str(image_filename))
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"警告: 无法打开图片 {image_path}, 将跳过此样本。错误: {e}")
            embeddings.append(None) # 使用 None 作为占位符
            continue

        with torch.no_grad():
            img_embedding = pipeline.forward({'img': image})['img_embedding']
            text_embedding = pipeline.forward({'text': [text]})['text_embedding']

        image_features_np = img_embedding.squeeze().cpu().numpy()
        text_features_np = text_embedding.squeeze().cpu().numpy()


        embeddings.append((image_features_np, text_features_np))
        
    return embeddings


def calculate_and_save_similarity(query_data, ref_data, output_filename, k=10):

    print(f"\n--- 正在处理: {output_filename} ---")
    
    print("正在为查询集生成特征嵌入...")
    query_embeddings_list = get_embeddings(query_data, "查询集嵌入")

    print("正在为参考集生成特征嵌入...")
    ref_embeddings_list = get_embeddings(ref_data, "参考集嵌入")
    
    # --- 数据预处理，分离有效样本和模态特征 ---
    valid_query_indices = [i for i, emb in enumerate(query_embeddings_list) if emb is not None]
    valid_ref_indices = [i for i, emb in enumerate(ref_embeddings_list) if emb is not None]
    
    if not valid_query_indices or not valid_ref_indices:
        print(f"错误: 未能生成有效的特征嵌入，无法继续。退出。")
        return
        

    query_image_embeddings = np.array([query_embeddings_list[i][0] for i in valid_query_indices])
    query_text_embeddings = np.array([query_embeddings_list[i][1] for i in valid_query_indices])
    
    ref_image_embeddings = np.array([ref_embeddings_list[i][0] for i in valid_ref_indices])
    ref_text_embeddings = np.array([ref_embeddings_list[i][1] for i in valid_ref_indices])


    print("正在独立计算各模态的相似度分数...")

    image_similarity_scores = np.dot(query_image_embeddings, ref_image_embeddings.T)
    

    text_similarity_scores = np.dot(query_text_embeddings, ref_text_embeddings.T)

    print("正在融合各模态的相似度分数...")

    weight_image = 0.5
    weight_text = 0.5
    final_similarity_scores = (weight_image * image_similarity_scores) + (weight_text * text_similarity_scores)
    

    similarity_scores_copy = copy.deepcopy(final_similarity_scores)

    results = []
    print(f"正在提取 Top-{k} 相似样本...")
    for i in tqdm(range(len(valid_query_indices)), desc="Top-K 提取"):
        is_self_comparison = (id(query_data) == id(ref_data))
        
        current_scores = similarity_scores_copy[i]


        if is_self_comparison:
            original_query_index = valid_query_indices[i]
            if original_query_index in valid_ref_indices:
                self_idx_in_valid_list = valid_ref_indices.index(original_query_index)
                current_scores[self_idx_in_valid_list] = -2.0 # 设置一个极小值以排除


        k_to_fetch = min(k, len(current_scores))
        top_k_indices = np.argpartition(current_scores, -k_to_fetch)[-k_to_fetch:]
        top_k_scores = current_scores[top_k_indices]
        

        sorted_order = np.argsort(top_k_scores)[::-1]
        samples_indices_in_valid_list = top_k_indices[sorted_order]
        scores = top_k_scores[sorted_order]
        
        original_query_index = valid_query_indices[i]
        query_item_id = query_data[original_query_index]['id']
        
        similar_ref_samples_info = []
        for j, score_val in zip(samples_indices_in_valid_list, scores):
            original_ref_index = valid_ref_indices[j]
            ref_item_id = ref_data[original_ref_index]['id']
                "id": query_item_id
            },
            "similar_ref_samples": similar_ref_samples_info,
        })

    with open(output_filename, "w", encoding='utf-8') as f:
        for result_item in results:
            json.dump(result_item, f, ensure_ascii=False)
            f.write("\n")
            
    print(f"结果已成功保存至: {output_filename}")



if __name__ == "__main__":
    test_json_path = "/mnt/workspace/test_datas.json"
    train_json_path = "/mnt/workspace/train_datas.json"
    
    try:
        print("正在加载数据集...")
        with open(test_json_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        with open(train_json_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        print("数据集加载完成。")
    except Exception as e:
        print(f"错误: 加载数据文件失败。请检查路径是否正确。错误信息: {e}")
        exit()

    os.makedirs("SSR_PaperMethod", exist_ok=True) 
    K_VALUE = 6
    

    output_file_v1 = "SSR/weight-weibo_PaperMethod_CLIP_train_vs_train.jsonl"
    calculate_and_save_similarity(
        query_data=train_data,
        ref_data=train_data,
        output_filename=output_file_v1,
        k=K_VALUE
    )


    output_file_v2 = "SSR/weight-weibo_PaperMethod_CLIP_test_vs_train.jsonl"
    calculate_and_save_similarity(
        query_data=test_data,
        ref_data=train_data,
        output_filename=output_file_v2,
        k=K_VALUE
    )

    print("\n所有任务已完成！")
