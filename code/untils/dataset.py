import os
from typing import Iterable, Sequence, Any, Dict, List, Tuple, Union
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
from tqdm import tqdm
import json, re
import torch.nn.functional as F
from torch import Tensor
import transformers
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, TextIteratorStreamer, AutoTokenizer, AutoModel
from modelscope import Model
from swift.tuners import Swift
from untils.functions import refine_checkpoint_path

from swift.llm import get_model_tokenizer, safe_snapshot_download

os.environ['USE_HF'] = 'True'

# Set seeds for reproducibility at module level
SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Enable deterministic algorithms (may impact performance)
torch.use_deterministic_algorithms(True, warn_only=True)


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def augment_ent(*, data_dir, output_dir, model_dir, seed=SEED):
    # Set seeds for this function call
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    data_dir = data_dir
    output_dir = output_dir
    with open(data_dir, "r") as f:
        entity = json.load(f)
    entity_name = list(entity.keys())
    entity_des = list(entity.values())
    data_name = entity_name
    data_des = entity_des

    model_dir = model_dir
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_dir,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    
    system = 'you are a helpful assistant!'
    PROMPT = """Please generate a one-sentence summary for the given entity, including entity name and description.
    entity name:{entity_name}
    entity description:{entity_des}
    Try your best to summarize the main content of the given entity. And generate a short summary in 1 sentence for it.
    Summary:
    """
    ent = []
    try:
        with open(output_dir, "r") as f:
            now_data = json.load(f)
        ent = now_data
    except:
        print("重新创建文件")

    all_data_name = []
    if len(ent) > 0:
        for da in ent:
            if da['ids'] not in all_data_name:
                all_data_name.append(da['ids'])
    writes = 0
    for i in tqdm(range(len(entity_name)), desc="Generating summaries for entities"):
        if data_name[i] in all_data_name:
            continue
        writes += 1
        if writes % 100 == 0:
            with open(output_dir, "w") as f:
                json.dump(ent, f)
        dict_data = {}
        dict_data['ids'] = data_name[i]
        dict_data['des'] = data_des[i]
        text = PROMPT.format(entity_name=data_name[i], entity_des=data_des[i])
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ]
        prompt = pipeline.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
            )
        terminators = [
                pipeline.tokenizer.eos_token_id,
                pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
        try:
            outputs = pipeline(
                prompt,
                max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                pad_token_id=128001
            )
            output = outputs[0]["generated_text"][len(prompt):]
            summary = output
            summary = summary.split(":")[-1]
            # print(summary)
            dict_data['sum'] = summary.replace('\n', '')
            ent.append(dict_data)
        except Exception as e:
            print("error!" + str(i) + ": " + str(e))

    with open(output_dir, "w") as f:
        json.dump(ent, f)


def run_emb(*, model_dir, data_dir, embed_dir, max_length, seed=SEED):
    # Set seeds for this function call
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    model_dir = model_dir
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)
    model.to("cuda")
    max_length = max_length
    data_dir = data_dir
    embed_dir = embed_dir
    with open(data_dir, "r") as f:
        data = json.load(f)
    ents = data
    
    embeds = []
    existing_ids = set()
    try:
        with open(embed_dir, "r") as f:
            existing_embeds = json.load(f)
        embeds = existing_embeds
        for item in existing_embeds:
            if 'ids' in item:
                existing_ids.add(item['ids'])
    except:
        pass

    for j, ent in enumerate(tqdm(ents, desc="Generating embeddings for entities")):
        if ent['ids'] in existing_ids:
            continue
        embed = {}
        embed['ids'] = ent['ids']
        text = ent['ids'] + ":" + ent['sum']
        text = text.replace("\n", " ")
        input_texts = text
        batch_dict = tokenizer(input_texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt").to("cuda")
        outputs = model(**batch_dict)
        em = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])[0]  # shape: (hidden_size,)
        embed['emb'] = em.tolist()
        embeds.append(embed)
    with open(embed_dir, "w") as f:
        json.dump(embeds, f)


def augment_men_img(*, mentions_dir, save_dir, model_id, image_dir, img_file_name_mapping, seed=SEED):
    # Set seeds for this function call
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    mentions_dir = mentions_dir
    save_dir = save_dir
    with open(mentions_dir, "r") as f:
        mentions = json.load(f)

    img_mapping = dict()
    if img_file_name_mapping is not None:
        with open(img_file_name_mapping, "r") as f:
            img_mapping = json.load(f)  # mapping of image file names to shorter file names

    model_id = model_id
    processor = LlavaNextProcessor.from_pretrained(model_id)

    model = LlavaNextForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.to("cuda")

    image_dir = image_dir
    PROMPT = """
    The image describes \"{mention_context}\"
    Introduce the image. Answer follow the format: "The image refers to..."
    """

    current_descriptions_dict = {}
    try:
        with open(save_dir, "r") as f:
            current_descriptions = json.load(f)
        for o in tqdm(current_descriptions, desc="Loading existing descriptions"):
            if 'des_llava' in o:
                current_descriptions_dict[o['ids']] = o['des_llava']
    except:
        pass

    for i in tqdm(range(len(mentions)), desc="Generating descriptions for mentions with images"):
        id_mention = mentions[i]['ids']
        cur_desc = current_descriptions_dict.get(id_mention, None)
        if cur_desc:
            mentions[i]['des_llava'] = cur_desc
        else:
            prompt = f"[INST] <image>\n{PROMPT.format(mention_context=mentions[i]['context'])} [/INST]"
            im_dir = image_dir + "/" + img_mapping.get(mentions[i]['image'], mentions[i]['image'])
            if os.path.exists(im_dir):
                try:
                    image = Image.open(im_dir).convert("RGB")
                    inputs = processor(prompt, image, return_tensors="pt").to("cuda")
                except Exception as e:
                    print("error:" + im_dir + ": " + str(e))
                    continue
            else:
                continue
            output = model.generate(**inputs, max_new_tokens=100).to("cuda")
            
            resp = processor.decode(output[0], skip_special_tokens=True)
            mentions[i]['des_llava'] = resp
    with open(save_dir, "w") as f:
        json.dump(mentions, f)


def augment_men_text(*, data_dir, output_dir, model_dir, seed=SEED):
    # Set seeds for this function call
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    data_dir = data_dir
    output_dir = output_dir
    with open(data_dir, "r") as f:
        entity = json.load(f)

    model_dir = model_dir
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_dir,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    
    system = 'you are a helpful assistant!'
    PROMPT = """Please make a brief description in 1 sentence for the entity under the background of context. 

    ### Entity
    Context:{mention_context}

    \# Description (Describe the entity without limiting or referring to context.)
    """

    ent = []
    # try:
    #     with open(output_dir,"r") as f:
    #         now_data = json.load(f)
    #     ent = now_data
    # except:
    #     print("重新创建文件")

    for i in tqdm(range(len(entity)), desc="Generating descriptions for entities from text if llava description is not available"):
        try:
            llava = entity[i]['des_llava']
            # if llava is not empty, use it as the description
        except:
            text = PROMPT.format(mention_context=entity[i]['context'])
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ]
            prompt = pipeline.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                )
            terminators = [
                    pipeline.tokenizer.eos_token_id,
                    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]
            try:
                outputs = pipeline(
                    prompt,
                    max_new_tokens=256,
                    eos_token_id=terminators,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    pad_token_id=128001
                )
                output = outputs[0]["generated_text"][len(prompt):]
                des = output
                entity[i]['des'] = des.replace('\n', '')
                # print(des)
            except Exception as e:
                print("error!" + str(i) + ": " + str(e))
        ent.append(entity[i])
    with open(output_dir, "w") as f:
        json.dump(ent, f)


def runtopK(*, K, model_dir, database_emb, database_sum, mention_dir, mention_topK_dir, max_length, seed=SEED):
    # Set seeds for this function call
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    model_dir = model_dir
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)
    model.to("cuda")
    ents = []
    database_emb = database_emb
    with open(database_emb, "r") as f:
        data = json.load(f)
    ents += data
    di = {}
    ent_ids = []
    ent_embs = []
    for ent in ents:
        di[ent['ids']] = ent['emb']
        ent_ids.append(ent['ids'])
        ent_embs.append(ent['emb'])
    ent_embs = torch.tensor(ent_embs).to("cuda")  # shape: (num_entities, embedding_dim)
    ents2 = []

    database_sum = database_sum
    with open(database_sum, "r") as f:
        data = json.load(f)
    ents2 += data

    already_computed = dict()
    try:
        with open(mention_topK_dir, "r") as f:
            existing_mentions = json.load(f)
        for mention in existing_mentions:
            if 'new_cands' in mention:
                already_computed[mention['ids']] = mention
    except:
        pass

    mention_dir = mention_dir
    with open(mention_dir, "r") as f:
        mentions = json.load(f)

    max_length = max_length
    K = K
    for i in tqdm(range(len(mentions)), desc=f"Calculating top K candidates for mentions in {mention_dir}"):
        # name = mentions[i]['name']
        # context = mentions[i]['context']
        if mentions[i]['ids'] in already_computed:
            mentions[i] = already_computed[mentions[i]['ids']]
        else:
            try:
                description = mentions[i].get('des_llava', mentions[i]['des']) # it should use 'des_llava' if available, otherwise fallback to 'des'. If none of them exist, it will raise an exception and exit.
            except:
                # description = mentions[i]['des']
                print(f"Warning: 'des_llava' not found for mention {mentions[i]}. Exiting.")
                exit(1)
            text = description
            # text = context + "\n"+ name
            input_texts = text
            batch_dict = tokenizer(input_texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt").to("cuda")
            outputs = model(**batch_dict)
            mention_emb = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])[0].tolist()
            mention_emb = torch.tensor(mention_emb).to("cuda")  # shape: (embedding_dim,)
            # cands_scores = []
            # for cand in mentions[i]['cands']:
            #     entity_emb = di[cand]
            #     entity_emb = torch.tensor(entity_emb)
            #     score = torch.dot(mention_emb,entity_emb)
            #     cands_scores.append(score)
            ent_scores = torch.matmul(mention_emb.unsqueeze(0), ent_embs.T).squeeze(0)  # shape: (num_entities,)
            # mentions[i]['score'] = cands_scores.tolist()
            _, idx = torch.topk(ent_scores, min(K, len(ent_scores)))
            idx = idx.tolist()
            new_cands = []
            new_cand_scores = []
            for id in idx:
                new_cands.append(ent_ids[id])
                new_cand_scores.append(ent_scores[id].item())
            mentions[i]['new_cands'] = new_cands
            mentions[i]['score'] = new_cand_scores

        if (i+1) % 100 == 0:
            with open(mention_topK_dir, "w") as f:
                json.dump(mentions, f)
    with open(mention_topK_dir, "w") as f:
        json.dump(mentions, f)

    # with open(mention_topK_dir,"r") as f:
    #     mentions = json.load(f)
    # acc =0 
    # wrong_list = []
    # for i in tqdm(range(len(mentions)), desc=f"Calculating accuracy for top K candidates in {mention_topK_dir}"):
    #     if mentions[i]['ids'] in mentions[i]['new_cands']:
    #         acc+=1
    # print(acc/len(mentions))


def infer(*, model_id, ckpt_id, max_length, database_sum, mention_topK_dir, res_output_dir, seed=SEED):
    # Set seeds for this function call
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    device = "cuda"
    model_id = model_id
    ckpt_id = ckpt_id

    print(f"Refining checkpoint path for {ckpt_id}...")
    ckpt_id = refine_checkpoint_path(ckpt_id).as_posix()
    print(f"Using refined checkpoint path: {ckpt_id}")

    checkpoint_lora = safe_snapshot_download(ckpt_id)

    model, tokenizer = get_model_tokenizer(model_id)

    model = Swift.from_pretrained(model, checkpoint_lora, inference_mode=True, max_length=max_length)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        model_kwargs={"torch_dtype": torch.bfloat16},
        tokenizer=tokenizer
    )
    
    ents2 = []
    database = database_sum
    with open(database, "r") as f:
        data = json.load(f)
    ents2 += data

    di2 = {}
    for ent in ents2:
        ent_dict = {}
        ent_dict['name'] = ent['ids']
        ent_dict['sum'] = ent['sum']
        di2[ent['ids']] = ent_dict

    with open(mention_topK_dir, "r") as f:
        mentions = json.load(f)

    PROMPT = """
    You are an expert in knowledge graph, and matching at top k specifically. Your task is to create matches between mention and entity tables to select the best-matched entities to match the given mention. 
    ###Mention
    Context: {mention_context}
    Description: {mention_des}

    ###Entity table
    0. {entity_0}
    1. {entity_1}
    2. {entity_2}
    3. {entity_3}
    4. {entity_4}

    Just give the serial number and do not give me any other information.
    The most matched serial number is:
    """

    acc = 0
    pred = []
    truth = []
    bad_cases = []
    for i in tqdm(range(len(mentions)), desc=f"Generating predictions for mentions in {mention_topK_dir}"):
        entity_table = ["", "", "", "", ""]
        cands = mentions[i]['new_cands']
        try:
            true = cands.index(mentions[i]['ids'])
        except:
            true = -1
        for idx, ca in enumerate(cands):
            try:
                ent_str = di2[ca]['name'] + ": " + di2[ca]['sum']
                entity_table[idx] = ent_str
            except:
                continue
        try:
            description = mentions[i]['des_llava']
        except:
            description = mentions[i]['des']
        text = PROMPT.format(mention_context=mentions[i]['context'], mention_des=description, entity_0=entity_table[0], entity_1=entity_table[1], entity_2=entity_table[2], entity_3=entity_table[3], entity_4=entity_table[4])
        # outputs = pipeline(text)
        # response = outputs[0]["generated_text"][len(text):]
        # pred.append(response)
        # truth.append(true)
        # bad_cases.append(text)
    # for text in tqdm(bad_cases):
        messages = [
            {"role": "system", "content": 'you are a helpful assistant!'},
            {"role": "user", "content": text},]
        prompt = pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = pipeline(
            prompt,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.9,
            top_p=0.5,
            pad_token_id=128001
        )
        response = outputs[0]["generated_text"][len(prompt):]
        # print(response)
        res_object = {}
        res_object['ids'] = mentions[i]['ids']
        res_object['pred'] = response
        pred.append(res_object)

    with open(res_output_dir, "w") as f:
        json.dump(pred, f)


def eval(*, inference_res, topk_file, mentions_file, save_reranked=None, ks=(1, 5, 10, 30, 50, 100)):
    """Evaluate re-ranked top-K using inference results.

    Parameters
    - inference_res: path to inference results file (list of {"ids","pred"})
    - topk_file: path to the runtopK output (mentions with `new_cands`)
    - mentions_file: original mentions file (contains ground-truth in `ids`)
    - save_reranked: optional path to save re-ranked results (json)
    - ks: tuple of recall cutoffs to compute

    Behavior:
    - Parse inference `pred` values. If `pred` contains Q-ids (Q\d+), use them as predicted entities.
      Otherwise, extract integers and treat them as indices into the `new_cands` list (0-based).
    - Place the inferred predictions on top in the inferred order, then append the remaining
      candidates from the original `new_cands` preserving their order. Truncate to original K.
    - If ground-truth entity is not found in the final ranking, treat its rank as 101.
    - Returns a dict with recall@k, MRR and MAP.
    """
    missing_rank = 101
    def iter_mentions(obj: Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]
                          ) -> Iterable[Tuple[str, Dict[str, Any]]]:
        return obj.items() if isinstance(obj, dict) else ((m.get("ids"), m) for m in obj)

    def parse_predictions(raw: Any, candidates: Sequence[str]) -> List[str]:
        if not raw:
            return []
        if isinstance(raw, list):
            seq = [str(x) for x in raw]
        else:
            s = str(raw)
            qids = re.findall(r"Q\d+", s)
            if qids:
                seq = qids
            else:
                ints = [int(i) for i in re.findall(r"\d+", s)]
                if ints and candidates:
                    seq = [candidates[i] for i in ints if 0 <= i < len(candidates)]
                else:
                    seq = [tok for tok in re.split(r"[\s,;]+", s) if tok in candidates]
        seen, out = set(), []
        for item in seq:
            if item not in seen:
                seen.add(item)
                out.append(item)
        return out

    def relevant_ranks(ranking: Sequence[str], relevant: Sequence[str]) -> List[int]:
        return [
            (ranking.index(rel) + 1) if rel in ranking else missing_rank
            for rel in relevant
        ]

    def average_precision(ranking: Sequence[str], relevant: Sequence[str]) -> float:
        if not relevant:
            return 0.0
        relevant_set, hits, total = set(relevant), 0, 0.0
        for idx, cand in enumerate(ranking, start=1):
            if cand in relevant_set:
                hits += 1
                total += hits / idx
        return total / len(relevant)

    def first_relevant_rank(ranks: Sequence[int]) -> int:
        return min(ranks) if ranks else missing_rank

    def mean(values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    # --- load inputs -------------------------------------------------------
    with open(inference_res, "r", encoding="utf-8") as f:
        inference = json.load(f)
    with open(topk_file, "r", encoding="utf-8") as f:
        topk_mentions = json.load(f)
    with open(mentions_file, "r", encoding="utf-8") as f:
        mentions = json.load(f)

    topk_map = {m["ids"]: m for m in topk_mentions}
    inf_map = {r["ids"]: r for r in inference}

    evaluations = []

    for mid, mention in iter_mentions(mentions):
        if mid is None:
            continue
        topk_list = list(topk_map.get(mid, {}).get("new_cands", []))
        # if not topk_list:
        #     continue

        gt = mention.get("depicted_entities", []) or []
        filtered_gt = [g for g in gt if g in topk_list]
        # if not filtered_gt:
        #     continue

        preds = parse_predictions(inf_map.get(mid, {}).get("pred"), topk_list)
        reranked = (preds + [c for c in topk_list if c not in preds])[: len(topk_list)]

        baseline_ranks = relevant_ranks(topk_list, filtered_gt)
        reranked_ranks = relevant_ranks(reranked, filtered_gt)

        evaluations.append(
            {
                "ids": mid,
                "reranked": reranked,
                "baseline_rel_ranks": baseline_ranks,
                "reranked_rel_ranks": reranked_ranks,
                "baseline_rank": first_relevant_rank(baseline_ranks),
                "reranked_rank": first_relevant_rank(reranked_ranks),
                "baseline_AP": average_precision(topk_list, filtered_gt),
                "AP": average_precision(reranked, filtered_gt),
                "num_rel": len(gt),
            }
        )

    if not evaluations:
        results = {f"recall_before@{k}": 0.0 for k in ks}
        results.update({f"recall_after@{k}": 0.0 for k in ks})
        results.update({"MRR_before": 0.0, "MRR_after": 0.0, "MAP_before": 0.0, "MAP_after": 0.0})
        print("No mentions with ground truth in top-K; nothing to evaluate")
        print(json.dumps(results, indent=2))
        return results

    def recall_macro(rank_lists: Sequence[Sequence[int]], k: int, num_rels: Sequence[int]) -> float:
        per_query = [
            sum(1 for r in ranks if r <= k) / n_rel if n_rel else 0.0
            for ranks, n_rel in zip(rank_lists, num_rels)
        ]
        return mean(per_query)

    baseline_rank_lists = [e["baseline_rel_ranks"] for e in evaluations]
    reranked_rank_lists = [e["reranked_rel_ranks"] for e in evaluations]
    num_rels = [e["num_rel"] for e in evaluations]
    baseline_first = [e["baseline_rank"] for e in evaluations]
    reranked_first = [e["reranked_rank"] for e in evaluations]
    baseline_APs = [e["baseline_AP"] for e in evaluations]
    reranked_APs = [e["AP"] for e in evaluations]

    results: Dict[str, float] = {}
    for k in ks:
        results[f"recall@{k}_before"] = recall_macro(baseline_rank_lists, k, num_rels)
        results[f"recall@{k}_after"] = recall_macro(reranked_rank_lists, k, num_rels)
    results["MRR_before"] = mean([1 / r for r in baseline_first])
    results["MRR_after"] = mean([1 / r for r in reranked_first])
    results["MAP_before"] = mean(baseline_APs)
    results["MAP_after"] = mean(reranked_APs)

    if save_reranked:
        with open(save_reranked, "w", encoding="utf-8") as f:
            json.dump(evaluations, f)

    print(json.dumps(results, indent=2))
    return results