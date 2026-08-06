import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
from tqdm import tqdm
import json,re
import torch.nn.functional as F
from torch import Tensor
import transformers
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, TextIteratorStreamer,AutoTokenizer, AutoModel
from modelscope import Model
from swift.tuners import Swift

from swift.llm import get_model_tokenizer, safe_snapshot_download

os.environ['USE_HF']='True'


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def augment_ent(*, data_dir, output_dir, model_dir):
    data_dir = data_dir
    output_dir = output_dir
    with open(data_dir,"r") as f:
        entity = json.load(f)
    entity_name =  list(entity.keys())
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
        with open(output_dir,"r") as f:
            now_data = json.load(f)
        ent = now_data
    except:
        print("重新创建文件")

    all_data_name = []
    if len(ent)>0:
        for da in ent:
            if da['ids'] not in all_data_name:
                all_data_name.append(da['ids'])
    writes=0
    for i in tqdm(range(len(entity_name)), desc="Generating summaries for entities"):
        # all_data_name = []
        # if len(ent)>0:
        #     for da in ent:
        #         if da['ids'] not in all_data_name:
        #             all_data_name.append(da['ids'])
        if data_name[i] in all_data_name:
            continue
        writes+=1
        if writes%100==0:
            with open(output_dir,"w") as f:
                json.dump(ent,f)
        dict = {}
        dict['ids'] = data_name[i]
        dict['des'] = data_des[i]
        text = PROMPT.format(entity_name=data_name[i],entity_des = data_des[i])
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
            sum = output
            sum = sum.split(":")[-1]
            #print(sum)
            dict['sum'] = sum.replace('\n','')
            ent.append(dict)
        except:
            print("error!"+str(i))

    with open(output_dir,"w") as f:
        json.dump(ent,f)


def run_emb(*, model_dir, data_dir, embed_dir, max_length):
    model_dir=model_dir
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)
    model.to("cuda")
    max_length = max_length
    data_dir = data_dir
    embed_dir = embed_dir
    with open(data_dir,"r")as f:
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

    for j,ent in enumerate(tqdm(ents, desc="Generating embeddings for entities")):
        if ent['ids'] in existing_ids:
            continue
        embed = {}
        embed['ids'] = ent['ids']
        text = ent['ids']+":"+ent['sum']
        text = text.replace("\n", " ")
        input_texts = text
        batch_dict = tokenizer(input_texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt").to("cuda")
        outputs = model(**batch_dict)
        em = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])[0] # shape: (hidden_size,)
        embed['emb'] = em.tolist()
        embeds.append(embed)
    with open(embed_dir,"w") as f:
        json.dump(embeds,f)

def augment_men_img(*, mentions_dir, save_dir, model_id, image_dir, img_file_name_mapping):
    mentions_dir = mentions_dir
    save_dir = save_dir
    with open(mentions_dir,"r") as f:
        mentions = json.load(f)

    with open(img_file_name_mapping, "r") as f:
        img_mapping = json.load(f) # mapping of image file names to shorter file names

    model_id = model_id
    processor = LlavaNextProcessor.from_pretrained(model_id)

    model = LlavaNextForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.to("cuda")


    image_dir = image_dir
    PROMPT = """
    The image describes \"{mention_context}\"
    Introduce the image. Answer follow the format: "The image refers to..."
    """

    current_descriptions_dict={}
    with open(save_dir,"r") as f:
        current_descriptions = json.load(f)
        for o in tqdm(current_descriptions, desc="Loading existing descriptions"):
            if 'des_llava' in o:
                current_descriptions_dict[o['ids']]=o['des_llava']

    for i in tqdm(range(len(mentions)), desc="Generating descriptions for mentions with images"):
        id_mention=mentions[i]['ids']
        cur_desc=current_descriptions_dict.get(id_mention, None)
        if cur_desc:
            mentions[i]['des_llava'] = cur_desc
        else:
            prompt = f"[INST] <image>\n{PROMPT.format(mention_context=mentions[i]['context'])} [/INST]"
            im_dir = image_dir + "/" + img_mapping.get(mentions[i]['image'], mentions[i]['image'])
            if os.path.exists(im_dir):
                try:
                    image = Image.open(im_dir).convert("RGB")
                    inputs = processor(prompt, image, return_tensors="pt").to("cuda")
                except:
                    print("error:"+im_dir)
                    continue
            else:
                continue
            output = model.generate(**inputs, max_new_tokens=100).to("cuda")
            
            resp = processor.decode(output[0], skip_special_tokens=True)
            mentions[i]['des_llava'] = resp
    with open(save_dir,"w") as f:
        json.dump(mentions,f)

def augment_men_text(*, data_dir, output_dir, model_dir):
    data_dir = data_dir
    output_dir = output_dir
    with open(data_dir,"r") as f:
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
                entity[i]['des'] = des.replace('\n','')
                #print(des)
            except:
                print("error!"+str(i))
        ent.append(entity[i])
    with open(output_dir,"w") as f:
        json.dump(ent,f)


def runtopK(*, K, model_dir, database_emb, database_sum, mention_dir, mention_topK_dir, max_length):
    model_dir = model_dir
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)
    model.to("cuda")
    ents = []
    database_emb = database_emb
    with open(database_emb,"r") as f:
        data = json.load(f)
    ents+=data
    di = {}
    ent_ids = []
    ent_embs = []
    for ent in ents:
        di[ent['ids']] = ent['emb']
        ent_ids.append(ent['ids'])
        ent_embs.append(ent['emb'])
    ent_embs = torch.tensor(ent_embs).to("cuda") # shape: (num_entities, embedding_dim)
    ents2 = []

    database_sum = database_sum
    with open(database_sum,"r") as f:
        data = json.load(f)
    ents2+=data

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
    with open(mention_dir,"r") as f:
        mentions = json.load(f)

    max_length=max_length
    K = K
    for i in tqdm(range(len(mentions)), desc=f"Calculating top K candidates for mentions in {mention_dir}"):
        # name = mentions[i]['name']
        # context = mentions[i]['context']
        if mentions[i]['ids'] in already_computed:
            mentions[i] = already_computed[mentions[i]['ids']]
        else:
            try:
                description = mentions[i]['des_llava']
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
            mention_emb = torch.tensor(mention_emb).to("cuda") # shape: (embedding_dim,)
            # cands_scores = []
            # for cand in mentions[i]['cands']:
            #     entity_emb = di[cand]
            #     entity_emb = torch.tensor(entity_emb)
            #     score = torch.dot(mention_emb,entity_emb)
            #     cands_scores.append(score)
            ent_scores= torch.matmul(mention_emb.unsqueeze(0), ent_embs.T).squeeze(0)  # shape: (num_entities,)
            #mentions[i]['score'] = cands_scores.tolist()
            _,idx = torch.topk(ent_scores,min(K,len(ent_scores)))
            idx = idx.tolist()
            new_cands = []
            new_cand_scores = []
            for id in idx:
                new_cands.append(ent_ids[id])
                new_cand_scores.append(ent_scores[id].item())
            mentions[i]['new_cands'] = new_cands
            mentions[i]['score'] = new_cand_scores

        if (i+1) % 100 == 0:
            with open(mention_topK_dir,"w") as f:
                json.dump(mentions,f)
    with open(mention_topK_dir,"w") as f:
        json.dump(mentions,f)

    # with open(mention_topK_dir,"r") as f:
    #     mentions = json.load(f)
    # acc =0 
    # wrong_list = []
    # for i in tqdm(range(len(mentions)), desc=f"Calculating accuracy for top K candidates in {mention_topK_dir}"):
    #     if mentions[i]['ids'] in mentions[i]['new_cands']:
    #         acc+=1
    # print(acc/len(mentions))



def infer(*, model_id, ckpt_id, max_length, database_sum, mention_topK_dir, res_output_dir):
    device = "cuda"
    model_id = model_id
    ckpt_id = ckpt_id

    # model = AutoModel.from_pretrained(
    #     model_id,
    #     device_map="auto",
    #     max_length=max_length
    # )

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
    with open(database,"r") as f:
        data = json.load(f)
    ents2+=data

    di2 = {}
    for ent in ents2:
        ent_dict = {}
        ent_dict['name'] = ent['ids']
        ent_dict['sum'] = ent['sum']
        di2[ent['ids']] = ent_dict

    with open(mention_topK_dir,"r")as f:
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
        entity_table = ["","","","",""]
        cands = mentions[i]['new_cands']
        try:
            true = cands.index(mentions[i]['ids'])
        except:
            true = -1
        for idx,ca in enumerate(cands):
            try:
                ent_str = di2[ca]['name'] +": "+di2[ca]['sum']
                entity_table[idx] = ent_str
            except:
                continue
        try:
            description = mentions[i]['des_llava']
        except:
            description = mentions[i]['des']
        text = PROMPT.format(mention_context=mentions[i]['context'],mention_des=description,entity_0=entity_table[0],entity_1=entity_table[1],entity_2=entity_table[2],entity_3=entity_table[3],entity_4=entity_table[4])
        # outputs = pipeline(text)
        # response = outputs[0]["generated_text"][len(text):]
        # pred.append(response)
        #truth.append(true)
        #bad_cases.append(text)
    #for text in tqdm(bad_cases):
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
        #print(response)
        res_object={}
        res_object['ids'] = mentions[i]['ids']
        res_object['pred'] = response
        pred.append(res_object)

    with open(res_output_dir,"w") as f:
        json.dump(pred,f)
    # res = []
    # for i in range(len(pred)):
    #     res_dict = {}
    #     res_dict['pred'] = pred[i]
    #     res_dict['true'] = truth[i]
    #     res_dict['bad_case'] = bad_cases[i]
    #     res.append(res_dict)
    # with open(res_output_dir,"w") as f:
    #     json.dump(res,f)

    # with open(res_output_dir,"r") as f:
    #     data = json.load(f)
    # acc=0
    # for idx,m in enumerate(data):
    #     pred=-1
    #     t = m['true']
    #     prompt = m['bad_case']
    #     p = m['pred'].split('is:\n\n')[-1]
    #     try:
    #         pred = int(re.findall(r'\d',p)[0])
    #     except:
    #         # print(m['pred'],'\n',p,'\n-----------\n')
    #         pred=-1
    #     # print(f'pred={pred}  true={t} raw_p={p}\n')
    #     if pred==t and t != -1:
    #         acc+=1
    #     else:
    #         print(f"id={idx} , pred={pred} , true={t}\n-----------\n")
    # print(acc,len(data),acc/len(data))

    # acc_dict = {}
    # acc_dict['acc'] = acc/len(data)

