from untils.functions import setup_parser
import json
from tqdm import tqdm
from pathlib import Path
if __name__=='__main__':
    args = setup_parser()

    # Prepare mentions
    input_mentions_train = args.orig_datset.train_mentions_dir
    input_mentions_val = args.orig_datset.val_mentions_dir
    input_mentions_test = args.orig_datset.test_mentions_dir

    entity_sums=dict()
    entity_sum_file=args.ent.train_output_dir

    with open(entity_sum_file, 'r', encoding='utf-8') as f:
        entity_sums_list = json.load(f)
        for entity in entity_sums_list:
            ide = entity.get("ids")
            summary = entity.get("sum", None)
            if ide and summary:
                entity_sums[ide] = ". ".join([ide, summary])

    for split, input_mentions in zip(['train', 'val'], [input_mentions_train, input_mentions_val]):
        mentions_desc_dir = getattr(args.mention, f"{split}_output_dir")
        mentions_output_dir = getattr(args.finetune, f"finetuning_{split}_data_dir")
        Path(mentions_output_dir).parent.mkdir(parents=True, exist_ok=True)

        with open(input_mentions, 'r', encoding='utf-8') as f:
            orig_mentions = json.load(f)

        descs = {}
        with open(mentions_desc_dir, 'r', encoding='utf-8') as f:
            orig_mentions_descs = json.load(f)
            for mentions_desc in orig_mentions_descs:
                mention_id = mentions_desc.get("ids")
                desc=mentions_desc.get("des_llava", mentions_desc.get("des", None))
                if mention_id and desc:
                    descs[mention_id] = desc

        new_mentions = []
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
        
            Just give the ids and do not give me any other information.
            The most matched ids are:
            """
        for id, mention in tqdm(orig_mentions.items(), desc=f"Processing {split} mentions"):
            """
            "Q18891111": {
                "depicted_entities": [
                    "Q7307",
                    "Q123456"
                ],
                "iconographic_depictions": [],
                "preiconographic_depictions": [
                    "Q7307"
                ],
                "article": "The artwork \"The Kiss\" represents a couple embracing in darkness, their faces merged into a single, featureless shape. The oil painting on canvas, created by Norwegian symbolist artist Edvard Munch in 1897, is characterized by long, slurpy brush strokes and depicts the unity of the lovers amidst a somber atmosphere.",
                "img": [
                    "Edvard%20Munch%20-%20The%20Kiss%20-%20Google%20Art%20Project.jpg"
                ]
            }

            to 

            {
                "messages": [
                    {
                        "role": "system",
                        "content": "you are a helpful assistant!"
                    },
                    {
                        "role": "user",
                        "content": "You are an expert in knowledge graph, ..."
                    },
                    {
                        "role": "assistant",
                        "content": "Q7307, Q123456"
                    }
                ]
            }

            Take into account that the context is the article but the description is extracted from the 
            """

            context = mention.get("article", "")
            description = descs.get(id, context)
            gt_ids=mention.get("depicted_entities", [])

            gt_sums = [entity_sums.get(gt_id, "") for gt_id in gt_ids]
            
            new_mention = {
                "messages": [
                    {
                        "role": "system",
                        "content": "you are a helpful assistant!"
                    },
                    {
                        "role": "user",
                        "content": PROMPT.format(
                            mention_context=context,
                            mention_des=description,
                            entity_0=gt_sums[0] if len(gt_sums) > 0 else "",
                            entity_1=gt_sums[1] if len(gt_sums) > 1 else "",
                            entity_2=gt_sums[2] if len(gt_sums) > 2 else "",
                            entity_3=gt_sums[3] if len(gt_sums) > 3 else "",
                            entity_4=gt_sums[4] if len(gt_sums) > 4 else ""
                        )
                    },
                    {
                        "role": "assistant",
                        "content": ", ".join(gt_ids)
                    }
                ]
            }
            new_mentions.append(new_mention)

        with open(mentions_output_dir, 'w', encoding='utf-8') as f: # jsonl
            for mention in new_mentions:
                f.write(json.dumps(mention, ensure_ascii=False) + '\n')
