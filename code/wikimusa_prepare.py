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

    for split, input_mentions in zip(['train', 'val', 'test'], [input_mentions_train, input_mentions_val, input_mentions_test]):
        mentions_output_dir = getattr(args.mention, f"{split}_mentions_dir")
        Path(mentions_output_dir).parent.mkdir(parents=True, exist_ok=True)

        with open(input_mentions, 'r', encoding='utf-8') as f:
            orig_mentions = json.load(f)
        new_mentions = []
        for id, mention in tqdm(orig_mentions.items(), desc=f"Processing {split} mentions"):
            """
            "Q18891111": {
                "depicted_entities": [
                    "Q7307"
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
                "ids": "Q18891111",
                "context": "The artwork \"The Kiss\" represents a couple embracing in darkness, their faces merged into a single, featureless shape. The oil painting on canvas, created by Norwegian symbolist artist Edvard Munch in 1897, is characterized by long, slurpy brush strokes and depicts the unity of the lovers amidst a somber atmosphere.",
                "image": "Edvard%20Munch%20-%20The%20Kiss%20-%20Google%20Art%20Project.jpg"
            }
            """
            
            new_mention = {
                "ids": id,
                "context": mention.get("article", ""),
                "image": mention.get("img", [""])[0] if mention.get("img") else ""
            }
            new_mentions.append(new_mention)

        with open(mentions_output_dir, 'w', encoding='utf-8') as f:
            json.dump(new_mentions, f, ensure_ascii=False, indent=4)