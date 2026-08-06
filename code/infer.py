from untils.dataset import augment_ent, augment_men_img, augment_men_text, run_emb, runtopK, infer
from untils.functions import setup_parser
import os

if __name__=='__main__':
    args = setup_parser()
    print(os.environ.get('USE_HF', '0'))
    infer(model_id=args.infer.model_id, ckpt_id=args.infer.ckpt_id, max_length=args.infer.max_length, database_sum=args.infer.test_database_sum, mention_topK_dir=args.infer.test_mention_topK_dir, res_output_dir=args.infer.test_res_topK_dir)