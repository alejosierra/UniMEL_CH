from untils.dataset import eval, infer
from untils.functions import setup_parser
import os

if __name__=='__main__':
    args = setup_parser()
    eval(inference_res=args.infer.test_res_topK_dir, topk_file=args.top.test_mention_topK_dir, mentions_file=args.orig_datset.test_mentions_dir)