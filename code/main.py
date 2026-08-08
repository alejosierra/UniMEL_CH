from untils.dataset import augment_ent, augment_men_img, augment_men_text, run_emb, runtopK, infer
from untils.functions import setup_parser
if __name__=='__main__':
    args = setup_parser()
    # train
    augment_ent(data_dir=args.ent.train_data_dir, output_dir=args.ent.train_output_dir, model_dir=args.ent.model_dir)
    # valid
    #augment_ent(data_dir=args.ent.val_data_dir, output_dir=args.ent.val_output_dir, model_dir=args.ent.model_dir)
    # test
    #augment_ent(data_dir=args.ent.test_data_dir, output_dir=args.ent.test_output_dir, model_dir=args.ent.model_dir)

    # train
    augment_men_img(mentions_dir=args.mention.train_mentions_dir, save_dir=args.mention.train_save_dir, model_id=args.mention.model_dir_img, image_dir=args.mention.train_kb_img_folder, img_file_name_mapping=args.mention.img_file_name_mapping)
    # valid
    augment_men_img(mentions_dir=args.mention.val_mentions_dir, save_dir=args.mention.val_save_dir, model_id=args.mention.model_dir_img, image_dir=args.mention.val_kb_img_folder, img_file_name_mapping=args.mention.img_file_name_mapping)
    # test
    augment_men_img(mentions_dir=args.mention.test_mentions_dir, save_dir=args.mention.test_save_dir, model_id=args.mention.model_dir_img, image_dir=args.mention.test_kb_img_folder, img_file_name_mapping=args.mention.img_file_name_mapping)

    # train
    augment_men_text(data_dir=args.mention.train_data_dir, output_dir=args.mention.train_output_dir, model_dir=args.mention.model_dir_text)
    # valid
    augment_men_text(data_dir=args.mention.val_data_dir, output_dir=args.mention.val_output_dir, model_dir=args.mention.model_dir_text)
    # test
    augment_men_text(data_dir=args.mention.test_data_dir, output_dir=args.mention.test_output_dir, model_dir=args.mention.model_dir_text)

    # train
    run_emb(model_dir=args.embed.emb_model_dir, data_dir=args.embed.train_data_dir, embed_dir=args.embed.train_embed_dir, max_length=args.embed.max_length)
    # valid
    #run_emb(model_dir=args.embed.emb_model_dir, data_dir=args.embed.val_data_dir, embed_dir=args.embed.val_embed_dir, max_length=args.embed.max_length)
    # test
    #run_emb(model_dir=args.embed.emb_model_dir, data_dir=args.embed.test_data_dir, embed_dir=args.embed.test_embed_dir, max_length=args.embed.max_length)

    # train
    runtopK(K=args.top.K, model_dir=args.top.model_dir, database_emb=args.top.train_database_emb, database_sum=args.top.train_database_sum, mention_dir=args.top.train_mention_dir, mention_topK_dir=args.top.train_mention_topK_dir, max_length=args.top.max_length)
    # valid
    runtopK(K=args.top.K, model_dir=args.top.model_dir, database_emb=args.top.val_database_emb, database_sum=args.top.val_database_sum, mention_dir=args.top.val_mention_dir, mention_topK_dir=args.top.val_mention_topK_dir, max_length=args.top.max_length)
    # test
    runtopK(K=args.top.K, model_dir=args.top.model_dir, database_emb=args.top.test_database_emb, database_sum=args.top.test_database_sum, mention_dir=args.top.test_mention_dir, mention_topK_dir=args.top.test_mention_topK_dir, max_length=args.top.max_length)