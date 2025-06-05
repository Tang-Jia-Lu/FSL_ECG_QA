from omegaconf import OmegaConf
from fairseq_signals import tasks
import torch

def hydra_main() :
    cfg={'common': {'fp16': False, 'log_format': 'json', 'log_interval': 10, 'all_gather_list_size': 2048000}, 'checkpoint': {'save_dir': 'checkpoints', 'save_interval': 1, 'keep_last_epochs': 1, 'save_interval_updates': 0, 'no_epoch_checkpoints': False}, 'task': {'_name': 'ecg_classification', 'data': '/home/jtang/data/ecg_qa_500/ecg_qa/manifest/finetune', 'leads_to_load': None, 'leads_bucket': None, 'bucket_selection': 'uniform', 'sample_rate': None, 'filter': False, 'normalize': False, 'mean_path': None, 'std_path': None, 'enable_padding': True, 'enable_padding_leads': False, 'max_sample_size': None, 'min_sample_size': None, 'num_batch_buckets': 0, 'precompute_mask_indices': False, 'perturbation_mode': None, 'p': [1.0], 'max_amplitude': 0.1, 'min_amplitude': 0.0, 'dependency': True, 'shift_ratio': 0.2, 'num_segment': 1, 'max_freq': 0.2, 'min_freq': 0.01, 'k': 3, 'mask_leads_selection': 'random', 'mask_leads_prob': 0.5, 'mask_leads_condition': [4, 5], 'inferred_w2v_config': None, 'inferred_3kg_config': None, 'criterion_name': 'binary_cross_entropy_with_logits', 'model_name': None, 'clocs_mode': None, 'path_dataset': False, 'load_specific_lead': False}, 'dataset': {'num_workers': 6, 'max_tokens': None, 'batch_size': 128, 'valid_subset': 'valid,test'}, 'distributed_training': {'distributed_world_size': 1}, 'criterion': {'_name': 'binary_cross_entropy_with_logits', 'weight': None, 'threshold': 0.5, 'report_auc': True, 'auc_average': 'macro', 'pos_weight': None, 'report_cinc_score': True, 'weights_file': '/home/jtang/data/ecg_qa_500/ecg_qa/fairseq-signals/examples/w2v_cmsc/weights.csv', 'per_log_keys': []}, 'optimization': {'max_epoch': 100, 'max_update': 320000, 'lr': [5e-05]}, 'optimizer': {'_name': 'adam', 'adam_betas': '(0.9, 0.98)', 'adam_eps': 1e-08, 'weight_decay': 0.0, 'use_old_adam': False, 'lr': [5e-05]}, 'lr_scheduler': {'_name': 'fixed', 'force_anneal': None, 'lr_shrink': 0.1, 'warmup_updates': 0, 'lr': [5e-05]}, 'model': {'_name': 'ecg_transformer_classifier', 'all_gather': False, 'normalize': False, 'filter': False, 'data': '/home/jtang/data/ecg_qa_500/ecg_qa/manifest/finetune', 'args': None, 'encoder_layers': 12, 'encoder_embed_dim': 768, 'encoder_ffn_embed_dim': 3072, 'encoder_attention_heads': 12, 'layer_norm_first': False, 'dropout': 0.0, 'attention_dropout': 0.0, 'activation_dropout': 0.1, 'encoder_layerdrop': 0.0, 'dropout_input': 0.0, 'dropout_features': 0.0, 'apply_mask': False, 'mask_length': 10, 'mask_prob': 0.0, 'mask_selection': 'static', 'mask_other': 0.0, 'no_mask_overlap': False, 'mask_min_space': 1, 'mask_channel_length': 10, 'mask_channel_prob': 0.0, 'mask_channel_selection': 'static', 'mask_channel_other': 0.0, 'no_mask_channel_overlap': False, 'mask_channel_min_space': 1, 'extractor_mode': 'default', 'conv_feature_layers': '[(256, 2, 2)] * 4', 'in_d': 12, 'conv_bias': False, 'feature_grad_mult': 0.0, 'conv_pos': 128, 'conv_pos_groups': 16, 'model_path': './ecg_checkpoint/checkpoint_ecg.pt', 'no_pretrained_weights': False, 'freeze_finetune_updates': 0, 'final_dropout': 0.0, 'num_labels': 26}, 'job_logging_cfg': {'version': 1, 'formatters': {'simple': {'format': '[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'}}, 'handlers': {'console': {'class': 'logging.StreamHandler', 'formatter': 'simple', 'stream': 'ext://sys.stdout'}, 'file': {'class': 'logging.FileHandler', 'formatter': 'simple', 'filename': 'extract.log'}}, 'root': {'level': 'INFO', 'handlers': ['console', 'file']}, 'disable_existing_loggers': False}}
    cfg = OmegaConf.create(cfg)
    OmegaConf.set_struct(cfg, True)
    
    task = tasks.setup_task(cfg.task)
    model = task.build_model(cfg.model)
    
    return model

if __name__ == "__main__":
    model = hydra_main()
    model.cuda()

    # Prepare padding mask and input tensor
    # Generate a random test ECG input
    truncated_ecg = torch.randn(1, 12, 5000)  # batch_size=1, leads=12, length=5000
    padding_mask = torch.zeros(truncated_ecg.shape[0], 12, 5000, dtype=torch.bool).cuda()
    inp = truncated_ecg.cuda()

    # Pass the test ECG through the model
    proj_out = model(**{
        'source': inp,
        'padding_mask': padding_mask
    })
    print("Model output shape:", proj_out['out'].shape)


