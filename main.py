from processors.ddp_mix_processor import DDPApexProcessor

# python -m torch.distributed.launch --nproc_per_node=4 main.py

if __name__ == '__main__':
    processor = DDPApexProcessor(cfg_path="config/centernet.yaml")
    processor.run()
