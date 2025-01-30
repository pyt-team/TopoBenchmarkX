## GCNEXT
python -m topobenchmark \
    dataset=graph/H36MDataset \
    model=graph/gcnext \
    dataset.dataloader_params.batch_size=256 \
    trainer.max_epochs=150 \
    optimizer.parameters.lr=0.0006 \
    trainer.check_val_every_n_epoch=5 \
    model.backbone.config.motion_mlp.use_skeletal_hyperedges=False\
    dataset.loader.parameters.keep_splits_on_disk=False\
    model.backbone.config.temporal_fc_in=False\
    model.backbone.config.temporal_fc_out=False\
    test=True
    
    
    # python -m topobenchmark \
#     dataset=graph/H36MDataset \
#     model=graph/gcn \
#     dataset.dataloader_params.batch_size=256 \
#     trainer.max_epochs=15 \
#     optimizer.parameters.lr=0.0006 \
#     trainer.check_val_every_n_epoch=5 \
#     dataset.loader.parameters.keep_splits_on_disk=False\
#     test=True 
    
# python -m topobenchmark \
#     dataset=graph/H36MDataset \
#     model=graph/gcnext \
#     dataset.dataloader_params.batch_size=256 \
#     trainer.max_epochs=15 \
#     optimizer.parameters.lr=0.0006 \
#     trainer.check_val_every_n_epoch=5 \
#     model.backbone.config.motion_mlp.use_skeletal_hyperedges=True\
#     dataset.loader.parameters.keep_splits_on_disk=False\
#     model.backbone.config.temporal_fc_in=False\
#     model.backbone.config.temporal_fc_out=False\
#     test=True

# python -m topobenchmark \
#     dataset=graph/H36MDataset \
#     model=graph/gcnext \
#     dataset.dataloader_params.batch_size=256 \
#     trainer.max_epochs=15 \
#     optimizer.parameters.lr=0.0006 \
#     trainer.check_val_every_n_epoch=5 \
#     model.backbone.config.motion_mlp.use_skeletal_hyperedges=False\
#     dataset.loader.parameters.keep_splits_on_disk=False\
#     model.backbone.config.temporal_fc_in=False\
#     model.backbone.config.temporal_fc_out=False\
#     test=True
   
# python -m topobenchmark \
#     dataset=graph/H36MDataset \
#     model=graph/gcnext \
#     dataset.dataloader_params.batch_size=256 \
#     trainer.max_epochs=30 \
#     optimizer.parameters.lr=0.0006 \
#     trainer.check_val_every_n_epoch=5 \
#     model.backbone.config.motion_mlp.use_skeletal_hyperedges=True\
#     dataset.loader.parameters.keep_splits_on_disk=False\
#     model.backbone.config.temporal_fc_in=True\
#     model.backbone.config.temporal_fc_out=True\
#     test=True

# python -m topobenchmark \
#     dataset=graph/H36MDataset \
#     model=graph/gcnext \
#     dataset.dataloader_params.batch_size=256 \
#     trainer.max_epochs=15 \
#     optimizer.parameters.lr=0.0006 \
#     trainer.check_val_every_n_epoch=5 \
#     model.backbone.config.motion_mlp.use_skeletal_hyperedges=False\
#     dataset.loader.parameters.keep_splits_on_disk=False\
#     model.backbone.config.temporal_fc_in=True\
#     model.backbone.config.temporal_fc_out=True\
#     test=True