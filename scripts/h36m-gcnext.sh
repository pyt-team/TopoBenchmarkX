python -m topobenchmarkx \
    dataset=graph/H36MDataset \
    model=graph/gcnext \
    dataset.dataloader_params.batch_size=256 \
    trainer.max_epochs=5 \
    optimizer.parameters.lr=0.0006 \
    trainer.check_val_every_n_epoch=2 \
    model.backbone.config.motion_mlp.use_skeletal_hyperedges=True\
    test=True

python -m topobenchmarkx \
    dataset=graph/H36MDataset \
    model=graph/gcnext \
    dataset.dataloader_params.batch_size=256 \
    trainer.max_epochs=5 \
    optimizer.parameters.lr=0.0006 \
    trainer.check_val_every_n_epoch=2 \
    model.backbone.config.motion_mlp.use_skeletal_hyperedges=False\
    test=True