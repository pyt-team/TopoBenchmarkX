# GAT
# Train rest of the TU graph datasets
datasets=( 'PROTEINS_TU' 'NCI1' 'NCI109' 'REDDIT-BINARY' 'IMDB-BINARY' 'IMDB-MULTI' )

for dataset in ${datasets[*]}
do
  python train.py \
    dataset=$dataset \
    model=graph/gat \
    model.optimizer.lr=0.01,0.001 \
    model.feature_encoder.out_channels=32,64,128 \
    model.backbone.num_layers=1,2,3,4 \
    model.feature_encoder.proj_dropout=0.25,0.5 \
    dataset.parameters.data_seed=7,9 \
    dataset.parameters.batch_size=128,256 \
    logger.wandb.project=TopoBenchmarkX_Graph \
    trainer.max_epochs=500 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=5 \
    callbacks.early_stopping.patience=10 \
    --multirun
done

python train.py \
  dataset=MUTAG \
  model=graph/gat \
  model.optimizer.lr=0.01,0.001 \
  model.feature_encoder.out_channels=32,64,128 \
  model.backbone.num_layers=1,2,3,4 \
  model.feature_encoder.proj_dropout=0.25,0.5 \
  dataset.parameters.data_seed=7,9 \
  dataset.parameters.batch_size=32,64 \
  trainer.max_epochs=500 \
  trainer.min_epochs=50 \
  trainer.check_val_every_n_epoch=1 \
  logger.wandb.project=TopoBenchmarkX_Graph \
  callbacks.early_stopping.patience=25 \
  tags="[MainExperiment]" \
  --multirun



### GIN 
# Train rest of the TU graph datasets
datasets=( 'PROTEINS_TU' 'NCI1' 'NCI109' 'REDDIT-BINARY' 'IMDB-BINARY' 'IMDB-MULTI' )

for dataset in ${datasets[*]}
do
  python train.py \
    dataset=$dataset \
    model=graph/gin \
    model.optimizer.lr=0.01,0.001 \
    model.feature_encoder.out_channels=32,64,128 \
    model.backbone.num_layers=1,2,3,4 \
    model.feature_encoder.proj_dropout=0.25,0.5 \
    dataset.parameters.data_seed=7,9 \
    dataset.parameters.batch_size=128,256 \
    logger.wandb.project=TopoBenchmarkX_Graph \
    trainer.max_epochs=500 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=5 \
    callbacks.early_stopping.patience=10 \
    --multirun
done

# ----TU graph datasets----
# MUTAG have very few samples, so we use a smaller batch size
# Train on MUTAG dataset
python train.py \
  dataset=MUTAG \
  model=graph/gin \
  model.optimizer.lr=0.01,0.001 \
  model.feature_encoder.out_channels=32,64,128 \
  model.backbone.num_layers=1,2,3,4 \
  model.feature_encoder.proj_dropout=0.25,0.5 \
  dataset.parameters.data_seed=7,9 \
  dataset.parameters.batch_size=32,64 \
  trainer.max_epochs=500 \
  trainer.min_epochs=50 \
  trainer.check_val_every_n_epoch=1 \
  logger.wandb.project=TopoBenchmarkX_Graph \
  callbacks.early_stopping.patience=25 \
  tags="[MainExperiment]" \
  --multirun


# ----Heterophilic datasets----

datasets=( roman_empire amazon_ratings tolokers questions minesweeper )

for dataset in ${datasets[*]}
do
  python train.py \
    dataset=$dataset \
    model=graph/gin \
    model.optimizer.lr=0.01,0.001 \
    model.feature_encoder.out_channels=32,64,128 \
    model.backbone.num_layers=1,2,3,4 \
    model.feature_encoder.proj_dropout=0.25,0.5 \
    dataset.parameters.data_seed=7,9 \
    dataset.parameters.batch_size=128 \
    logger.wandb.project=TopoBenchmarkX_Graph \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=128 \
    callbacks.early_stopping.patience=50 \
    tags="[MainExperiment]" \
    --multirun
done


### GCN

# Train rest of the TU graph datasets
datasets=( 'PROTEINS_TU' 'NCI1' 'NCI109' 'REDDIT-BINARY' 'IMDB-BINARY' 'IMDB-MULTI' )

for dataset in ${datasets[*]}
do
  python train.py \
    dataset=$dataset \
    model=graph/gcn \
    model.optimizer.lr=0.01,0.001 \
    model.feature_encoder.out_channels=32,64,128 \
    model.backbone.num_layers=1,2,3,4 \
    model.feature_encoder.proj_dropout=0.25,0.5 \
    dataset.parameters.data_seed=7,9 \
    dataset.parameters.batch_size=128,256 \
    logger.wandb.project=TopoBenchmarkX_Graph \
    trainer.max_epochs=500 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=5 \
    callbacks.early_stopping.patience=10 \
    --multirun
done


# ----TU graph datasets----
# MUTAG have very few samples, so we use a smaller batch size
# Train on MUTAG dataset
python train.py \
  dataset=MUTAG \
  model=graph/gcn \
  model.optimizer.lr=0.01,0.001 \
  model.feature_encoder.out_channels=32,64,128 \
  model.backbone.num_layers=1,2,3,4 \
  model.feature_encoder.proj_dropout=0.25,0.5 \
  dataset.parameters.data_seed=7,9 \
  dataset.parameters.batch_size=32,64 \
  trainer.max_epochs=500 \
  trainer.min_epochs=50 \
  trainer.check_val_every_n_epoch=1 \
  logger.wandb.project=TopoBenchmarkX_Graph \
  callbacks.early_stopping.patience=25 \
  tags="[MainExperiment]" \
  --multirun



for dataset in ${datasets[*]}
do
  python train.py \
    dataset=$dataset \
    model=graph/gcn \
    model.optimizer.lr=0.01,0.001 \
    model.feature_encoder.out_channels=32,64,128 \
    model.backbone.num_layers=1,2,3,4 \
    model.feature_encoder.proj_dropout=0.25,0.5 \
    dataset.parameters.data_seed=7,9 \
    dataset.parameters.batch_size=128 \
    logger.wandb.project=TopoBenchmarkX_Graph \
    trainer.max_epochs=500 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=5 \
    callbacks.early_stopping.patience=10 \
    --multirun
done
