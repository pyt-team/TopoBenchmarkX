# DONE
# python train.py dataset=cocitation_cora model=graph/gin model.optimizer.lr=0.01,0.001 model.backbone.hidden_channels=64,128,256 model.backbone.num_layers=1,2,3,4 dataset.parameters.data_seed=0,3,5 model.backbone.dropout=0,0.25,0.5 callbacks.early_stopping.patience=50 dataset.parameters.data_seed=0,3,5 logger.wandb.project=topobenchmark_search --multirun
# python train.py dataset=cocitation_citeseer model=graph/gin model.optimizer.lr=0.01,0.001 model.backbone.hidden_channels=64,128,256 model.backbone.num_layers=1,2,3,4 dataset.parameters.data_seed=0,3,5 model.backbone.dropout=0,0.25,0.5 callbacks.early_stopping.patience=50 dataset.parameters.data_seed=0,3,5 logger.wandb.project=topobenchmark_search --multirun
# python train.py dataset=cocitation_pubmed model=graph/gin model.optimizer.lr=0.01,0.001 model.backbone.hidden_channels=64,128,256 model.backbone.num_layers=1,2,3,4 dataset.parameters.data_seed=0,3,5 model.backbone.dropout=0,0.25,0.5 callbacks.early_stopping.patience=50 dataset.parameters.data_seed=0,3,5 logger.wandb.project=topobenchmark_search --multirun

# TO RUN

python train.py dataset=PROTEINS_TU model=graph/gin model.optimizer.lr=0.01,0.001 model.backbone.hidden_channels=64,128,256 model.backbone.num_layers=1,2,3,4 dataset.parameters.data_seed=0,3,5 model.backbone.dropout=0,0.25,0.5 callbacks.early_stopping.patience=30 dataset.parameters.data_seed=0,3,5 dataset.parameters.batch_size=128,256 logger.wandb.project=topobenchmark_search --multirun
python train.py dataset=NCI1 model=graph/gin model.optimizer.lr=0.01,0.001 model.backbone.hidden_channels=64,128,256 model.backbone.num_layers=1,2,3,4 dataset.parameters.data_seed=0,3,5 model.backbone.dropout=0,0.25,0.5 callbacks.early_stopping.patience=30 dataset.parameters.data_seed=0,3,5 dataset.parameters.batch_size=128,256 logger.wandb.project=topobenchmark_search --multirun
python train.py dataset=IMDB-BINARY model=graph/gin model.optimizer.lr=0.01,0.001 model.backbone.hidden_channels=64,128,256 model.backbone.num_layers=1,2,3,4 dataset.parameters.data_seed=0,3,5 model.backbone.dropout=0,0.25,0.5 callbacks.early_stopping.patience=30 dataset.parameters.data_seed=0,3,5 dataset.parameters.batch_size=128,256 logger.wandb.project=topobenchmark_search --multirun
python train.py dataset=IMDB-MULTI model=graph/gin model.optimizer.lr=0.01,0.001 model.backbone.hidden_channels=64,128,256 model.backbone.num_layers=1,2,3,4 dataset.parameters.data_seed=0,3,5 model.backbone.dropout=0,0.25,0.5 callbacks.early_stopping.patience=30 dataset.parameters.data_seed=0,3,5 dataset.parameters.batch_size=128,256 logger.wandb.project=topobenchmark_search --multirun
python train.py dataset=MUTAG model=graph/gin model.optimizer.lr=0.01,0.001 model.backbone.hidden_channels=64,128,256 model.backbone.num_layers=1,2,3,4 dataset.parameters.data_seed=0,3,5 model.backbone.dropout=0,0.25,0.5 callbacks.early_stopping.patience=30 dataset.parameters.data_seed=0,3,5 dataset.parameters.batch_size=32,64 logger.wandb.project=topobenchmark_search --multirun
python train.py dataset=REDDIT-BINARY model=graph/gin model.optimizer.lr=0.01,0.001 model.backbone.hidden_channels=64,128,256 model.backbone.num_layers=1,2,3,4 dataset.parameters.data_seed=0,3,5 model.backbone.dropout=0,0.25,0.5 callbacks.early_stopping.patience=30 dataset.parameters.data_seed=0,3,5 dataset.parameters.batch_size=128,256 logger.wandb.project=topobenchmark_search --multirun
python train.py dataset=ZINC model=graph/gin model.optimizer.lr=0.01,0.001 model.optimizer.weight_decay=0 model.backbone.hidden_channels=16,32,64,128 model.backbone.num_layers=1,2,3,4 dataset.parameters.batch_size=128,256 dataset.parameters.data_seed=0 model.backbone.dropout=0,0.25,0.5 logger.wandb.project=topobenchmark_search callbacks.early_stopping.patience=30 --multirun










# python train.py dataset=ZINC model=graph/gin model.optimizer.lr=0.01,0.001 model.optimizer.weight_decay=0 model.backbone.hidden_channels=16,32,64,128 model.backbone.num_layers=1,2,3,4 dataset.parameters.batch_size=128,256 dataset.parameters.data_seed=0 model.backbone.dropout=0,0.25,0.5 logger.wandb.project=ZINC_sanityckeck --multirun

#python train.py dataset=cocitation_citeseer model=hypergraph/edgnn model.optimizer.lr=0.01,0.001 model.backbone.num_features=64,128,256 model.backbone.All_num_layers=1,2 model.backbone.MLP_num_layers=0,1,2 model.backbone.input_dropout=0,0.25,0.5 callbacks.early_stopping.patience=25 dataset.parameters.data_seed=0,3,5 logger.wandb.project=topobenchmark_search --multirun
#python train.py dataset=cocitation_pubmed model=hypergraph/edgnn model.optimizer.lr=0.01,0.001 model.backbone.num_features=64,128,256 model.backbone.All_num_layers=1,2 model.backbone.MLP_num_layers=0,1,2 model.backbone.input_dropout=0,0.25,0.5 callbacks.early_stopping.patience=25 dataset.parameters.data_seed=0,3,5 logger.wandb.project=topobenchmark_search --multirun

# python train.py dataset=ZINC model=hypergraph/edgnn model.optimizer.lr=0.01,0.001 model.backbone.num_features=16,32,64,128 model.backbone.All_num_layers=1,2,3 model.backbone.MLP_num_layers=0,1,2 model.backbone.input_dropout=0,0.25,0.5 dataset.parameters.batch_size=128,256 callbacks.early_stopping.patience=5 dataset.parameters.data_seed=0 logger.wandb.project=topobenchmark_search --multirun

