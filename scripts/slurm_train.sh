# modify these augments if you want to try other datasets, splits or methods
# dataset: ['pascal', 'cityscapes', 'ade20k', 'coco']
# method: ['CTFS_train', 'fixmatch', 'supervised']
# exp: just for specifying the 'save_path'
# split: ['92', '1_16', ...]. Please check directory './splits/$dataset' for concrete splits
dataset='pascal'
method='CTFS_train'
exp='dinov2_small'
split='02'

config=configs/${dataset}.yaml
labeled_id_path=splits/$dataset/$split/labeled.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
save_path=exp/$dataset/$method/$exp/$split

mkdir -p $save_path

srun --mpi=pmi2 -p $3 -n $1 --gres=gpu:$1 --ntasks-per-node=$1 --job-name=$job
    --open-mode=append -o $save_path/out.log --quotatype=reserved \
    python3 -u $method.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $2
