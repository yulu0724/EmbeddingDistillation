DATASET=cub # car, product
TYPE=relative #absolute
TNET=resnet101
SNET=resnet18
GPU=1
LOSS=triplet
LR=1e-05 # 1e-05 for cub and car; 1e-06 for products
for LAMDA in 0.10  # cub: (abs:1.00, rel:0.30); car: (abs:1.50, rel: 0.10); product: (abs:2.00, rel: 1.00)
do

python train_distillation.py -lamda ${LAMDA} -data ${DATASET}  -TNet ${TNET} -net ${SNET} -lr ${LR} -dim 512   -num_instances 8 -BatchSize 32 -loss ${LOSS}  -epochs 2001  -Ttype ${TYPE} -save_step 200 -gpu ${GPU} #-start 2400 -continue_train 
# cub: 2000 epochs; car: 3000 epochs; product: 50 epochs

python test.py -gpu ${GPU} -data ${DATASET} -r  checkpoints/${DATASET}_${LOSS}_dis_${SNET}_${TNET}_${TYPE}_${LAMDA}_${LR}/0_model.pkl >./results/${DATASET}_${LOSS}_${SNET}_${TNET}_${TYPE}_${LAMDA}_test.txt 
python test.py -gpu ${GPU} -data ${DATASET} -r  checkpoints/${DATASET}_${LOSS}_dis_${SNET}_${TNET}_${TYPE}_${LAMDA}_${LR}/2000_model.pkl >>./results/${DATASET}_${LOSS}_${SNET}_${TNET}_${TYPE}_${LAMDA}_test.txt 



done
