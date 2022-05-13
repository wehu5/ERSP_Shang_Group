set -e

gpu=$1
dataset=$2
CUDA_VISIBLE_DEVICES=${gpu} python static_representations.py --dataset_name ${dataset}
CUDA_VISIBLE_DEVICES=${gpu} python document_representations.py --dataset_name ${dataset}
CUDA_VISIBLE_DEVICES=${gpu} python document_representations.py --dataset_name ${dataset} --attention_mechanism none
python document_class_alignment.py --dataset_name ${dataset}


python evaluate.py --dataset ${dataset} --stage Rep --suffix bbu-12-mixture-100
python evaluate.py --dataset ${dataset} --stage Align --suffix pca64.clusgmm.bbu-12.mixture-100.42
python prepare_text_classifer_training.py --dataset_name ${dataset}
./run_train_text_classifier.sh ${gpu} ${dataset} pca64.clusgmm.bbu-12.mixture-100.42.0.5
python evaluate.py --dataset ${dataset} --stage Rep --suffix bbu-12-mixture-100
python evaluate.py --dataset ${dataset} --stage Align --suffix pca64.clusgmm.bbu-12.mixture-100.42