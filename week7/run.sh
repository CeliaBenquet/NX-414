experiments_100424(){

    # python fit_model.py \
    #     --num_epochs 100 --model_architecture cnn-2

    python fit_model.py \
        --num_epochs 1000 --model_architecture cnn-2 --learning_rate 0.001

    # python fit_model.py \
    #     --num_epochs 100 --model_architecture cnn-3 --learning_rate 0.0001

    # python fit_model.py \
    #     --num_epochs 100 --model_architecture cnn-3

    python fit_model.py \
       --num_epochs 1000 --model_architecture cnn-2 --learning_rate 0.001
}


experiments_100424\
| parallel --jobs 4 \
'CUDA_VISIBLE_DEVICES=$PARALLEL_JOBSLOT bash -c {}'