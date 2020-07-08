show_help() {
cat << EOF
Usage: 
    ${0##*/} [-h/--help] [-s/--start] [-e/--end] [-d/--dataset] [-m/--model]  [-g/--gpu]
    e.g.
        bash ${0##*/} -s 1 -e 20 -d VOT2018 -g 4 # for test models
        bash ${0##*/} -m snapshot/checkpoint_e10.pth -n 8 -g 4 # for tune models
EOF
}

ROOT=`git rev-parse --show-toplevel`
export PYTHONPATH=$ROOT:$PYTHONPATH
export PYTHONPATH=$PWD:$PYTHONPATH

dataset=VOT2018
NUM=4
START=1
END=20
GPU=0

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        -h|--help)
            show_help
            exit
            ;;
        -d|--dataset)
            dataset=$2
            shift 2
            ;;
        -n|--num)
            NUM=$2
            shift 2
            ;;
        -s|--start)
            START=$2
            shift 2
            ;;
        -e|--end)
            END=$2
            shift 2
            ;;
        -m|--model)
            model=$2
            shift 2
            ;;
        -g|--gpu)
            GPU=$2
            shift 2
            ;;
        *)
            echo invalid arg [$1]
            show_help
            exit 1
            ;;
    esac
done

set -e

gpu_arr=($GPU)
gpu_len=${#gpu_arr[@]}

if ! [ -z "$model" ]; then
    echo test davisconfig $START ~ $END on dataset $dataset with $GPU gpus.
    for i in $(seq $START $END)
    do 
        # bash test_mask_refine.sh config_vot18.json snapshot/checkpoint_e$i.pth $dataset $((${gpu_arr[($i-1) % $gpu_len]})) &
        bash test_mask_refine.sh config_davis$i.json $model $dataset $((${gpu_arr[($i-1) % $gpu_len]})) &
    done
fi