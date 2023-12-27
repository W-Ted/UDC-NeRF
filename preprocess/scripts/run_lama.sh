cd preprocess/lama
export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)
echo $(pwd)


scane_names=("toydesk1" "toydesk2" "scannet0024" "scannet0038" "scannet0113" "scannet0192" "scannet0113_multi")  
for i in "${scane_names[@]}"
do  
    echo "$i"
    ~/anaconda3/envs/lama/bin/python -u bin/predict.py \
    model.path=$(pwd)/big-lama \
    indir=$(pwd)/../"$i"_lamain \
    outdir=$(pwd)/../"$i"_lamaout
done