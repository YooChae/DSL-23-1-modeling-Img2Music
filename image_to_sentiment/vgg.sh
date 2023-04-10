if [ ! -d "%s/log"%base_dir ]; then
  mkdir ./log
fi
# 뭐라고 주소 정할지 아직 생각 안해봄
for lr in 0.001 0.0005 0.0001 0.00025; do
    for network in vgg; do
        for num_epoch in 10 30 50; do   
            for layers in 16 19; do
                python -u /content/drive/MyDrive/modeling/run.py \
                  -model_id vgg_$layers'_'$num_epoch \
                  --network $network \
                  --layers $layers \
                  --num_epoch $num_epoch \
                  --lr $lr \
                  --batch_size $batch_size >log/$model_name'_'vgg_$layers'_'$num_epoch.log
            done
        done
    done
done
