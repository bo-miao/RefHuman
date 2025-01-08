cd ..

GPU_NUM=4
BATCH=24

CKPT="../datasets/RefHuman/UniPHD_RefHuman_SwinT.pth" # PATH to Checkpoint
OUT_DIR="../results/UniPHD_Results"

python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port 29233 \
    main.py -c config/uniphd.py \
    --backbone swin_T_224_1k \
    --output_dir ${OUT_DIR} \
    --options batch_size=${BATCH} \
    --resume ${CKPT} \
    --eval \
    --eval_trigger 'text'\

echo "**** Finish text prompt evaluation. ****"


python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port 29233 \
    main.py -c config/uniphd.py \
    --backbone swin_T_224_1k \
    --output_dir ${OUT_DIR} \
    --options batch_size=${BATCH} \
    --resume ${CKPT} \
    --eval \
    --eval_trigger 'point'\

echo "**** Finish point prompt evaluation. ****"


python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port 29233 \
    main.py -c config/uniphd.py \
    --backbone swin_T_224_1k \
    --output_dir ${OUT_DIR} \
    --options batch_size=${BATCH} \
    --resume ${CKPT} \
    --eval \
    --eval_trigger 'scribble'\

echo "**** Finish scribble prompt evaluation. ****"
