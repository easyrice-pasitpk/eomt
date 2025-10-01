source .venv/bin/activate
export WANDB_ENTITY="easyrice"

python main.py fit \
  -c configs/dinov3/coco/instance/eomt_small_640.yaml \
  --trainer.devices 1 \
  --trainer.accumulate_grad_batches 16 \
  --data.num_workers 4 \
  --data.batch_size 4 \
  --data.path /project/lt200377-mpind/segment