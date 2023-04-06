CUDA_VISIBLE_DEVICES=2 \
ns-train nerfacto --data /data/hyzhou/data/kitti/kitti360_nerfacto \
                  --experiment_name kitti \
                  --pipeline.datamanager.camera-optimizer.mode off \
                  --vis tensorboard

#ns-train instant-ngp --data /data/hyzhou/data/kitti/kitti360_nerfacto \
#                  --experiment_name kitti \
#                  --pipeline.datamanager.camera-optimizer.mode off \
#                  --vis tensorboard