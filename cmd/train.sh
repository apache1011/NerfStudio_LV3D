CUDA_VISIBLE_DEVICES=2 \
ns-train nerfacto --data /data/hyzhou/data/kitti360_nerfacto_20 \
                  --vis tensorboard \
                  --experiment_name kitti_short \
                  --pipeline.datamanager.camera-optimizer.mode off


#ns-train instant-ngp --data /data/hyzhou/data/kitti/kitti360_nerfacto \
#                  --experiment_name kitti \
#                  --pipeline.datamanager.camera-optimizer.mode off \
#                  --vis tensorboard

#CUDA_VISIBLE_DEVICES=2 \
#ns-train neus --data /data/hyzhou/data/kitti/kitti360_neus \
#                  --experiment_name kitti \
#                  --pipeline.datamanager.camera-optimizer.mode off \
#                  --vis tensorboard