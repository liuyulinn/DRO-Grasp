uv run multi_gpu_inference_DGN.py -g 0 1 2 3 --bodex-hand sim_backallegro/fc_left \
      --bodex-input-root data/bodex/DGN_allegro_left_lifted/graspdata \
      --bodex-scene-kind tabletop_ur10e \
      --checkpoint ckpt/model/model_3lefthand.pth \
      --batch-size 20 --split-batch-size 20


# sharpa
python multi_gpu_inference_DGN.py -g 0 --bodex-hand sim_fixsharpa/fc_left \
    --bodex-input-root data/bodex/DGN_fixsharpa_left_lifted/graspdata
