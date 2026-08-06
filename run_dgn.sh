uv run multi_gpu_inference_DGN.py -g 0 --bodex-hand sim_backallegro/fc_left \
      --bodex-input-root data/bodex/DGN_allegro_left_lifted/graspdata \
      --bodex-scene-kind tabletop_ur10e \
      --checkpoint output/model_3lefthands/state_dict/epoch_20.pth \
      --batch-size 20 --split-batch-size 20


# sharpa
python multi_gpu_inference_DGN.py -g 0 --bodex-hand sim_fixsharpa/fc_left \
    --bodex-input-root data/bodex/DGN_fixsharpa_left_lifted/graspdata
