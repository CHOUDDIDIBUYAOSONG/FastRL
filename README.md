# Learn from the Gap: Differential-Aware Advantage Pruning with Adaptive Rollout Sampling for GRPO
<p align="center">
  ☑️Accepted by ``NeurIPS 2026'' (CCF A)
</p>

:triangular_flag_on_post: If possible, could you please star this project. :star:  :arrow_upper_right: 

#### We have open-sourced a subset of the baseline training scripts and code. The complete implementation of our method, as well as all related code, will be released soon.

### Installation


```bash
cd EasyR1-FastRL
pip install -e .
```

### FastRL Training

```bash
bash examples/scr/qwen2_5_vl_7b_fastrl.sh
```


### Baseline Training

```bash
bash examples/scr/qwen2_5_vl_7b_cppo.sh
bash examples/scr/qwen2_5_vl_7b_grpo.sh
bash examples/scr/qwen2_5_vl_7b_dapo.sh
bash examples/scr/qwen2_5_vl_7b_gspo.sh
```

### Merge Checkpoint in Hugging Face Format

```bash
python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
```

### Please cite this paper as follows (BibTeX): 
```
@inproceedings{yang2026fastrl,
  title={Learn from the Gap: Differential-Aware Advantage Pruning with Adaptive Rollout Sampling for GRPO},
  author={Yang, Jiahua and Yang, Zhiwei and Zhang, Xianpeng and Chen, Dongyu and Chen, Xing and Su, Tianhuang and Lu, Haonan and Guan, Quanlong and Tang, Kai and Wang, Chuangchuang},
  booktitle={Proceedings of the 40th Annual Conference on Neural Information Processing Systems (NeurIPS)},
  pages={xx--xx},
  month={December},
  year={2026}
}
```

The Fortieth Annual Conference on Neural Information Processing Systems (NeurIPS) will take place in Sydney (Main) Dec. 6-12, 2026.
