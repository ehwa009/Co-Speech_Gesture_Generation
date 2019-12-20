# Co-Speech Gesture Generation
A PyTorch implementation of "Robots Learn Social Skills: End-to-End Learning of Co-Speech Gesture Generation for Humanoid Robots".
```
@article{yoon2018robots,
  title={Robots Learn Social Skills: End-to-End Learning of Co-Speech Gesture Generation for Humanoid Robots},
  author={Yoon, Youngwoo and Ko, Woo-Ri and Jang, Minsu and Lee, Jaeyeon and Kim, Jaehong and Lee, Geehyuk},
  journal={arXiv preprint arXiv:1810.12541},
  year={2018}
}
```

---
## Seq2pos Model Strucuture
![seq2pos](./figures/seq2pos_diagram.PNG)

---
## TED Dataset Visualization
 - The dataset was pre-processed by l2 norm, relocating neck coordination, and shoulder length normalization in order 
 - The below equation was used to nomralize shoulder length:

![sh_equation](./figures/equation_sh_norm.png)

- After l2 norm
![l2_norm](./figures/l2_norm.png)

- After re-locating neck coordination
![neck_reloc](./figures/neck_re_loc.png)

- After normalizing shoulder length
![sh_norm](./figures/sh_len_norm.png)

---
## Result
- Model overfitting was observed. 
  But the model still is able to produce gesture output like Youngwoo.
![loss_graph](./log/loss.png)