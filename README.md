# Co-Speech_Gesture_Generation
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
## Seq2pos model strucuture
![seq2pos](./figures/seq2pos_diagram.PNG)

---
## TED dataset visualization
 - The dataset was pre-processed by l2 norm, relocating neck coordination, and shoulder length normalization in order 
 - The below equation was used to nomralize shoulder length:
 ![sh_equation](./figures/equation_sh_norm.png)

![sh_norm](./figures/sh_len_norm.png)

---
## Result
- Model overfitting was observed.
![loss_graph](./log/loss.png)