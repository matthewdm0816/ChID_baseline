# ChID_baseline
计算语言学22-23学年秋季学期课程大作业实现

本次作业采用ChID中文成语完型填空数据集，要求模型能够从一系列成语候选中选择最正确的成语填入语篇的特定位置。
```
{
	"groundTruth": ["一目了然", "先入为主"], 
	"candidates": [["明明白白", "添油加醋", "一目了然", "残兵败将", "杂乱无章", "心中有数", "打抱不平"], ["矫揉造作", "死不瞑目", "先入为主", "以偏概全", "期期艾艾", "似是而非", "追根究底"]], 
	"content": "【分析】父母对孩子的期望这一点可以从第一段中找到“即使是学校也只是我们送孩子去接受实用教育的地方，而不是让他们为了知识而去追求知识的地方。”至此，答案选项[C]#idiom#。而选项[B]显然错误。选项[A]这个干扰项是出题人故意拿出一个本身没有问题，但是不适合本处的说法来干扰考生。考生一定要警惕#idiom#的思维模式，在做阅读理解的时候，不能按照自己的直觉和知识瞎猜，一定要以原文为根据。选项[D]显然也是不符合家长的期望的。", 
	"realCount": 2
}
```
如上所示，在`content`中有两处`#idiom#`标志。以第一个标志处为例，模型要从候选成语`["明明白白", "添油加醋", "一目了然", "残兵败将", "杂乱无章", "心中有数", "打抱不平"]`选择最合适的成语`一目了然`填入此处。

## 数据集下载
本文数据集ChID由 **[ChID: A Large-scale Chinese IDiom Dataset for Cloze Test](https://www.aclweb.org/anthology/P19-1075)** 提出。训练集包含句子50w条，验证集和测试集各有2w条句子。

[下载链接（北大网盘）](https://disk.pku.edu.cn:443/link/3510A73BA4793A830B0179DF795330C8)


## 模型介绍
请参见实验报告。


## 实验结果
请参见实验报告。

##  额外文件说明

1. `idiom.json`: 成语释义文件。
2. `idiom_vocab.json`: 成语词汇编号文件。

## 参考文献
* ChID: A Large-scale Chinese IDiom Dataset for Cloze Test. Chujie Zheng, Minlie Huang, Aixin Sun. ACL (1) 2019: 778-787
* It's Not Just Size That Matters: Small Language Models Are Also Few-Shot Learners. Timo Schick, Hinrich Schütze. NAACL-HLT 2021: 2339-2352

