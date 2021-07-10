# 基于GPT2的闲聊模型

## TODO
1. 性格画像模块：可以自定义名字，年龄，爱好，性别，
是否有亲人等个性。 这样当用户不断询问相关问题的时候，机器人
的回答可以保持一致。 关于这些信息，可以制定一个模板来让用户自定义。
定义完了模板之后，可以使用一个轻量级的二分类器，来判断是否是询问
性格相关问题，如果是的话，再定义一个多分类器，来判断到底循环的是哪个
特征，之后做出相应的回答。 可以考虑手工提取特征，比如关键词的命中，关键词
的前后顺序等，或者直接使用LSTM。

2. 敏感词检测模块：当机器人做出应答时，应该先用一个敏感词检测模块来判断是否含有相关
敏感词。如果有的话，可以按敏感词的类型做出相应的改动，替换之类的。可以参考UNIT的做法。

3. 机器人调度模块。

4. （解决）解决log每次都生成两次的问题

    因为生成了两个logger

5. （解决）保存dataset文件，节省之后再次处理的时间

6. （解决）保存最后一个epoch的model， optimizer，scheduler，可以随时打断训练并重启训练
    
    [参考文档](https://zhuanlan.zhihu.com/p/133250753)
7. distributed data parallel

8. inference 回答的选择策略， top-k&top-p


## 关于warning
1. ``UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; 
will instead unsqueeze and return a vector.``

每张卡上的loss都是要汇总到第0张卡上求梯度，更新好以后把权重分发到其余卡。但是为什么会出现这个warning，
这其实和nn.DataParallel中最后一个参数dim有关，其表示tensors被分散的维度，默认是0，
nn.DataParallel将在dim0（批处理维度）中对数据进行分块，并将每个分块发送到相应的设备。单卡的没有这个warning，
多卡的时候采用nn.DataParallel训练会出现这个warning，由于计算loss的时候是分别在多卡计算的，那么返回的也就是多个loss，
你使用了多少个gpu，就会返回多少个loss。
关于这个问题，更详细的解答[在这里](https://zhuanlan.zhihu.com/p/102697821)

2. 
