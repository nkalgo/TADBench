

class MyConfig(object):
    # 模型参数
    embedding_size = 1
    hidden_dims = 1
    num_layers = 1
    num_classes = 1        # 类别数

    def __str__(self):
        return f"{self.embedding_size=}, {self.hidden_dims=}, {self.num_layers=}"

    def __repr__(self):
        return self.__str__()
