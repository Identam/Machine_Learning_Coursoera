class LinearRegression:
        def __init__(self, l_p_metric=2, seed=42):
            """
            :param l_p_metric: Задаёт метрику для оптимизации.
            Значение 1 соответсвует MAE, 2 — MSE.
            :param seed: radom_seed для случайной инициализации весов
            """
            # Используйте np.linalg.norm
            if l_p_metric==2: 
                self.metric = lambda preds, y: np.sum((preds-y)**l_p_metric)/y.shape[0]

            if l_p_metric==1: 
                self.metric = lambda preds, y: np.sum((preds-y)**l_p_metric)/y.shape[0]
            self.seed = seed

            self.W = None
            self.b = None

        def init_weights(self, input_size, output_size):
            """
            Инициализирует параметры модели
            :param W: - матрица размерности (input_size, output_size)
            инициализируется рандомными числами из
            нормального распределения со средним 0 и стандартным отклонением 0.01
            :param b: - вектор размерности (1, output_size)
            инициализируется нулями
            """
            np.random.seed(self.seed)
            self.W = np.random.normal(0, 0.01, (input_size, output_size))
            self.b = np.zeros((1, output_size))

        def fit(self, X, y, num_epochs=1000, lr=0.001):
            """
            Обучение модели линейной регрессии методом градиентного спуска
            :param X: размерности (num_samples, input_shape)
            :param y: размерности (num_samples, output_shape)
            :param num_epochs: количество итераций градиентного спуска
            :param lr: шаг градиентного спуска
            :return metrics: вектор значений метрики на каждом шаге градиентного
            спуска. Метрика контролируется параметром l_p_metric в конструкторе
            """
            self.init_weights(X.shape[1], y.shape[1])
            metrics = []
            for _ in range(num_epochs):
                preds = self.predict(X)
                # сделайте вычисления градиентов без циклов,
                # используя только numpy
                b_grad =  2 * np.sum((preds - y))/X.shape[0]
                W_grad =  2 * np.sum(X(Т)@(preds - y))/X.shape[0]
                self.b -=  lr * b_grad
                self.W -=  lr * W_grad
                metrics.append(self.metric(preds, y))
            return metrics

        
        def predict(self, X):
            preds = X@self.W+self.b
            return preds