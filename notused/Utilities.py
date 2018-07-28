
class Utils:
    @staticmethod
    def optimizer_update(optimizers):
        for o in optimizers:
            o.update()

    @staticmethod
    def backward(losses):
        for loss in losses:
            loss.backward()

    @staticmethod
    def reset_all(models):
        for model in models:
            model.reset_state()
            model.cleargrads()


