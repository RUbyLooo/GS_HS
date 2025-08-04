import abc


class WheelController(abc.ABC):

    @abc.abstractmethod
    def ctrl(self, vx, vy, wz):
        pass
