import os
os.environ["MUJOCO_GL"] = "egl"  # 如果你有 GUI 环境
import mujoco
import mujoco.viewer



class BaseViewer:
    def __init__(self, model_path, distance=3, azimuth=0, elevation=-30):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.distance = distance
        self.azimuth = azimuth
        self.elevation = elevation
        self.handle = mujoco.viewer.launch_passive(self.model, self.data)
        self.handle.cam.distance = distance
        self.handle.cam.azimuth = azimuth
        self.handle.cam.elevation = elevation

        self._cam = mujoco.MjvCamera()
        self._scn = mujoco.MjvScene(self.model, maxgeom=1000)
        self.opt = mujoco.MjvOption()


    def is_running(self):
        return self.handle.is_running()

    def sync(self):
        self.handle.sync()

    @property
    def cam(self):
        return self.handle.cam

    @property
    def viewport(self):
        return self.handle.viewportW

    def run_loop(self):
        self.run_before()
        while self.is_running():
            mujoco.mj_forward(self.model, self.data)
            self.run_func()
            mujoco.mj_step(self.model, self.data)
            self.sync()
    
    def run_before(self):
        pass
        

    def run_func(self):
        pass