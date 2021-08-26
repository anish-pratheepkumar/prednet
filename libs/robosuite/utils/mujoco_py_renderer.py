import time
from collections import defaultdict

import glfw
import numpy as np
from mujoco_py import MjViewer, functions
from mujoco_py.generated import const


class CustomMjViewer(MjViewer):
    keypress = defaultdict(list)
    keyup = defaultdict(list)
    keyrepeat = defaultdict(list)

    def __init__(self, parent_viewer, *args):
        self.parent_viewer = parent_viewer
        super(CustomMjViewer, self).__init__(*args)

        self._last_button_left_pressed = False
        self._last_button_right_pressed = False
        self._last2_button_left_pressed = False
        self._last2_button_right_pressed = False
        self._press_moment = True

    def key_callback(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            tgt = self.keypress
        elif action == glfw.RELEASE:
            tgt = self.keyup
        elif action == glfw.REPEAT:
            tgt = self.keyrepeat
        else:
            return
        if tgt.get(key):
            for fn in tgt[key]:
                fn(window, key, scancode, action, mods)
        if tgt.get("any"):
            for fn in tgt["any"]:
                fn(window, key, scancode, action, mods)
            # retain functionality for closing the viewer
            if key == glfw.KEY_ESCAPE:
                super().key_callback(window, key, scancode, action, mods)
        else:
            # only use default mujoco callbacks if "any" callbacks are unset
            super().key_callback(window, key, scancode, action, mods)

    # def _cursor_pos_callback(self, window, xpos, ypos):
    #     super(CustomMjViewer, self)._cursor_pos_callback(window, xpos, ypos)
    #     self.select(window, self._last_mouse_x, self._last_mouse_y)

    def _mouse_button_callback(self, window, button, act, mods):
        self._last2_button_left_pressed = self._last_button_left_pressed
        self._last2_button_right_pressed = self._last_button_right_pressed
        self._last_button_left_pressed = self._button_left_pressed
        self._last_button_right_pressed = self._button_right_pressed
        super(CustomMjViewer, self)._mouse_button_callback(window, button, act, mods)
        # print(self._button_left_pressed, self._last_button_left_pressed, self._last2_button_left_pressed)
        if self._button_left_pressed:
            self._press_moment = time.time()
        if self.parent_viewer.edit_key_enabled and (
                self._button_left_pressed == False
                and self._last_button_left_pressed == True
                and self._last2_button_left_pressed == False
        ):
            release_time = time.time() - self._press_moment
            was_dragging = release_time > 0.2
            if was_dragging:
                # print("Was dragging: {}s".format(release_time))
                return
            else:
                # or (                self._button_right_pressed and not self._last_button_right_pressed):
                # key pressed then released
                self.select(window, self._last_mouse_x, self._last_mouse_y)
                # print("Left Key pressed then released")
                self._last2_button_left_pressed = False
                return

    def select(self, window, cursor_x, cursor_y):
        """Returns bodies and geoms visible at given coordinates in the frame.
        Args:
          cursor_position:  A `tuple` containing x and y coordinates, normalized to
            between 0 and 1, and where (0, 0) is bottom-left.
        Returns:
          A `Selected` namedtuple. Fields are None if nothing is selected.
        """
        # self.update()
        width, height = glfw.get_framebuffer_size(window)
        aspect_ratio = width / height
        pos = np.empty(3, np.double)
        pos = np.asarray(pos, order='C')
        scn = self.scn  # self.sim.render_contexts[0].scn
        # geom_id_arr = np.intc([0])
        # skin_id_arr = np.intc([0])
        geom_id_arr = np.asarray([-1], dtype=np.intc,
                                 order='C')  # np.intc([-1])# np.array([-1], dtype=np.int32)  # np.intc([0])
        skin_id_arr = np.asarray([-1], dtype=np.intc,
                                 order='C')  # np.intc([-1]) #np.array([-1], dtype=np.int32)  # np.intc([0])
        # def _mjv_select(PyMjModel m, PyMjData d, PyMjvOption vopt, float aspectratio, float relx, float rely, PyMjvScene scn, np.ndarray[np.float64_t, mode="c", ndim=1] selpnt, uintptr_t geomid, uintptr_t skinid):
        #     return mjv_select(m.ptr, d.ptr, vopt.ptr, aspectratio, relx, rely, scn.ptr, &selpnt[0], <int*>geomid, <int*>skinid)
        # # Select geom or skin with mouse, return bodyid; -1: none selected.
        #     int mjv_select(const mjModel* m, const mjData* d, const mjvOption* vopt,
        #                    mjtNum aspectratio, mjtNum relx, mjtNum rely,
        #                    const mjvScene* scn, mjtNum* selpnt, int* geomid, int* skinid);
        # ---------------------
        x = cursor_x / width
        y = 1. - cursor_y / height
        viewport_pos = np.array([x, y], np.float32)
        # ----------------------
        np.asarray([-1], dtype=np.intc, order='C')
        # print("here")
        body_id = functions.mjv_select(
            self.sim.model,
            self.sim.data,
            self.vopt,
            np.float32(aspect_ratio),
            viewport_pos[0],
            viewport_pos[1],
            scn,
            pos,
            geom_id_arr.__array_interface__['data'][0],
            skin_id_arr.__array_interface__['data'][0])
        # print("passed")
        [geom_id] = geom_id_arr
        [skin_id] = skin_id_arr
        if body_id > 0:
            # print("selected body: ", self.sim.model.body_id2name(body_id))

            self.pert.select = body_id
            self.pert.active = 0

            functions.mjv_initPerturb(self.sim.model, self.sim.data, self.scn,
                                      self.pert)
        # Validate IDs
        if body_id != -1:
            assert 0 <= body_id < self.sim.model.nbody
        else:
            body_id = None
        if geom_id != -1:
            assert 0 <= geom_id < self.sim.model.ngeom
        else:
            geom_id = None
        if skin_id != -1:
            assert 0 <= skin_id < self.sim.model.nskin
        else:
            skin_id = None

        if all(id_ is None for id_ in (body_id, geom_id, skin_id)):
            pos = None

        return body_id, geom_id, skin_id, pos  # Selected(            body=body_id, geom=geom_id, skin=skin_id, world_position=pos)


class MujocoPyRenderer:
    def __init__(self, sim):
        """
        Args:
            sim: MjSim object
        """
        self.viewer = CustomMjViewer(self, sim)
        self.callbacks = {}
        self.edit_key_enabled = False

        self.add_keyup_callback(glfw.KEY_X, self.callback)
        self.add_keyup_callback(glfw.KEY_L, self.visualize_labels)
        self.add_keyup_callback(glfw.KEY_V, self.do_nothing)
    def do_nothing(self,*_):
        pass
    def visualize_labels(self, *_):
        self.viewer.vopt.label = int(not self.viewer.vopt.label)

    def callback(self, *_):
        # print("X pressed")
        self.edit_key_enabled = not self.edit_key_enabled

    def set_camera(self, camera_id):
        """
        Set the camera view to the specified camera ID.
        """
        self.viewer.cam.fixedcamid = camera_id
        self.viewer.cam.type = const.CAMERA_FIXED

    def render(self):
        # safe for multiple calls
        self.viewer.render()

    def close(self):
        """
        Destroys the open window and renders (pun intended) the viewer useless.
        """
        glfw.destroy_window(self.viewer.window)
        self.viewer = None

    def add_keypress_callback(self, key, fn):
        """
        Allows for custom callback functions for the viewer. Called on key down.
        Parameter 'any' will ensure that the callback is called on any key down,
        and block default mujoco viewer callbacks from executing, except for
        the ESC callback to close the viewer.
        """
        self.viewer.keypress[key].append(fn)

    def add_keyup_callback(self, key, fn):
        """
        Allows for custom callback functions for the viewer. Called on key up.
        Parameter 'any' will ensure that the callback is called on any key up,
        and block default mujoco viewer callbacks from executing, except for
        the ESC callback to close the viewer.
        """
        self.viewer.keyup[key].append(fn)

    def add_keyrepeat_callback(self, key, fn):
        """
        Allows for custom callback functions for the viewer. Called on key repeat.
        Parameter 'any' will ensure that the callback is called on any key repeat,
        and block default mujoco viewer callbacks from executing, except for
        the ESC callback to close the viewer.
        """
        self.viewer.keyrepeat[key].append(fn)
