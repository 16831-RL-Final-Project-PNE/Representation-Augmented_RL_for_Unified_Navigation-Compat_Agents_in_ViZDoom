# doom_env.py
# Headless ViZDoom env with RGB input, frame stacking, and a unified 12-action discrete space.
import numpy as np
from collections import deque
from typing import Tuple, Optional, List
from PIL import Image
import vizdoom as vzd
from pkg_resources import resource_filename

def _save_gif(out_file: str, frames: list[np.ndarray], fps: int) -> None:
    """
    Save a list of HWC uint8 RGB frames as an animated GIF using Pillow.

    This bypasses imageio's GIF writer to avoid transparency / palette issues.
    """
    if not frames:
        return

    pil_frames = [Image.fromarray(f) for f in frames]

    # duration is in milliseconds per frame
    duration_ms = int(1000 / max(1, fps))

    pil_frames[0].save(
        out_file,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )

def _to_rgb(buf):
    import numpy as np
    arr = np.asarray(buf)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    # CHW -> HWC if needed
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    elif arr.ndim == 2:  # HxW -> HxWx1
        arr = arr[..., None]

    # drop alpha if present; expand gray to RGB
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    elif arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    return arr.copy()


# ---- unified 12-action space built from {FORWARD, LEFT, RIGHT, SHOOT} (no contradictory combos) ----
BUTTONS = [vzd.Button.MOVE_FORWARD, vzd.Button.TURN_LEFT,
           vzd.Button.TURN_RIGHT, vzd.Button.ATTACK]
ACTIONS: List[List[int]] = [
    [0,0,0,0],  # 0 noop
    [1,0,0,0],  # 1 forward
    [0,1,0,0],  # 2 left
    [0,0,1,0],  # 3 right
    [0,0,0,1],  # 4 shoot
    [1,1,0,0],  # 5 forward+left
    [1,0,1,0],  # 6 forward+right
    [1,0,0,1],  # 7 forward+shoot
    [0,1,0,1],  # 8 left+shoot
    [0,0,1,1],  # 9 right+shoot
    [1,1,0,1],  # 10 forward+left+shoot
    [1,0,1,1],  # 11 forward+right+shoot
]

class DoomEnv:
    """
    Minimal Gym-like wrapper (without Gym dependency):
      - reset() -> obs
      - step(a) -> obs, reward, done, info
      - render('rgb_array') -> HxWx3 uint8
    Observations: RGB, downsampled to (H,W) and frame-stacked along channels: (frame_stack, 3, H, W) uint8.
    """
    def __init__(self,
                 scenario: str = "basic",    # "basic" or "my_way_home"
                 frame_repeat: int = 4,      # repeat the chosen action for N game tics
                 frame_stack: int = 4,       # concatenate last N frames
                 width: int = 84,
                 height: int = 84,
                 seed: int = 0,
                 base_res: str="320x240",
                 window_visible: bool = False,
                 sound_enabled: bool = False):
        assert scenario in ("basic", "my_way_home", "mywayhome", "mwh")
        self.scenario = "my_way_home" if scenario in ("mywayhome", "mwh") else scenario
        self.frame_repeat = int(frame_repeat)
        self.frame_stack = int(frame_stack)
        self.width, self.height = int(width), int(height)
        self._stack: deque = deque(maxlen=self.frame_stack)
        self._last_rgb: Optional[np.ndarray] = None
        self._last_rgb_native: Optional[np.ndarray] = None

        # --- build and init the ViZDoom game ---
        self.game = vzd.DoomGame()
        cfg_path = resource_filename(vzd.__name__,
                                     f"scenarios/{'my_way_home' if self.scenario=='my_way_home' else 'basic'}.cfg")
        self.game.load_config(cfg_path)
        self.game.set_window_visible(bool(window_visible))
        self.game.set_sound_enabled(bool(sound_enabled))
        self.game.set_seed(int(seed))

        # RGB frames
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        res_map = {
                    "160x120":  vzd.ScreenResolution.RES_160X120,
                    "320x240":  vzd.ScreenResolution.RES_320X240,
                    "800x600":  vzd.ScreenResolution.RES_800X600,
        }
        # Use a higher base resolution; we will downsample to (width, height)
        self.game.set_screen_resolution(res_map.get(base_res, vzd.ScreenResolution.RES_320X240))

        # restrict to our unified buttons
        self.game.clear_available_buttons()
        for b in BUTTONS:
            self.game.add_available_button(b)

        self.game.set_mode(vzd.Mode.PLAYER)
        self.game.init()

        # zeros frame for stack bootstrapping
        self._zero = np.zeros((3, self.height, self.width), dtype=np.uint8)
        self._reset_episode_stats()

    def _reset_episode_stats(self):
        self._ep_return: float = 0.0
        self._ep_tics: int = 0
        self._ep_kills: int = 0
        self._ep_dead: bool = False
        self._ep_reason: Optional[str] = None

    # ------------- helpers -------------
    def _read_rgb(self) -> np.ndarray:
        state = self.game.get_state()
        if state is None or state.screen_buffer is None:
            # return last image if available
            if self._last_rgb is not None:
                return np.transpose(self._last_rgb, (2, 0, 1))
            return self._zero
        # ViZDoom gives (C,H,W) uint8
        return state.screen_buffer

    def _proc_frame(self, buf: np.ndarray) -> np.ndarray:
        """Convert any buffer to RGB HWC, resize, then return CHW uint8."""
        # 1) normalize to HWC RGB uint8 (handles CHW/GRAY/alpha/odd shapes)
        hwc = _to_rgb(buf)                     # -> (H, W, 3) uint8
        self._last_rgb_native = hwc  # store native HWC frame before resizing

        # 2) resize to (self.width, self.height)
        img = Image.fromarray(hwc)             # safe now: strictly HWC RGB
        img = img.resize((self.width, self.height), Image.BILINEAR)
        hwc_resized = np.asarray(img, dtype=np.uint8)

        # 3) back to CHW
        chw = np.transpose(hwc_resized, (2, 0, 1))
        return chw

    @property
    def observation_shape(self):
        return (self.frame_stack, 3, self.height, self.width)  # (T, C, H, W)

    def _get_obs(self) -> np.ndarray:
        if len(self._stack) < self.frame_stack:
            frames = [self._zero]*(self.frame_stack - len(self._stack)) + list(self._stack)
        else:
            frames = list(self._stack)
        return np.stack(frames, axis=0)  # (T, C, H, W)

    # ------------- public API -------------
    def episode_summary(self) -> dict:
        """Return last-known episode summary (valid after reset() and during/after step())."""
        return {
            "return": float(self._ep_return),
            "tics": int(self._ep_tics),
            "secs": float(self._ep_tics / 35.0),
            "kills": int(self._ep_kills),
            "dead": bool(self._ep_dead),
            "reason": self._ep_reason,
        }        

    @property
    def action_space_n(self) -> int:
        return len(ACTIONS)

    def reset(self) -> np.ndarray:
        self.game.new_episode()
        self._stack.clear()
        self._reset_episode_stats()

        # fill stack with zeros except the last frame from the env
        for _ in range(self.frame_stack - 1):
            self._stack.append(self._zero)
        obs = self._proc_frame(self._read_rgb())
        self._stack.append(obs)
        self._last_rgb = self._last_rgb_native  # render will return native size
        return self._get_obs()

    def step(self, action: int):
        action = int(action)

        # ---- move forward through every tic frame ----
        self.game.set_action(ACTIONS[action])
        step_reward = 0.0
        tic_frames = []
        last_hwc = None

        for _ in range(self.frame_repeat):
            self.game.advance_action()
            step_reward += self.game.get_last_reward()
            s = self.game.get_state()
            if s is not None and s.screen_buffer is not None:
                hwc = _to_rgb(s.screen_buffer)
                tic_frames.append(hwc)          # ← tic 
                self._last_rgb_native = hwc     # ← render() using original resolution
                last_hwc = hwc

        done = self.game.is_episode_finished()

        # ---- accumulate ----
        self._ep_return += step_reward
        self._ep_tics   += self.frame_repeat
        try:
            self._ep_kills = int(self.game.get_game_variable(vzd.GameVariable.KILLCOUNT))
        except Exception:
            self._ep_kills = 0
        self._ep_dead = bool(self.game.is_player_dead())
        info = {"kills": self._ep_kills, "dead": self._ep_dead, "tic_frames": tic_frames}

        if not done:
            # use last tic frame to generate obs
            if last_hwc is not None:
                obs = self._proc_frame(last_hwc)  # _last_rgb_native
            else:
                obs = self._proc_frame(self._read_rgb())
            self._stack.append(obs)
            self._last_rgb = self._last_rgb_native
        else:
            # terminate reason
            if self._ep_dead:
                self._ep_reason = "death"
            elif (self.scenario == "basic" and self._ep_kills >= 1):
                self._ep_reason = "goal: kill"
            elif (self.scenario == "my_way_home" and self._ep_return > 0):
                self._ep_reason = "goal: reached vest"
            else:
                self._ep_reason = "timeout"

            info.update({
                "reason": self._ep_reason,
                "tics": self._ep_tics,
                "secs": self._ep_tics / 35.0,
                "return": self._ep_return,
                "episode": self.episode_summary(),
            })
            if self._last_rgb_native is not None:
                # Make sure we keep a copy of the last visible RGB frame
                info["last_screen"] = self._last_rgb_native.copy()

        return self._get_obs(), float(step_reward), bool(done), info

    '''
    def render(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        if mode != "rgb_array":
            return None
        if self._last_rgb is None:
            return np.transpose(self._zero, (1, 2, 0))
        return self._last_rgb
    '''

    def render(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        """
        Return the current ViZDoom RGB frame in HWC uint8 format.

        We always try to pull the latest screen_buffer from the engine.
        If for some reason it's not available, we fall back to the last
        native frame we saw, and finally to a black image.
        """
        if mode != "rgb_array":
            return None

        if self.game is not None:
            state = self.game.get_state()
        else:
            state = None

        if state is not None and state.screen_buffer is not None:
            # Convert ViZDoom's CHW buffer to HWC RGB uint8
            frame = _to_rgb(state.screen_buffer)     # (H, W, 3)
            self._last_rgb_native = frame
            return frame

        # Fallbacks if state/screen_buffer is not available
        if self._last_rgb_native is not None:
            return self._last_rgb_native

        # Last resort: black frame in HWC
        return np.transpose(self._zero, (1, 2, 0))

    def close(self):
        if self.game is not None:
            self.game.close()
            self.game = None
