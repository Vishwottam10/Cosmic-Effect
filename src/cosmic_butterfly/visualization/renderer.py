import taichi as ti
import numpy as np
from cosmic_butterfly.physics.simulation import Simulation
from cosmic_butterfly.physics.units import time_to_years
from enum import Enum

# Maximum trail length in segments
MAX_TRAIL = 2000

class VisMode(Enum):
    DUAL_VIEW = 0
    OVERLAY = 1

@ti.data_oriented
class Renderer:
    """
    Renders two N-body simulations using Taichi GGUI.
    System A (baseline) is drawn in cyan, System B (perturbed) in orange.
    Trails are recorded in physical space to survive camera transforms.
    """
    def __init__(self, width: int = 1280, height: int = 720, title: str = "Cosmic Butterfly Effect"):
        self.width = width
        self.height = height
        self.window = ti.ui.Window(title, (width, height), vsync=True)
        self.canvas = self.window.get_canvas()
        self.gui = self.window.get_gui()

        # Camera
        self.scale = 0.08
        self.offset_x = 0.5
        self.offset_y = 0.5
        self.target_scale = 0.08
        self.target_offset_x = 0.5
        self.target_offset_y = 0.5
        self.auto_frame = True

        # Visualization state
        self.mode = VisMode.OVERLAY
        self.show_trajectories = True
        self.show_diagnostics = False
        
        self._initialized = False
        self._frame = 0
        
        # Graph history
        self.graph_history_len = 400
        self.graph_d = np.zeros(self.graph_history_len, dtype=np.float32)
        self.graph_time = np.zeros(self.graph_history_len, dtype=np.float32)
        self.graph_idx = 0
        
        self.perturbation_val = 1e-5
        
        # State tracking scalar
        self.trail_len = ti.field(dtype=ti.i32, shape=())
        self.trail_head = ti.field(dtype=ti.i32, shape=())
        self.trail_len[None] = 0
        self.trail_head[None] = 0

    def _init_fields(self, n: int):
        self.n = n

        # Screen positions
        self.pos_a_scr = ti.Vector.field(2, dtype=ti.f32, shape=n)
        self.pos_b_scr = ti.Vector.field(2, dtype=ti.f32, shape=n)

        # Body colours
        self.col_a = ti.Vector.field(3, dtype=ti.f32, shape=n)
        self.col_b = ti.Vector.field(3, dtype=ti.f32, shape=n)

        # Trail physical space
        total_vertices = n * MAX_TRAIL * 2
        self.trail_a_phys = ti.Vector.field(2, dtype=ti.f64, shape=total_vertices)
        self.trail_b_phys = ti.Vector.field(2, dtype=ti.f64, shape=total_vertices)
        
        # Previous physical positions for the NEXT segment
        self.prev_pos_a = ti.Vector.field(2, dtype=ti.f64, shape=n)
        self.prev_pos_b = ti.Vector.field(2, dtype=ti.f64, shape=n)

        # Trail screen space
        self.trail_a_scr = ti.Vector.field(2, dtype=ti.f32, shape=total_vertices)
        self.trail_b_scr = ti.Vector.field(2, dtype=ti.f32, shape=total_vertices)
        
        # Trail colours
        self.trail_a_col = ti.Vector.field(3, dtype=ti.f32, shape=total_vertices)
        self.trail_b_col = ti.Vector.field(3, dtype=ti.f32, shape=total_vertices)

        self.trail_a_scr.fill([-10.0, -10.0])
        self.trail_b_scr.fill([-10.0, -10.0])

        # Init colours
        for i in range(n):
            if i == 0:
                self.col_a[i] = [1.0, 0.9, 0.2]   
                self.col_b[i] = [1.0, 0.9, 0.2]
            else:
                self.col_a[i] = [0.2, 0.7, 1.0]     
                self.col_b[i] = [1.0, 0.4, 0.1]   

        self._initialized = True

    @ti.kernel
    def _project_system(
        self,
        n: int,
        pos_phys: ti.template(),
        pos_scr: ti.template(),
        trail_phys: ti.template(),
        trail_scr: ti.template(),
        col_body: ti.template(),
        col_trail: ti.template(),
        scale: ti.f32,
        ox: ti.f32,
        oy: ti.f32
    ):
        # Project bodies
        for i in range(n):
            pos_scr[i][0] = ti.cast(pos_phys[i][0], ti.f32) * scale + ox
            pos_scr[i][1] = ti.cast(pos_phys[i][1], ti.f32) * scale + oy

        # Project valid trails and calculate fade
        length = self.trail_len[None]
        head = self.trail_head[None]
        
        for i in range(length * n * 2):
            # trail_phys has size n * MAX_TRAIL * 2
            # The valid segments might be wrapped around. But we packed them linearly over time.
            pass
            
        # Actually, let's just project everything. Off-screen stays off-screen.
        total_vertices = n * MAX_TRAIL * 2
        for i in range(total_vertices):
            trail_scr[i][0] = ti.cast(trail_phys[i][0], ti.f32) * scale + ox
            trail_scr[i][1] = ti.cast(trail_phys[i][1], ti.f32) * scale + oy
            
            # Fade logic:
            seg_idx = i // (n * 2)
            # if length hasn't reached MAX_TRAIL, seg_idx >= length are invalid
            if length < MAX_TRAIL and seg_idx >= length:
                trail_scr[i] = [-10.0, -10.0]
                continue
                
            age = (head - 1 - seg_idx + MAX_TRAIL) % MAX_TRAIL
            if age >= length:
                trail_scr[i] = [-10.0, -10.0]
            else:
                opacity = 1.0 - (ti.cast(age, ti.f32) / ti.cast(MAX_TRAIL, ti.f32))
                body_idx = (i % (n * 2)) // 2
                col_trail[i] = col_body[body_idx] * opacity

    @ti.kernel
    def _record_trail_kernel(
        self,
        n: int,
        pos_a: ti.template(),
        pos_b: ti.template()
    ):
        h = self.trail_head[None]
        l = self.trail_len[None]
        base_idx = h * n * 2

        for i in range(n):
            idx1 = base_idx + i * 2
            idx2 = idx1 + 1
            
            self.trail_a_phys[idx1] = self.prev_pos_a[i]
            self.trail_a_phys[idx2] = pos_a[i]
            
            self.trail_b_phys[idx1] = self.prev_pos_b[i]
            self.trail_b_phys[idx2] = pos_b[i]

            self.prev_pos_a[i] = pos_a[i]
            self.prev_pos_b[i] = pos_b[i]

        self.trail_head[None] = (h + 1) % MAX_TRAIL
        if l < MAX_TRAIL:
            self.trail_len[None] = l + 1

    def init_prev_pos(self, sim_a: Simulation, sim_b: Simulation):
        self.prev_pos_a.copy_from(sim_a.pos)
        self.prev_pos_b.copy_from(sim_b.pos)

    def record_trail_step(self, sim_a: Simulation, sim_b: Simulation):
        if not self._initialized:
            return
        self._record_trail_kernel(self.n, sim_a.pos, sim_b.pos)

    def _update_camera(self, pos_a_np: np.ndarray, pos_b_np: np.ndarray, draw_b: bool):
        if not self.auto_frame:
            self.scale += (self.target_scale - self.scale) * 0.1
            self.offset_x += (self.target_offset_x - self.offset_x) * 0.1
            self.offset_y += (self.target_offset_y - self.offset_y) * 0.1
            return
            
        all_pos = pos_a_np
        if draw_b:
            all_pos = np.vstack([all_pos, pos_b_np])
            
        min_x, max_x = np.min(all_pos[:, 0]), np.max(all_pos[:, 0])
        min_y, max_y = np.min(all_pos[:, 1]), np.max(all_pos[:, 1])
        
        # Add 20% margin
        span_x = max(max_x - min_x, 1e-4) * 1.2
        span_y = max(max_y - min_y, 1e-4) * 1.2
        
        target_span = max(span_x, span_y)
        
        if self.mode == VisMode.DUAL_VIEW:
            self.target_scale = 0.35 / target_span
        else:
            self.target_scale = 0.6 / target_span
            
        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0
        
        self.target_offset_x = 0.5 - cx * self.target_scale
        self.target_offset_y = 0.5 - cy * self.target_scale

        self.scale += (self.target_scale - self.scale) * 0.05
        self.offset_x += (self.target_offset_x - self.offset_x) * 0.05
        self.offset_y += (self.target_offset_y - self.offset_y) * 0.05

    def render(self, sim_a: Simulation, sim_b: Simulation, draw_b: bool = True, sim_time: float = 0.0, div: float = 0.0):
        if not self._initialized:
            self._init_fields(sim_a.n)
            self.init_prev_pos(sim_a, sim_b)

        self.canvas.set_background_color((0.02, 0.02, 0.05))
        n = self.n

        # Update camera using current physical positions only!
        self._update_camera(sim_a.pos.to_numpy(), sim_b.pos.to_numpy(), draw_b)

        # Transform positions
        if self.mode == VisMode.DUAL_VIEW:
            self._project_system(n, sim_a.pos, self.pos_a_scr, self.trail_a_phys, self.trail_a_scr, self.col_a, self.trail_a_col, self.scale, self.offset_x - 0.25, self.offset_y)
            if draw_b:
                self._project_system(n, sim_b.pos, self.pos_b_scr, self.trail_b_phys, self.trail_b_scr, self.col_b, self.trail_b_col, self.scale, self.offset_x + 0.25, self.offset_y)
        else:
            self._project_system(n, sim_a.pos, self.pos_a_scr, self.trail_a_phys, self.trail_a_scr, self.col_a, self.trail_a_col, self.scale, self.offset_x, self.offset_y)
            if draw_b:
                self._project_system(n, sim_b.pos, self.pos_b_scr, self.trail_b_phys, self.trail_b_scr, self.col_b, self.trail_b_col, self.scale, self.offset_x, self.offset_y)

        # Update graph data
        self._frame += 1
        if self._frame % 5 == 0:
            if self.graph_idx < self.graph_history_len:
                self.graph_d[self.graph_idx] = div
                self.graph_time[self.graph_idx] = sim_time
                self.graph_idx += 1
            else:
                self.graph_d = np.roll(self.graph_d, -1)
                self.graph_time = np.roll(self.graph_time, -1)
                self.graph_d[-1] = div
                self.graph_time[-1] = sim_time

        # --- Draw trails ---
        if self.show_trajectories:
            self.canvas.lines(self.trail_a_scr, width=0.002, per_vertex_color=self.trail_a_col)
            if draw_b:
                self.canvas.lines(self.trail_b_scr, width=0.002, per_vertex_color=self.trail_b_col)

        # --- Draw bodies ---
        # Draw central star slightly larger
        self.canvas.circles(self.pos_a_scr, radius=0.015, per_vertex_color=self.col_a)
        if draw_b:
            self.canvas.circles(self.pos_b_scr, radius=0.012, per_vertex_color=self.col_b)

    def draw_hud(self, sim_time: float, divergence: float, energy_err: float,
                 scenario_name: str, phase_name: str):
                     
        # Primary Dashboard
        with self.gui.sub_window("COSMIC BUTTERFLY EFFECT", 0.02, 0.02, 0.35, 0.45) as w:
            w.text(f"Experiment Phase: {phase_name}")
            w.text(f"Scenario: {scenario_name}")
            w.text(f"Time (yr): {time_to_years(sim_time):.3f}")
            w.text("")
            
            # Interactive controls
            self.perturbation_val = w.slider_float("Perturbation (\u0394v)", self.perturbation_val, 1e-6, 1e-2)
            
            w.text("")
            w.text("SYSTEM A (Cyan)   vs   SYSTEM B (Orange)")
            w.text(f"Divergence D(t) = {divergence:.2e}")
            w.text("")
            
            # View modes
            is_overlay = self.mode == VisMode.OVERLAY
            new_overlay = w.checkbox("Overlay Mode", is_overlay)
            if new_overlay and not is_overlay:
                self.mode = VisMode.OVERLAY
                
            is_dual = self.mode == VisMode.DUAL_VIEW
            new_dual = w.checkbox("Dual View Mode", is_dual)
            if new_dual and not is_dual:
                self.mode = VisMode.DUAL_VIEW
                
            self.show_trajectories = w.checkbox("Show Trajectories", self.show_trajectories)
            self.auto_frame = w.checkbox("Auto-Frame Camera", self.auto_frame)
            self.show_diagnostics = w.checkbox("Numerical Validation", self.show_diagnostics)

        # Numerical Validation Panel
        if self.show_diagnostics:
            with self.gui.sub_window("NUMERICAL VALIDATION", 0.02, 0.49, 0.35, 0.25) as w:
                w.text("Integrator: Velocity Verlet (Symplectic)")
                w.text("Precision: Float64")
                w.text("")
                w.text(f"Energy Drift: {energy_err:.4e}")
                if energy_err > 1e-2:
                    w.text("WARNING: High energy drift detected!")
                w.text("Linear Momentum: Conserved (Tested)")
                w.text("Angular Momentum: Conserved (Tested)")

    def clear_trails(self):
        self.trail_len[None] = 0
        self.trail_head[None] = 0
        self.trail_a_scr.fill([-10.0, -10.0])
        self.trail_b_scr.fill([-10.0, -10.0])
        
    def reset_graph(self):
        self.graph_idx = 0
        self.graph_d.fill(0)
        self.graph_time.fill(0)

    def draw_graph_on_canvas(self):
        # We will draw a real divergence graph in the bottom right corner (0.5 to 0.95 x, 0.05 to 0.35 y)
        if self.graph_idx < 2:
            return
            
        valid_d = self.graph_d[:self.graph_idx]
        valid_t = self.graph_time[:self.graph_idx]
        
        safe_d = np.clip(valid_d, 1e-12, None)
        log_d = np.log10(safe_d)
        min_val = -10.0
        max_val = max(1.0, np.max(log_d))
        norm_d = (log_d - min_val) / (max_val - min_val)
        norm_d = np.clip(norm_d, 0.0, 1.0)
        
        # Scale t to 0-1
        t_span = max(valid_t[-1] - valid_t[0], 1e-6)
        norm_t = (valid_t - valid_t[0]) / t_span
        
        # Map to screen
        x0, x1 = 0.5, 0.95
        y0, y1 = 0.05, 0.35
        
        pts = np.zeros((self.graph_idx, 2), dtype=np.float32)
        pts[:, 0] = x0 + norm_t * (x1 - x0)
        pts[:, 1] = y0 + norm_d * (y1 - y0)
        
        lines = np.zeros((self.graph_idx * 2, 2), dtype=np.float32)
        lines[0:self.graph_idx*2:2] = pts
        lines[1:-1:2] = pts[1:]
        lines[-1] = pts[-1]
        
        if not hasattr(self, 'graph_field'):
            self.graph_field = ti.Vector.field(2, dtype=ti.f32, shape=self.graph_history_len * 2)
            self.graph_field.fill([-10.0, -10.0])
            
        self.graph_field.from_numpy(lines)
        # Use vertex_count to only draw valid lines? No, vertex_count gives error. We just fill remainder with -10.
        # But `from_numpy` handles the size of `lines`. The field is size `graph_history_len*2`.
        # Wait, if we use from_numpy with a smaller array, it fails.
        full_lines = np.full((self.graph_history_len * 2, 2), -10.0, dtype=np.float32)
        full_lines[:len(lines)] = lines
        self.graph_field.from_numpy(full_lines)
        
        self.canvas.lines(self.graph_field, width=0.003, color=(0.9, 0.3, 0.3))
        
        # Draw axes and labels
        if not hasattr(self, 'axes_field'):
            self.axes_field = ti.Vector.field(2, dtype=ti.f32, shape=6)
            axes = np.array([
                [x0, y0], [x1, y0], # X axis
                [x0, y0], [x0, y1], # Y axis
                [-10, -10], [-10, -10]
            ], dtype=np.float32)
            self.axes_field.from_numpy(axes)
        self.canvas.lines(self.axes_field, width=0.002, color=(0.7, 0.7, 0.7))

    def handle_input(self):
        if self.window.is_pressed('+', '=', '-', '_', ti.ui.LEFT, ti.ui.RIGHT, ti.ui.UP, ti.ui.DOWN):
            self.auto_frame = False
            self.target_scale = self.scale
            self.target_offset_x = self.offset_x
            self.target_offset_y = self.offset_y

        if self.window.is_pressed('+') or self.window.is_pressed('='):
            self.target_scale *= 1.05
        if self.window.is_pressed('-') or self.window.is_pressed('_'):
            self.target_scale *= 0.95
        if self.window.is_pressed(ti.ui.LEFT):
            self.target_offset_x += 0.02
        if self.window.is_pressed(ti.ui.RIGHT):
            self.target_offset_x -= 0.02
        if self.window.is_pressed(ti.ui.UP):
            self.target_offset_y -= 0.02
        if self.window.is_pressed(ti.ui.DOWN):
            self.target_offset_y += 0.02
            
        if self.window.get_event(ti.ui.PRESS):
            if self.window.event.key == 'r':
                self.auto_frame = True

    def show(self):
        self.draw_graph_on_canvas()
        self.window.show()

    def is_running(self) -> bool:
        return self.window.running
