import pygame
import sys
import math
import random

# --- Constants ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

# --- PALETTE ---
BACKGROUND = (135, 206, 235)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK_GREY = (50, 50, 50) 

LIGHT_TOP = 1.0
LIGHT_RIGHT = 0.8
LIGHT_LEFT = 0.6

C_GRASS_TOP = (118, 215, 60)
C_ROAD_TOP = (90, 90, 100)
C_WATER_TOP = (100, 190, 255)
C_TREE_TRUNK = (120, 80, 40)
C_LOG = (160, 100, 50)
C_CAR_WIN = (100, 200, 250) 

C_TREE_VARIANTS = [(50, 160, 50), (30, 140, 40), (60, 180, 80), (40, 130, 60)]
C_CAR_VARIANTS = [(230, 80, 80), (80, 80, 230), (230, 230, 80), (230, 150, 50), (150, 50, 200)]

# Chicken Colors
C_CHICKEN_BODY = (255, 255, 240)
C_CHICKEN_RED = (255, 50, 50)
C_CHICKEN_BEAK = (255, 180, 50)

class Lane:
    def __init__(self, y_index, cols):
        self.y_index = y_index
        self.cols = cols
        self.type = "base"
        self.grid = [None] * cols 
        self.cars = []
        self.logs = []
        self.side_trees = []
        
        # --- FOREST GENERATION ---
        for cx in range(-6, 0):
            self.side_trees.append({'x': cx, 'color': random.choice(C_TREE_VARIANTS)})
        for cx in range(cols, cols + 6):
            self.side_trees.append({'x': cx, 'color': random.choice(C_TREE_VARIANTS)})
    
    def update(self):
        pass

    def check_collision(self, p_x):
        return False
    
    def is_blocked(self, x):
        return False
        
    def get_log_velocity(self, p_x):
        return None

class GrassLane(Lane):
    def __init__(self, y_index, cols):
        super().__init__(y_index, cols)
        self.type = "grass"
        n_trees = random.randint(1, 3)
        for _ in range(n_trees):
            idx = random.randint(0, cols-1)
            self.grid[idx] = random.choice(C_TREE_VARIANTS)

    def is_blocked(self, x):
        if 0 <= x < self.cols:
            return self.grid[x] is not None
        return True

class RoadLane(Lane):
    def __init__(self, y_index, cols):
        super().__init__(y_index, cols)
        self.type = "road"
        self.direction = 1 if random.random() > 0.5 else -1
        self.speed = random.uniform(0.03, 0.08)
        self.spawn_timer = 0
        self.spawn_rate = random.randint(120, 240)

    def update(self):
        for car in self.cars:
            car['x'] += self.speed * self.direction
        self.cars = [c for c in self.cars if -2 < c['x'] < self.cols + 2]
        
        self.spawn_timer += 1
        if self.spawn_timer > self.spawn_rate:
            self.spawn_timer = 0
            start_x = -1 if self.direction == 1 else self.cols
            self.cars.append({
                'x': start_x, 
                'width': random.uniform(1.4, 1.8),
                'color': random.choice(C_CAR_VARIANTS) 
            })

    def check_collision(self, p_x):
        for car in self.cars:
            if abs(p_x - car['x']) < (0.5 + car['width']/2): 
                return True
        return False

class WaterLane(Lane):
    def __init__(self, y_index, cols):
        super().__init__(y_index, cols)
        self.type = "water"
        self.direction = 1 if random.random() > 0.5 else -1
        self.speed = random.uniform(0.03, 0.06)
        
        dist = 0
        while dist < cols + 2:
            log_size = random.randint(2, 3)
            gap = random.randint(2, 4)
            self.logs.append({'x': dist - 2, 'width': log_size})
            dist += log_size + gap

    def update(self):
        for log in self.logs:
            log['x'] += self.speed * self.direction
            if self.direction == 1 and log['x'] > self.cols + 2:
                log['x'] = -3
            elif self.direction == -1 and log['x'] < -3:
                log['x'] = self.cols + 2

    def get_log_velocity(self, p_x):
        for log in self.logs:
            center_log = log['x'] + log['width']/2
            if abs(p_x - center_log) < (log['width']/2 + 0.3):
                return self.speed * self.direction
        return None 


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Trimetric Crossy Road")
        self.clock = pygame.time.Clock()
        
        self.screen_center_x = SCREEN_WIDTH // 2
        self.screen_center_y = SCREEN_HEIGHT // 2 

        # --- PROJECTION ---
        self.angle_x_rad = math.radians(15)  
        self.angle_y_rad = math.radians(125) 
        self.grid_scale = 50
        self.z_scale = 50

        # --- GAMEPLAY ---
        self.cols = 9
        self.lanes = {} 
        
        self.p_grid_x = 4
        self.p_grid_y = 0 
        self.visual_x = 4.0
        self.visual_y = 0.0
        self.visual_z = 0.0
        self.facing = (0, -1) 
        
        self.is_moving = False
        self.anim_phase = 0.0
        self.move_speed = 0.1
        self.start_pos = (4,0)
        self.target_pos = (4,0)
        self.move_drift_speed = 0.0
        
        self.is_dead = False

        for i in range(-20, 5): 
            self.add_lane(i, force_safe=(abs(i) < 3))

    def add_lane(self, y_index, force_safe=False):
        if force_safe:
            self.lanes[y_index] = GrassLane(y_index, self.cols)
            if abs(y_index) < 3:
                self.lanes[y_index].grid[4] = None
        else:
            r = random.random()
            if r < 0.4:
                self.lanes[y_index] = GrassLane(y_index, self.cols)
            elif r < 0.7:
                self.lanes[y_index] = RoadLane(y_index, self.cols)
            else:
                self.lanes[y_index] = WaterLane(y_index, self.cols)

    def get_basis_vectors(self):
        vx_x = math.cos(self.angle_x_rad) * self.grid_scale
        vx_y = math.sin(self.angle_x_rad) * self.grid_scale
        vy_x = math.cos(self.angle_y_rad) * self.grid_scale
        vy_y = math.sin(self.angle_y_rad) * self.grid_scale
        return (vx_x, vx_y), (vy_x, vy_y)

    def grid_to_screen(self, grid_x, grid_y, grid_z, cam_x_offset, cam_y_offset):
        (vx_x, vx_y), (vy_x, vy_y) = self.get_basis_vectors()

        screen_x = (grid_x * vx_x) + (grid_y * vy_x)
        screen_y = (grid_x * vx_y) + (grid_y * vy_y)
        screen_y -= grid_z * self.z_scale
        
        focus_x = cam_x_offset + 0.5
        focus_y = cam_y_offset + 0.5 - 3.0 
        
        cam_px_x = (focus_x * vx_x) + (focus_y * vy_x)
        cam_px_y = (focus_x * vx_y) + (focus_y * vy_y)

        final_x = screen_x - cam_px_x + self.screen_center_x
        final_y = screen_y - cam_px_y + self.screen_center_y
        
        return final_x, final_y

    def apply_lighting(self, color, factor):
        r, g, b = color
        return (int(r * factor), int(g * factor), int(b * factor))

    def draw_cube(self, gx, gy, gz, cam_x, cam_y, size_x=1.0, size_y=1.0, size_z=1.0, color=WHITE):
        c_top = self.apply_lighting(color, LIGHT_TOP)
        c_right = self.apply_lighting(color, LIGHT_RIGHT)
        c_left = self.apply_lighting(color, LIGHT_LEFT)
        
        b0 = self.grid_to_screen(gx,        gy,        gz, cam_x, cam_y)
        b1 = self.grid_to_screen(gx+size_x, gy,        gz, cam_x, cam_y)
        b2 = self.grid_to_screen(gx+size_x, gy+size_y, gz, cam_x, cam_y)
        b3 = self.grid_to_screen(gx,        gy+size_y, gz, cam_x, cam_y)

        t0 = self.grid_to_screen(gx,        gy,        gz+size_z, cam_x, cam_y)
        t1 = self.grid_to_screen(gx+size_x, gy,        gz+size_z, cam_x, cam_y)
        t2 = self.grid_to_screen(gx+size_x, gy+size_y, gz+size_z, cam_x, cam_y)
        t3 = self.grid_to_screen(gx,        gy+size_y, gz+size_z, cam_x, cam_y)

        pygame.draw.polygon(self.screen, c_right, [b1, b2, t2, t1])
        pygame.draw.polygon(self.screen, c_left, [b2, b3, t3, t2])
        pygame.draw.polygon(self.screen, c_top, [t0, t1, t2, t3])

    def draw_tree(self, x, y, z, cam_x, cam_y, leaves_color):
        self.draw_cube(x+0.25, y+0.25, z, cam_x, cam_y, 0.5, 0.5, 0.5, C_TREE_TRUNK)
        self.draw_cube(x, y, z+0.5, cam_x, cam_y, 1.0, 1.0, 1.0, leaves_color)

    def draw_car(self, x, y, z, width, color, cam_x, cam_y):
        chassis_height = 0.4
        self.draw_cube(x, y, z, cam_x, cam_y, size_x=width, size_y=1.0, size_z=chassis_height, color=color)
        
        cabin_width = width * 0.6
        cabin_offset = (width - cabin_width) / 2
        cabin_height = 0.35
        cabin_start_z = z + chassis_height
        
        self.draw_cube(x + cabin_offset, y + 0.1, cabin_start_z, cam_x, cam_y, 
                       size_x=cabin_width, size_y=0.8, size_z=cabin_height, color=color)
        
        self.draw_cube(x + cabin_offset + 0.05, y + 0.15, cabin_start_z + 0.1, cam_x, cam_y,
                       size_x=cabin_width - 0.1, size_y=0.7, size_z=cabin_height - 0.05, color=C_CAR_WIN)


    def draw_player(self, x, y, z, cam_x, cam_y):
        b_w, b_d, b_h = 0.6, 0.6, 0.8
        self.draw_cube(x, y, z, cam_x, cam_y, b_w, b_d, b_h, C_CHICKEN_BODY)
        
        dx, dy = self.facing
        beak_size = 0.2
        wing_size = 0.12 
        wing_z = z + 0.15 
        
        cx = x + b_w/2
        cy = y + b_d/2
        
        if (dx, dy) == (0, -1): # UP
            self.draw_cube(cx - beak_size/2, y - 0.1, z + 0.5, cam_x, cam_y, beak_size, 0.1, 0.2, C_CHICKEN_BEAK)
            self.draw_cube(cx - 0.1, y, z + b_h, cam_x, cam_y, 0.2, 0.2, 0.15, C_CHICKEN_RED)
            self.draw_cube(x - wing_size, cy - 0.2, wing_z, cam_x, cam_y, wing_size, 0.4, 0.3, WHITE)
            self.draw_cube(x + b_w, cy - 0.2, wing_z, cam_x, cam_y, wing_size, 0.4, 0.3, WHITE)
        elif (dx, dy) == (0, 1): # DOWN
            self.draw_cube(cx - beak_size/2, y + b_d, z + 0.5, cam_x, cam_y, beak_size, 0.1, 0.2, C_CHICKEN_BEAK)
            self.draw_cube(cx - 0.1, y + b_d - 0.2, z + b_h, cam_x, cam_y, 0.2, 0.2, 0.15, C_CHICKEN_RED)
            self.draw_cube(x - wing_size, cy - 0.2, wing_z, cam_x, cam_y, wing_size, 0.4, 0.3, WHITE)
            self.draw_cube(x + b_w, cy - 0.2, wing_z, cam_x, cam_y, wing_size, 0.4, 0.3, WHITE)
        elif (dx, dy) == (-1, 0): # LEFT
            self.draw_cube(x - 0.1, cy - beak_size/2, z + 0.5, cam_x, cam_y, 0.1, beak_size, 0.2, C_CHICKEN_BEAK)
            self.draw_cube(x, cy - 0.1, z + b_h, cam_x, cam_y, 0.2, 0.2, 0.15, C_CHICKEN_RED)
            self.draw_cube(cx - 0.2, y - wing_size, wing_z, cam_x, cam_y, 0.4, wing_size, 0.3, WHITE)
            self.draw_cube(cx - 0.2, y + b_d, wing_z, cam_x, cam_y, 0.4, wing_size, 0.3, WHITE)
        elif (dx, dy) == (1, 0): # RIGHT
            self.draw_cube(x + b_w, cy - beak_size/2, z + 0.5, cam_x, cam_y, 0.1, beak_size, 0.2, C_CHICKEN_BEAK)
            self.draw_cube(x + b_w - 0.2, cy - 0.1, z + b_h, cam_x, cam_y, 0.2, 0.2, 0.15, C_CHICKEN_RED)
            self.draw_cube(cx - 0.2, y - wing_size, wing_z, cam_x, cam_y, 0.4, wing_size, 0.3, WHITE)
            self.draw_cube(cx - 0.2, y + b_d, wing_z, cam_x, cam_y, 0.4, wing_size, 0.3, WHITE)

    def trigger_move(self, dx, dy):
        if self.is_moving or self.is_dead: return

        self.facing = (dx, dy)
        tx = self.p_grid_x + dx
        ty = self.p_grid_y + dy
        
        if not (0 <= tx < self.cols): return

        current_lane = self.lanes.get(self.p_grid_y)
        target_lane = self.lanes.get(ty)

        if target_lane and target_lane.type == 'grass' and target_lane.is_blocked(tx):
            return

        self.move_drift_speed = 0.0
        if dy == 0 and current_lane and current_lane.type == 'water':
            vel = current_lane.get_log_velocity(self.visual_x)
            if vel is not None:
                self.move_drift_speed = vel

        self.is_moving = True
        self.anim_phase = 0.0
        self.start_pos = (self.visual_x, self.visual_y)
        self.target_pos = (float(tx), float(ty))
        self.p_grid_x = tx
        self.p_grid_y = ty

        min_y = min(self.lanes.keys())
        if self.p_grid_y < min_y + 15:
            for i in range(min_y - 1, min_y - 6, -1):
                self.add_lane(i)
        
        max_y = max(self.lanes.keys())
        if max_y > self.p_grid_y + 25:
            del self.lanes[max_y]

    def update(self):
        if self.is_dead: return

        for lane in self.lanes.values():
            lane.update()

        if self.is_moving:
            if self.move_drift_speed != 0:
                self.start_pos = (self.start_pos[0] + self.move_drift_speed, self.start_pos[1])
                self.target_pos = (self.target_pos[0] + self.move_drift_speed, self.target_pos[1])

            self.anim_phase += self.move_speed
            if self.anim_phase >= 1.0:
                self.anim_phase = 1.0
                self.is_moving = False
                self.visual_x = self.target_pos[0]
                self.visual_y = self.target_pos[1]
                self.visual_z = 0.0
                self.p_grid_x = int(round(self.visual_x))
            else:
                s_x, s_y = self.start_pos
                t_x, t_y = self.target_pos
                self.visual_x = s_x + (t_x - s_x) * self.anim_phase
                self.visual_y = s_y + (t_y - s_y) * self.anim_phase
                self.visual_z = 4 * self.anim_phase * (1 - self.anim_phase)

        current_lane = self.lanes.get(self.p_grid_y)
        
        if current_lane and current_lane.type == 'water' and not self.is_moving:
            vel = current_lane.get_log_velocity(self.visual_x)
            if vel is None:
                self.is_dead = True
            else:
                self.visual_x += vel
                self.p_grid_x = int(round(self.visual_x)) 
                if self.p_grid_x < 0 or self.p_grid_x >= self.cols:
                    self.is_dead = True

        if current_lane and current_lane.type == 'road':
            if current_lane.check_collision(self.visual_x):
                self.is_dead = True

    def run(self):
        running = True
        font = pygame.font.SysFont("Arial", 40, bold=True)

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.KEYDOWN:
                    if not self.is_dead:
                        if event.key == pygame.K_UP or event.key == pygame.K_w:
                            self.trigger_move(0, -1) 
                        if event.key == pygame.K_DOWN or event.key == pygame.K_s:
                            self.trigger_move(0, 1)  
                        if event.key == pygame.K_LEFT or event.key == pygame.K_a:
                            self.trigger_move(-1, 0)
                        if event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                            self.trigger_move(1, 0)
                    else:
                        if event.key == pygame.K_r:
                            self.__init__()

            self.update()
            self.screen.fill(BACKGROUND)

            # --- RENDER LOOP ---
            center_y_int = int(round(self.visual_y))
            start_row = center_y_int - 15  
            end_row = center_y_int + 20 

            player_render_row = int(math.floor(self.visual_y + 0.5))
            cam_fixed_x = self.cols // 2 

            for row in range(start_row, end_row):
                if row not in self.lanes: continue
                lane = self.lanes[row]

                base_color = C_GRASS_TOP
                if lane.type == 'road': base_color = C_ROAD_TOP
                if lane.type == 'water': base_color = C_WATER_TOP
                floor_z = -0.2 if lane.type == 'water' else 0
                
                # Draw Floor First (Background)
                for col in range(-6, self.cols + 6):
                    self.draw_cube(col, row, floor_z, cam_fixed_x, self.visual_y, size_z=1.0, color=base_color)

                # --- DEPTH SORTING FOR ROW OBJECTS ---
                # We collect all renderable objects in this row into a list
                # format: (sort_x, type, data)
                render_list = []

                # 1. Side Trees
                for tree in lane.side_trees:
                    render_list.append((tree['x'], 'tree', tree['color']))

                # 2. Lane Objects
                if lane.type == 'grass':
                    for col in range(self.cols):
                        if lane.grid[col] is not None:
                             render_list.append((col, 'tree', lane.grid[col]))
                elif lane.type == 'road':
                    for car in lane.cars:
                        render_list.append((car['x'], 'car', car))
                elif lane.type == 'water':
                    for log in lane.logs:
                        render_list.append((log['x'], 'log', log))

                # 3. Player
                if not self.is_dead and row == player_render_row:
                    render_list.append((self.visual_x, 'player', None))

                # --- SORT BY X ---
                # This ensures objects with higher X (visually "in front") are drawn later
                render_list.sort(key=lambda x: x[0])

                # --- DRAW IN ORDER ---
                for item in render_list:
                    sx, type, data = item
                    
                    if type == 'tree':
                        self.draw_tree(sx, row, 1, cam_fixed_x, self.visual_y, data)
                    
                    elif type == 'car':
                        self.draw_car(sx, row, 1, data['width'], data['color'], cam_fixed_x, self.visual_y)
                    
                    elif type == 'log':
                        self.draw_cube(sx, row, 0.2, cam_fixed_x, self.visual_y, size_x=data['width'], size_z=0.5, color=C_LOG)
                    
                    elif type == 'player':
                        pz = 1.0 + self.visual_z
                        if self.lanes.get(self.p_grid_y) and self.lanes[self.p_grid_y].type == 'water' and not self.is_moving:
                            pz = 0.7 + self.visual_z
                        self.draw_player(self.visual_x, self.visual_y, pz, cam_fixed_x, self.visual_y)


            if self.is_dead:
                shad = font.render("GAME OVER - Press R", True, BLACK)
                txt = font.render("GAME OVER - Press R", True, WHITE)
                self.screen.blit(shad, (SCREEN_WIDTH//2 - 102, SCREEN_HEIGHT//2 + 2))
                self.screen.blit(txt, (SCREEN_WIDTH//2 - 100, SCREEN_HEIGHT//2))
            
            score_shadow = font.render(f"Score: {-self.p_grid_y}", True, BLACK)
            score_txt = font.render(f"Score: {-self.p_grid_y}", True, WHITE)
            self.screen.blit(score_shadow, (22, 22))
            self.screen.blit(score_txt, (20, 20))

            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = Game()
    game.run()
