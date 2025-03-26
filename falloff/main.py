import pygame
import random, time, sys, h5py
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize
from skimage import data, color

pygame.init()

# Constants
height = pygame.display.Info().current_h - 100
width = int(height * 9 / 16)
size_window = (width, height)
color_window = (56, 182, 255)

# Create window
window = pygame.display.set_mode(size_window)
pygame.display.set_caption("Falloff")

class Button(pygame.sprite.Sprite):
    def __init__(self, img, x=0, y=0, scale=1):
        super().__init__()
        self.image = pygame.image.load(img)
        self.image = pygame.transform.scale(self.image, (int(self.image.get_width() * scale), int(self.image.get_height() * scale)))
        self.rect = self.image.get_rect(center=(x, y))

    def pressed(self, mouse):
        return self.rect.collidepoint(mouse)

class Player(pygame.sprite.Sprite):
    def __init__(self, x, y, texture_path, folded_texture_path):
        super().__init__()
        scale = 1.7

        self.normal_image = pygame.image.load(texture_path).convert_alpha()
        self.folded_image = pygame.image.load(folded_texture_path).convert_alpha()
        
        width = int(self.normal_image.get_width() * scale)
        height = int(self.normal_image.get_height() * scale)
        
        self.normal_image = pygame.transform.scale(self.normal_image, (width, height))
        self.folded_image = pygame.transform.scale(self.folded_image, (width, height))
        
        self.image = self.normal_image
        self.rect = self.image.get_rect(center=(x, y))
        self.mask = pygame.mask.from_surface(self.image)
        
        self.move_speed = 4
        self.bonus_activated = False

    def update(self):
        keys = pygame.key.get_pressed()
        
        # Change texture based on down arrow
        self.image = self.folded_image if keys[pygame.K_DOWN] else self.normal_image
        
        # Left/Right movement
        if keys[pygame.K_LEFT] and self.rect.left > 0:
            self.rect.x -= self.move_speed
        if keys[pygame.K_RIGHT] and self.rect.right < width:
            self.rect.x += self.move_speed

class Obstacle(pygame.sprite.Sprite):
    def __init__(self, x, y, obstacle_type, texture_path):
        super().__init__()
        scale = 4 if obstacle_type == 1 else 0.1
        
        self.image = pygame.image.load(texture_path).convert_alpha()
        width = int(self.image.get_width() * scale)
        height = int(self.image.get_height() * scale)
        self.image = pygame.transform.scale(self.image, (width, height))
        
        self.rect = self.image.get_rect(center=(x, y))
        self.mask = pygame.mask.from_surface(self.image)
        
        self.speed = 2

    def update(self):
        self.rect.y -= self.speed

class Bonus(pygame.sprite.Sprite):
    def __init__(self, x, y, texture_path):
        super().__init__()
        scale = 0.01
        
        self.image = pygame.image.load(texture_path).convert_alpha()
        width = int(self.image.get_width() * scale)
        height = int(self.image.get_height() * scale)
        self.image = pygame.transform.scale(self.image, (width, height))
        
        self.rect = self.image.get_rect(center=(x, y))
        self.mask = pygame.mask.from_surface(self.image)
        
        self.speed = 2

    def update(self):
        self.rect.y -= self.speed

class Cloud(pygame.sprite.Sprite):
    def __init__(self, x, y, scale, speed_factor, texture_path):
        super().__init__()
        self.image = pygame.image.load(texture_path).convert_alpha()
        
        width = int(self.image.get_width() * scale)
        height = int(self.image.get_height() * scale)
        self.image = pygame.transform.scale(self.image, (width, height))
        
        self.rect = self.image.get_rect(center=(x, y))
        self.speed_factor = speed_factor

    def update(self, base_speed):
        self.rect.y -= base_speed * self.speed_factor

class GameOver:

    def __init__(self, score, elapsed_time=None):
        self.running = True
        self.score = score
        self.elapsed_time = elapsed_time
        # Center the buttons in the lower half of the screen
        self.replay_button = Button("assets/rejouer.png", x=width//2, y=3*height//4 - 50, scale=0.5)
        self.menu_button = Button("assets/menu.png", x=width//2, y=3*height//4 + 50, scale=0.5)
        scale = 14
        self.parapluie = pygame.image.load("assets/parapluie.png")
        w = int(self.parapluie.get_width() * scale)
        h = int(self.parapluie.get_height() * scale)
        # Center the umbrella in the upper half of the screen
        self.parapluie = pygame.transform.scale(self.parapluie, (w, h))
        self.parapluie_rect = self.parapluie.get_rect(center=(width//2, height//4))

        self.all_sprites = pygame.sprite.Group(self.replay_button, self.menu_button)
        self.font = pygame.font.SysFont('Arial', 30)

    def run(self):
        while self.running:
            window.fill(color_window)
            window.blit(self.parapluie, self.parapluie_rect)
            self.all_sprites.draw(window)

            score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
            window.blit(score_text, (width//2 - score_text.get_width()//2, height//2))

            if self.elapsed_time is not None:
                time_text = self.font.render(f"Time: {self.elapsed_time:.2f} seconds", True, (255, 255, 255))
                window.blit(time_text, (width//2 - time_text.get_width()//2, height//2 + 40))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                    
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.replay_button.pressed(event.pos):
                        Aventure().run()  # Assuming replay starts a new Aventure game
                    if self.menu_button.pressed(event.pos):
                        MenuHumain().run()

            pygame.display.flip()


class Aventure:
    def __init__(self):
        self.running = True
        self.start_time = time.time()
        self.base_speed = 3
        self.speed_increase = 0.1
        
        self.assets = {
            "player": "assets/parapluie.png",
            "player_folded": "assets/parapluie_plier.png",
            "obstacle1": "assets/vache.png",
            "obstacle2": "assets/ballon.webp",
            "bonus": "assets/bonus.png",
            "cloud": "assets/nuage.png"
        }
                
        # Sprite groups
        self.all_sprites = pygame.sprite.Group()
        self.obstacles_group = pygame.sprite.Group()
        self.bonus_group = pygame.sprite.Group()
        self.clouds_group = pygame.sprite.Group()
        
        # Create player
        self.player = Player(width // 2, 200, self.assets["player"], self.assets["player_folded"])
        self.all_sprites.add(self.player)
        
        # Game parameters
        self.obstacle_spawn_delay = 150
        self.obstacle_spawn_time = self.obstacle_spawn_delay - 1
        self.obstacle_delay_decrease = 1
        self.gap_width = 100
        
        self.bonus_spawn_time = 0
        self.bonus_spawn_delay = 1000
        
        self.cloud_spawn_time = 0
        self.cloud_spawn_delay = 500
        
        # Font for displaying score
        self.font = pygame.font.SysFont('Arial', 30)
        
        self.score = 0
        self.bonus_activated = False
        self.bonus_duration = 500  # Duration of bonus effect in frames
        self.bonus_time = 0

        # record
        self.record_delay = 5
        self.record_time = 0
        self.precedant_img = None
        self.precedant_pos = None
        self.speed_history = np.zeros((self.record_delay))
        self.dataset = None
        
        self.cloud = False
        #self.spawn_cloud()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.running = False

    def spawn_obstacles(self):
        self.obstacle_spawn_time += 1
        if self.obstacle_spawn_time >= self.obstacle_spawn_delay:
            num_obstacles = random.randint(2, 5)
            obstacle_positions = [random.randint(80, width - 80) for _ in range(num_obstacles)]
            
            obstacle_positions.sort()
            for i in range(len(obstacle_positions) - 1):
                if obstacle_positions[i + 1] - obstacle_positions[i] < self.gap_width:
                    obstacle_positions[i + 1] = obstacle_positions[i] + self.gap_width
            
            for obstacle_x in obstacle_positions:
                obstacle_type = random.randint(1, 2)
                texture_path = self.assets["obstacle1"] if obstacle_type == 1 else self.assets["obstacle2"]
                new_obstacle = Obstacle(obstacle_x, height + 50, obstacle_type, texture_path)
                self.obstacles_group.add(new_obstacle)
                self.all_sprites.add(new_obstacle)
            
            self.obstacle_spawn_time = 0
            self.obstacle_spawn_delay = self.obstacle_spawn_delay-self.obstacle_delay_decrease

    def spawn_bonus(self):
        self.bonus_spawn_time += 1
        if self.bonus_spawn_time >= self.bonus_spawn_delay:
            bonus_x = random.randint(50, width - 50)
            new_bonus = Bonus(bonus_x, height + 50, self.assets["bonus"])
            self.bonus_group.add(new_bonus)
            self.all_sprites.add(new_bonus)
            
            self.bonus_spawn_time = 0

    def spawn_cloud(self):
        cloud_x = random.randint(-50, width + 50)
        cloud_size = random.uniform(10, 20)
        cloud_speed = random.uniform(0.5, 1.0)
        
        new_cloud = Cloud(cloud_x, height + 100, cloud_size, cloud_speed, self.assets["cloud"])
        self.clouds_group.add(new_cloud)

    def manage_clouds(self):
        self.cloud_spawn_time += 1
        if self.cloud_spawn_time >= self.cloud_spawn_delay:
            self.spawn_cloud()
            self.cloud_spawn_time = 0
        
        current_speed = self.get_current_speed()
        self.clouds_group.update(current_speed)

    def get_current_speed(self):
        return self.base_speed + (self.score * self.speed_increase)

    def update_entities(self):
        current_speed = self.get_current_speed()
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_DOWN]:
            current_speed *= 2
            self.speed_history[self.record_time] = 1
        
        for obstacle in self.obstacles_group:
            obstacle.speed = current_speed
            if obstacle.rect.bottom < 0:
                obstacle.kill()
                if self.bonus_activated:
                    self.score += 2
                else:
                    self.score += 1
                
        for bonus in self.bonus_group:
            bonus.speed = current_speed
            if bonus.rect.bottom < 0:
                bonus.kill()

    def check_collisions(self):
        hits = pygame.sprite.spritecollide(self.player, self.obstacles_group, False, pygame.sprite.collide_mask)
        if hits:
            self.game_over()
        
        bonus_hits = pygame.sprite.spritecollide(self.player, self.bonus_group, True, pygame.sprite.collide_mask)
        for bonus in bonus_hits:
            self.score += 10
            self.bonus_activated = True
            self.bonus_time = 0  # Reset bonus time

    def game_over(self):
        move = self.dataset["move"]
        frames = self.dataset["frame"]
        with h5py.File("data.h5", "a") as h5f:
            # Handling 'frame' dataset
            if "frame" in h5f:
                dataset = h5f["frame"]
                dataset.resize((dataset.shape[0], dataset.shape[1], dataset.shape[2] + frames.shape[2]))  # Expand along the 3rd axis
                dataset[:, :, -frames.shape[2]:] = frames  # Append along the last dimension
            else:
                h5f.create_dataset("frame", data=frames, maxshape=(frames.shape[0], frames.shape[1], None))  # 3rd dim is dynamic

            # Handling 'move' dataset
            if "move" in h5f:
                dataset = h5f["move"]
                dataset = np.concatenate([dataset, move], axis=0)  # Append new values
            else:
                h5f.create_dataset("move", data=move, maxshape=(None, move.shape[1]))

        elapsed_time = time.time() - self.start_time
        print(f"Game Over! Time Survived: {elapsed_time:.2f} seconds, Score: {self.score}")
        self.running = False
        GameOver(self.score, elapsed_time).run()

    def draw(self):
        window.fill(color_window)
        
        for cloud in self.clouds_group:
            window.blit(cloud.image, cloud.rect)
        
        self.all_sprites.draw(window)
        
        elapsed_time = time.time() - self.start_time
        speed_text = self.font.render(f"Vitesse: {self.get_current_speed():.1f}", True, (255, 255, 255))
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        window.blit(speed_text, (10, 20))
        window.blit(score_text, (10, 50))
        
        if self.bonus_activated:
            x2_text = self.font.render("x2", True, (255, 255, 0))
            window.blit(x2_text, (10, 80))
        
        pygame.display.flip()

    def run(self):
        clock = pygame.time.Clock()
        while self.running:
            self.handle_events()
            self.player.update()
            self.spawn_obstacles()
            if self.cloud:
                self.spawn_bonus()
                self.manage_clouds()
            self.update_entities()
            self.check_collisions()
            self.all_sprites.update()
            self.draw()
            self.record()
            
            # Update bonus time and deactivate bonus if duration is exceeded
            if self.bonus_activated:
                self.bonus_time += 1
                if self.bonus_time >= self.bonus_duration:
                    self.bonus_activated = False
            
            clock.tick(40)
    
    def record(self):
        self.record_time += 1
        if self.record_time >= self.record_delay:
            self.record_time = 0
            if self.precedant_pos is not None:
                frame  = self.precedant_img
                frame = pygame.surfarray.array3d(frame) # np array (x, y, 3)
                
                #frame = self.rgb2gray(frame) # convert to gray
                frame = color.rgb2gray(frame)
                frame = resize(frame, (76, 119)) / np.max(frame)
                #frame = frame[::10, ::10] # downscale image by 10
                frame = np.expand_dims(frame[:, 15:], axis=0).astype(np.int16) # remove writing
                #plt.imshow(frame)
                #plt.show()

                """
                out = 0 # idx in a 6 of lenght
                pos = self.player.rect.centerx
                if pos > self.precedant_pos: # right
                    if np.sum(self.speed_history) == 0:
                        out = 0
                    elif np.sum(self.speed_history) < self.record_delay/2:
                        out = 2
                    elif np.sum(self.speed_history) >= self.record_delay/2:
                        out = 4

                elif pos < self.precedant_pos: # left
                    if np.sum(self.speed_history) == 0:
                        out = 1
                    elif np.sum(self.speed_history) < self.record_delay/2:
                        out = 3
                    elif np.sum(self.speed_history) >= self.record_delay/2:
                        out = 5
                """
                pos = self.player.rect.centerx
                out = np.zeros((1, 1))
                out[0, 0] = width

                if self.dataset is not None:
                    self.dataset["frame"] = np.concatenate([self.dataset["frame"], frame], axis=0)
                    #self.dataset["move"].append(out)
                    self.dataset["move"] = np.concatenate([self.dataset["move"], out], axis=0)
                else:
                    self.dataset = {}
                    self.dataset["frame"] = frame
                    #self.dataset["move"] = [out]
                    self.dataset["move"] = out

            self.precedant_img  = window.copy()
            self.speed_history = np.zeros((self.record_delay))
            self.precedant_pos = self.player.rect.centerx


class Course(Aventure):
    def __init__(self):
        super().__init__()
        self.win_score = 100  # Score needed to win the game

    def check_collisions(self):
        super().check_collisions()
        if self.score >= self.win_score:
            self.win_game()

    def win_game(self):
        elapsed_time = time.time() - self.start_time
        print(f"Congratulations! You won! Time: {elapsed_time:.2f} seconds, Score: {self.score}")
        self.running = False
        GameOver(self.score, elapsed_time).run()

class MenuHumain:
    def __init__(self):
        self.running = True
        # Center the buttons in the lower half of the screen
        self.course = Button("assets/button_course.png", x=width//2, y=3*height//4 - 50, scale=0.5)
        self.aventure = Button("assets/button_aventure.png", x=width//2, y=3*height//4 + 50, scale=0.5)
        scale = 14
        self.parapluie = pygame.image.load("assets/parapluie.png")
        w = int(self.parapluie.get_width() * scale)
        h = int(self.parapluie.get_height() * scale)
        # Center the umbrella in the upper half of the screen
        self.parapluie = pygame.transform.scale(self.parapluie, (w, h))
        self.parapluie_rect = self.parapluie.get_rect(center=(width//2, height//4))

        self.all_sprites = pygame.sprite.Group(self.course, self.aventure)

    def run(self):
        while self.running:
            window.fill(color_window)
            window.blit(self.parapluie, self.parapluie_rect)
            self.all_sprites.draw(window)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                    
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.course.pressed(event.pos):
                        Course().run()
                    if self.aventure.pressed(event.pos):
                        Aventure().run()

            pygame.display.flip()

class Menu:
    def __init__(self):
        self.running = True
        # Center the buttons in the lower half of the screen
        self.button_humain = Button("assets/button_humain.png", x=width//2, y=3*height//4 - 50, scale=0.5)
        self.button_ia = Button("assets/button_ia.png", x=width//2, y=3*height//4 + 50, scale=0.5)
        scale = 14
        self.parapluie = pygame.image.load("assets/parapluie.png")
        w = int(self.parapluie.get_width() * scale)
        h = int(self.parapluie.get_height() * scale)
        # Center the umbrella in the upper half of the screen
        self.parapluie = pygame.transform.scale(self.parapluie, (w, h))
        self.parapluie_rect = self.parapluie.get_rect(center=(width//2, height//4))

        self.all_sprites = pygame.sprite.Group(self.button_humain, self.button_ia)

    def run(self):
        while self.running:
            window.fill(color_window)
            window.blit(self.parapluie, self.parapluie_rect)
            self.all_sprites.draw(window)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                    
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.button_humain.pressed(event.pos):
                        MenuHumain().run()

            pygame.display.flip()

Menu().run()