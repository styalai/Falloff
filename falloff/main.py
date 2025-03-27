import pygame
import random, time, sys, h5py
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize
from skimage import data, color
import torch
import torch.nn as nn
import torch.nn.functional as F

pygame.init()

# Constants
height = 650 #pygame.display.Info().current_h - 100
width = int(height*9/16) #int(height * 9 / 16)
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
        
        self.move_speed = 5
        self.bonus_activated = False

    def update(self):
        self.move_speed += 0.01/30 # augmente de 0.01 toute les secondes: 1 toute les 100 secondes
        keys = pygame.key.get_pressed()
        
        # Change texture based on down arrow
        self.image = self.folded_image if keys[pygame.K_DOWN] else self.normal_image
        self.mask = pygame.mask.from_surface(self.folded_image) if keys[pygame.K_DOWN] else pygame.mask.from_surface(self.normal_image)
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

class GameOverHumain:

    def __init__(self, score, elapsed_time=None):
        self.running = True
        self.score = score
        self.elapsed_time = elapsed_time
        # Center the buttons in the lower half of the screen
        self.replay_button = Button("assets/rejouer.png", x=width//2, y=3*height//4 - 50, scale=0.5)
        self.menu_button = Button("assets/menu.png", x=width//2, y=3*height//4 + 50, scale=0.5)
        scale = 10
        self.parapluie = pygame.image.load("assets/parapluie.png")
        w = int(self.parapluie.get_width() * scale)
        h = int(self.parapluie.get_height() * scale)
        # Center the umbrella in the upper half of the screen
        self.parapluie = pygame.transform.scale(self.parapluie, (w, h))
        self.parapluie_rect = self.parapluie.get_rect(center=(width//2, height//4))

        self.all_sprites = pygame.sprite.Group(self.replay_button, self.menu_button)
        self.font = pygame.font.SysFont('Arial', 20)

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
                        Menu().run()

            pygame.display.flip()

class GameOverIA:

    def __init__(self, score, elapsed_time=None):
        self.running = True
        self.score = score
        self.elapsed_time = elapsed_time
        # Center the buttons in the lower half of the screen
        self.replay_button = Button("assets/rejouer.png", x=width//2, y=3*height//4 - 50, scale=0.5)
        self.menu_button = Button("assets/menu.png", x=width//2, y=3*height//4 + 50, scale=0.5)
        scale = 10
        self.parapluie = pygame.image.load("assets/parapluie.png")
        w = int(self.parapluie.get_width() * scale)
        h = int(self.parapluie.get_height() * scale)
        # Center the umbrella in the upper half of the screen
        self.parapluie = pygame.transform.scale(self.parapluie, (w, h))
        self.parapluie_rect = self.parapluie.get_rect(center=(width//2, height//4))

        self.all_sprites = pygame.sprite.Group(self.replay_button, self.menu_button)
        self.font = pygame.font.SysFont('Arial', 20)

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
                        Aventure(ia=True, model_pth="model.pth").run()  # Assuming replay starts a new Aventure game
                    if self.menu_button.pressed(event.pos):
                        Menu().run()

            pygame.display.flip()

class Aventure:
    def __init__(self, ia=False, model_pth=None):
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
        self.ia = ia
        if ia:
            self.player = IAplayer(model_pth, width // 2, 200, self.assets["player"], self.assets["player_folded"])
            self.all_sprites.add(self.player)
        else:
            self.player = Player(width // 2, 200, self.assets["player"], self.assets["player_folded"])
            self.all_sprites.add(self.player)
            
        # Game parameters
        self.obstacle_spawn_delay = 110
        self.obstacle_spawn_time = self.obstacle_spawn_delay - 1
        self.obstacle_delay_decrease = 0.7
        
        self.bonus_spawn_time = 0
        self.bonus_spawn_delay = 1000
        
        self.cloud_spawn_time = 0
        self.cloud_spawn_delay = 500
        self.cloud = False
        
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
            # Load images if not already loaded
            if not hasattr(self, 'obstacle1_image'):
                self.obstacle1_image = pygame.image.load(self.assets["obstacle1"])
                self.obstacle2_image = pygame.image.load(self.assets["obstacle2"])
            
            # Predefined positions as percentages of screen width
            screen_positions = [
                width * 0.2,  # 20% from left
                width * 0.4,  # 40% from left
                width * 0.6,  # 60% from left
                width * 0.8   # 80% from left
            ]
            
            # Randomly select how many obstacles to spawn (2-4)
            num_obstacles = random.randint(1, 3)
            
            # Randomly select positions
            selected_positions = random.sample(screen_positions, num_obstacles)
            if num_obstacles == 1:
                obtacle_types = [random.randint(1, 2)]
            elif num_obstacles == 2:
                obtacle_types = [random.randint(1, 2), random.randint(1, 2)]
            elif num_obstacles == 3:
                obtacle_types = random.choice([[1, 1, 2], [2, 1, 2], [1, 2, 1], [2, 1, 1]])
            
            # Create obstacles at selected positions
            for position in selected_positions:
                # Randomly choose obstacle type
                obstacle_type = obtacle_types[selected_positions.index(position)]
                
                # Select texture based on type
                texture_path = self.assets["obstacle1"] if obstacle_type == 1 else self.assets["obstacle2"]
                
                # Create obstacle
                new_obstacle = Obstacle(position, height + 50, obstacle_type, texture_path)
                self.obstacles_group.add(new_obstacle)
                self.all_sprites.add(new_obstacle)
            
            # Reset spawn timer
            self.obstacle_spawn_time = 0
            self.obstacle_spawn_delay = self.obstacle_spawn_delay - self.obstacle_delay_decrease
                
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

    def update_entities(self, speed=None):
        current_speed = self.get_current_speed()
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_DOWN] or speed:
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
                self.obstacle_spawn_delay -= self.obstacle_delay_decrease
                
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
        if not self.ia:
            move = self.dataset["move"]
            frames = self.dataset["frame"]
            with h5py.File("data.h5", "a") as h5f:
                # Handling 'frame' dataset
                if "frame" in h5f:
                    dataset = h5f["frame"]
                    #dataset.resize((dataset.shape[0], dataset.shape[1], dataset.shape[2] + frames.shape[2]))  # Expand along the 3rd axis
                    dataset = np.concatenate([dataset, frames], axis=0) 
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
        if not self.ia:
            GameOverHumain(self.score, elapsed_time).run()
        else:
            GameOverIA(self.score, elapsed_time).run()

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

            if self.ia:
                frame = window.copy()
                frame = pygame.surfarray.array3d(frame) # np array (x, y, 3)
                frame = color.rgb2gray(frame)
                frame = resize(frame, (76, 119)) #/ np.max(frame)
                frame = torch.tensor(torch.from_numpy(np.expand_dims(frame[:, 15:], axis=0)), dtype=torch.float) # remove writing and convert it to int 16
                frame = frame.unsqueeze(0)
                speed = self.player.update(frame)

            else:
                self.player.update()
            
            self.spawn_obstacles()
            if self.cloud:
                self.spawn_bonus()
                self.manage_clouds()
            
            if self.ia:
                self.update_entities(speed)
            else:
                self.update_entities()

            self.check_collisions()
            self.all_sprites.update()
            self.draw()

            if not self.ia:
                self.record()
            
            # Update bonus time and deactivate bonus if duration is exceeded
            if self.bonus_activated:
                self.bonus_time += 1
                if self.bonus_time >= self.bonus_duration:
                    self.bonus_activated = False
            
            clock.tick(30)
    
    def record(self):
        self.record_time += 1
        if self.record_time >= self.record_delay:
            self.record_time = 0
            if self.precedant_pos is not None:
                if self.precedant_pos != self.player.rect.centerx:
                    frame  = self.precedant_img
                    frame = pygame.surfarray.array3d(frame) # np array (x, y, 3)
                    
                    #frame = self.rgb2gray(frame) # convert to gray
                    frame = color.rgb2gray(frame)
                    frame = resize(frame, (76, 119)) #/ np.max(frame)
                    #frame = frame[::10, ::10] # downscale image by 10
                    frame = np.expand_dims(frame[:, 15:], axis=0).astype(np.float16) # remove writing and convert it to int 16
                    
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
                    out = np.zeros((1, 2))
                    out[0, 0] = int(pos / width * 100) # % of the screen
                    out[0, 1] = False in self.speed_history

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
        if not self.ia:
            GameOverHumain(self.score, elapsed_time).run()
        else:
            GameOverIA(self.score, elapsed_time).run()

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # input: (batch_size, 1, 76, 104)
        # Expanded convolutional layers with more depth and channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(24, 30), stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(10, 10), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(5, 5), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Fully connected layers with dropout for regularization
        self.fc = nn.Sequential(
            nn.Linear(256, 304),  # Increased input and hidden layer sizes
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(304, 204),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(204, 102),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = torch.flatten(x, start_dim=1)  # Flatten the tensor
        x = self.fc(x)
        return x

class IAplayer(pygame.sprite.Sprite):
    def __init__(self, model_pth, x, y, texture_path, folded_texture_path):
        super().__init__()
        scale = 1.7
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Model()
        self.model.load_state_dict(torch.load(model_pth, weights_only=True))
        self.model.eval()
        self.model = self.model.to(self.device)

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

    def update(self, frame=None):
        if frame is not None:
            self.move_speed += 0.01/30 # augmente de 0.01 toute les secondes: 1 toute les 100 secondes

            out = self.model(frame)
            pos = F.softmax(out[0, :100]*0.01 * width, dim=-1)
            pos = torch.argmax(pos).item()
            speed = torch.argmax(out[0, 100:], dim=-1).item()
            
            # Change texture based on down arrow
            self.image = self.folded_image if bool(speed) else self.normal_image
            # Left/Right movement
            if pos < self.rect.x and self.rect.left > 0: # left
                self.rect.x -= self.move_speed
            if pos > self.rect.x and self.rect.right < width: # right
                self.rect.x += self.move_speed
            return bool(speed)

class TextInput:
    def __init__(self, x, y, width, height, default_text='', font_size=30):
        self.rect = pygame.Rect(x, y, width, height)
        self.color_inactive = pygame.Color('blue')
        self.color_active = pygame.Color('dodgerblue2')
        self.color = self.color_inactive
        self.text = default_text
        self.font = pygame.font.SysFont('Arial', font_size)
        self.txt_surface = self.font.render(self.text, True, self.color)
        self.active = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = not self.active
            else:
                self.active = False
            self.color = self.color_active if self.active else self.color_inactive
        
        if event.type == pygame.KEYDOWN:
            if self.active:
                if event.key == pygame.K_RETURN:
                    self.active = False
                elif event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:
                    self.text += event.unicode
                self.txt_surface = self.font.render(self.text, True, self.color)

    def update(self):
        width = max(self.rect.w, self.txt_surface.get_width()+10)
        self.rect.w = width

    def draw(self, screen):
        screen.blit(self.txt_surface, (self.rect.x+5, self.rect.y+5))
        pygame.draw.rect(screen, self.color, self.rect, 2)

class Trainer:
    def __init__(self):
        self.running = True
        
        # UI Elements
        self.font = pygame.font.SysFont('Arial', 15)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Create text input fields
        self.batch_input = TextInput(width//2 - 100, height//2 - 100, 200, 40, default_text='32')
        self.epochs_input = TextInput(width//2 - 100, height//2 - 20, 200, 40, default_text='10')
        self.model_path_input = TextInput(width//2 - 100, height//2 + 60, 200, 40, default_text='model.pth')
        
        # Train button
        self.train_button = Button("assets/button_entrainer.png", x=width//2, y=height//2 + 150, scale=0.5)
        self.back_button = Button("assets/menu.png", x=width//2, y=height//2 + 240 , scale=0.4)
        
        # Result text 
        self.result_text = None

    def train(self, batch_size, num_epochs, model_path):
        try:
            hf = h5py.File('data.h5', 'r')
            frames = torch.tensor(torch.from_numpy(hf["frame"][:]), dtype=torch.float)
            moves = torch.tensor(torch.from_numpy(hf["move"][:]), dtype=torch.long)

            model = Model()
            try:
                model.load_state_dict(torch.load(model_path, weights_only=True))
            except:
                print("Can't load model")
            model = model.to(self.device)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

            # Training loop
            for epoch in range(num_epochs):
                model.train()
                running_loss = 0.0
                for i in range(0, len(frames), batch_size):
                    inputs = frames[i:i+batch_size].unsqueeze(1).to(self.device)
                    labels = moves[i:i+batch_size].to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss1 = criterion(outputs[:, :100], labels[:, 0])
                    loss2 = criterion(outputs[:, 100:], labels[:, 1])
                    loss = loss1+loss2

                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()

                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(frames):.4f}")

            torch.save(model.state_dict(), model_path)
            return f"Training completed. Model saved to {model_path}"
        except Exception as e:
            return f"Training failed: {str(e)}"

    def run(self):
        while self.running:
            window.fill(color_window)
            
            # Draw titles for input fields
            batch_title = self.font.render("Batch Size", True, (255, 255, 255))
            epochs_title = self.font.render("Number of Epochs", True, (255, 255, 255))
            model_path_title = self.font.render("Model Save Path", True, (255, 255, 255))
            
            window.blit(batch_title, (width//2 - 50, height//2 - 130))
            window.blit(epochs_title, (width//2 - 50, height//2 - 50))
            window.blit(model_path_title, (width//2 - 50, height//2 + 30))
            
            # Draw input fields and buttons
            self.batch_input.draw(window)
            self.epochs_input.draw(window)
            self.model_path_input.draw(window)
            
            # Draw buttons
            #self.train_button.rect.center = (width//2, height//2 + 150)
            #self.back_button.rect.center = (width//2, height//2 + 240)
            window.blit(self.train_button.image, self.train_button.rect)
            window.blit(self.back_button.image, self.back_button.rect)
            
            # Draw result text if exists
            if self.result_text:
                result_surf = self.font.render(self.result_text, True, (255, 255, 255))
                window.blit(result_surf, (width//2 - result_surf.get_width()//2, height//2 + 270))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                self.batch_input.handle_event(event)
                self.epochs_input.handle_event(event)
                self.model_path_input.handle_event(event)
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.train_button.pressed(event.pos):
                        try:
                            batch_size = int(self.batch_input.text)
                            num_epochs = int(self.epochs_input.text)
                            model_path = self.model_path_input.text
                            
                            self.result_text = self.train(batch_size, num_epochs, model_path)
                        except ValueError:
                            self.result_text = "Invalid input. Please enter valid numbers."
                    
                    if self.back_button.pressed(event.pos):
                        MenuIA().run()

            pygame.display.flip()

class MenuIA:
    def __init__(self):
        self.running = True
        # Center the buttons in the lower half of the screen
        self.aventure = Button("assets/button_aventure.png", x=width//2, y=3*height//4 - 70, scale=0.5)
        self.train = Button("assets/button_entrainer.png", x=width//2, y=3*height//4 + 30, scale=0.5)
        self.retour = Button("assets/button_retour.png", x=width//2, y=3*height//4 + 118, scale=0.4)
        scale = 10
        self.parapluie = pygame.image.load("assets/parapluie.png")
        w = int(self.parapluie.get_width() * scale)
        h = int(self.parapluie.get_height() * scale)
        # Center the umbrella in the upper half of the screen
        self.parapluie = pygame.transform.scale(self.parapluie, (w, h))
        self.parapluie_rect = self.parapluie.get_rect(center=(width//2, height//4))

        self.all_sprites = pygame.sprite.Group(self.train, self.aventure, self.retour)

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
                    if self.train.pressed(event.pos):
                        Trainer().run()
                    elif self.aventure.pressed(event.pos):
                        Aventure(ia=True, model_pth="model.pth").run()
                    elif self.retour.pressed(event.pos):
                        Menu().run()

            pygame.display.flip()

class MenuHumain:
    def __init__(self):
        self.running = True
        # Center the buttons in the lower half of the screen
        self.course = Button("assets/button_course.png", x=width//2, y=3*height//4 - 70, scale=0.5)
        self.aventure = Button("assets/button_aventure.png", x=width//2, y=3*height//4 + 30, scale=0.5)
        self.retour = Button("assets/button_retour.png", x=width//2, y=3*height//4 + 118, scale=0.4)
        scale = 10
        self.parapluie = pygame.image.load("assets/parapluie.png")
        w = int(self.parapluie.get_width() * scale)
        h = int(self.parapluie.get_height() * scale)
        # Center the umbrella in the upper half of the screen
        self.parapluie = pygame.transform.scale(self.parapluie, (w, h))
        self.parapluie_rect = self.parapluie.get_rect(center=(width//2, height//4))

        self.all_sprites = pygame.sprite.Group(self.course, self.aventure, self.retour)

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
                    if self.retour.pressed(event.pos):
                        Menu().run()

            pygame.display.flip()

class Menu:
    def __init__(self):
        self.running = True
        # Center the buttons in the lower half of the screen
        self.button_humain = Button("assets/button_humain.png", x=width//2, y=3*height//4 - 50, scale=0.5)
        self.button_ia = Button("assets/button_ia.png", x=width//2, y=3*height//4 + 50, scale=0.5)
        scale = 10
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
                    elif self.button_ia.pressed(event.pos):
                        MenuIA().run()

            pygame.display.flip()

Menu().run()