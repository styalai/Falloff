import pygame
import sys

pygame.init()

# Constantes
height = pygame.display.Info().current_h - 75
width = int(height * 9 / 16)
size_window = (width, height)
color_window = (56, 182, 255)

# Création de la fenêtre
window = pygame.display.set_mode(size_window)
pygame.display.set_caption("Falloff")

### Une classe pour tout les boutton
class Button(pygame.sprite.Sprite):
    def __init__(self, img, x=0, y=0, scale=1):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load(img)
        self.image = pygame.transform.scale(self.image, (int(self.image.get_width() * scale), int(self.image.get_height() * scale)))

        self.rect = self.image.get_rect()
        self.setCords(x, y)

    def setCords(self, x, y):
        self.rect.center = (x, y)

    def pressed(self, mouse):
        return self.rect.collidepoint(mouse)  # Utilisation de collidepoint() pour simplifier


### classe du jeu course
class Course:
    def __init__(self):
        self.running = True
    
    # coder le jeu
    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            pygame.display.flip()

### classe du jeu aventure
class Aventure:
    def __init__(self):
        self.running = True
    
    # coder le jeu
    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            pygame.display.flip()


### Menu pour choisir entre course et aventure
class MenuHumain:
    def __init__(self):
        self.running = True
        self.course = Button("assets/button_course.png", x=width//2, y=50, scale=0.5)
        self.aventure = Button("assets/button_aventure.png", x=width//2, y=150, scale=0.5)
        self.all_sprites = pygame.sprite.Group(self.course, self.aventure)

    def run(self):
        while self.running:
            window.fill(color_window) 
            self.all_sprites.draw(window)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                    
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.course.pressed(event.pos):
                        Course().run() # run course
                    if self.aventure.pressed(event.pos):
                        Aventure().run() # run aventure

            pygame.display.flip()


### Premier menu
class Menu:
    def __init__(self):
        self.running = True
        self.button_humain = Button("assets/button_humain.png", x=width//2, y=50, scale=0.5)
        self.button_ia = Button("assets/button_ia.png", x=width//2, y=150, scale=0.5)
        self.all_sprites = pygame.sprite.Group(self.button_humain, self.button_ia)

    def run(self):
        while self.running:
            window.fill(color_window)  # Remplir l'écran avec une couleur
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