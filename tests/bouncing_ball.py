import arcade

window = arcade.Window(title="Window")


class GameView(arcade.View): # 1 usage
    def __init__(self):
        super().__init__()
        self.circle_x = self.width//2
        self.circle_y = self.height//2
        self.speed_x = 500
        self.speed_y = 500
        self.radius = 40

    def on_draw(self):
        self.clear()
        arcade.draw_circle_filled(
            self.circle_x, self.circle_y, self.radius, arcade.color.AERO_BLUE
        )
        arcade.draw_circle_outline(
            self.circle_x, self.circle_y, self.radius, arcade.color.BLUE_BELL, 4
        )

        arcade.draw_text(f"x: {self.circle_x:.1f} - y: {self.circle_y:.1f}",
                         x=10,
                         y=self.height-30,
                         color=arcade.color.WHITE,
                         font_size=20
                         )
    
    def on_update(self, delta_time: float):
        self.circle_x += self.speed_x * delta_time
        self.circle_y += self.speed_y * delta_time

        if self.circle_x > self.width - self.radius: # right side collision
            self.circle_x = self.width - self.radius
            self.speed_x *= -1

        if self.circle_x < self.radius: # left side collision
            self.circle_x = self.radius
            self.speed_x *= -1

        if self.circle_y > self.height - self.radius:
            self.circle_y = self.height - self.radius
            self.speed_y *= -1

        if self.circle_y < self.radius:
            self.circle_y = self.radius
            self.speed_y *= -1

game = GameView()
window.show_view(game)
arcade.run()