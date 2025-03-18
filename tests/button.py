### Button IA
        self.button_ia_h = 70
        self.button_ia_w = 190
        self.humain = UIFlatButton(
            x=(self.width-self.button_ia_w)//2,
            y=10,
            width=self.button_ia_w,
            height=self.button_ia_h,
            color=(92, 225, 230),
            text="button",
        )
        self.manager.add(self.humain)