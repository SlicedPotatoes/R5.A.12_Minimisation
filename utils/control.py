import ipywidgets as widgets
from IPython.display import display

class Control:
    def __init__(self, fig, anim, update, size):        
        self.frame = 0
        
        previous_btn = widgets.Button(description="⏮️")
        next_btn = widgets.Button(description="⏭️")
            
        def _prev(_):
            anim.event_source.stop()

            if self.frame > 0:
                self.frame -= 1
                update(self.frame)
                fig.canvas.draw_idle()
                
        def _next(_):
            anim.event_source.stop()

            if self.frame < size - 1:
                self.frame += 1
                update(self.frame)
                fig.canvas.draw_idle()          

        previous_btn.on_click(_prev)
        next_btn.on_click(_next)

        display(widgets.HBox([previous_btn, next_btn]))