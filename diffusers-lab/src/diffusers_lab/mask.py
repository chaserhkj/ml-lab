"""
Mask Drawing Widget for Jupyter Notebooks
Uses ipycanvas for drawing and ipywidgets for UI controls
"""
# Assisted by agent continue.dev w/ minimax/minimax-m2.1

import ipywidgets as widgets
from ipycanvas import MultiCanvas, hold_canvas
from PIL import Image
import numpy as np
import time


class MaskDrawer:
    """
    A widget for manually drawing masks on top of images.
    
    Features:
    - Pen tool for drawing
    - Eraser tool for removing mask
    - Rectangle masking tool
    - Grab/Pan tool for moving around the canvas
    - Adjustable pen size
    - Undo functionality
    - Reset mask
    - Pan with grab tool
    """
    
    def __init__(self, image, mask_color=(0, 0, 0), 
                pending_mask_color=(255, 0, 0),
                mask_alpha=128, pending_mask_alpha=200,
                initial_mask=None):
        """
        Initialize the mask drawer.
        
        Args:
            image: PIL Image or path to image
            mask_color: RGB tuple for mask color (default: black)
            pending_mask_color: RGB tuple for pending mask color (default: red)
            mask_alpha: Alpha value for mask (0-255, default: 128)
            pending_mask_alpha: Alpha value for pending mask (0-255, default: 200)
            initial_mask: Optional numpy array to use as initial mask
        """
        # Load image
        if isinstance(image, str):
            self.image = Image.open(image)
        else:
            self.image = image
        
        # Convert to RGBA if needed
        if self.image.mode != 'RGBA':
            self.image = self.image.convert('RGBA')
        
        self.width, self.height = self.image.size
        self.mask_color = mask_color
        self.pending_mask_color = pending_mask_color
        self.mask_alpha = mask_alpha
        self.pending_mask_alpha = pending_mask_alpha
        
        # Drawing state
        self.current_tool = 'pen'
        self.pen_size = 10
        self.is_drawing = False
        self.start_pos = None
        self.mask_history = []
        self.max_history = 20
        
        # Zoom and Pan state
        self.zoom = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 10.0
        self.pan_offset = (0.0, 0.0)
        self.is_panning = False
        self.pan_start = None
        self._last_move = 0.0
        
        # Create mask (initialize as transparent, or use provided mask)
        if initial_mask is not None:
            self.mask_array = initial_mask.astype(bool)
            # Validate dimensions match
            if self.mask_array.shape != (self.height, self.width):
                raise ValueError(f"initial_mask shape {(self.mask_array.shape)} must match image dimensions ({self.height}, {self.width})")
        else:
            self.mask_array = np.zeros((self.height, self.width), dtype=bool)
        
        self._last_move = 0.0

        # Create composite image for display
        self._update_composite()
        
        # Setup UI
        self._setup_canvas()
        self._setup_controls()
        self._setup_layout()
        self._setup_event_handlers()
        
        # Save initial state
        self._save_state()
    
    def _update_composite(self):
        """Create composite image with mask overlay."""

        # Create color array with alpha
        color = np.array([*self.mask_color, self.mask_alpha], dtype=np.uint8)  # (4,)
        
        # Display composite mask as inverted
        mask_inverted = ~self.mask_array

        mask_rgba = color * mask_inverted[..., np.newaxis]  # (H, W, 4)
        
        # Convert to PIL and composite with original image
        mask_img = Image.fromarray(mask_rgba, mode='RGBA')
        self.composite = Image.alpha_composite(self.image, mask_img)
    
    def _setup_canvas(self):
        """Setup the canvas layers."""
        # Main canvas with 2 layers: 0=image, 1=mask
        self.canvas = MultiCanvas(2, width=self.width, height=self.height)
        
        # Get references to each layer
        self.image_canvas = self.canvas[0]
        self.mask_canvas = self.canvas[1]
        
        # Draw initial image
        self._draw_image()
        
        # Initialize mask canvas with current mask
        self._draw_mask_canvas()
    
    def _draw_image(self):
        """Draw the composite image on the base canvas - optimized version."""
        with hold_canvas(self.image_canvas):
            self.image_canvas.clear()
            # Convert PIL image directly to bytes
            self.image_canvas.put_image_data(np.array(self.composite))
        with hold_canvas(self.mask_canvas):
            self.mask_canvas.clear()
    
    def _draw_mask_canvas(self):
        """Draw current mask state on mask canvas - optimized version."""
        with hold_canvas(self.mask_canvas):
            self.mask_canvas.clear()
            
            # Only draw if there's a mask
            if np.any(self.mask_array):
                # Create colored mask for display
                color = np.array([*self.pending_mask_color, self.pending_mask_alpha], dtype=np.uint8)  # (4,)
                
                # Bool array * scalar array = scalar where True, 0 where False
                mask_rgba = color * self.mask_array[..., np.newaxis]  # (H, W, 4)
                
                # Use put_image_data for fast pixel-level updates
                # ipycanvas supports direct image data manipulation
                self.mask_canvas.put_image_data(mask_rgba)
    
    def _setup_controls(self):
        """Setup the UI controls."""
        # Tool selection
        self.pen_btn = widgets.Button(
            description='Pen',
            button_style='primary',
            icon='pencil',
            layout=widgets.Layout(width='80px')
        )
        
        self.eraser_btn = widgets.Button(
            description='Eraser',
            button_style='',
            icon='eraser',
            layout=widgets.Layout(width='80px')
        )
        
        self.rect_btn = widgets.Button(
            description='Rectangle',
            button_style='',
            icon='square-o',
            layout=widgets.Layout(width='100px')
        )
        
        self.grab_btn = widgets.Button(
            description='Grab',
            button_style='',
            icon='hand-paper-o',
            layout=widgets.Layout(width='80px')
        )
        
        # Undo and Reset
        self.undo_btn = widgets.Button(
            description='Undo',
            button_style='',
            icon='undo',
            layout=widgets.Layout(width='80px')
        )
        
        self.reset_btn = widgets.Button(
            description='Reset',
            button_style='danger',
            icon='trash',
            layout=widgets.Layout(width='80px')
        )
        self.invert_btn = widgets.Button(
            description='Invert',
            button_style='',
            icon='refresh',
            layout=widgets.Layout(width='80px')
        )
        
        # Zoom controls
        self.zoom_in_btn = widgets.Button(
            description='+',
            button_style='',
            icon='plus',
            layout=widgets.Layout(width='40px')
        )
        
        self.zoom_out_btn = widgets.Button(
            description='-',
            button_style='',
            icon='minus',
            layout=widgets.Layout(width='40px')
        )
        
        self.zoom_reset_btn = widgets.Button(
            description='100%',
            button_style='',
            layout=widgets.Layout(width='60px')
        )
        
        # Zoom level display
        self.zoom_label = widgets.Label(value=f'Zoom: {int(self.zoom * 100)}%')
        
        # Pen size slider
        self.pen_size_slider = widgets.IntSlider(
            value=self.pen_size,
            min=1,
            max=50,
            description='Size:',
            layout=widgets.Layout(width='200px'),
            style={'description_width': '50px'}
        )
        
        # Current tool display
        self.tool_label = widgets.Label(value=f'Tool: {self.current_tool.capitalize()}')
    
    def _setup_layout(self):
        """Setup the widget layout."""
        # Tool buttons row
        tool_row = widgets.HBox([
            self.pen_btn, 
            self.eraser_btn, 
            self.rect_btn,
            self.grab_btn,
            widgets.HTML("&nbsp;" * 10),
            self.undo_btn,
            self.invert_btn,
            self.reset_btn
        ])
        
        # Zoom controls row
        zoom_row = widgets.HBox([
            self.zoom_out_btn,
            self.zoom_reset_btn,
            self.zoom_in_btn,
            widgets.HTML("&nbsp;" * 10),
            self.zoom_label
        ])
        
        # Pen size row
        size_row = widgets.HBox([self.pen_size_slider, self.tool_label])
        
        # Main layout
        self.widget = widgets.VBox([
            tool_row,
            zoom_row,
            size_row,
            self.canvas
        ])
        
        # Set some styling
        self.widget.layout.align_items = 'center'
    
    def _setup_event_handlers(self):
        """Setup event handlers for controls and canvas."""
        # Button handlers
        self.pen_btn.on_click(self._on_pen_click)
        self.eraser_btn.on_click(self._on_eraser_click)
        self.rect_btn.on_click(self._on_rect_click)
        self.grab_btn.on_click(self._on_grab_click)
        self.undo_btn.on_click(self._on_undo_click)
        self.invert_btn.on_click(self._on_invert_click)
        self.reset_btn.on_click(self._on_reset_click)
        self.zoom_in_btn.on_click(self._on_zoom_in_click)
        self.zoom_out_btn.on_click(self._on_zoom_out_click)
        self.zoom_reset_btn.on_click(self._on_zoom_reset_click)
        
        # Slider handler
        self.pen_size_slider.observe(self._on_size_change, names='value')
        
        # Canvas mouse handlers - use throttled versions if available
        self.canvas.on_mouse_down(self._on_mouse_down)
        self.canvas.on_mouse_move(self._on_mouse_move)
        self.canvas.on_mouse_up(self._on_mouse_up)
        self._setup_keyboard_handler()

    def _setup_keyboard_handler(self):
        """Setup keyboard event handlers."""
        self.canvas.on_key_down(self._on_key_down)

    def _on_key_down(self, key, shift, ctrl, meta):
        """Handle keyboard input for tool selection."""
        key = key.lower()
        
        if key == 'd' and not shift and  ctrl and not meta:
            self._on_undo_click(None)
        elif key == 'b' and not shift and not ctrl and not meta:
            self._on_pen_click(None)
        elif key == 'v' and not shift and not ctrl and not meta:
            self._on_eraser_click(None)
        elif key == 'd' and not shift and not ctrl and not meta:
            self._on_rect_click(None)
        elif key == 'g' and not shift and not ctrl and not meta:
            self._on_grab_click(None)
        elif key == 'b' and shift and not ctrl and not meta:
            self._on_zoom_in_click(None)
        elif key == 'v' and shift and not ctrl and not meta:
            self._on_zoom_out_click(None)
        elif key == 'd' and shift and not ctrl and not meta:
            if self.pen_size_slider.value > 2:
                self.pen_size_slider.value -= 2
        elif key == 'g' and shift and not ctrl and not meta:
            self.pen_size_slider.value += 2

    def _update_tool_buttons(self):
        """Update button styles to show current tool."""
        style_map = {
            'pen': self.pen_btn,
            'eraser': self.eraser_btn,
            'rectangle': self.rect_btn,
            'grab': self.grab_btn
        }
        
        for tool, btn in style_map.items():
            if tool == self.current_tool:
                btn.button_style = 'primary'
            else:
                btn.button_style = ''
        
        self.tool_label.value = f'Tool: {self.current_tool.capitalize()}'
    
    def _update_zoom_label(self):
        """Update the zoom label."""
        self.zoom_label.value = f'Zoom: {int(self.zoom * 100)}%'
    
    def _on_pen_click(self, b):
        """Handle pen tool selection."""
        self.current_tool = 'pen'
        self._update_tool_buttons()
    
    def _on_eraser_click(self, b):
        """Handle eraser tool selection."""
        self.current_tool = 'eraser'
        self._update_tool_buttons()
    
    def _on_rect_click(self, b):
        """Handle rectangle tool selection."""
        self.current_tool = 'rectangle'
        self._update_tool_buttons()
    
    def _on_grab_click(self, b):
        """Handle grab/pan tool selection."""
        self.current_tool = 'grab'
        self._update_tool_buttons()
    
    def _on_undo_click(self, b):
        """Handle undo action."""
        if len(self.mask_history) > 1:
            self.mask_history.pop()  # Remove current state
            previous_state = self.mask_history[-1]
            self.mask_array = previous_state.copy()
            self._draw_mask_canvas()
            self._update_composite()
            self._draw_image()

    def _on_invert_click(self, b):
        """Handle invert action."""
        self.mask_array = ~self.mask_array
        self._draw_mask_canvas()
        self._save_state()
        self._update_composite()
        self._draw_image()

    def _on_reset_click(self, b):
        """Handle reset action."""
        self.mask_array = np.zeros((self.height, self.width), dtype=bool)
        self._update_composite()
        self._draw_image()
        self.mask_history = [self.mask_array.copy()]
    
    def _on_size_change(self, change):
        """Handle pen size change."""
        self.pen_size = change['new']
    
    def _on_zoom_in_click(self, b):
        """Handle zoom in button click."""
        self._set_zoom(self.zoom * 1.2)
    
    def _on_zoom_out_click(self, b):
        """Handle zoom out button click."""
        self._set_zoom(self.zoom / 1.2)
    
    def _on_zoom_reset_click(self, b):
        """Handle zoom reset button click."""
        self.pan_offset = (0, 0)
        self._set_zoom(1.0)
    
    def _set_zoom(self, new_zoom):
        """Set zoom level with bounds."""
        self.zoom = max(self.min_zoom, min(self.max_zoom, new_zoom))
        self._update_zoom_label()
        self._apply_view_transform()
        self._draw_image()
    
    def _apply_view_transform(self):
        """Apply zoom and pan transform to canvas using context transforms."""
        # Apply transform to both canvas layers
        for layer_canvas in [self.image_canvas, self.mask_canvas]:
            # Reset transform first
            layer_canvas.reset_transform()
            
            # Apply pan offset (translate)
            layer_canvas.translate(self.pan_offset[0], self.pan_offset[1])
            
            # Apply zoom (scale)
            layer_canvas.scale(self.zoom, self.zoom)
    
    def _screen_to_canvas_coords(self, screen_x, screen_y):
        """Convert screen coordinates to canvas coordinates accounting for zoom and pan."""
        canvas_x = int((screen_x - self.pan_offset[0]) / self.zoom)
        canvas_y = int((screen_y - self.pan_offset[1]) / self.zoom)
        return canvas_x, canvas_y
    
    def _on_mouse_down(self, x, y):
        """Handle mouse press - optimized."""
        # Check if grab tool is selected for panning
        if self.current_tool == 'grab':
            self.is_panning = True
            self.pan_start = (x, y)
            return
        x, y = int(x), int(y)
        
        x ,y = self._screen_to_canvas_coords(x, y)
        self.is_drawing = True
        self.start_pos = (x, y)
        self.last_x, self.last_y = x, y
        
        if self.current_tool in ('pen', 'eraser'):
            # Draw/erase immediately without saving state
            if self.current_tool == 'pen':
                self._draw_at_position(x, y)
            else:
                self._erase_at_position(x, y)
                
            # Only update canvas, not full redraw
            self._draw_mask_canvas()
            
        elif self.current_tool == 'rectangle':
            self.rect_start = (x, y)
    
    def _on_mouse_move(self, x, y):
        """Handle mouse drag - optimized with throttling."""
        if time.time() - self._last_move <= 0.05:
            return
        self._last_move = time.time()
        x, y = int(x), int(y)
        
        # Handle panning
        if self.is_panning and self.pan_start:
            self._draw_pan_preview(x, y)
            return
        
        if not self.is_drawing:
            return

        x ,y = self._screen_to_canvas_coords(x, y)
        
        if self.current_tool == 'pen':
            self._draw_line(self.last_x, self.last_y, x, y)
            self.last_x, self.last_y = x, y
            # Fast update - only mask canvas
            self._draw_mask_canvas()
            
        elif self.current_tool == 'eraser':
            self._erase_line(self.last_x, self.last_y, x, y)
            self.last_x, self.last_y = x, y
            # Fast update - only mask canvas
            self._draw_mask_canvas()
            
        elif self.current_tool == 'rectangle':
            # Preview rectangle - only redraw this layer
            self._draw_rectangle_preview(self.rect_start[0], self.rect_start[1], x, y)
    
    def _on_mouse_up(self, x, y):
        """Handle mouse release."""
        # End panning
        if self.is_panning:
            self.is_panning = False
            dx = x - self.pan_start[0]
            dy = y - self.pan_start[1]
            self.pan_offset = (self.pan_offset[0] + dx, self.pan_offset[1] + dy)
            self.pan_start = None
            self._apply_view_transform()
            self._draw_image()
            return
        
        x, y = int(x), int(y)

        x ,y = self._screen_to_canvas_coords(x, y)
        if not self.is_drawing:
            return
        
        self.is_drawing = False
        
        if self.current_tool == 'rectangle':
            self._apply_rectangle(self.rect_start[0], self.rect_start[1], x, y)
            self._draw_mask_canvas()
        
        # Save state after drawing is complete
        self._save_state()
        
        # Update composite image for final display
        self._update_composite()
        self._draw_image()
    
    def _draw_at_position(self, x, y):
        """Draw at a single position - optimized."""
        
        # Create a circular brush
        size = self.pen_size
        y_min = max(0, y - size//2)
        y_max = min(self.height, y + size//2)
        x_min = max(0, x - size//2)
        x_max = min(self.width, x + size//2)
        
        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
        circle = (xx - (x - size//2))**2 + (yy - (y - size//2))**2 <= (size//2)**2
        
        self.mask_array[y_min:y_max, x_min:x_max][circle] = 255
    
    def _draw_line(self, x1, y1, x2, y2):
        """Draw a line between two points - optimized."""
        
        # Bresenham's line algorithm for smooth lines
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        sx = -1 if x1 > x2 else 1
        sy = -1 if y1 > y2 else 1
        
        if dx > dy:
            err = dx / 2.0
            while x != x2:
                self._draw_circle_at(x, y)
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y2:
                self._draw_circle_at(x, y)
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        
        self._draw_circle_at(x2, y2)
    
    def _draw_circle_at(self, x, y):
        """Draw a filled circle at the given position on the mask array."""
        size = self.pen_size
        y_min = max(0, y - size//2)
        y_max = min(self.height, y + size//2)
        x_min = max(0, x - size//2)
        x_max = min(self.width, x + size//2)
        
        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
        circle = (xx - x)**2 + (yy - y)**2 <= (size//2)**2
        
        self.mask_array[y_min:y_max, x_min:x_max][circle] = 255
    
    def _erase_at_position(self, x, y):
        """Erase mask at a position."""
        self._erase_circle_at(x, y)
    
    def _erase_line(self, x1, y1, x2, y2):
        """Erase along a line between two points."""
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        sx = -1 if x1 > x2 else 1
        sy = -1 if y1 > y2 else 1
        
        if dx > dy:
            err = dx / 2.0
            while x != x2:
                self._erase_circle_at_coords(x, y)
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y2:
                self._erase_circle_at_coords(x, y)
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        
        self._erase_circle_at_coords(x2, y2)
    
    def _erase_circle_at(self, x, y):
        """Erase at position using current mask array."""
        self._erase_circle_at_coords(x, y)
    
    def _erase_circle_at_coords(self, x, y):
        """Erase a circle at coordinates on the mask array."""
        size = self.pen_size
        y_min = max(0, y - size//2)
        y_max = min(self.height, y + size//2)
        x_min = max(0, x - size//2)
        x_max = min(self.width, x + size//2)
        
        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
        circle = (xx - x)**2 + (yy - y)**2 <= (size//2)**2
        
        self.mask_array[y_min:y_max, x_min:x_max][circle] = 0
    
    def _draw_rectangle_preview(self, x1, y1, x2, y2):
        """Draw a preview rectangle (used during rectangle tool drag)."""
        # Clear and redraw for preview
        with hold_canvas(self.mask_canvas):
            self.mask_canvas.clear()
            self.mask_canvas.stroke_style = 'white'
            self.mask_canvas.line_width = 2
            self.mask_canvas.stroke_rect(
                min(x1, x2), min(y1, y2),
                abs(x2 - x1), abs(y2 - y1)
            )
    
    def _draw_pan_preview(self, x, y):
        with hold_canvas(self.mask_canvas):
            self.mask_canvas.clear()
            self.mask_canvas.stroke_style = 'white'
            self.mask_canvas.line_width = 2
            x1, y1 = self._screen_to_canvas_coords(*self.pan_start)
            x2, y2 = self._screen_to_canvas_coords(x, y)
            self.mask_canvas.stroke_line(x1, y1, x2, y2)
    
    def _apply_rectangle(self, x1, y1, x2, y2):
        """Apply the rectangle mask."""
        x_start = max(0, min(x1, x2))
        y_start = max(0, min(y1, y2))
        x_end = min(self.width, max(x1, x2))
        y_end = min(self.height, max(y1, y2))
        
        if x_start < x_end and y_start < y_end:
            self.mask_array[y_start:y_end, x_start:x_end] = True
    
    def _save_state(self):
        """Save current mask state for undo."""
        self.mask_history.append(self.mask_array.copy())
        if len(self.mask_history) > self.max_history:
            self.mask_history.pop(0)
    
    def get_mask(self):
        """
        Get the current mask as a PIL Image.
        
        Returns:
            PIL Image: Grayscale mask (255 = masked, 0 = unmasked)
        """
        return Image.fromarray(self.mask_array)
    
    def get_mask_array(self):
        """
        Get the current mask as a numpy array.
        
        Returns:
            numpy.ndarray: Grayscale mask array
        """
        return self.mask_array.copy()
    
    def display(self):
        """Display the widget."""
        return self.widget