# CSS Structure for AI Job Scraper

This directory contains the organized CSS files for the AI Job Scraper Streamlit application.

## File Structure

- **`main.css`** - Main entry point that imports all other CSS modules

- **`theme.css`** - CSS variables and global styles (colors, typography, scrollbars)

- **`components.css`** - Component-specific styles (buttons, cards, inputs, badges)

- **`responsive.css`** - Media queries and responsive design rules

- **`animations.css`** - Keyframe animations and transitions

## Benefits of This Structure

1. **Maintainability**: Each file has a clear purpose, making it easy to find and update styles
2. **Reusability**: CSS modules can be reused across different Streamlit projects
3. **Performance**: External CSS files can be cached by browsers
4. **Developer Experience**: Proper syntax highlighting and CSS tooling in your editor
5. **Separation of Concerns**: Clean separation between Python logic and styling

## Usage

The CSS is loaded in `app.py` using the `css_loader` utility:

```python
from utils.css_loader import load_css
load_css("static/css/main.css")
```

## Customization

To modify the theme:

1. Edit CSS variables in `theme.css`
2. Add new component styles in `components.css`
3. Add responsive rules in `responsive.css`
4. Add animations in `animations.css`

## CSS Variables

The theme uses CSS custom properties (variables) for consistent styling:

- Primary colors: `--primary-color`, `--primary-hover`, `--primary-dark`

- Status colors: `--success-color`, `--danger-color`, `--warning-color`

- Background colors: `--bg-primary`, `--bg-secondary`, `--bg-card`

- Text colors: `--text-primary`, `--text-secondary`, `--text-muted`

- Shadows: `--shadow-sm`, `--shadow-md`, `--shadow-lg`
