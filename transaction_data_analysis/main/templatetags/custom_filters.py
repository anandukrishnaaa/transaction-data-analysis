from django import template
import plotly.io as pio

register = template.Library()

"""
Custom filter `to_div_html` used to render charts and plots in `div` tag, overriding the default behavior of rendering as a full html page.
"""


@register.filter
def to_div_html(figure):
    # Modify other parameters or perform additional operations
    html_output = pio.to_html(
        figure,
        full_html=False,
        # Remove the select tools, save as image and plotly icons
        config={
            "modeBarButtonsToRemove": ["select2d", "lasso2d", "toImage"],
            "displaylogo": False,
        },
    )
    return html_output
