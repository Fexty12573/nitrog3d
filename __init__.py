def register():
    from .blender import operator, texture_props, texture_ui

    operator.register()
    texture_props.register()
    texture_ui.register()


def unregister():
    from .blender import operator, texture_props, texture_ui

    texture_ui.unregister()
    texture_props.unregister()
    operator.unregister()
