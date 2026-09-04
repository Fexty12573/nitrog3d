def register():
    from .blender import operator, texture_props

    operator.register()
    texture_props.register()


def unregister():
    from .blender import operator, texture_props

    texture_props.unregister()
    operator.unregister()
