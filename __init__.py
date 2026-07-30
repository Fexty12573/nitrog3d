
def register():
    from .blender import operator
    operator.register()


def unregister():
    from .blender import operator
    operator.unregister()
