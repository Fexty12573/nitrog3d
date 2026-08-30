local has_blender, blender = pcall(require, "blender")
if has_blender then
	blender.setup({
		profiles = {
			{
				name = "nitrog3d",
				cmd = "blender",
			},
		},
	})
end
