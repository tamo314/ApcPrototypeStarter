"""Narrow headless compatibility fix for ManiSkill 3.0.1 / SAPIEN 3.0.3.

Those versions skip render components at build time but eagerly allocate some
materials earlier. Only visual construction is omitted when scene.can_render()
is false. Physics builders, URDF inertials/collisions and task logic stay upstream.
The scene-guarded hooks persist for later reset/reconfigure calls; rendered scenes
always delegate to the original functions. Installed packages are not edited.
"""
from copy import copy
from importlib.metadata import version

_installed = False


def install_headless_compat():
    global _installed
    if _installed:
        return
    if (version("mani_skill"), version("sapien")) != ("3.0.1", "3.0.3"):
        raise RuntimeError("Headless compatibility requires ManiSkill 3.0.1 / SAPIEN 3.0.3")

    from mani_skill.utils.building import actors
    from mani_skill.utils.building.actors import common
    from mani_skill.utils.building.urdf_loader import URDFLoader

    original_link = URDFLoader._build_link
    original_cube = common.build_cube
    original_sphere = common.build_sphere

    def build_link(loader, link, builder):
        if not loader.scene.can_render():
            # Keep the parsed URDF unchanged; the shallow copy shares inertials
            # and collision data with the original and replaces only visuals.
            link = copy(link)
            link.visuals = []
        return original_link(loader, link, builder)

    def build_cube(scene, half_size, color, name, body_type="dynamic",
                   add_collision=True, scene_idxs=None, initial_pose=None):
        if scene.can_render():
            return original_cube(scene, half_size, color, name, body_type,
                                 add_collision, scene_idxs, initial_pose)
        builder = scene.create_actor_builder()
        if add_collision:
            builder.add_box_collision(half_size=[half_size] * 3)
        return common._build_by_type(builder, name, body_type, scene_idxs, initial_pose)

    def build_sphere(scene, radius, color, name, body_type="dynamic",
                     add_collision=True, scene_idxs=None, initial_pose=None):
        if scene.can_render():
            return original_sphere(scene, radius, color, name, body_type,
                                   add_collision, scene_idxs, initial_pose)
        builder = scene.create_actor_builder()
        if add_collision:
            builder.add_sphere_collision(radius=radius)
        return common._build_by_type(builder, name, body_type, scene_idxs, initial_pose)

    URDFLoader._build_link = build_link
    # PickCube uses the package re-exports, not common directly.
    actors.build_cube = common.build_cube = build_cube
    actors.build_sphere = common.build_sphere = build_sphere
    _installed = True
