"""Resize the Easy XML using its existing agent and force-actuator templates."""

from copy import deepcopy
import math
import re
import xml.etree.ElementTree as ET


def build_model_xml(xml, num_agents):
    if isinstance(num_agents, bool) or not isinstance(num_agents, int) or not 1 <= num_agents <= 16:
        raise ValueError("num_agents must be an integer from 1 to 16")
    root = ET.fromstring(xml)
    world = root.find("worldbody")
    actuator = root.find("actuator")
    if world is None or actuator is None:
        raise ValueError("Easy XML needs worldbody and actuator sections")
    bodies = [b for b in world.findall("body") if re.fullmatch(r"agent_\d+", b.get("name", ""))]
    names = {b.get("name") for b in bodies}
    if names == {f"agent_{i}" for i in range(num_agents)}:
        return xml  # Preserve snapshots, including their exact hashes, when already sized.
    template = next((b for b in bodies if b.get("name") == "agent_0"), None)
    motors = [actuator.find(f"motor[@name='agent_0_f{axis}']") for axis in ("x", "y")]
    if template is None or any(m is None for m in motors):
        raise ValueError("Easy XML needs agent_0 and its X/Y force motors as templates")
    template = deepcopy(template)
    motors = [deepcopy(m) for m in motors]
    for body in bodies:
        world.remove(body)
    for motor in list(actuator):
        if re.fullmatch(r"agent_\d+_f[xy]", motor.get("name", "")):
            actuator.remove(motor)
    obj = world.find("body[@name='t_shape']")
    if obj is None:
        raise ValueError("Easy XML needs a t_shape body")
    ox, oy, _ = map(float, obj.get("pos", "0 0 0").split())
    colors = ("0.1 0.65 0.25 1", "0.15 0.45 0.9 1", "0.65 0.2 0.8 1", "0.9 0.2 0.2 1")
    for i in range(num_agents):
        body = deepcopy(template)
        for element in body.iter():
            for attribute in ("name", "joint"):
                if attribute in element.attrib:
                    element.set(attribute, element.get(attribute).replace("agent_0", f"agent_{i}"))
        # Valid separated XML reference positions; the environment sets actual spawns.
        angle = 2 * math.pi * i / num_agents
        z = template.get("pos", "0 0 .04").split()[2]
        body.set("pos", f"{ox + .8 * math.cos(angle)} {oy + .8 * math.sin(angle)} {z}")
        body.find("geom").set("rgba", colors[i % len(colors)])
        world.append(body)
        for template_motor in motors:
            motor = deepcopy(template_motor)
            for attribute in ("name", "joint"):
                motor.set(attribute, motor.get(attribute).replace("agent_0", f"agent_{i}"))
            actuator.append(motor)
    return ET.tostring(root, encoding="unicode")
