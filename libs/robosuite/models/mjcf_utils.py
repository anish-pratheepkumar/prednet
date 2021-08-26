# utility functions for manipulating MJCF XML models

import os
import xml.etree.ElementTree as ET

import numpy as np

from . import assets_root

transparent_5 = 0.05
transparent_10 = 0.10
RED = [1, 0, 0, 0.8]
ORANGE = [1, 0.63, 0.42, transparent_10]
GREEN = [0, 1, 0, 0.0]
NONE = [0, 0, 0, 0.0]
BLUE = [0, 0, 1, 1]
YELLOW = [1., 0.65, 0, transparent_10]
GREY = [0.75, 0.75, 0.75, 0.3]
BLACK = [0.9, 0.9, 0.9, 0.3]
INVISIBLE = [0., 0., 0., 0.]


def xml_path_completion(xml_path, given_assets_root=None):
    """
    Takes in a local xml path and returns a full path.
        if @xml_path is absolute, do nothing
        if @xml_path is not absolute, load xml that is shipped by the package
    """
    if os.path.isfile(xml_path): #xml_path.startswith("/"):
        full_path = xml_path
    else:
        if given_assets_root is None:
            given_assets_root = assets_root
        else:
            pass
        full_path = os.path.abspath(os.path.join(given_assets_root, xml_path))

    return full_path


def array_to_string(array, delimiter=" ", format="{}", precision=None):
    """
    Converts a numeric array into the string format in mujoco.

    Examples:
        [0, 1, 2] => "0 1 2"
    """
    if precision is not None and format == "{}":
        return delimiter.join([format.format(round(x, precision)) for x in array])
    else:
        return delimiter.join([format.format(x, precision) for x in array])


def string_to_array(string, type=float, delimiter=" "):
    """
    Converts a array string in mujoco xml to np.array.

    Examples:
        "0 1 2" => [0, 1, 2]
    """
    return np.array([type(x) for x in string.split(delimiter) if x!=delimiter if x])


def set_alpha(node, alpha=0.1):
    """
    Sets all a(lpha) field of the rgba attribute to be @alpha
    for @node and all subnodes
    used for managing display
    """
    for child_node in node.findall(".//*[@rgba]"):
        rgba_orig = string_to_array(child_node.get("rgba"))
        child_node.set("rgba", array_to_string(list(rgba_orig[0:3]) + [alpha]))


def new_joint(**kwargs):
    """
    Creates a joint tag with attributes specified by @**kwargs.
    """

    element = ET.Element("joint", attrib=kwargs)
    return element


def new_actuator(joint, act_type="actuator", **kwargs):
    """
    Creates an actuator tag with attributes specified by @**kwargs.

    Args:
        joint: element_type of actuator transmission.
            see all types here: http://mujoco.org/book/modeling.html#actuator
        act_type (str): actuator element_type. Defaults to "actuator"

    """
    element = ET.Element(act_type, attrib=kwargs)
    element.set("joint", joint)
    return element


def new_mocap(name):
    """
<!--body mocap="true" name="robot0:mocap" pos="0 0 0" quat="1. 0. 0. 0.">
            <geom conaffinity="0" contype="0" pos="0 0 0" rgba="0 0 0 0.7" size="0.005 0.005 0.005" type="box"/>
            <geom conaffinity="0" contype="0" pos="0 0 0" rgba="0.5 0 0 0.1" size="1 0.005 0.005" type="box"/>
            <geom conaffinity="0" contype="0" pos="0 0 0" rgba="0 0.5 0 0.1" size="0.005 1 0.001" type="box"/>
            <geom conaffinity="0" contype="0" pos="0 0 0" rgba="0 0 0.5 0.1" size="0.005 0.005 1" type="box"/>
    </body-->
    """
    # print(mocap_body)
    body = new_body(name, pos=[0., 0., 0.], quat="1. 0. 0. 0.", mocap="true")
    geom_1 = new_geom(geom_type="box", size=[0.025, 0.025, 0.025], rgba=[1, 1, 0, 0.05],
                      conaffinity="0", contype="0")
    geom_2 = new_geom(geom_type="box", size=[0.005, 0.005, 0.005], rgba=[0.5, 0, 0, 0.05],
                      conaffinity="0", contype="0")
    geom_3 = new_geom(geom_type="box", size=[0.005, 0.005, 0.005], rgba=[0, 0.5, 0, 0.05],
                      conaffinity="0", contype="0")
    geom_4 = new_geom(geom_type="box", size=[0.005, 0.005, 0.005], rgba=[0, 0, 0.5, 0.05],
                      conaffinity="0", contype="0")
    body.append(geom_1)
    body.append(geom_2)
    body.append(geom_3)
    body.append(geom_4)
    return body


def new_sensor(site, sen_type="sensor", **kwargs):
    element = ET.Element(sen_type, attrib=kwargs)
    element.set("site", site)
    return element


def new_site(name=None, rgba=None, pos=(0, 0, 0), size=None, **kwargs):
    """
    Creates a site element with attributes specified by @**kwargs.

    Args:
        name (str): site name.
        rgba: color and transparency. Defaults to solid red.
        pos: 3d position of the site.
        size ([float]): site size (sites are spherical by default).
    """
    if rgba is not None:
        if isinstance(rgba, str):
            kwargs["rgba"] = rgba
        else:
            kwargs["rgba"] = array_to_string(rgba)
    if pos is not None:
        kwargs["pos"] = array_to_string(pos)
    if size is not None:
        kwargs["size"] = array_to_string(size)
    if name is not None:
        kwargs["name"] = name
    element = ET.Element("site", attrib=kwargs)
    return element


def new_element_from_args(element_type, **kwargs):
    element = ET.Element(element_type, attrib=kwargs)
    return element


def new_geom(geom_type, size, pos=(0, 0, 0), rgba=None, group=0, **kwargs):
    """
    Creates a geom element with attributes specified by @**kwargs.

    Args:
        geom_type (str): element_type of the geom.
            see all types here: http://mujoco.org/book/modeling.html#geom
        size: geom size parameters.
        pos: 3d position of the geom frame.
        rgba: color and transparency. Defaults to solid red.
        group: the integrer group that the geom belongs to. useful for
            separating visual and physical elements.
    """
    kwargs["type"] = str(geom_type)
    kwargs["size"] = array_to_string(size)
    if rgba:
        if isinstance(pos, str):
            kwargs["rgba"] = rgba
        else:
            kwargs["rgba"] = array_to_string(rgba)
    kwargs["group"] = str(group)
    if isinstance(pos, str):
        kwargs["pos"] = pos
    else:
        kwargs["pos"] = array_to_string(pos, format="{0:0.4f}")
    element = ET.Element("geom", attrib=kwargs)
    return element


def new_body(name=None, pos=None, **kwargs):
    """
    Creates a body element with attributes specified by @**kwargs.

    Args:
        name (str): body name.
        pos: 3d position of the body frame.
    """
    if name is not None:
        kwargs["name"] = name
    if pos is not None:
        kwargs["pos"] = array_to_string(pos)
    element = ET.Element("body", attrib=kwargs)
    return element


def new_inertial(pos=(0, 0, 0), mass=None, **kwargs):
    """
    Creates a inertial element with attributes specified by @**kwargs.

    Args:
        mass: The mass of inertial
    """
    if mass is not None:
        kwargs["mass"] = str(mass)
    kwargs["pos"] = array_to_string(pos)
    element = ET.Element("inertial", attrib=kwargs)
    return element


def exclude_contact(body1, body2):
    """
    Creates a site element with attributes specified by @**kwargs.

    Args:
        body1 (str): name of body1
        body2 (str):name body2
    """
    kwargs = {}
    kwargs["body1"] = body1
    kwargs["body2"] = body2
    element = ET.Element("exclude", attrib=kwargs)
    return element


def add_equality_constraint(body1, body2, solimp="0.9 0.95 0.001", solref="0.02 1", **kwargs):
    """
    Creates a site element with attributes specified by @**kwargs.

    Args:
        name (str): site name.
        rgba: color and transparency. Defaults to solid red.
        pos: 3d position of the site.
        size ([float]): site size (sites are spherical by default).

    Example:
        <equality>
        <weld body1="robot0:mocap" body2="right_gripper" solimp="0.9 0.95 0.001" solref="0.02 1"></weld>
        </equality>

    """
    kwargs.update({"body1": body1, "body2": body2, "solimp": solimp, "solref": solref})
    element = ET.Element("weld", attrib=kwargs)
    return element


def postprocess_model_xml(xml_str):
    """
    This function postprocesses the model.xml collected from a MuJoCo demonstration
    in order to make sure that the STL files can be found.
    """

    path = os.path.split("../")[0]
    path_split = path.split("/")

    # replace mesh and texture file paths
    tree = ET.fromstring(xml_str)
    root = tree
    asset = root.find("asset")
    meshes = asset.findall("mesh")
    textures = asset.findall("texture")
    all_elements = meshes + textures

    for elem in all_elements:
        old_path = elem.get("file")
        if old_path is None:
            continue
        old_path_split = old_path.split("/")
        ind = max(
            loc for loc, val in enumerate(old_path_split) if val == "robosuite"
        )  # last occurrence index
        new_path_split = path_split + old_path_split[ind + 1:]
        new_path = "/".join(new_path_split)
        elem.set("file", new_path)

    return ET.tostring(root, encoding="utf8").decode("utf8")


def new_element(element_type: str, **kwargs):
    element = ET.Element(element_type, attrib=kwargs)
    return element
