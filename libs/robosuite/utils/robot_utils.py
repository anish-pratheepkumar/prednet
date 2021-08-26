from xml.etree.ElementTree import Element

import numpy as np


def get_children_bodies(worldbody, include_bodies_with_all_children: list) -> object:
    """
    Returns all children bodies of a given body
    :param include_bodies_with_all_children: name of body in xml
    :return: list of names (latest) including parent body and all its children
    """
    all_collision_bodies = set()
    for collision_body_name in include_bodies_with_all_children:
        parent_body = find_element_with_property(worldbody, element_tag="body",
                                                 property_tag="name",
                                                 property_prefix=collision_body_name)

        all_collision_bodies.add(parent_body.get("name"))
        children_bodies = parent_body.findall(".//body")
        for child_body in children_bodies:
            child_body_name = child_body.get("name")
            if child_body_name:
                all_collision_bodies.add(child_body_name)
    return list(all_collision_bodies)


def get_joint(worldbody, name: str):
    """
    Returns all children bodies of a given body
    :param include_bodies_with_all_children: name of body in xml
    :return: list of names (latest) including parent body and all its children
    """
    return find_element_with_property(worldbody, element_tag="joint", property_tag="name", property_prefix=name)


def get_actuator(parent, actuator_name, type="position"):
    return find_element_with_property(parent, element_tag=type, property_tag="name", property_prefix=actuator_name)


def get_actuator_from_joint(parent, joint_name, type="position"):
    return find_element_with_property(parent, element_tag=type, property_tag="joint", property_prefix=joint_name)


def get_geom(worldbody, geom_name: str):
    """
    Returns all children bodies of a given body
    :param include_bodies_with_all_children: name of body in xml
    :return: list of names (latest) including parent body and all its children
    """
    return find_element_with_property(worldbody, element_tag="geom", property_tag="name", property_prefix=geom_name)


def get_mocap(parent, body_name):
    """
    Finds mocap with prefix body_name
    :param parent: worldbody or parent body
    :param body_name: prefix or exact mocap name
    :return:
    """
    mocaps = find_element_with_property(parent, element_tag="body", property_tag="mocap", property_prefix="true")
    if mocaps is None:
        return None
    if isinstance(mocaps, list):
        for mocap in mocaps:
            if body_name in mocap.get("name"):
                return mocap
    else:
        if body_name in mocaps.get("name"):
            return mocaps


def get_something(parent, type, name, all: bool = False):
    return find_element_with_property(parent, element_tag=type, property_tag="name", property_prefix=name, all=all)


def get_body(parent, body_name):
    """
    Finds body with body_name
    :param parent:
    :param body_name:
    :return:
    """
    return find_element_with_property(parent, element_tag="body", property_tag="name", property_prefix=body_name)


def get_default_class(parent, body_name):
    """
    Finds body with body_name
    :param parent:
    :param body_name:
    :return:
    """
    return find_element_with_property(parent, element_tag="default", property_tag="class", property_prefix=body_name)


def get_site(parent, body_name):
    """
    Finds body with body_name
    :param parent:
    :param body_name:
    :return:
    """
    return find_element_with_property(parent, element_tag="site", property_tag="name", property_prefix=body_name)


def get_sensor(parent, sensor_type, name):
    """
    Finds sensor with given name and tag. Example
    <sensor>
        <force name="force_value" site="force_sensor"/> <!-- sensor_type: force, name: force_value-->
        <torque name="torque_value" site="force_sensor"/> <!-- sensor_type: torque, name: torque_value-->
    </sensor>
    :param parent:
    :param sensor_type:
    :param name:
    :return:
    """
    return find_element_with_property(parent, element_tag=sensor_type, property_tag="name", property_prefix=name)


def get_sensors_model_detail(sim, sensor_names: list) -> (np.array, np.array, np.array):
    """
    Get the sensor ID, sensor ADR and sensor DIM of a list of sensor names as string

    Args:
        sim (MjSim): mujoco py simulation
        sensor_names (list): list with sensor names as strings, e.g. ['force_sensor', 'touch_sensor']

    Returns:
        sensor_id, sensor_adr, sensor_dim: returns a tuple with the id, adr and dim of the sensors
    """
    if sim is None or sensor_names is None or sim.model.sensor_adr is None:
        return None, None, None
    else:
        sensor_id = np.array([sim.model.sensor_name2id(sensor_name) for sensor_name in sensor_names])
        sensor_address = sim.model.sensor_adr[sensor_id]
        sensor_dim = sim.model.sensor_dim[sensor_id]
        return sensor_id, sensor_address, sensor_dim


def read_sensors(sim, sensor_address: np.array, sensor_dim: np.array):
    """
    Return sensor readings
    Args:
        sim:
        sensor_id: from get_sensors_model_detail
        sensor_address: sensor address from get_sensors_model_detail
        sensor_dim: sensor dimension from get_sensors_model_detail

    Returns:

    """
    readings = np.array(
        [
            sim.data.sensordata[sensor_add_:sensor_add_ + sensor_dim_]
            for sensor_add_, sensor_dim_ in
            zip(sensor_address, sensor_dim)
        ]
    )
    return readings


def get_parent_node(root, child):
    """
    Return parent element of a child
    Args:
        model (XMLElement):
        child (XMLElement):

    Returns (XMLElement): direct parent node of child

    """
    parent_map = {c: p for p in root.iter() for c in p}
    return parent_map.get(child)


def find_element_with_property(parent, element_tag, property_tag, property_prefix, all=False):
    """
    parent.findall(.//{element_tag}@[{property_tag} CONTAINS {property_prefix}])
    :param parent:
    :param element_tag:
    :param property_tag:
    :param property_prefix:
    :param all: all elements with this property_tag=property_prefix
    :return: element(s) with exact property prefix or the first element with prefix occurance
    """
    if all:
        all_elements = parent.findall(".//{}[@{}='{}']".format(element_tag, property_tag, property_prefix))
    else:
        all_elements = parent.find(".//{}[@{}='{}']".format(element_tag, property_tag, property_prefix))
    if all_elements is None or len(all_elements)==0:
        all_elements = parent.findall(".//{}".format(element_tag))
        return_elements = []
        for element in all_elements:
            property = element.get(property_tag)
            if property is not None and property_prefix in property:
                if all:
                    return_elements.append(element)
                else:
                    return element
        return return_elements
    else:
        return all_elements


def get_first_body_with_name(parent, name):
    all_bodies = parent.findall(".//body")
    for body_ in all_bodies:
        if name in body_.get("name"):
            body_with_tag = body_
            return body_with_tag


def enable_collisions_geoms(sim, geoms_1, geoms_2, contype_1=4, ignore_disabled_geoms=True):
    """
    Enable collisions between geoms_1 and geoms_2 by setting geoms_1.contype = geoms_2.conaffinity = contype_1.
    This affects only the contype of geoms_1 and the conaffinity of geoms_2. However, if the geom has the contyp or
     conaffinity set to 0, the geom will be ignored
    Args:
        sim:
        geoms_1:
        geoms_2:
        contype_1:

    Returns:

    """
    for geom_1_ in geoms_1:
        geom_1_name = geom_1_.get("name")
        if geom_1_name:
            geom_1_id = sim.model.geom_name2id(geom_1_name)
            if sim.model.geom_contype[geom_1_id] == 0 and ignore_disabled_geoms:
                continue
            else:
                sim.model.geom_contype[geom_1_id] = contype_1
    for geom_2_ in geoms_2:
        geom_2_name = geom_2_.get("name")
        if geom_2_name:
            geom_2_id = sim.model.geom_name2id(geom_2_name)
            if sim.model.geom_conaffinity[geom_2_id] == 0 and ignore_disabled_geoms:
                continue
            else:
                sim.model.geom_conaffinity[geom_2_id] = contype_1


def enable_collisions(sim, bodies_1, bodies_2, contype=4, conaffinity=0):
    enabled_bodies = set()
    for body in bodies_1:
        # set contype to 4
        if isinstance(body, Element):
            geoms = body.findall(".geom")
        else:
            geoms = body.model.worldbody.findall(".//geom")
        for geom in geoms:
            if (geom.get("contype") is None) or (geom.get("contype") is not None and geom.get("contype") is not "0"):
                geom_id = sim.model.geom_name2id(geom.get("name"))
                sim.model.geom_contype[geom_id] = contype
                sim.model.geom_conaffinity[geom_id] = conaffinity
                parent_body = get_body_name(sim, geom_id)
                enabled_bodies.add(parent_body)
        # set_conaffinity to 0
    for body in bodies_2:
        # set conaffinity to 4
        # set contype to 0
        # set contype to 4
        if isinstance(body, Element):
            geoms = body.findall(".geom")
        else:
            geoms = body.model.worldbody.findall(".//geom")
        for geom in geoms:
            if (geom.get("conaffinity") is None) or (
                    geom.get("conaffinity") is not None and geom.get("conaffinity") is not "0"):
                geom_id = sim.model.geom_name2id(geom.get("name"))
                sim.model.geom_conaffinity[geom_id] = contype
                sim.model.geom_contype[geom_id] = conaffinity
                parent_body = get_body_name(sim, geom_id)
                enabled_bodies.add(parent_body)
    return list(enabled_bodies)


def get_body_name(sim, child_geom_id):
    contact_body_id = sim.model.geom_bodyid[child_geom_id]
    contact_body_name = sim.model.body_id2name(contact_body_id)
    return contact_body_name
