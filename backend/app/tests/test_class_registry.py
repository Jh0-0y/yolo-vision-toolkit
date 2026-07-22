from app.core.class_registry import ClassRegistry


def test_union_by_normalized_name():
    reg = ClassRegistry()
    map_a = reg.add_model("model_a", {0: "person", 1: "Car"})
    map_b = reg.add_model("model_b", {0: "car", 1: "dog"})

    assert reg.names == {0: "person", 1: "Car", 2: "dog"}
    assert map_a == {0: 0, 1: 1}
    assert map_b == {0: 1, 1: 2}  # "car" merged into "Car"


def test_sources_tracked_per_class():
    reg = ClassRegistry()
    reg.add_model("a", {0: "person", 1: "car"})
    reg.add_model("b", {0: "car", 1: "dog"})

    sources = reg.class_sources()
    assert sources[0] == ["a"]
    assert sources[1] == ["a", "b"]
    assert sources[2] == ["b"]


def test_ids_append_only_when_adding_models_later():
    reg = ClassRegistry()
    reg.add_model("a", {0: "person"})
    snapshot = reg.to_dict()

    # simulate later session: reload and add another model
    reg2 = ClassRegistry.from_dict(snapshot)
    reg2.add_model("b", {0: "zebra", 1: "person"})

    assert reg2.names[0] == "person"  # existing id untouched
    assert reg2.names[1] == "zebra"  # new class appended


def test_roundtrip_serialization():
    reg = ClassRegistry()
    reg.add_model("a", {0: "person", 1: "car"})
    reg2 = ClassRegistry.from_dict(reg.to_dict())
    assert reg2.names == reg.names
    assert reg2.class_sources() == reg.class_sources()
