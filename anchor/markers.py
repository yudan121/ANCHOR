from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class MarkerTreeNode:
    name: str
    positive_markers: tuple[str, ...] = ()
    negative_markers: tuple[str, ...] = ()
    children: tuple["MarkerTreeNode", ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MarkerTreeNode":
        if "name" not in data:
            raise ValueError("marker tree node must contain `name`")
        return cls(
            name=str(data["name"]),
            positive_markers=tuple(str(x) for x in data.get("positive_markers", ()) or ()),
            negative_markers=tuple(str(x) for x in data.get("negative_markers", ()) or ()),
            children=tuple(cls.from_dict(x) for x in data.get("children", ()) or ()),
            metadata={
                k: v
                for k, v in data.items()
                if k not in {"name", "positive_markers", "negative_markers", "children"}
            },
        )

    def to_dict(self) -> dict[str, Any]:
        out = {
            "name": self.name,
            "positive_markers": list(self.positive_markers),
            "negative_markers": list(self.negative_markers),
            "children": [child.to_dict() for child in self.children],
        }
        out.update(dict(self.metadata))
        return out

    def walk(self) -> Iterable["MarkerTreeNode"]:
        yield self
        for child in self.children:
            yield from child.walk()


@dataclass(frozen=True)
class MarkerTree:
    root: MarkerTreeNode

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MarkerTree":
        return cls(root=MarkerTreeNode.from_dict(data))

    def to_dict(self) -> dict[str, Any]:
        return self.root.to_dict()

    def infer_hidden_branch(self) -> bool:
        """Return whether the public schema explicitly declares a hidden branch."""
        for node in self.root.walk():
            if node.metadata.get("hidden_branch") is True:
                return True
        return False


def load_marker_tree(source: str | Path | Mapping[str, Any] | MarkerTree) -> MarkerTree:
    if isinstance(source, MarkerTree):
        return source
    if isinstance(source, Mapping):
        return MarkerTree.from_dict(source)
    payload = json.loads(Path(source).read_text())
    if not isinstance(payload, Mapping):
        raise ValueError("marker tree JSON must be an object")
    return MarkerTree.from_dict(payload)
