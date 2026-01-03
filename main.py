#!/bin/env python3

import json
from abc import abstractmethod
from argparse import ArgumentParser
from dataclasses import dataclass, field
from os import path
from pathlib import Path
from sys import stderr
from typing import (
	IO,
	Any,
	Dict,
	List,
	Literal,
	Optional,
	Protocol,
	Set,
	Tuple,
	cast,
	override,
)

import pydash
from pygments import formatters, highlight, lexers


class JSONEncoder(json.JSONEncoder):
	def default(self, o: Any) -> Any:
		if isinstance(o, Obj):
			return {
				"type": o.type,
				"properties": o.properties,
				"items": o.items,
				"required": o.required,
			}
		return super().default(o)


def prettify(obj: Any):
	print(
		highlight(
			json.dumps(obj, cls=JSONEncoder, indent=2),
			lexers.JsonLexer(),
			formatters.Terminal256Formatter(),
		)
	)


def get(obj, path: str):
	return pydash.get({"#": obj}, path.replace("/", "."))


PrimitiveSchemaTypes = Literal[
	"string",
	"number",
	"integer",
	"boolean",
	"any",
]

ComplexSchemaTypes = Literal[
	"enum",
	"ref",
	"array",
	"object",
	"tuple",
]


SchemaTypes = PrimitiveSchemaTypes | ComplexSchemaTypes


IncludeTypes = Literal["system", "local"]
ExtTypes = Literal["h", "hpp", "hxx"]


@dataclass(frozen=True)
class Include:
	name: str
	ext: Optional[ExtTypes] = None
	_opener: str = field(init=False)
	_closer: str = field(init=False)

	def to_string(self) -> str:
		ext = "" if self.ext is None else "." + self.ext
		return f"#include {self._opener}{ext}{self._closer}"


@dataclass(frozen=True)
class SystemInclude(Include):
	_opener: str = field(init=False, default="<")
	_closer: str = field(init=False, default=">")


@dataclass(frozen=True)
class ProjectInclude(Include):
	_opener: str = field(init=False, default='"')
	_closer: str = field(init=False, default='"')


@dataclass
class Obj:
	type: SchemaTypes
	properties: Optional[Dict[str, "Obj"]] = None
	items: Optional["Obj"] = None
	prefix_items: Optional[List["Obj"]] = None
	required: bool = False
	ref: Optional[str] = None
	enum: Optional[List[str]] = None


@dataclass
class SchemaType(Protocol):
	name: str

	@abstractmethod
	def to_string(self) -> str: ...

	@abstractmethod
	def get_lang_type(self) -> str: ...

	def get_includes(self) -> Set[Include]:
		return set()


@dataclass
class Enum(SchemaType):
	members: List[str]

	@override
	def to_string(self) -> str:
		lines = [
			f"enum class {self.get_lang_type()}",
			"{",
			",\n".join(["\t" + member for member in self.members]),
			"};",
		]
		return "\n".join(lines)

	@override
	def get_lang_type(self) -> str:
		return f"{self.name}"


@dataclass
class Ref(SchemaType):
	ref_type: str

	@override
	def to_string(self) -> str:
		return f"\n{self.get_lang_type()} {self.name};"

	@override
	def get_lang_type(self) -> str:
		return self.ref_type

	@override
	def get_includes(self) -> Set[Include]:
		return {ProjectInclude(self.ref_type, "hpp")}


@dataclass
class Array(SchemaType):
	children_types: SchemaType

	@override
	def to_string(self) -> str:
		return f"\n{self.get_lang_type()} {self.name}"

	@override
	def get_lang_type(self) -> str:
		return f"std::vector<{self.children_types.get_lang_type()}>"

	@override
	def get_includes(self) -> Set[Include]:
		return {
			SystemInclude("vector"),
			*self.children_types.get_includes(),
		}


@dataclass
class Obj_(SchemaType):
	properties: Dict[str, SchemaType]

	@override
	def to_string(self) -> str:
		return "\n".join(
			[
				f"struct {self.name}",
				"{",
				*[
					f"\t{prop_type} {prepare_prop_name(prop_name)};"
					for prop_name, prop_type in self.properties.items()
				],
				"};",
			]
		)

	@override
	def get_lang_type(self) -> str:
		return f"{self.name}"

	@override
	def get_includes(self) -> Set[Include]:
		includes = set()
		for prop in self.properties.values():
			includes = includes.union(prop.get_includes())
		return {
			SystemInclude("tuple"),
			*includes,
		}


@dataclass
class TupleType(SchemaType):
	member_types: List[SchemaType]

	@override
	def to_string(self) -> str:
		return f"{self.get_lang_type()} {self.name};"

	@override
	def get_lang_type(self) -> str:
		types = ", ".join([m.get_lang_type() for m in self.member_types])
		return f"std::tuple<{types}>"

	@override
	def get_includes(self) -> Set[Include]:
		includes = set()
		for m in self.member_types:
			includes = includes.union(m.get_includes())
		return {
			SystemInclude("tuple"),
			*includes,
		}


@dataclass
class Primitive(SchemaType):
	@override
	def to_string(self) -> str:
		return f"{self.get_lang_type()} {self.name};"


@dataclass
class String(Primitive):
	@override
	def get_lang_type(self) -> str:
		return "std::string"

	@override
	def get_includes(self) -> Set[Include]:
		return {SystemInclude("string")}


@dataclass
class Number(Primitive):
	@override
	def get_lang_type(self) -> str:
		return "double"


@dataclass
class Integer(Primitive):
	@override
	def get_lang_type(self) -> str:
		return "long"


@dataclass
class Boolean(Primitive):
	@override
	def get_lang_type(self) -> str:
		return "bool"


@dataclass
class AnySchemaType(Primitive):
	@override
	def get_lang_type(self) -> str:
		return "std::any"

	@override
	def get_includes(self) -> Set[Include]:
		return {SystemInclude("any")}


def parse_schema(root: Dict[str, Any], obj: Dict[str, Any]) -> Obj:
	items = None
	properties = None
	prefix_items = None
	type_: SchemaTypes
	enum = None

	if "allOf" in obj:
		ref = obj["allOf"][0]["$ref"]
		return Obj("ref", ref=ref)
	if "$ref" in obj:
		ref = obj["$ref"]
		return Obj("ref", ref=ref)
	if "type" not in obj:
		return Obj("any")
	try:
		match obj["type"]:
			case "array":
				if "prefixItems" in obj:
					prefix_items = []
					for prefix_item in obj["prefixItems"]:
						prefix_items.append(parse_schema(root, prefix_item))
					type_ = "tuple"
				else:
					items = parse_schema(root, obj["items"])
					type_ = "array"
			case "object":
				type_ = "object"
				properties = {}
				required: List[str] = obj.get("required") or []
				if "properties" in obj:
					for name, value in obj["properties"].items():
						properties[name] = parse_schema(root, value)
						properties[name].required = name in required
			case _:
				if "enum" in obj:
					type_ = "enum"
					enum = obj["enum"]
				else:
					type_ = obj["type"]
	except Exception:
		prettify(obj)
		raise

	return Obj(
		type=type_,
		items=items,
		properties=properties,
		prefix_items=prefix_items,
		enum=enum,
	)


def parse_schemas(root):
	schemas: dict[str, Obj] = {}
	for name, schema in pydash.get(root, "components.schemas").items():
		if name in schemas:
			continue
		schemas[name] = parse_schema(root, schema)
		schemas[name].required = True
	return schemas


def base_name(ref: str):
	return ref.split("/")[-1]


def get_basic_cxx_name(schema_type: str) -> str:
	match schema_type:
		case "any":
			prop_class = "std::any"
		case "integer":
			prop_class = "int"
		case "number":
			prop_class = "float"
		case "string":
			prop_class = "std::string"
		case "boolean":
			prop_class = "bool"
		case _:
			print(schema_type)
			prop_class = "void*"
	return prop_class


def get_prop_class_and_include(
	parent: Obj, prop_val: Obj
) -> Tuple[str, Optional[str]]:
	prop_include = None
	match prop_val.type:
		case "ref":
			prop_class = base_name(cast(str, prop_val.ref))
		case "array":
			items = cast(Obj, prop_val.items)
			if items.ref is not None:
				array_items = base_name(cast(str, items.ref))
			else:
				array_items = get_basic_cxx_name(items.type)
			prop_class = f"std::vector<{array_items}>"
			prop_include = f"{array_items}.hxx"
		case _:
			prop_class = get_basic_cxx_name(parent.type)
	return prop_class, prop_include


def prepare_prop_name(name: str) -> str:
	reserved_names = ["operator"]
	if name in reserved_names:
		return name + "_"
	return name


def write_schema(
	class_type: str,
	class_name: str,
	system_include: Set[str],
	project_include: Set[str],
	properties: Dict[str, str],
	file: IO,
):
	file.writelines(
		[
			"#pragma once\n\n",
			*[f"#include <{sys_incl}>\n" for sys_incl in system_include],
			"\n",
			*[f'#include "{proj_incl}"\n' for proj_incl in project_include],
			"\n",
			f"{class_type} {class_name}\n{{\n",
			*[
				f"\t{prop_type} {prepare_prop_name(prop_name)};\n"
				for prop_name, prop_type in properties.items()
			],
			"};\n",
		]
	)


def dump_schema(name: str, obj: Obj, file: IO):
	class_type = None
	system_include = {"nlohmann/json.hpp", "cpr/cpr.h"}
	project_include = set()
	properties = {}

	match obj.type:
		case "object":
			class_type = "struct"

			props = cast(Dict[str, Obj], obj.properties).items()
			for prop_name, prop_val in props:
				match prop_val.type:
					case "ref":
						project_include.add(
							f"{base_name(cast(str, prop_val.ref))}.hxx"
						)
					case "any":
						system_include.add("any")
					case "string":
						system_include.add("string")
					case "array":
						system_include.add("vector")
				properties[prop_name], prop_include = (
					get_prop_class_and_include(obj, prop_val)
				)
				if prop_include is not None:
					project_include.add(prop_include)
		case "enum":
			class_type = "enum class"
		case _:
			class_type = "error"

	write_schema(
		class_type, name, system_include, project_include, properties, file
	)


def create_folder(dest_dir: Path) -> Path:
	basename = "schemas"
	num = -1

	if not dest_dir.exists():
		raise Exception("folder is not found")
	if not dest_dir.is_dir():
		raise Exception("must be a folder")

	while True:
		path = dest_dir / (basename + (str(num) if num != -1 else ""))
		if path.exists():
			num += 1
			continue
		path.mkdir()
		return path


def main():
	parser = ArgumentParser()
	parser.add_argument("file")
	args = parser.parse_args()
	if not path.exists(args.file):
		print(f"File '{args.file}' not found", file=stderr)
		exit(1)
	with open(args.file) as file:
		oapi = json.load(file)
	schemas = parse_schemas(oapi)
	folder = create_folder(Path("./"))
	for name, schema in schemas.items():
		if not (schema.type == "object" or schema.type == "enum"):
			continue
		with open(folder / (name + ".hxx"), "w") as f:
			dump_schema(name, schema, f)


if __name__ == "__main__":
	main()
