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
	Set,
	Type,
	override,
)

import humps
import pydash
from pygments import formatters, highlight, lexers


class JSONEncoder(json.JSONEncoder):
	def default(self, o: Any) -> Any:
		if isinstance(o, SchemaType):
			return {
				"type": o.__class__,
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
	"string", "number", "integer", "boolean", "any", "array", "tuple", "ref"
]
ComplexSchemaTypes = Literal["object", "enum"]
SchemaTypes = PrimitiveSchemaTypes | ComplexSchemaTypes


IncludeTypes = Literal["system", "local"]
ExtTypes = Literal["h", "hpp", "hxx"]


@dataclass(frozen=True)
class Include:
	name: str
	ext: Optional[ExtTypes] = None
	_opener: str = field(init=False)
	_closer: str = field(init=False)

	def to_cxx(self) -> str:
		ext = "" if self.ext is None else "." + self.ext
		return f"#include {self._opener}{self.name}{ext}{self._closer}"


@dataclass(frozen=True)
class SystemInclude(Include):
	_opener: str = field(init=False, default="<")
	_closer: str = field(init=False, default=">")


@dataclass(frozen=True)
class ProjectInclude(Include):
	_opener: str = field(init=False, default='"')
	_closer: str = field(init=False, default='"')

	def __post_init__(self):
		object.__setattr__(self, "name", humps.depascalize(self.name))


@dataclass
class SchemaType:
	name: str
	required: bool = field(init=False, default=True)

	@abstractmethod
	def to_cxx(self) -> str: ...

	@abstractmethod
	def get_schema_target_type(self) -> str: ...

	@abstractmethod
	def get_tag_target_type(sefl) -> str: ...

	def get_includes(self) -> Set[Include]:
		if self.required:
			return set()
		return {SystemInclude("optional")}


TAG_SUFFIX = "Tag"
OPTIONAL_TAG_SUFFIX = "OptionalTag"


@dataclass
class Enum(SchemaType):
	members: List[str]

	@override
	def to_cxx(self) -> str:
		lines = [
			f"enum class {self.get_schema_target_type()}",
			"{",
			",\n".join(["\t" + member for member in self.members]),
			"};",
			"",
			f"struct {self.get_schema_target_type()}{TAG_SUFFIX} {{}};",
			f"struct {self.get_schema_target_type()}{OPTIONAL_TAG_SUFFIX} {{}};",
		]
		return "\n".join(lines)

	@override
	def get_schema_target_type(self) -> str:
		return f"{self.name}"

	@override
	def get_tag_target_type(self) -> str:
		if self.required:
			return f"{self.get_schema_target_type()}{TAG_SUFFIX}"
		return f"{self.get_schema_target_type()}{OPTIONAL_TAG_SUFFIX}"


@dataclass
class Obj(SchemaType):
	class_name: str
	properties: Dict[str, SchemaType]

	@override
	def to_cxx(self) -> str:
		return "\n".join(
			[
				f"struct {self.class_name}",
				"{",
				*[
					f"\t{field_type.to_cxx()}"
					for field_type in self.properties.values()
				],
				"};",
				"",
				f"struct {self.get_schema_target_type()}{TAG_SUFFIX} {{}};",
				f"struct {self.get_schema_target_type()}{OPTIONAL_TAG_SUFFIX} {{}};",
			]
		)

	@override
	def get_schema_target_type(self) -> str:
		return f"{self.class_name}"

	@override
	def get_includes(self) -> Set[Include]:
		includes = set()
		for prop in self.properties.values():
			includes = includes.union(prop.get_includes())
		return {
			SystemInclude("nlohmann/json", "hpp"),
			*super().get_includes(),
			*includes,
		}

	@override
	def get_tag_target_type(self) -> str:
		if self.required:
			return f"{self.get_schema_target_type()}{TAG_SUFFIX}"
		return f"{self.get_schema_target_type()}{OPTIONAL_TAG_SUFFIX}"


@dataclass
class Primitive(SchemaType):
	@override
	def to_cxx(self) -> str:
		return f"{self.get_schema_target_type()} {to_cxx_name(self.name)};"

	@override
	def get_schema_target_type(self) -> str:
		if self.required:
			return self._get_primitive_lang_type()
		return f"std::optional<{self._get_primitive_lang_type()}>"

	@override
	def get_includes(self) -> Set[Include]:
		return {*super().get_includes(), ProjectInclude("helpers", "hxx")}

	@abstractmethod
	def _get_primitive_lang_type(self) -> str: ...


@dataclass
class Ref(Primitive):
	ref_type: str

	@override
	def _get_primitive_lang_type(self) -> str:
		return self.ref_type

	@override
	def get_includes(self) -> Set[Include]:
		return {*super().get_includes(), ProjectInclude(self.ref_type, "hxx")}


@dataclass
class Array(Primitive):
	children_type: SchemaType

	@override
	def _get_primitive_lang_type(self) -> str:
		return f"std::vector<{self.children_type.get_schema_target_type()}>"

	@override
	def get_includes(self) -> Set[Include]:
		return {
			*super().get_includes(),
			SystemInclude("vector"),
			*self.children_type.get_includes(),
		}


@dataclass
class TupleType(Primitive):
	member_types: List[SchemaType]

	@override
	def _get_primitive_lang_type(self) -> str:
		types = ", ".join(
			[m.get_schema_target_type() for m in self.member_types]
		)
		return f"std::tuple<{types}>"

	@override
	def get_includes(self) -> Set[Include]:
		includes = set()
		for m in self.member_types:
			includes = includes.union(m.get_includes())
		return {
			*super().get_includes(),
			SystemInclude("tuple"),
			*includes,
		}


@dataclass
class String(Primitive):
	@override
	def _get_primitive_lang_type(self) -> str:
		return "std::string"

	@override
	def get_includes(self) -> Set[Include]:
		return {*super().get_includes(), SystemInclude("string")}


@dataclass
class Number(Primitive):
	@override
	def _get_primitive_lang_type(self) -> str:
		return "double"


@dataclass
class Integer(Primitive):
	@override
	def _get_primitive_lang_type(self) -> str:
		return "long"


@dataclass
class Boolean(Primitive):
	@override
	def _get_primitive_lang_type(self) -> str:
		return "bool"


@dataclass
class AnySchemaType(Primitive):
	@override
	def _get_primitive_lang_type(self) -> str:
		return "std::any"

	@override
	def get_includes(self) -> Set[Include]:
		return {*super().get_includes(), SystemInclude("any")}


def make_primitive_schema_type(
	field_name: str, primitive: PrimitiveSchemaTypes
) -> SchemaType:
	view: Dict[str, Type[Primitive]] = {
		"string": String,
		"number": Number,
		"integer": Integer,
		"boolean": Boolean,
		"any": AnySchemaType,
	}
	return view[primitive](field_name)


def parse_schema(
	root: Dict[str, Any], field_name: str, obj: Dict[str, Any]
) -> SchemaType:
	if "allOf" in obj:
		ref = base_name(obj["allOf"][0]["$ref"])
		return Ref(field_name, ref)
	if "$ref" in obj:
		ref = base_name(obj["$ref"])
		return Ref(field_name, ref)

	if "type" not in obj:
		return AnySchemaType(field_name)

	if "enum" in obj:
		return Enum(field_name, obj["enum"])

	match obj["type"]:
		case "array":
			if "prefixItems" in obj:
				member_types = []
				for member in obj["prefixItems"]:
					member_types.append(
						parse_schema(root, "/* logical error */", member)
					)
				return TupleType(field_name, member_types)
			else:
				children_types = parse_schema(
					root, "/* logical error */", obj["items"]
				)
				return Array(field_name, children_types)
		case "object":
			properties = {}
			required: List[str] = obj.get("required") or []
			if "properties" in obj:
				for prop_name, value in obj["properties"].items():
					properties[prop_name] = parse_schema(root, prop_name, value)
					properties[prop_name].required = prop_name in required
			return Obj(field_name, field_name, properties)

	return make_primitive_schema_type(field_name, obj["type"])


def parse_schemas(root):
	schemas: dict[str, SchemaType] = {}
	for name, schema in pydash.get(root, "components.schemas").items():
		if name in schemas:
			continue
		schemas[name] = parse_schema(root, name, schema)
	return schemas


def base_name(ref: str):
	return ref.split("/")[-1]


def to_cxx_name(name: str) -> str:
	reserved_names = ["operator"]
	if name in reserved_names:
		return name + "_"
	return name


def from_cxx_name(name: str) -> str:
	return name.rstrip("_")


def dump_schema(schema: SchemaType, file: IO):
	includes = sorted(list(schema.get_includes()), key=lambda i: i.to_cxx())
	file.write(
		"\n".join(
			[
				"#pragma once",
				"",
				*[sys_incl.to_cxx() for sys_incl in includes],
				"",
				schema.to_cxx(),
			]
		)
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


def create_helpers_file(folder: Path):
	def get_struct(name):
		return [
			f"struct {name}{TAG_SUFFIX} {{}};",
			f"struct {name}{OPTIONAL_TAG_SUFFIX} {{}};",
		]

	helpers = "\n".join(
		[
			"#pragma once",
			"",
			"#include <nlohmann/json.hpp>",
			"",
			*get_struct("String"),
			*get_struct("Number"),
			*get_struct("Integer"),
			*get_struct("Boolean"),
			*get_struct("AnySchema"),
			"template<class...>",
			get_struct("Tuple")[0],
			"template<class...>",
			get_struct("Tuple")[1],
		]
	)
	with open(folder / "helpers.hxx", "w") as f:
		f.write(helpers)


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
	create_helpers_file(folder)
	for name, schema in schemas.items():
		if isinstance(schema, Obj) or isinstance(schema, Enum):
			with open(folder / (humps.depascalize(name) + ".hxx"), "w") as f:
				dump_schema(schema, f)


if __name__ == "__main__":
	main()
