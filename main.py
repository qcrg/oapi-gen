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
	def _get_tag_target_type(self) -> str: ...

	def get_tag_target_type(self) -> str:
		tag = self._get_tag_target_type()
		if self.required:
			return tag
		return f"{CXX_OPTIONAL_TAG_TYPE}<{tag}>"

	def get_includes(self) -> Set[Include]:
		base_includes: Set[Include] = {ProjectInclude("helpers", "hxx")}
		if self.required:
			return base_includes
		return {
			*base_includes,
			SystemInclude("optional"),
		}


CXX_TAG_SUFFIX = "Tag"
CXX_OPTIONAL_TAG_TYPE = "OptionalTag"


@dataclass
class Enum(SchemaType):
	members: List[str]

	@override
	def to_cxx(self) -> str:
		lines: List[str] = [
			*self._make_cxx_enum_lines(),
			"",
			*self._make_cxx_helper_lines(),
			"",
			*self._make_cxx_from_json_lines(),
			"",
			*self._make_cxx_optional_from_json_lines(),
		]
		return "\n".join(lines)

	def _make_cxx_enum_lines(self) -> List[str]:
		return [
			f"enum class {self.get_schema_target_type()}",
			"{",
			",\n".join(["\t" + member for member in self.members]),
			"};",
		]

	def _make_cxx_helper_lines(self) -> List[str]:
		return [
			f"struct {self._get_tag_target_type()} {{}};",
		]

	def _make_cxx_from_json_lines(self) -> List[str]:
		def make_stmt(m):
			return f'\t case simple_hash("{m}"): return {self.get_schema_target_type()}::{m};'

		return [
			f"inline constexpr {self.get_schema_target_type()}",
			"from_json(",
			"\tconst nlohmann::json& json,",
			f"\t{self._get_tag_target_type()}",
			") {",
			"\tconst std::string& value = json;",
			"\tswitch (simple_hash(value)) {",
			*[make_stmt(m) for m in self.members],
			"\t}",
			"\tthrow value;",
			"}",
		]

	def _make_cxx_optional_from_json_lines(self) -> List[str]:
		return [
			f"inline constexpr std::optional<{self.get_schema_target_type()}>",
			"from_json(",
			"\tconst nlohmann::json& json,",
			f"\t{CXX_OPTIONAL_TAG_TYPE}<{self._get_tag_target_type()}>",
			") {",
			"\treturn json.is_string()",
			f"\t\t? std::optional{'{'}from_json(json, {self._get_tag_target_type()}{'{}'}){'}'}",
			"\t\t: std::nullopt;",
			"}",
		]

	@override
	def get_schema_target_type(self) -> str:
		return f"{self.name}"

	@override
	def _get_tag_target_type(self) -> str:
		return f"{self.get_schema_target_type()}{CXX_TAG_SUFFIX}"


@dataclass
class Obj(SchemaType):
	class_name: str
	properties: Dict[str, SchemaType]

	@override
	def to_cxx(self) -> str:
		lines: List[str] = [
			*self._make_cxx_class_lines(),
			"",
			*self._make_cxx_schema_tag(),
			"",
			*self._make_cxx_tag_helper(),
			"",
			*self._make_cxx_from_json_lines(),
			"",
			*self._make_cxx_optional_from_json_lines(),
			"",
			*self._make_cxx_class_from_json(),
		]
		return "\n".join(lines)

	def _make_cxx_class_lines(self) -> List[str]:
		return [
			f"struct {self.class_name}",
			"{",
			*[
				f"\t{field_type.to_cxx()}"
				for field_type in self.properties.values()
			],
			"",
			f"\tstatic {self.class_name} from_json(const nlohmann::json &json);",
			"};",
		]

	def _make_cxx_schema_tag(self) -> List[str]:
		return [
			f"struct {self._get_tag_target_type()} {{}};",
		]

	def _make_cxx_tag_helper(self) -> List[str]:
		return [
			"template<>",
			f"struct TagHelper<{self.get_schema_target_type()}>",
			"{",
			f"\tusing Tag = {self._get_tag_target_type()};",
			"};",
		]

	def _make_cxx_from_json_lines(self) -> List[str]:
		def from_json_param(name: str, prop: SchemaType):
			return f'\t\t.{to_cxx_name(name)} = from_json(json["{name}"], {prop._get_tag_target_type()}{"{}"})'

		return [
			f"inline constexpr {self.get_schema_target_type()}",
			"from_json(",
			"\tconst nlohmann::json& json,",
			f"\t{self._get_tag_target_type()}",
			") {",
			"\treturn {",
			",\n".join(
				[
					from_json_param(name, prop)
					for name, prop in self.properties.items()
				]
			),
			"\t};",
			"}",
		]

	def _make_cxx_optional_from_json_lines(self) -> List[str]:
		return [
			f"inline constexpr std::optional<{self.get_schema_target_type()}>",
			"from_json(",
			"\tconst nlohmann::json& json,",
			f"\t{CXX_OPTIONAL_TAG_TYPE}<{self._get_tag_target_type()}>",
			") {",
			"\treturn json.is_string()",
			f"\t\t? std::optional{'{'}from_json(json, {self._get_tag_target_type()}{'{}'}){'}'}",
			"\t\t: std::nullopt;",
			"}",
		]

	def _make_cxx_class_from_json(self) -> List[str]:
		cxx_type = self.get_schema_target_type()
		cxx_tag_type = self._get_tag_target_type()
		return [
			f"inline {cxx_type}",
			f"{cxx_type}::from_json(const nlohmann::json &json)",
			"{",
			f"\treturn ::from_json(json, {cxx_tag_type}{{}});",
			"}",
		]

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
	def _get_tag_target_type(self) -> str:
		return f"{self.get_schema_target_type()}{CXX_TAG_SUFFIX}"


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

	@abstractmethod
	def _get_primitive_lang_type(self) -> str: ...


@dataclass
class Ref(Primitive):
	ref_type: str

	@override
	def _get_primitive_lang_type(self) -> str:
		return self.ref_type

	@override
	def _get_tag_target_type(self) -> str:
		return self.ref_type + CXX_TAG_SUFFIX

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
	def _get_tag_target_type(self) -> str:
		return f"Vector{CXX_TAG_SUFFIX}<{self.children_type.get_schema_target_type()}>"

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
	def _get_tag_target_type(self) -> str:
		types = map(lambda m: m.get_schema_target_type(), self.member_types)
		return f"Vector{CXX_TAG_SUFFIX}<{', '.join(list(types))}>"

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


class PrimitiveHelper:
	def _get_tag_target_type(self) -> str:
		return self.__class__.__name__ + CXX_TAG_SUFFIX


@dataclass
class String(PrimitiveHelper, Primitive):
	@override
	def _get_primitive_lang_type(self) -> str:
		return "std::string"

	@override
	def get_includes(self) -> Set[Include]:
		return {*super().get_includes(), SystemInclude("string")}


@dataclass
class Number(PrimitiveHelper, Primitive):
	@override
	def _get_primitive_lang_type(self) -> str:
		return "double"


@dataclass
class Integer(PrimitiveHelper, Primitive):
	@override
	def _get_primitive_lang_type(self) -> str:
		return "long"


@dataclass
class Boolean(PrimitiveHelper, Primitive):
	@override
	def _get_primitive_lang_type(self) -> str:
		return "bool"


@dataclass
class AnySchemaType(PrimitiveHelper, Primitive):
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


def make_helpers_for_primitive(prim: Type) -> str:
	class_name = prim.__name__
	class_tag = class_name + CXX_TAG_SUFFIX
	cxx_type = prim("")._get_primitive_lang_type()
	same = ["{", "\treturn json;", "}", ""]
	return "\n".join(
		[
			f"struct {class_tag} {{}};",
			"",
			"template<>",
			f"struct TagHelper<{cxx_type}>",
			"{",
			f"\tusing Tag = {class_tag};",
			"};",
			"",
			f"inline {cxx_type} from_json(const nlohmann::json& json, {class_tag})",
			*same,
			f"inline std::optional<{cxx_type}> from_json(const nlohmann::json& json, {CXX_OPTIONAL_TAG_TYPE}<{class_tag}>)",
			*same,
		]
	)


HELPERS = f"""#pragma once

#include <nlohmann/json.hpp>

constexpr size_t simple_hash(const char *str)
{{
	size_t hash = 5381;
	while (*str) {{
		hash = ((hash << 5) + hash) + static_cast<uint8_t>(*str);
		str++;
	}}
	return hash;
}}

constexpr size_t simple_hash(const std::string &str)
{{
	return simple_hash(str.c_str());
}}

template<class>
struct {CXX_OPTIONAL_TAG_TYPE} {{}};

template<class>
struct TagHelper {{}};

{make_helpers_for_primitive(String)}
{make_helpers_for_primitive(Number)}
{make_helpers_for_primitive(Integer)}
{make_helpers_for_primitive(Boolean)}
{make_helpers_for_primitive(AnySchemaType)}
template<class>
struct Vector{CXX_TAG_SUFFIX} {{}};

template<class T>
struct TagHelper<std::vector<T>>
{{
	using Tag = VectorTag<T>;
}};

template<class T>
inline std::vector<T>
from_json(
	const nlohmann::json &json,
	Vector{CXX_TAG_SUFFIX}<T>
) {{
	std::vector<T> res;
	res.reserve(json.size());
	for (const nlohmann::json &elem : json) {{
		res.push_back(from_json(elem, typename TagHelper<T>::Tag{{}}));
	}}
	return res;
}}

template<class T>
inline std::optional<std::vector<T>>
from_json(
	const nlohmann::json &json,
	{CXX_OPTIONAL_TAG_TYPE}<Vector{CXX_TAG_SUFFIX}<T>>
) {{
	if (json.is_array())
		return from_json(json, Vector{CXX_TAG_SUFFIX}<T>{{}});
	return {{}};
}}

template<class...>
struct Tuple{CXX_TAG_SUFFIX} {{}};

template<class... Ts>
struct TagHelper<std::tuple<Ts...>>
{{
	using Tag = TupleTag<Ts...>;
}};

template<class... Ts, size_t... Is>
inline std::tuple<Ts...>
from_json_tuple_impl(
	const nlohmann::json& json,
	std::index_sequence<Is...>
) {{
	return std::tuple<Ts...>{{
		from_json(json[Is], typename TagHelper<Ts>::Tag{{}})...
	}};
}}

template<class... Ts>
inline std::tuple<Ts...>
from_json(
	const nlohmann::json &json,
	Tuple{CXX_TAG_SUFFIX}<Ts...>
) {{
	return from_json_tuple_impl(
		json,
		std::index_sequence_for<Ts...>{{}}
	);
}}

template<class... Ts>
inline std::optional<std::tuple<Ts...>>
from_json(
	const nlohmann::json &json,
	{CXX_OPTIONAL_TAG_TYPE}<Tuple{CXX_TAG_SUFFIX}<Ts...>>
) {{
	if (json.is_null())
		return {{}};
	return from_json(json, TupleTag<Ts...>{{}});
}}
"""


def create_helpers_file(folder: Path):
	with open(folder / "helpers.hxx", "w") as f:
		f.write(HELPERS)


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
