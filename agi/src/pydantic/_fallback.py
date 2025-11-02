from __future__ import annotations

import inspect
import json
from collections.abc import Iterable as ABCIterable, Mapping as ABCMapping, Sequence as ABCSequence
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple, Type, TypeVar, Union, get_args, get_origin, get_type_hints

__all__ = [
    "BaseModel",
    "ConfigDict",
    "Field",
    "ValidationError",
    "field_validator",
    "USING_PYDANTIC_FALLBACK",
]

_MISSING = object()
T = TypeVar("T", bound="BaseModel")

USING_PYDANTIC_FALLBACK = True


class ValidationError(Exception):
    """Lightweight drop-in replacement for pydantic's ValidationError."""

    def __init__(self, errors: Iterable[Mapping[str, Any]]) -> None:
        self.errors = list(errors)
        message = "Validation failed"
        if self.errors:
            first = self.errors[0]
            loc = " -> ".join(str(part) for part in first.get("loc", ()))
            detail = first.get("msg") or first.get("type") or ""
            if loc:
                message = f"{message} for {loc}: {detail}"
            else:
                message = f"{message}: {detail}"
        super().__init__(message)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        lines = ["ValidationError:"]
        for error in self.errors:
            loc = " -> ".join(str(part) for part in error.get("loc", ()))
            detail = error.get("msg") or error.get("type") or "invalid value"
            if loc:
                lines.append(f"  {loc}: {detail}")
            else:
                lines.append(f"  {detail}")
        return "\n".join(lines)


def ConfigDict(**kwargs: Any) -> Dict[str, Any]:
    """Compatibility helper that mirrors pydantic.ConfigDict."""

    return dict(kwargs)


class _FieldInfo:
    __slots__ = ("default", "default_factory", "alias", "metadata")

    def __init__(
        self,
        default: Any = _MISSING,
        *,
        default_factory: Callable[[], Any] | None = None,
        alias: str | None = None,
        **metadata: Any,
    ) -> None:
        self.default = default
        self.default_factory = default_factory
        self.alias = alias
        self.metadata = metadata


def Field(
    default: Any = _MISSING,
    *aliases: str,
    default_factory: Callable[[], Any] | None = None,
    alias: str | None = None,
    **metadata: Any,
) -> _FieldInfo:
    """Return field metadata describing defaults and aliases."""

    if aliases and alias is None:
        alias = aliases[0]
    return _FieldInfo(default, default_factory=default_factory, alias=alias, **metadata)


def field_validator(*field_names: str, mode: str = "after") -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that registers field-level validators on BaseModel subclasses."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        target = func
        if isinstance(func, (classmethod, staticmethod)):
            target = func.__func__
        setattr(target, "__field_validator_config__", {"fields": field_names, "mode": mode, "wrapper": func})
        return func

    return decorator


class _ModelField:
    __slots__ = ("name", "alias", "annotation", "default", "default_factory", "required", "metadata")

    def __init__(
        self,
        name: str,
        *,
        alias: str | None,
        annotation: Any,
        default: Any,
        default_factory: Callable[[], Any] | None,
        required: bool,
        metadata: Mapping[str, Any],
    ) -> None:
        self.name = name
        self.alias = alias
        self.annotation = annotation
        self.default = default
        self.default_factory = default_factory
        self.required = required
        self.metadata = dict(metadata)


class _Validator:
    __slots__ = ("mode", "func", "is_classmethod")

    def __init__(self, mode: str, func: Callable[..., Any], *, is_classmethod: bool) -> None:
        self.mode = mode
        self.func = func
        self.is_classmethod = is_classmethod

    def call(self, cls: Type["BaseModel"], value: Any) -> Any:
        if self.is_classmethod:
            return self.func.__get__(None, cls)(value)
        return self.func(value)


def _collect_validators(namespace: Mapping[str, Any], bases: Tuple[type, ...]) -> Dict[str, List[_Validator]]:
    validators: Dict[str, List[_Validator]] = {}
    for base in bases:
        base_validators = getattr(base, "__validators__", None)
        if base_validators:
            for field, items in base_validators.items():
                validators.setdefault(field, []).extend(items)

    for attr in namespace.values():
        func = attr
        is_classmethod = isinstance(attr, classmethod)
        if is_classmethod:
            func = attr.__func__
        config = getattr(func, "__field_validator_config__", None)
        if not config:
            continue
        mode = config.get("mode", "after")
        wrapper = config.get("wrapper", attr)
        for field in config.get("fields", ()):  # type: ignore[arg-type]
            validators.setdefault(field, []).append(_Validator(mode, wrapper, is_classmethod=is_classmethod))
    return validators


def _collect_fields(cls: Type["BaseModel"]) -> Dict[str, _ModelField]:
    fields: Dict[str, _ModelField] = {}
    try:
        annotations = get_type_hints(cls, include_extras=True)
    except Exception:
        annotations = {}
        for base in reversed(cls.__mro__):
            annotations.update(getattr(base, "__annotations__", {}))
    else:
        fallback: Dict[str, Any] = {}
        for base in reversed(cls.__mro__):
            fallback.update(getattr(base, "__annotations__", {}))
        for name, annotation in fallback.items():
            annotations.setdefault(name, annotation)

    namespace = {}
    for base in reversed(cls.__mro__):
        namespace.update(getattr(base, "__dict__", {}))

    for name, annotation in annotations.items():
        if name.startswith("_"):
            continue
        raw_default = namespace.get(name, _MISSING)
        if isinstance(raw_default, _FieldInfo):
            info = raw_default
            default = info.default
            default_factory = info.default_factory
            alias = info.alias
            metadata = info.metadata
        else:
            info = None
            default = raw_default
            default_factory = None
            alias = None
            metadata = {}
        required = default is _MISSING and default_factory is None
        fields[name] = _ModelField(
            name,
            alias=alias,
            annotation=annotation,
            default=default,
            default_factory=default_factory,
            required=required,
            metadata=metadata,
        )
    return fields


class BaseModelMeta(type):
    def __new__(mcls, name: str, bases: Tuple[type, ...], namespace: MutableMapping[str, Any]) -> type:
        validators = _collect_validators(namespace, bases)
        cls = super().__new__(mcls, name, bases, dict(namespace))
        cls.__validators__ = validators
        cls.__fields__ = _collect_fields(cls)
        return cls


class BaseModel(metaclass=BaseModelMeta):
    """Minimal subset of the pydantic BaseModel API used in tests."""

    __fields__: Dict[str, _ModelField]
    __validators__: Dict[str, List[_Validator]]

    def __init__(self, **values: Any) -> None:
        errors = []
        resolved: Dict[str, Any] = {}
        for name, field in self.__fields__.items():
            value = self._extract_value(name, field, values)
            before_validators = [
                validator for validator in self.__validators__.get(name, []) if validator.mode == "before"
            ]
            after_validators = [
                validator for validator in self.__validators__.get(name, []) if validator.mode != "before"
            ]
            try:
                for validator in before_validators:
                    value = validator.call(self.__class__, value)
                value = self._convert_value(value, field.annotation, field)
                for validator in after_validators:
                    value = validator.call(self.__class__, value)
            except ValidationError:
                raise
            except Exception as exc:  # pragma: no cover - defensive
                errors.append({"loc": (name,), "msg": str(exc)})
                continue
            resolved[name] = value
        if errors:
            raise ValidationError(errors)
        for key, value in resolved.items():
            object.__setattr__(self, key, value)

    def _extract_value(self, name: str, field: _ModelField, values: Mapping[str, Any]) -> Any:
        sentinel = object()
        value = values.get(name, sentinel)
        if value is sentinel and field.alias:
            value = values.get(field.alias, sentinel)
        if value is sentinel:
            if field.default is not _MISSING:
                return field.default
            if field.default_factory is not None:
                return field.default_factory()
            if field.required:
                raise ValidationError([{"loc": (name,), "msg": "field required"}])
            return None
        return value

    def _convert_value(self, value: Any, annotation: Any, field: _ModelField) -> Any:
        if value is None:
            return None
        origin = get_origin(annotation)
        if origin is Union:
            args = [arg for arg in get_args(annotation) if arg is not type(None)]
            if not args:
                return value
            for arg in args:
                try:
                    return self._convert_value(value, arg, field)
                except ValidationError:
                    raise
                except Exception:
                    continue
            return value
        if origin in (dict, Dict, Mapping, MutableMapping, ABCMapping):
            return dict(value)
        if origin in (list, List):
            subtype = get_args(annotation)[0] if get_args(annotation) else Any
            if isinstance(value, str):
                items = [value]
            else:
                items = list(value) if isinstance(value, Iterable) else [value]
            return [self._convert_value(item, subtype, field) for item in items]
        if origin in (tuple, Tuple):
            subtype = get_args(annotation)[0] if get_args(annotation) else Any
            if isinstance(value, str):
                items = [value]
            else:
                items = list(value) if isinstance(value, Iterable) else [value]
            return tuple(self._convert_value(item, subtype, field) for item in items)
        if origin in (Sequence, ABCSequence, Iterable, ABCIterable):
            subtype = get_args(annotation)[0] if get_args(annotation) else Any
            if isinstance(value, str):
                items = [value]
            else:
                items = list(value) if isinstance(value, Iterable) else [value]
            return [self._convert_value(item, subtype, field) for item in items]
        if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
            if isinstance(value, annotation):
                return value
            if isinstance(value, Mapping):
                return annotation(**value)
        if annotation in (str, int, float, bool):
            return self._coerce_primitive(value, annotation)
        return value

    @staticmethod
    def _coerce_primitive(value: Any, typ: type) -> Any:
        if isinstance(value, typ):
            return value
        if typ is bool:
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "yes", "1"}:
                    return True
                if lowered in {"false", "no", "0"}:
                    return False
            return bool(value)
        try:
            return typ(value)
        except Exception:  # pragma: no cover - fallback
            return value

    def model_dump(self, *, exclude_none: bool = False, by_alias: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        for name, field in self.__fields__.items():
            key = field.alias if by_alias and field.alias else field.alias or name
            value = getattr(self, name, None)
            if exclude_none and value is None:
                continue
            payload[key] = self._serialise(value, exclude_none=exclude_none, by_alias=by_alias)
        return payload

    def model_dump_json(self, *, indent: int | None = None, exclude_none: bool = False) -> str:
        return json.dumps(self.model_dump(exclude_none=exclude_none, by_alias=True), indent=indent)

    @classmethod
    def model_json_schema(cls) -> Dict[str, Any]:
        properties: Dict[str, Any] = {}
        required: List[str] = []
        for name, field in cls.__fields__.items():
            key = field.alias or name
            properties[key] = {"title": key}
            if field.required:
                required.append(key)
        schema: Dict[str, Any] = {
            "title": cls.__name__,
            "type": "object",
            "properties": properties,
        }
        if required:
            schema["required"] = required
        return schema

    @classmethod
    def model_validate(cls: Type[T], data: Mapping[str, Any]) -> T:
        return cls(**dict(data))

    @classmethod
    def model_validate_json(cls: Type[T], payload: str) -> T:
        data = json.loads(payload)
        if not isinstance(data, Mapping):
            raise ValidationError([{"loc": (), "msg": "JSON payload must be an object"}])
        return cls.model_validate(data)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        args = ", ".join(f"{name}={getattr(self, name)!r}" for name in self.__fields__)
        return f"{self.__class__.__name__}({args})"

    @staticmethod
    def _serialise(value: Any, *, exclude_none: bool, by_alias: bool) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump(exclude_none=exclude_none, by_alias=by_alias)
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, list):
            return [
                BaseModel._serialise(item, exclude_none=exclude_none, by_alias=by_alias)
                for item in value
            ]
        if isinstance(value, tuple):
            return [
                BaseModel._serialise(item, exclude_none=exclude_none, by_alias=by_alias)
                for item in value
            ]
        if isinstance(value, dict):
            return {
                key: BaseModel._serialise(item, exclude_none=exclude_none, by_alias=by_alias)
                for key, item in value.items()
            }
        return value


BaseModel.__fields__ = {}
BaseModel.__validators__ = {}
