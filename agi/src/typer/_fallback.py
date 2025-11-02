from __future__ import annotations

import inspect
import sys
from collections.abc import Sequence as ABCSequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union, get_args, get_origin, get_type_hints


__all__ = [
    "Argument",
    "Exit",
    "Option",
    "Typer",
    "colors",
    "echo",
    "secho",
    "USING_TYPER_FALLBACK",
]


class Exit(SystemExit):
    """Exception raised to abort command execution with a specific exit code."""

    def __init__(self, code: int = 0) -> None:
        super().__init__(code)
        self.code = code


class _Colors:
    RED = "red"
    GREEN = "green"
    YELLOW = "yellow"


colors = _Colors()


def echo(value: Any = "", **kwargs: Any) -> None:
    print(value, **kwargs)


def secho(value: Any = "", *, fg: str | None = None, err: bool = False, **kwargs: Any) -> None:
    stream = sys.stderr if err else sys.stdout
    print(value, file=stream, **kwargs)


@dataclass
class ArgumentInfo:
    default: Any
    exists: bool = False
    resolve_path: bool = False
    help: str | None = None


@dataclass
class OptionInfo:
    default: Any
    flags: Tuple[str, ...]
    help: str | None = None
    is_flag: bool = False
    resolve_path: bool = False
    exists: bool = False
    min: int | None = None
    max: int | None = None

    def __post_init__(self) -> None:
        expanded: List[str] = []
        for flag in self.flags:
            expanded.extend(part for part in flag.split("/") if part)
        self.flags = tuple(expanded)


def Argument(
    default: Any,
    *names: str,
    exists: bool = False,
    resolve_path: bool = False,
    help: str | None = None,
) -> ArgumentInfo:
    return ArgumentInfo(default=default, exists=exists, resolve_path=resolve_path, help=help)


def Option(
    default: Any,
    *flags: str,
    help: str | None = None,
    is_flag: bool | None = None,
    resolve_path: bool = False,
    exists: bool = False,
    min: int | None = None,
    max: int | None = None,
) -> OptionInfo:
    if is_flag is None and isinstance(default, bool):
        is_flag = True
    return OptionInfo(
        default=default,
        flags=tuple(flags) if flags else (),
        help=help,
        is_flag=bool(is_flag),
        resolve_path=resolve_path,
        exists=exists,
        min=min,
        max=max,
    )


@dataclass
class _CommandParam:
    name: str
    annotation: Any
    kind: str  # "argument" or "option"
    info: ArgumentInfo | OptionInfo | None
    default: Any
    required: bool


@dataclass
class _Command:
    func: Any
    params: List[_CommandParam]
    flag_map: Dict[str, _CommandParam]


class Typer:
    """Minimal Typer-compatible dispatcher used for tests."""

    def __init__(self, *, help: str | None = None) -> None:
        self.help = help or ""
        self._commands: Dict[str, _Command] = {}
        self._subcommands: Dict[str, "Typer"] = {}

    def command(self, name: str | None = None) -> CallableDecorator:
        def decorator(func: Any) -> Any:
            cmd_name = name or func.__name__.replace("_", "-")
            command = self._build_command(func)
            self._commands[cmd_name] = command
            return func

        return decorator

    def add_typer(self, app: "Typer", *, name: str) -> None:
        self._subcommands[name] = app

    def _execute(self, args: Sequence[str]) -> Tuple[int, Any]:
        command, remaining = self._resolve(list(args))
        params = self._parse_arguments(command, remaining)
        result = command.func(**params)
        return 0, result

    def _resolve(self, args: List[str]) -> Tuple["_Command", List[str]]:
        app: Typer = self
        tokens = list(args)
        while tokens and tokens[0] in app._subcommands:
            app = app._subcommands[tokens.pop(0)]
        if not tokens:
            raise Exit(code=1)
        command_name = tokens.pop(0)
        command = app._commands.get(command_name)
        if command is None:
            raise Exit(code=1)
        return command, tokens

    def _build_command(self, func: Any) -> _Command:
        signature = inspect.signature(func)
        params: List[_CommandParam] = []
        flag_map: Dict[str, _CommandParam] = {}
        type_hints = {}
        try:
            type_hints = get_type_hints(func)
        except Exception:
            type_hints = {}
        for param in signature.parameters.values():
            annotation = type_hints.get(param.name, param.annotation)
            if annotation is inspect._empty:
                annotation = Any
            default = param.default
            info: ArgumentInfo | OptionInfo | None = None
            kind = "argument"
            required = False
            default_value: Any

            if isinstance(default, ArgumentInfo):
                info = default
                default_value = default.default
                required = default_value is ...
            elif isinstance(default, OptionInfo):
                info = default
                kind = "option"
                default_value = default.default
                required = default_value is ...
            else:
                required = default is inspect._empty
                default_value = None if default is inspect._empty else default

            if default_value is ...:
                default_value = None

            cmd_param = _CommandParam(
                name=param.name,
                annotation=annotation,
                kind=kind,
                info=info,
                default=default_value,
                required=required,
            )
            params.append(cmd_param)
            if isinstance(info, OptionInfo):
                for flag in info.flags:
                    flag_map[flag] = cmd_param
        return _Command(func=func, params=params, flag_map=flag_map)

    def _parse_arguments(self, command: _Command, tokens: List[str]) -> Dict[str, Any]:
        option_values: Dict[str, Any] = {}
        positional: List[str] = []
        idx = 0
        while idx < len(tokens):
            token = tokens[idx]
            if token.startswith("-") and token != "-":
                flag, value = self._split_flag(token)
                idx += 1
                param = command.flag_map.get(flag)
                if param is None:
                    raise Exit(code=2)
                info = param.info if isinstance(param.info, OptionInfo) else None
                if info is not None and info.is_flag:
                    option_values[param.name] = not flag.lstrip("-").startswith("no-")
                    continue
                if value is None:
                    if idx >= len(tokens):
                        raise Exit(code=2)
                    value = tokens[idx]
                    idx += 1
                if self._allows_multiple(param.annotation):
                    option_values.setdefault(param.name, []).append(value)
                else:
                    option_values[param.name] = value
            else:
                positional.append(token)
                idx += 1

        result: Dict[str, Any] = {}
        pos_idx = 0
        for param in command.params:
            if param.kind == "argument":
                if pos_idx < len(positional):
                    raw_value = positional[pos_idx]
                    pos_idx += 1
                elif not param.required:
                    raw_value = param.default
                else:
                    raise Exit(code=2)
            else:
                if param.name in option_values:
                    raw_value = option_values[param.name]
                else:
                    raw_value = param.default
                    if raw_value is None and isinstance(param.info, OptionInfo) and param.info.is_flag:
                        raw_value = bool(param.default)
            result[param.name] = self._apply_type_conversion(raw_value, param.annotation, param.info)
        return result

    @staticmethod
    def _split_flag(token: str) -> Tuple[str, str | None]:
        if "=" in token:
            parts = token.split("=", 1)
            return parts[0], parts[1]
        return token, None

    @staticmethod
    def _allows_multiple(annotation: Any) -> bool:
        origin = get_origin(annotation)
        if origin in (list, List, Sequence, ABCSequence):
            return True
        if origin in (tuple, Tuple):
            return True
        if origin is Union:
            return any(Typer._allows_multiple(arg) for arg in get_args(annotation) if arg is not type(None))
        return False

    @staticmethod
    def _apply_type_conversion(value: Any, annotation: Any, info: ArgumentInfo | OptionInfo | None) -> Any:
        if value is None:
            return None
        converted = _coerce_value(value, annotation)
        if isinstance(info, (ArgumentInfo, OptionInfo)) and info.resolve_path:
            if isinstance(converted, list):
                converted = [Path(item).resolve() for item in converted]
            elif isinstance(converted, (str, Path)):
                converted = Path(converted).resolve()
        if isinstance(info, (ArgumentInfo, OptionInfo)) and info.exists:
            paths = converted if isinstance(converted, list) else [converted]
            for path in paths:
                if isinstance(path, Path) and not path.exists():
                    raise Exit(code=2)
        return converted


CallableDecorator = Any


def _coerce_value(value: Any, annotation: Any) -> Any:
    if annotation is Any or annotation is inspect._empty:
        return value
    origin = get_origin(annotation)
    if origin in (list, List, Sequence):
        (subtype,) = get_args(annotation) or (Any,)
        if isinstance(value, list):
            return [_coerce_value(item, subtype) for item in value]
        return [_coerce_value(value, subtype)]
    if origin in (tuple, Tuple):
        (subtype,) = get_args(annotation) or (Any,)
        if isinstance(value, tuple):
            return tuple(_coerce_value(item, subtype) for item in value)
        if isinstance(value, list):
            return tuple(_coerce_value(item, subtype) for item in value)
        return (_coerce_value(value, subtype),)
    if origin in (dict, Dict, Mapping, MutableMapping):
        return dict(value)
    if origin is Union:
        for option in get_args(annotation):
            if option is type(None):
                if value in (None, "None", "null"):
                    return None
                continue
            try:
                return _coerce_value(value, option)
            except Exception:
                continue
        return value
    if annotation is bool:
        if isinstance(value, str):
            lowered = value.lower()
            if lowered in {"true", "yes", "1"}:
                return True
            if lowered in {"false", "no", "0"}:
                return False
        return bool(value)
    if annotation is int:
        return int(value)
    if annotation is float:
        return float(value)
    if annotation is Path:
        return Path(value)
    return value

USING_TYPER_FALLBACK = True
