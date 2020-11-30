import copy
import errno
import json
import os
import re
from enum import EnumMeta
from itertools import groupby
from json.decoder import JSONDecodeError
from pathlib import Path
from string import Template
from typing import Any, Callable, Dict, List, MutableMapping

import click
import requests
from click import Choice
from click._compat import term_len
from click.formatting import iter_rows, measure_table, wrap_text
from toml import TomlDecodeError, load
from web3.gas_strategies.time_based import fast_gas_price_strategy

from raiden.constants import ServerListType
from raiden.exceptions import ConfigurationError, InvalidChecksummedAddress
from raiden.network.rpc.middleware import faster_gas_price_strategy
from raiden.utils.formatting import address_checksum_and_decode
from raiden_contracts.constants import CHAINNAME_TO_ID

LOG_CONFIG_CLI_NAME = "log-config"
LOG_CONFIG_OPTION_NAME = LOG_CONFIG_CLI_NAME.replace("-", "_")


class HelpFormatter(click.HelpFormatter):
    """
    Subclass that allows multiple (option) sections to be formatted with pre-determined
    widths.
    """

    def write_dl(self, rows, col_max=30, col_spacing=2, widths=None):
        """Writes a definition list into the buffer.  This is how options
        and commands are usually formatted.

        :param rows: a list of two item tuples for the terms and values.
        :param col_max: the maximum width of the first column.
        :param col_spacing: the number of spaces between the first and
                            second column.
        :param widths: optional pre-calculated line widths
        """
        rows = list(rows)
        if widths is None:
            widths = measure_table(rows)
        if len(widths) != 2:
            raise TypeError("Expected two columns for definition list")

        first_col = min(widths[0], col_max) + col_spacing

        for first, second in iter_rows(rows, len(widths)):
            self.write("%*s%s" % (self.current_indent, "", first))
            if not second:
                self.write("\n")
                continue
            if term_len(first) <= first_col - col_spacing:
                self.write(" " * (first_col - term_len(first)))
            else:
                self.write("\n")
                self.write(" " * (first_col + self.current_indent))

            text_width = max(self.width - first_col - 2, 10)
            lines = iter(wrap_text(second, text_width).splitlines())
            if lines:
                self.write(next(lines) + "\n")
                for line in lines:
                    self.write("%*s%s\n" % (first_col + self.current_indent, "", line))
            else:
                self.write("\n")


class Context(click.Context):
    def make_formatter(self):
        return HelpFormatter(width=self.terminal_width, max_width=self.max_content_width)


class CustomContextMixin:
    def make_context(self, info_name, args, parent=None, **extra):
        """
        This function when given an info name and arguments will kick
        off the parsing and create a new :class:`Context`.  It does not
        invoke the actual command callback though.

        :param info_name: the info name for this invokation.  Generally this
                          is the most descriptive name for the script or
                          command.  For the toplevel script it's usually
                          the name of the script, for commands below it it's
                          the name of the script.
        :param args: the arguments to parse as list of strings.
        :param parent: the parent context if available.
        :param extra: extra keyword arguments forwarded to the context
                      constructor.
        """
        for key, value in iter(self.context_settings.items()):  # type: ignore
            if key not in extra:
                extra[key] = value

        ctx = Context(self, info_name=info_name, parent=parent, **extra)  # type: ignore
        with ctx.scope(cleanup=False):
            self.parse_args(ctx, args)  # type: ignore
        return ctx


class ParseConfigFileMixin:
    def _args_merge_config(self, ctx, args):
        initial_args = copy.copy(args)
        path_options = {
            param.name for param in self.params if isinstance(param.type, (click.Path, click.File))  # type: ignore
        }
        # consume and parse args, fills ctx.params with parsed, internal representation
        self.parse_args(ctx, args)  # type: ignore

        file_path = ctx.params["config_file"]
        config_file = Path(file_path)
        config_file_values: MutableMapping[str, Any] = dict()
        try:
            with config_file.open() as config_file:
                config_file_values = load(config_file)
        except OSError as ex:
            if "--config-file" in initial_args or ex.errno != errno.ENOENT:
                raise ConfigurationError(f"Error opening config file: {ex}")
        except TomlDecodeError as ex:
            raise ConfigurationError(f"Error loading config file: {ex}")

        config_parameters = dict()
        log_config = dict()
        for config_name, config_value in config_file_values.items():
            # simply convert to CLI option
            if config_name == LOG_CONFIG_CLI_NAME:
                # Uppercase log level names
                for k, v in config_value.items():
                    log_config[k] = v.upper()
            else:
                if config_name in path_options:
                    config_value = os.path.expanduser(config_value)
                config_parameters[f"--{config_name}"] = config_value
        # overload log config string
        # therefore, we have to parse it first
        cli_log_dict = ctx.params[LOG_CONFIG_OPTION_NAME]
        if "--log-config" in initial_args:
            # let the original values overwrite the config file values
            log_config.update(cli_log_dict)
        else:
            cli_log_dict.update(log_config)
            log_config = cli_log_dict
        # HACK parse to string again
        log_str = ""
        for logger_name, logger_level in log_config.items():
            log_str += f"{logger_name}:{logger_level.upper()},"

        # delete the entries that are explicitly provided from the CLI
        for val in initial_args:
            if val in config_parameters:
                assert val.startswith("--")
                # remove overwrites
                del config_parameters[val]
        # since this is the merged dict already, always use this

        log_config_option_name = f"--{LOG_CONFIG_CLI_NAME}"
        # replace or append the merged log-config string
        try:
            i = initial_args.index(log_config_option_name)
            initial_args[i + 1] = log_str
        except ValueError:
            # only append when explicit config file was given!
            if "--config_file" in initial_args:
                initial_args += [log_config_option_name, log_str]

        # now append all config parameters that were not present in the cli
        for k, v in config_parameters.items():
            initial_args += [k, v]
        # FIXME we blindly insert args from the config to the CLI args,
        # which can be a problem when later on subcommands are handled!
        return initial_args

    def make_context(self, info_name, args, parent=None, **extra):
        """
        FIXME maybe this is not the best place to read and parse
        the config file
        """
        for key, value in iter(self.context_settings.items()):  # type: ignore
            if key not in extra:
                extra[key] = value

        initial_ctx = Context(self, info_name=info_name, parent=parent, **extra)  # type: ignore
        # TODO ML should we enforce the cleanup here?
        with initial_ctx.scope(cleanup=True):
            args_and_config_args = self._args_merge_config(initial_ctx, args)

        # Now make the initial ctx with the modified args
        return super().make_context(info_name, args_and_config_args, parent=parent, **extra)


class GroupableOption(click.Option):
    def __init__(
        self,
        param_decls=None,
        show_default=False,
        prompt=False,
        confirmation_prompt=False,
        hide_input=False,
        is_flag=None,
        flag_value=None,
        multiple=False,
        count=False,
        allow_from_autoenv=True,
        option_group=None,
        **attrs,
    ):
        super().__init__(
            param_decls,
            show_default,
            prompt,
            confirmation_prompt,
            hide_input,
            is_flag,
            flag_value,
            multiple,
            count,
            allow_from_autoenv,
            **attrs,
        )
        self.option_group = option_group


class GroupableOptionCommand(CustomContextMixin, click.Command):
    def format_options(self, ctx, formatter):
        def keyfunc(o):
            value = getattr(o, "option_group", None)
            return value if value is not None else ""

        grouped_options = groupby(sorted(self.get_params(ctx), key=keyfunc), key=keyfunc)

        options: Dict = {}
        for option_group, params in grouped_options:
            for param in params:
                rv = param.get_help_record(ctx)
                if rv is not None:
                    options.setdefault(option_group, []).append(rv)

        if options:
            widths_a, widths_b = list(
                zip(*[measure_table(group_options) for group_options in options.values()])
            )
            widths = (max(widths_a), max(widths_b))

            for option_group, group_options in options.items():
                with formatter.section(option_group if option_group else "Options"):
                    formatter.write_dl(group_options, widths=widths)


class GroupableOptionCommandGroup(CustomContextMixin, click.Group):
    def format_options(self, ctx, formatter):
        GroupableOptionCommand.format_options(self, ctx, formatter)  # type: ignore
        self.format_commands(ctx, formatter)

    def command(self, *args, **kwargs):
        return super().command(*args, **{"cls": GroupableOptionCommand, **kwargs})

    def group(self, *args, **kwargs):
        return super().group(*args, **{"cls": self.__class__, **kwargs})


class ParseConfigGroupableOptionCommandGroup(
    ParseConfigFileMixin, CustomContextMixin, click.Group
):
    def format_options(self, ctx, formatter):
        GroupableOptionCommand.format_options(self, ctx, formatter)  # type: ignore
        self.format_commands(ctx, formatter)

    def command(self, *args, **kwargs):
        return super().command(*args, **{"cls": GroupableOptionCommand, **kwargs})

    def group(self, *args, **kwargs):
        return super().group(*args, **{"cls": GroupableOptionCommandGroup, **kwargs})


def command(name=None, cls=GroupableOptionCommand, **attrs):
    return click.command(name, cls, **attrs)


def group(name=None, cls=GroupableOptionCommandGroup, **attrs):
    return click.group(name, **{"cls": cls, **attrs})  # type: ignore


def group_parse_config(name=None, **attrs):
    return click.group(name, **{"cls": ParseConfigGroupableOptionCommandGroup, **attrs})  # type: ignore


def option(*args, **kwargs):
    return click.option(*args, **{"cls": GroupableOption, **kwargs})  # type: ignore


def option_group(name: str, *options: Callable):
    def decorator(f):
        for option_ in reversed(options):
            for closure_cell in option_.__closure__:  # type: ignore
                if isinstance(closure_cell.cell_contents, dict):
                    closure_cell.cell_contents["option_group"] = name
                    break
            option_(f)
        return f

    return decorator


class AddressType(click.ParamType):
    name = "address"

    def convert(self, value, param, ctx):  # pylint: disable=unused-argument
        try:
            return address_checksum_and_decode(value)
        except InvalidChecksummedAddress as e:
            self.fail(str(e))


class LogLevelConfigType(click.ParamType):
    name = "log-config"
    _validate_re = re.compile(
        r"^(?:"
        r"(?P<logger_name>[a-zA-Z0-9._]+)?"
        r":"
        r"(?P<logger_level>debug|info|warn(?:ing)?|error|critical|fatal)"
        r",?)*$",
        re.IGNORECASE,
    )

    def convert(self, value, param, ctx):  # pylint: disable=unused-argument
        if not self._validate_re.match(value):
            self.fail("Invalid log config format")
        level_config = dict()
        if value.strip(" ") == "":
            return None  # default value

        if value.endswith(","):
            value = value[:-1]
        for logger_config in value.split(","):
            logger_name, logger_level = logger_config.split(":")
            level_config[logger_name] = logger_level.upper()
        return level_config


class ChainChoiceType(click.Choice):
    def convert(self, value, param, ctx):
        if isinstance(value, int):
            return value
        elif isinstance(value, str) and value.isnumeric():
            try:
                return int(value)
            except ValueError:
                self.fail(f"invalid numeric network id: {value}", param, ctx)
        else:
            network_name = super().convert(value, param, ctx)
            return CHAINNAME_TO_ID[network_name]


class EnumChoiceType(Choice):
    def __init__(self, enum_type: EnumMeta, case_sensitive=True):
        self._enum_type = enum_type
        # https://github.com/python/typeshed/issues/2942
        super().__init__(
            [choice.value for choice in enum_type], case_sensitive=case_sensitive  # type: ignore
        )

    def convert(self, value, param, ctx):
        try:
            return self._enum_type(value)
        except ValueError:
            self.fail(f"'{value}' is not a valid {self._enum_type.__name__.lower()}", param, ctx)


class GasPriceChoiceType(click.Choice):
    """ Returns a GasPriceStrategy for the choice """

    def convert(self, value, param, ctx):
        if isinstance(value, str) and value.isnumeric():
            try:
                gas_price = int(value)

                def fixed_gas_price_strategy(_web3, _transaction_params):
                    return gas_price

                return fixed_gas_price_strategy
            except ValueError:
                self.fail(f"invalid numeric gas price: {value}", param, ctx)
        else:
            gas_price_string = super().convert(value, param, ctx)
            if gas_price_string == "fast":
                return faster_gas_price_strategy
            else:
                return fast_gas_price_strategy


class MatrixServerType(click.Choice):
    def convert(self, value, param, ctx):
        if value.startswith("http"):
            return value
        return super().convert(value, param, ctx)


class HypenTemplate(Template):
    idpattern = r"(?-i:[_a-zA-Z-][_a-zA-Z0-9-]*)"


class PathRelativePath(click.Path):
    """
    `click.Path` subclass that can default to a value depending on
    another option of type `click.Path`.

    Uses :ref:`string.Template` to expand the parameters default value.

    Example::

        @click.option('--some-dir', type=click.Path())
        @click.option('--some-file', type=PathRelativePath(), default='${some-dir}/file.txt')
    """

    def convert(self, value, param, ctx):
        if value == param.default:
            try:
                value = self.expand_default(value, ctx.params)
            except KeyError as ex:
                raise RuntimeError(
                    "Subsitution parameter not found in context. "
                    "Make sure it's defined with `is_eager=True`."  # noqa: C812
                ) from ex

        return super().convert(value, param, ctx)

    @staticmethod
    def expand_default(default, params):
        return HypenTemplate(default).substitute(params)


def get_matrix_servers(
    url: str, server_list_type: ServerListType = ServerListType.ACTIVE_SERVERS
) -> List[str]:
    """Fetch a list of matrix servers from a URL

    The URL is expected to point to a JSON document of the following format::

        {
            "active_servers": [
                "url1",
                "url2",
                ...
            ],
            "all_servers": [
                "url1",
                "url2",
                ...
            ]
        }

    Which of the two lists is returned is controlled by the ``server_list_type`` argument.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as ex:
        raise RuntimeError(f"Could not fetch matrix servers list: {url!r} => {ex!r}") from ex

    try:
        known_servers: Dict[str, List[str]] = json.loads(response.text)
        msg = f"Unexpected format of known server list at {url}"
        assert {type_.value for type_ in ServerListType} == known_servers.keys(), msg
        active_servers = known_servers[server_list_type.value]
    except (JSONDecodeError, AssertionError) as ex:
        raise RuntimeError(
            f"Could not process list of known matrix servers: {url!r} => {ex!r}"
        ) from ex
    return [
        f"https://{server}" if not server.startswith("http") else server
        for server in active_servers
    ]


ADDRESS_TYPE = AddressType()
LOG_LEVEL_CONFIG_TYPE = LogLevelConfigType()
