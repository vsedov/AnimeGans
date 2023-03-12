from rich.traceback import install

from src.utils.core_helper import setup_constants, setup_extra_constants, setup_globals

install()

#  ╭────────────────────────────────────────────────────────────────────╮
#  │             Set up all global constants using hc class             │
#  │                                                                    │
#  ╰────────────────────────────────────────────────────────────────────╯
setup_constants()
setup_extra_constants()
setup_globals()
