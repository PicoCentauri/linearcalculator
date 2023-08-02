#!/usr/bin/env python3
"""A frontend to calculator linear models with equisolve and rascaline."""

import argparse
import contextlib
import logging
import os
import pickle
from logging.handlers import TimedRotatingFileHandler

import tomli

from linearcalculator import __version__, compute_linear_models
from linearcalculator.utils import plot_realization


_log_dt_fmt = "%Y-%m-%d %H:%M:%S"
_log_fmt = "[{asctime}] [{levelname}] {name}: {message}"
logging.basicConfig(format=_log_fmt, datefmt=_log_dt_fmt, style="{", level=logging.INFO)

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def setup_logging(logfile=None, debug=False):
    try:
        logging.getLogger("discord").setLevel(logging.WARNING)
        logging.getLogger("discord.http").setLevel(logging.WARNING)

        log = logging.getLogger()
        log.setLevel(logging.INFO if not debug else logging.DEBUG)
        if logfile:
            handler = TimedRotatingFileHandler(
                filename=logfile,
                when="midnight",
                utc=True,
                encoding="utf-8",
                backupCount=5,
            )
            fmt = logging.Formatter(_log_fmt, _log_dt_fmt, style="{")
            handler.setFormatter(fmt)
            log.addHandler(handler)
        else:
            logger.warning("Logging to file is disabled")

        yield
    finally:
        handlers = log.handlers[:]
        for handler in handlers:
            handler.close()
            log.removeHandler(handler)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-v",
        "--debug",
        dest="debug",
        action="store_true",
        help="Set loglevel to debug",
    )
    parser.add_argument(
        "-l",
        "--logfile",
        dest="logfile",
        action="store",
        help="Logfile (optional)",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=__version__,
    )
    parser.add_argument(
        "-p",
        "--plot",
        dest="plot",
        action="store_true",
        help="Plot and save regulerizer curves. Only works if potential_exponents has "
        "only one entry.",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        dest="config_file",
        action="store",
        default="config.toml",
        help="Configuration file to run bot with",
    )
    args = parser.parse_args()

    with setup_logging(logfile=args.logfile, debug=args.debug):
        with open(os.path.realpath(args.config_file), "rb") as f:
            config = tomli.load(f)
            results = compute_linear_models(config)

        if args.plot:
            if (
                type(config["potential_exponents"]) == list
                and len(config["potential_exponents"]) > 1
            ):
                logger.error(
                    "Plots requested but more than one potential_exponent given."
                )
            else:
                for name, realization in results.items():
                    plot_realization(
                        realization, fname=f"{config['output']}/{name}.png"
                    )

        results.config = config

        # Save models to pickle file
        with open(os.path.join(config["output"], "results.pickle"), "wb") as f:
            pickle.dump(results, f)


if __name__ == "__main__":
    main()
