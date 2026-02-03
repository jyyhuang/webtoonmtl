import click


@click.group()
@click.version_option("0.1.0", prog_name="webtoonmtl")
def cli():
    """Machine Translations for Korean webtoons to English."""


@cli.command()
def start():
    """Starts the Webtoon MTL GUI application."""
    from webtoonmtl.ui import run_gui

    run_gui()
