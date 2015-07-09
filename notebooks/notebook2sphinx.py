"""This script will look for all .ipynb files in source_dir and convert them on the same place to
.rst files.

It is used in order to integrate notebooks output to sphinx documentation.

Read it, use it, hack it and share it ! Or you can do better writing a sphinx extension :-)

:Url:
    https://gist.github.com/hadim/16e29b5848672e2e497c
:Author:
    HadiM <hadrien.mary@gmail.com>
:License:
    WTFPL
:Version:
    0.3 (2014.08.14):

Changelog
---------
- 0.3 (2014.08.14):
    - add --overwrite option to overwrite executed notebook to original path.
- 0.2.1 (2014.08.12):
    - Clear old output dirs
- 0.2 (2014.08.12):
    - add IPython 3 native notebook running with ExecutePreprocessor (with fallback to runipy)
- 0.1 (2014.08.11):
    - initial version
"""

import fnmatch
import shutil
import os
import argparse
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

from IPython.nbconvert import RSTExporter
from IPython.nbconvert.writers import FilesWriter
from IPython.nbformat import current as nbformat


def is_ipython_3():
    """
    """
    import IPython
    ipv = IPython.__version__
    ipv_major = int(ipv.split('.')[0])
    return ipv_major >= 3


def runipy_available():
    """Test if runipy is availabe
    """
    try:
        import runipy
    except ImportError:
        return False
    return True


def execute_notebook(nb, resources):
    """Execute notebook. With ExecutePreprocessor using IPython >= 3 or runipy instead.
    """

    if is_ipython_3():
        from IPython.nbconvert.preprocessors import ExecutePreprocessor
        nb, resources = ExecutePreprocessor().preprocess(nb, resources)
    elif runipy_available:
        from runipy.notebook_runner import NotebookRunner
        r = NotebookRunner(nb)
        r.run_notebook(skip_exceptions=True)
        nb = r.nb
    else:
        raise ImportError("Can't execute notebooks. Please install IPython >= 3 or runipy.")

    return nb

if __name__ == '__main__':

    desc = "Convert notebooks (.ipynb) to ready-to-build RsT (.rst) files with Sphinx."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--source_dir', type=str, nargs=1, default="",
                        help='Sphinx source directory')
    parser.add_argument('--execute', action='store_true',
                        help="Execute notebook before export (need 'runipy' or IPython >= 3)")
    parser.add_argument('--overwrite', action='store_true',
                        help="Overwrite notebook after it has been executed")
    parser.add_argument('--outputs_dir_suffix', type=str, nargs=1,
                        default="_notebook_output_files",
                        help='Suffix to output files directory (notebook images mainly)')

    args = parser.parse_args()
    execute = args.execute
    overwrite = args.overwrite

    if not args.source_dir:
#        source_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "source")
        source_dir = os.path.dirname(os.path.realpath(__file__))
    else:
        source_dir = args.source_dir

    for root, dirs, files in os.walk(source_dir):
        if ".ipynb_checkpoints" in root:
            continue

        nb_files = fnmatch.filter(files, '*.ipynb')
        for f in nb_files:

            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, source_dir)

            build_dir = os.path.dirname(full_path)
            rel_build_dir = os.path.relpath(build_dir, source_dir)

            resources = {}
            nb_name = os.path.splitext(os.path.basename(full_path))[0]
            nb_output_dirs = nb_name + args.outputs_dir_suffix[0]
            resources['output_files_dir'] = nb_output_dirs

            # Clear old output dir path
            if os.path.isdir(os.path.join(build_dir, nb_output_dirs)):
                shutil.rmtree(os.path.join(build_dir, nb_output_dirs))

            exporter = RSTExporter()

            nb = nbformat.reads_json(open(full_path).read())

            if execute:
                log.info("Execute notebook '{}'".format(rel_path))
                nb = execute_notebook(nb, resources)

                if overwrite and len(nbformat.validate(nb)) == 0:
                    with open(full_path, 'w') as f:
                        nbformat.write(nb, f, 'ipynb')
                elif overwrite and len(nbformat.validate(nb)) > 0:
                    log.error("Executed notebook is not a valid format. "
                              "Original notebook won't be overwritten.")

            log.info("Export notebook '{}'".format(rel_path))
            (output, resources) = exporter.from_notebook_node(nb, resources=resources)

            writer = FilesWriter()
            writer.build_directory = build_dir
            writer.write(output, resources, notebook_name=nb_name)
