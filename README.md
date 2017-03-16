knossos_cuber
=============

A Python application that converts images into a **KNOSSOS**-readable format.

Prerequisites
-------------

**knossos_cuber** depends on the following Python packages:

*   `numpy`
*   `scipy`
*   `pillow`
*   `pyqt5` (only if you want to use the GUI)
*   `future` (only if you need to use Python 2 - in this case, use `pip2`/`python2` instead of `pip3`/`python3` in the following instructions)

These dependencies can be installed using pip:

    pip3 install numpy scipy pillow pyqt5

Note: If you are using a Python version older than 3.5, `pyqt5` has to be separately installed using your operating system's package manager (`pacman`, `apt-get`, `brew`, etc.).


Usage
-----

### Commandline ###


If you run `python3 knossos_cuber.py` without any arguments, you get the following output:

    usage: knossos_cuber.py [-h] [--format FORMAT] [--config CONFIG]
                            source_dir target_dir
    knossos_cuber.py: error: too few arguments

`knossos_cuber.py` expects at least 3 arguments:

*   `--format`/`-f`: The format of your image files. Currently, the options `png`, `tif` and `jpg` are supported.
*   `source_dir`: The path to the directory where your images are located in.
*   `target_dir`: Path to the output directory.

For example: `python3 knossos_cuber.py -f png input_dir output_dir`

If you run it like this, `knossos_cuber` will use sane default parameters for dataset generation. These parameters can be found in `config.ini`, and can be overridden by supplying an own configuration file using the `--config`/`-c` argument:

For example: `python3 knossos_cuber.py -f png -c my_conf.ini input_dir output_dir`

### GUI ###

For a GUI version of this program, run `python3 knossos_cuber_gui.py`. It accepts an additional argument, `--config`/`-c`, that should be the path to another configuration file.


Afterword
---------

**knossos_cuber** is part of **KNOSSOS**. Head to **KNOSSOS'** [main site](http://www.knossostool.org) or our [Github Repository](https://github.com/knossos-project/knossos) for more information.
