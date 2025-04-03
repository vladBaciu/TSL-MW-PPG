import os
import json
import logging
from datafile import DataFile
from formats.format import *
import dialogs
from pathlib import Path

PROJECT_CONFIG = "project.json"


# Interacts with single file configurations
class FilesData:
    def __init__(self, files):
        self.files_list = files
        self.config_list = []
        self.current_file = 0
        self.current_label = 0
        self.modified = False
        self.bad_files = []

        self.datafile = None
        self.config = None
        self.init()
        self.read()

    def init(self):
        for file in self.files_list:
            conf = file + ".json"
            file_conf = conf if os.path.exists(conf) else None
            self.config_list.append(file_conf)

    # noinspection PyTypeChecker
    def read(self):
        conf_path = self.config_list[self.current_file]
        if conf_path:
            self.read_conf()
            self.read_file()
        else:
            self.config = {
                "labels": ["Label"],
                "colors": ["#1f77b4"],
            }
            self.read_file()
            self.config["plot"] = [[i] for i in self.datafile.get_data_columns()]
            self.config["normalize"] = []
            self.config["functions"] = []

        if self.current_label >= len(self.config["labels"]):
            self.current_label = 0
        self.modified = False

    def read_conf(self):
        conf_path = self.config_list[self.current_file]
        try:
            with open(conf_path) as in_file:
                self.config = json.load(in_file)
        except IOError:
            logger.error("Unable to read {}: permission denied".format(conf_path))
            exit(2)

    def read_file(self):
        if set(self.files_list) == set(self.bad_files):
            dialogs.report_no_files()
            exit(3)

        current = self.files_list[self.current_file]
        if current in self.bad_files:
            self.next_file()
            self.read_file()
            return

        try:
            self.datafile = DataFile(current, self.config["labels"])
        except (UnrecognizedFormatError, BadFileError):
            self.datafile = None
            self.bad_files.append(current)
            dialogs.notify_read_error(os.path.basename(current))

            self.next_file()
            self.read_file()

    def save_file(self):
        self.datafile.save()

    def save_config(self):
        if self.modified:
            conf_path = self.files_list[self.current_file] + ".json"
            try:
                with open(conf_path, 'w') as out_file:
                    json.dump(self.config, out_file)
            except IOError:
                logger.error("Unable to write {}: permission denied".format(conf_path))
                exit(2)
            self.config_list[self.current_file] = conf_path
            self.modified = False

    def next_label(self):
        self.current_label = (self.current_label + 1) % len(self.config["labels"])

    def prev_label(self):
        length = len(self.config["labels"])
        self.current_label = (self.current_label - 1 + length) % length

    def get_current_label(self):
        label = self.config["labels"][self.current_label]
        color = self.config["colors"][self.current_label]
        return label, color

    def get_label_color(self, label):
        try:
            index = self.config["labels"].index(label)
        except:
            #if label not existent, get default color
            index = 0
        return self.config["colors"][index]

    def get_labels_info(self):
        return self.config["labels"], self.config["colors"]

    def set_labels_info(self, names, colors):
        self.config["labels"] = names
        self.config["colors"] = colors
        self.modified = True

        if self.current_label >= len(names):
            self.current_label = 0

    def get_plot_info(self):
        return self.config["plot"], self.config["normalize"]

    def set_plot_info(self, plot_set, normalize):
        self.config["plot"] = plot_set
        self.config["normalize"] = normalize
        self.modified = True

    def next_file(self):
        self.current_file = (self.current_file + 1) % len(self.files_list)

    def prev_file(self):
        self.current_file = (self.current_file - 1 + len(self.files_list)) % len(self.files_list)

    def get_functions(self):
        return self.config["functions"]

    def add_function(self, fs):
        self.datafile.add_function(fs)
        self.config["functions"].append(fs.name)
        self.modified = True

    def remove_function(self, index):
        f_name = self.config["functions"][index]

        header = self.datafile.get_data_header()
        f_col = self.datafile.get_data_columns()[header.index(f_name)]
        for plot in self.config["plot"]:
            if f_col in plot:
                plot.remove(f_col)
            for i, col in enumerate(plot):
                if col > f_col:
                    plot[i] = col - 1

        self.datafile.remove_function(f_name)
        del self.config["functions"][index]
        self.modified = True


# Interacts with project.json
class ProjectData:
    def __init__(self, project):
        self.folder = os.path.dirname(project)
        self.project_file = project
        self.current_file = 0
        self.current_label = 0
        self.modified = False
        self.bad_files = []

        self.datafile = None
        self.config = None
        self.read_conf()
        self.read_file()

    def read(self):
        if self.modified:
            self.read_conf()
            self.modified = False
        self.read_file()

        if self.current_label >= len(self.config["labels"]):
            self.current_label = 0

    def read_conf(self):
        try:
            with open(self.project_file) as in_file:
                self.config = json.load(in_file)
        except IOError:
            logger.error("Unable to read {}: permission denied".format(self.project_file))
            exit(2)

    def read_file(self):
        if set(self.config["files"]) == set(self.bad_files):
            dialogs.report_no_files()
            exit(3)

        current = self.config["files"][self.current_file]
        file_path = os.path.join(self.folder, current)
        file_path_tsl = file_path

        #create potential tls_generated file name
        file_path_tsl = Path(file_path)
        file_path_tsl = file_path_tsl.parent / 'tsl_generated' / file_path_tsl.name

        try:
            #try reading tsl_generated file if exists
            self.datafile = DataFile(file_path_tsl, self.config["labels"])
            self.datafile.filename = file_path
            self.insert_header()
        except:
            #if there is no tsl_generated file yet, read normal file type
            try:
                self.datafile = DataFile(file_path, self.config["labels"])
                self.insert_header()
            except (UnrecognizedFormatError, BadFileError):
                #thow error if format is not recognized
                self.datafile = None
                self.bad_files.append(current)
                dialogs.notify_read_error(current)
                self.next_file()
                self.read_file()

    def insert_header(self):
        header = self.datafile.get_data_header()
        if str(header) not in self.config.keys():
            self.config[str(header)] = {
                "plot": [[i] for i in self.datafile.get_data_columns()],
                "normalize": [],
                "functions": []
            }
            self.modified = True
            self.save_config()

    def save_file(self):
        self.datafile.save()

    def save_config(self):
        if self.modified:
            try:
                with open(self.project_file, 'w') as out_file:
                    json.dump(self.config, out_file)
            except IOError:
                logger.error("Unable to write {}: permission denied".format(self.project_file))
                exit(2)
            self.modified = False

    def next_label(self):
        self.current_label = (self.current_label + 1) % len(self.config["labels"])

    def prev_label(self):
        length = len(self.config["labels"])
        self.current_label = (self.current_label - 1 + length) % length

    def get_current_label(self):
        label = self.config["labels"][self.current_label]
        color = self.config["colors"][self.current_label]
        return label, color

    def get_label_color(self, label):
        try:
            index = self.config["labels"].index(label)
        except:
            #if label not existent, get default color
            index = 0
        return self.config["colors"][index]

    def get_labels_info(self):
        return self.config["labels"], self.config["colors"]

    def get_additional_options(self):
        return self.config["binary_class"], self.config["independent_channels"]

    def set_labels_info(self, names, colors):
        self.config["labels"] = names
        self.config["colors"] = colors
        self.modified = True

        if self.current_label >= len(names):
            self.current_label = 0

    def get_plot_info(self):
        header = self.datafile.get_data_header()
        conf = self.config[str(header)]
        return conf["plot"], conf["normalize"]

    def set_plot_info(self, plot_set, normalize):
        header = self.datafile.get_data_header()
        conf = self.config[str(header)]
        conf["plot"] = plot_set
        conf["normalize"] = normalize
        self.modified = True

    def next_file(self):
        self.current_file = (self.current_file + 1) % len(self.config["files"])

    def prev_file(self):
        self.current_file = (self.current_file - 1 + len(self.config["files"])) % len(self.config["files"])

    def get_functions(self):
        header = self.datafile.get_data_header()
        return self.config[str(header)]["functions"]

    def add_function(self, fs):
        header = self.datafile.get_data_header()
        new_header = header + [fs.name]
        self.config[str(new_header)] = self.config[str(header)]
        self.config[str(new_header)]["functions"].append(fs.name)

        self.datafile.add_function(fs)
        self.modified = True

    def remove_function(self, index):
        header = self.datafile.get_data_header()
        conf = self.config[str(header)]

        f_name = conf["functions"][index]
        f_col = self.datafile.get_data_columns()[header.index(f_name)]
        for plot in conf["plot"]:
            if f_col in plot:
                plot.remove(f_col)
            for i, col in enumerate(plot):
                if col > f_col:
                    plot[i] = col - 1

        self.datafile.remove_function(f_name)
        del conf["functions"][index]
        self.modified = True


class Config:
    def __init__(self):
        self.path = None
        self.config = None
        self.default = {"autosave": False, "plot_height": 1.06}
        self.init()

    def init(self):
        default_path = './config.json'
        alternative_path = os.path.expanduser('~/.config/tsl/config.json')

        if os.path.exists(default_path):
            conf = read_json(default_path)
            if conf is not None:
                self.path = default_path
                self.config = conf
                return
        elif write_json(self.default, default_path):
            self.path = default_path
            self.config = self.default
            return

        if os.path.exists(alternative_path):
            conf = read_json(alternative_path)
            if conf is not None:
                self.path = alternative_path
                self.config = conf
                return
        else:
            if not os.path.isdir(os.path.dirname(alternative_path)):
                os.makedirs(os.path.dirname(alternative_path))
            if write_json(self.default, alternative_path):
                self.path = alternative_path
                self.config = self.default
                return

        logger.error("Unable to read or write a configuration file")
        exit(2)

    def save(self):
        write_json(self.config, self.path)


def read_json(path):
    try:
        with open(path) as in_file:
            target_config = json.load(in_file)
            return target_config
    except IOError:
        logger.error("Unable to read {}: permission denied".format(path))
        return None


def write_json(data, path):
    try:
        with open(path, 'w') as out_file:
            json.dump(data, out_file)
            return True
    except IOError:
        logger.error("Unable to write {}: permission denied".format(path))
        return False


def start_session(files=None, project=None):
    global data_config
    if files:
        data_config = FilesData(files)
    elif project:
        data_config = ProjectData(project)


def get_files_list(folder):
    files_list = []
    format_list = get_all_formats()
    for file in os.listdir(folder):
        for form in format_list:
            if file.endswith(form):
                files_list.append(file)
                break
    return files_list


def init_project(folder, project_dict):
    project_path = os.path.join(folder, PROJECT_CONFIG)

    try:
        with open(project_path, 'w') as out_file:
            json.dump(project_dict, out_file)
    except IOError:
        logger.error("Unable to write {}: permission denied".format(project_path))
        exit(2)

    if os.path.exists(project_path):
        return project_path
    return None


def init_logger(path):
    log = logging.getLogger(__name__)
    handler = logging.FileHandler(path)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)
    return log


try:
    logger = init_logger('./tsl.log')
except IOError:
    ALT_LOG = os.path.expanduser('~/.config/tsl/tsl.log')
    if not os.path.isdir(os.path.dirname(ALT_LOG)):
        os.makedirs(os.path.dirname(ALT_LOG))
    logger = init_logger(ALT_LOG)

tsl_config = Config()
data_config = None


# CONFIGURATION WRAPPERS
# The tsl_config and data_config instances shouldn't be accessible from the outside.
# The functions below abstract this module's logic and make it easier to use it.
# Using this wrappers allows to safely improve the methods without having to look for all their
#  usages among all the other modules and modify them (which would be a really poor design choice).
# Also note that data_config class type is not known a priori.
#
# Example:
#   import config
#   config.tsl_config.config["plot_height"]             [BAD: assumption on the dict structure]
#   config.get_plot_height()                            [GOOD: just need to change the function below]

# APPLICATION CONFIG (getters, setters, saver)
def get_autosave():
    return tsl_config.config["autosave"]


def get_plot_height():
    return tsl_config.config["plot_height"]


def set_tsl_config(autosave=None, plot_height=None):
    if autosave is not None:
        tsl_config.config["autosave"] = autosave
    if plot_height is not None:
        tsl_config.config["plot_height"] = plot_height


def save_tsl_config():
    tsl_config.save()


# DATA CONFIG (getters, setters, modified, I/O operations)
def get_datafile():
    return data_config.datafile


def get_plot_info():
    return data_config.get_plot_info()


def get_labels_info():
    return data_config.get_labels_info()


def get_current_label():
    return data_config.get_current_label()


def get_label_color(label_name):
    return data_config.get_label_color(label_name)

def get_additional_options():
    try:
        additional_options = data_config.get_additional_options()
    except:
        #case when no project is available and only one file is selected. to be modified manually
        #[binary_label, independent_channel_labeling]
        additional_options = ["true", "true"]
    return additional_options
def get_functions():
    return data_config.get_functions()


def is_modified():
    return data_config.modified


def set_plot_info(plot_set, normalize):
    data_config.set_plot_info(plot_set, normalize)


def set_labels_info(names, colors):
    data_config.set_labels_info(names, colors)


def read_data_config():
    data_config.read()


def save_data_config():
    data_config.save_config()


def save_file():
    data_config.save_file()


def next_file():
    data_config.next_file()


def prev_file():
    data_config.prev_file()


def next_label():
    data_config.next_label()


def prev_label():
    data_config.prev_label()
