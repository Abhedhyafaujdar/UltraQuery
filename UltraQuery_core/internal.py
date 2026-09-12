import ctypes
import time
import os
import platform
import sys
try:
    from . import plotengine as ple
except ImportError:
    import UltraQuery_core.plotengine as ple

# Determine platform and correct DLL/SO path
engine_dir = os.path.join(os.path.dirname(__file__), "engine_lib")
engine_path = os.path.join(engine_dir, "engine.dll") if platform.system() == "Windows" else os.path.join(engine_dir, "engine.so")

if not os.path.exists(engine_path):
    print(f"❌ Critical: Engine file not found at {engine_path}")
    sys.exit(1)

clib = ctypes.CDLL(engine_path)

clib.readcsv.argtypes = [ctypes.c_char_p, ctypes.c_int]
clib.columnsget.argtypes = [ctypes.c_char_p]
clib.getdata.argtypes = [ctypes.c_char_p, ctypes.c_int]
clib.dataframe.argtypes = [ctypes.c_char_p, ctypes.c_int]
clib.readcsv.restype = None
clib.columnsget.restype = None
clib.getdata.restype = None
clib.dataframe.restype = None

class UltraQuery:
    def __init__(self):
        pass

    def viewfile(self, csv, limit=None):
        return self.viewdata(csv, limit)

    def viewcolumn(self, csv, limit=None):
        return clib.columnsget(csv.encode())

    def viewdata(self, csv, limit=None):
        return clib.getdata(csv.encode(), limit if limit else 100)
    
    def df(self,csv,limit=100):
        if not os.path.exists(csv):
            print(f"❌ File '{csv}' not found.")
            sys.exit(1)
        return clib.dataframe(csv.encode(),limit)

    def dataframe(self, csv, limit=None):
        return self.df(csv, limit if limit is not None else 100)
    

    def plot(self, file, xcol, ycol, graph_type):
        fun = ple.UltraQuery_plot(file, xcol, ycol)
        match graph_type:
            case "bar":
                fun._bar()
            case "pie":
                fun._pie()
            case "line":
                fun._line()
            case "histogram":
                fun._histogram()
            case "scatter":
                fun._scatter()
            case _:
                print("❌ Invalid plot type. Supported types: bar, pie, line, histogram, scatter.")

    def auto_plot(self, file, graph_type="bar"):
        return ple.auto_plot(file, graph_type)

    def comparison_plot(self, file, xcol=None, ycols=None):
        return ple.comparison_plot(file, xcol, ycols)

    def multi_dataset_plot(self, files, metric=None, output=None):
        return ple.multi_dataset_plot(files, metric, output)

    def smart_dashboard(self, file, scraped_pages=None, ai_summary=None, output=None):
        return ple.smart_dashboard(file, scraped_pages, ai_summary, output)
