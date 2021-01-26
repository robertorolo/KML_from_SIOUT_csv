import pip

def import_or_install(package):
    try:
        __import__(package)
    except ImportError:
        pip.main(['install', package])  

packages = ['pandas', 'numpy', 'pykml', 'matplotlib', 'geopandas', 'descartes', 'openpyxl', 'pretty_html_table']

for package in packages:
    import_or_install(package)
