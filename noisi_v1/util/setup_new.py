import os
import io
import time
import yaml
import inspect


def setup_proj(args, comm, size, rank):

    project_name = args.project_name
    os.makedirs(os.path.join(project_name))

    noisi_path = os.path.dirname(inspect.stack()[1][1])

    with io.open(os.path.join(noisi_path,
                              'config', 'config.yml'), 'r+') as fh:

        conf = yaml.safe_load(fh)

    with io.open(os.path.join(noisi_path,
                              'config', 'config_comments.txt'), 'r+') as fh:
        comments = fh.read()

    with io.open(os.path.join(noisi_path,
                              'config', 'stationlist.csv'), 'r') as fh:
        stations = fh.read()

    conf['date_created'] = time.strftime("%Y.%m.%d")
    conf['project_name'] = project_name
    conf['project_path'] = os.path.abspath(project_name)

    with io.open(os.path.join(project_name, 'config.yml'), 'w') as fh:
        cf = yaml.dump(conf, sort_keys=False, indent=4)
        fh.write(cf)
        fh.write(comments)
    with io.open(os.path.join(project_name, 'stationlist.csv'), 'w') as fh:
        fh.write(stations)
    print("Created project directory {}.\nPlease edit config file and run \
setup_sourcegrid.".format(project_name))
