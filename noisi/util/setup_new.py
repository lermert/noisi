import os
import io
import time
import json
from noisi import _ROOT

def setup_proj(args):

    project_name = args.project_name
    os.makedirs(os.path.join(project_name))

    with io.open(os.path.join(_ROOT, 'config', 'config.json'), 'r+') as fh:
        conf = json.loads(fh.read())

    conf['date_created'] = time.strftime("%Y.%m.%d")
    conf['project_name'] = project_name
    conf['project_path'] = os.path.abspath(project_name)

    with io.open(os.path.join(project_name, 'config.json'), 'w') as fh:
        cf = json.dumps(conf, sort_keys=False, indent=4,
                        separators=(",", ": "))
        fh.write(cf)
    print("Created project directory {}. Please edit config file and run \
setup_sourcegrid.".format(project_name))


def setup_source(args):
    source_model = args.source_model

    if os.path.exists(source_model):
        raise ValueError("Directory exists already.")

    if not os.path.exists('config.json'):
        raise FileNotFoundError('File config.json does not exist.\
 Run setup_project first.')

    os.makedirs(os.path.join(source_model, 'step_0'))
    os.mkdir(os.path.join(source_model, 'observed_correlations'))
    for d in ['adjt', 'grad', 'corr', 'kern']:
        os.mkdir(os.path.join(source_model, 'step_0', d))

    with io.open(os.path.join(_ROOT, 'config', 'source_config.json'),
                              'r') as fh:
        conf = json.loads(fh.read())
        conf['date_created'] = str(time.strftime("%Y.%m.%d"))
        conf['project_name'] = os.path.basename(os.getcwd())
        conf['project_path'] = os.getcwd()
        conf['source_name'] = source_model
        conf['source_path'] = os.path.abspath(source_model)

    with io.open(os.path.join(source_model, 'source_config.json'), 'w') as fh:
        cf = json.dumps(conf, sort_keys=True, indent=4, separators=(",", ": "))
        fh.write(cf)

    with io.open(os.path.join(_ROOT, 'config', 'measr_config.json'),
                              'r') as fh:
        conf = json.loads(fh.read())
        conf['date_created'] = str(time.strftime("%Y.%m.%d"))

    with io.open(os.path.join(source_model, 'measr_config.json'), 'w') as fh:
        cf = json.dumps(conf, sort_keys=True, indent=4, separators=(",", ": "))
        fh.write(cf)

    os.system('cp {} {}'.format(os.path.join(_ROOT,
              'util/setup_noisesource.py'), source_model))
    os.system('cp {} {}'.format(os.path.join(_ROOT,
              'util/wavefield_from_instaseis.py'), source_model))
    print("Copied default source_config.json and measr_config.json \
to source model directory, please edit. \
Please run setup_noisesource.py after editing to \
create starting model.")