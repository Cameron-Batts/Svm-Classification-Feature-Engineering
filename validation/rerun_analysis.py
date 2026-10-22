"""Execute the original radial-classification notebook and capture fresh figures."""
import ast
import base64
import contextlib
import io
import json
import os
import platform
from datetime import date
from importlib.metadata import version
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

root = Path(__file__).resolve().parents[1]
os.chdir(root)
path = root/'Notebooks/Support Vector Machine Classification & Manual Feature Engineering.ipynb'
nb = json.loads(path.read_text())
scope = {}
count = 0
figure_count = 0
original_show = plt.show
outputs = []
def capture_figures(*args, **kwargs):
    global figure_count
    for number in plt.get_fignums():
        figure = plt.figure(number)
        buffer = io.BytesIO()
        figure.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        outputs.append({'output_type':'display_data','metadata':{},'data':{
            'image/png':base64.b64encode(buffer.getvalue()).decode(),
            'text/plain':[f'<Fresh figure {figure_count+1}>']}})
        figure_count += 1
        plt.close(figure)

plt.show = capture_figures
baseline = None
try:
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        count += 1
        print('Executing',cell['id'],flush=True)
        outputs = []
        stream = io.StringIO()
        tree = ast.parse(''.join(cell['source']))
        value = None
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                expression = tree.body.pop().value
                exec(compile(tree, cell['id'], 'exec'), scope)
                value = eval(compile(ast.Expression(expression), cell['id'], 'eval'), scope)
            else:
                exec(compile(tree, cell['id'], 'exec'), scope)
            capture_figures()
        if stream.getvalue():
            outputs.insert(0,{'output_type':'stream','name':'stdout','text':stream.getvalue().splitlines(True)})
        if value is not None and not hasattr(value,'fit') and not hasattr(value,'figure'):
            outputs.append({'output_type':'execute_result','execution_count':count,'metadata':{},'data':{'text/plain':[repr(value)]}})
        cell['execution_count'] = count
        cell['outputs'] = outputs
        if cell['id'] == 'project-006':
            baseline = scope['logistic_regression']
finally:
    plt.show = original_show

data = scope['data']
assert len(data) == 4000 and data.y.value_counts().to_dict() == {0:2000,1:2000}
assert np.allclose(data.x3,np.sqrt(data.x1**2+data.x2**2))
reports = {}
for name,model,columns in [
    ('Logistic regression: x1/x2',baseline,['x1','x2']),
    ('Tuned RBF SVM',scope['svc'],['x1','x2']),
    ('Logistic regression: radial feature',scope['logistic_regression'],['x1','x2','x3'])]:
    features = data[columns] if len(columns)==3 else data[columns].values
    pred = model.predict(features)
    reports[name] = {'accuracy':accuracy_score(data.y,pred),'precision':precision_score(data.y,pred),'recall':recall_score(data.y,pred)}
assert reports['Tuned RBF SVM']['accuracy'] == 1
assert reports['Logistic regression: radial feature']['accuracy'] == 1
search = scope['grid_search']
assert len(search.cv_results_['params']) == 20
assert np.isfinite(search.cv_results_['mean_test_score']).all()
assert figure_count == 6
record = {
    'date':date.today().isoformat(),'python':platform.python_version(),
    'versions':{p:version(p) for p in ['numpy','pandas','scikit-learn','scipy','matplotlib','seaborn','joblib','threadpoolctl']},
    'executed_code_cells':count,'generated_figures':figure_count,
    'observations':4000,'class_counts':{'0':2000,'1':2000},'random_seed':42,
    'plotting_helper':'Existing Utils.DA.py imported and used without modification.',
    'best_gamma':float(search.best_params_['svc__gamma']),'best_mean_cv_accuracy':float(search.best_score_),
    'cv_folds':5,'gamma_candidates':20,'models':reports,
    'limitations':'Model scores are fitting-data scores, not held-out performance. CV selected gamma only; no independent test set or noise-robustness analysis was added.'
}
(root/'Reports/rerun_metrics.json').write_text(json.dumps(record,indent=2)+'\n')
nb['metadata']['language_info']['version'] = platform.python_version()
path.write_text(json.dumps(nb,indent=1,ensure_ascii=False)+'\n')
print(json.dumps(record,indent=2))
