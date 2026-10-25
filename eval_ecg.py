"""Build the ECG task pool (results/ecg_pool.npz) and print the analytics/triage metrics.

Run this BEFORE run_experiments.py whenever quart/ecg.py changes."""
import json, os
import numpy as np
from quart.ecg import build_task_pool

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    r = build_task_pool('auto', out='results/ecg_pool.npz')
    m = r['metrics']
    d = np.load('results/ecg_pool.npz')
    t = d['test']
    print('source          :', r['source'])
    print('windows (all)   :', np.bincount(d['y'], minlength=4).tolist())
    print('windows (test)  :', np.bincount(d['y'][t], minlength=4).tolist())
    print('accuracy        : %.3f' % m['acc'])
    print('recall NSR/AF/PVC/VT:', [round(v, 3) for v in m['per_class_recall']])
    print('alarm sens/spec : %.3f / %.3f' % (m['sens_abnormal'], m['spec_abnormal']))
    print('triage sens / flag rate: %.3f / %.3f' % (m['triage_sens'], m['triage_flag_rate']))
    print('VT/run windows raising an alarm: %.3f' % m.get('alarm_any_vt', 0))
    if m.get('beat'):
        b = m['beat']
        print('beat level (held-out): V sens %.3f, V PPV %.3f, S sens %.3f, n=%d' % (b['V_sens'], b['V_ppv'], b['S_sens'], b['n_beats_test']))
    print('confusion matrix:', m['cm'])
    json.dump({'metrics': m, 'source': r['source']}, open('results/ecg_metrics.json', 'w'), indent=1)
