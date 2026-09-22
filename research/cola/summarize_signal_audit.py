"""Aggregate saved audit scalars without treating crops as independent images."""
import argparse
import json
from pathlib import Path
import statistics


def describe(values):
    v = sorted(x for x in values if x is not None)
    if not v:
        return None
    return {'n':len(v),'min':min(v),'median':statistics.median(v),'max':max(v),
            'mean':statistics.mean(v)}


def summarize(root):
    summary=json.loads((root/'summary.json').read_text())
    rows=[json.loads(line) for line in (root/'samples.jsonl').read_text().splitlines()]
    result={'checkpoint_epoch':summary['epoch'],'samples':len(rows),'weights':summary['weights'],
            'blocks':{},'projections':{},'output_scores':{},'crop_metrics':summary['crop_metrics']}
    for block in (2,5,8,11):
        r={}
        branches=[v['patch'] for row in rows for k,v in row['branches'].items() if k.split('.')[0]==str(block)]
        for key in ('cross_intra_ratio','cross_base_ratio','cross_intra_cosine'):
            r[key]=describe(v[key] for v in branches)
        for key in ('relative_centered_rms','mean_element_population_variance'):
            r['phi_'+key]=describe(v['all'][key] for k,v in summary['phi'].items() if k.split('.')[0]==str(block))
        r['phi_different_image_cosine']=describe(v['all']['different_image_pair_cosine']['mean'] for k,v in summary['phi'].items() if k.split('.')[0]==str(block))
        scales=[row['scales'][str(block)] for row in rows]
        for key in ('delta_cosine','retention_vs_weighted_norm_sum','retention_vs_orthogonal_sum','fusion_identity_max_error'):
            r[key]=describe(v[key] for v in scales)
        for stage in ('raw_optical','raw_sar','optical','sar','fused'):
            r[stage]={key:describe(v[stage][key] for v in scales) for key in ('delta_rms','relative_delta')}
        result['blocks'][str(block)]=r
    for name, values in summary['phi'].items():
        branches=[row['branches'][name]['patch'] for row in rows]
        result['projections'][name]={k:describe(v[k] for v in branches) for k in ('cross_intra_ratio','cross_intra_cosine')}
        result['projections'][name]['phi']=values
    for key in rows[0]['output_scores']:
        result['output_scores'][key]=describe(row['output_scores'][key] for row in rows)
    (root/'aggregate.json').write_text(json.dumps(result,indent=2))
    return result


def plot(root, data):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    blocks=(2,5,8,11)
    projections=('q','k','v','o','up','down')
    fig,axes=plt.subplots(3,2,figsize=(13,10),constrained_layout=True)
    for m, stream in enumerate(('Optical target','SAR target')):
        for metric in range(3):
            values=[]
            for b in blocks:
                row=[]
                for p in projections:
                    v=data['projections'][f'{b}.{m}.{p}']
                    row.append(v['cross_intra_ratio']['median'] if metric==0 else
                               v['cross_intra_cosine']['median'] if metric==1 else
                               v['phi']['all']['relative_centered_rms']*100)
                values.append(row)
            values=np.array(values)
            ax=axes[metric,m]
            if metric==0:
                picture=ax.imshow(np.log10(values.clip(1e-6)),cmap='viridis',vmin=-.5,vmax=1.5)
            elif metric==1:
                picture=ax.imshow(values,cmap='RdBu_r',vmin=-1,vmax=1)
            else:
                picture=ax.imshow(values,cmap='viridis',vmin=0,vmax=100)
            for i in range(4):
                for j in range(6):
                    ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',fontsize=9,
                            bbox=dict(facecolor='white',alpha=.65,edgecolor='none',pad=1))
            ax.set_xticks(range(6),projections)
            ax.set_yticks(range(4),[str(b) for b in blocks])
            ax.set_ylabel('Block (zero-based)')
            titles=('Cross/intra RMS median (color: log10)','Cosine(cross,intra) median','Phi centered RMS / RMS (%)')
            ax.set_title(stream+'\n'+titles[metric])
            fig.colorbar(picture,ax=ax,shrink=.8)
    fig.suptitle(f'Checkpoint e{data["checkpoint_epoch"]}: 32 fixed crops from 16 source images; eval B=1')
    fig.savefig(root/'signal_heatmaps.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--plot',action='store_true')
    args=parser.parse_args()
    result=summarize(args.root)
    if args.plot:
        plot(args.root,result)
    print(json.dumps({k:v for k,v in result.items() if k!='projections'},indent=2))
