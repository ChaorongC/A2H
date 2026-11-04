"""Reuse frozen GSE110821/PGP-UK definitions for GSE128734 by platform."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT = Path('.')
HIST = Path('.')
sys.path.insert(0, str(HIST))
from src.evaluate_pgpuk import file_hash, row_stats, scalar_metrics, reference_control_predictions
from src.run_pgpuk_disjoint_complete10 import make_groups
from src.tissue_only_transfer import epic_mapping

SOURCE = Path('external_validation')
NS = 'paired_validation_20261009'
MODEL = Path('legacy_model')
EPIC = Path('reference/EPIC.hg38.manifest.gencode.v36.tsv')
OUT = HIST/'results/gse128734_external_20261014/450K'
PYTHON = 'python'
TARGETS = ('CHALM', 'CAMDA')
CONTROL = HIST/'results/reference_controls_original'
TRANSFER = HIST/'results/tissue_only_r1_5_r1_6_20261008'
METHODS = ('A2H', 'm', 'm2', 'm3', 'reference_mean', 'reference_linear', 'universal_quadratic')


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, default=str)+'\n')
    temp.replace(path)


def record(path):
    return {'path': str(path), 'sha256': file_hash(path)}


def prepared(out):
    value = json.loads((out/'preparation.json').read_text())
    for info in value['frozen_inputs'].values():
        if file_hash(Path(info['path'])) != info['sha256']:
            raise ValueError(f'Frozen analysis input changed: {info["path"]}')
    return value


def source_titles(frame):
    frame = frame[['array_id','array_title']].drop_duplicates()
    if frame.array_id.duplicated().any():
        raise ValueError('Conflicting authoritative array titles')
    return frame


def select_pairs(paired, titles, runs, completed, platform):
    paired = paired.loc[paired.study.eq('gse128734') & paired.platform.eq(platform)].copy()
    paired = paired.rename(columns={'replicate':'curated_replicate'})
    paired = paired.merge(source_titles(titles),on='array_id',validate='one_to_one')
    replicate = paired.array_title.str.extract(r'_rep([12])$',expand=False)
    if platform=='450K' and replicate.isna().any():
        raise ValueError('Missing explicit 450K replicate label in source title')
    paired['replicate'] = replicate.fillna('1').astype(int)
    paired = paired.merge(runs[['wgbs_id','sample_id','run_accession']],on='wgbs_id',validate='many_to_one')
    if any(not title.startswith(sample+'_') for title,sample in zip(paired.array_title,paired.sample_id,strict=True)):
        raise ValueError('Array title/sample key mismatch')
    paired['primary'] = paired.replicate.eq(1) & paired.run_accession.isin(completed)
    primary = paired.loc[paired.primary].copy()
    if primary.wgbs_id.duplicated().any() or primary.array_id.duplicated().any():
        raise ValueError('Primary replicate is not unique per WGBS sample')
    excluded = paired.loc[~paired.run_accession.isin(completed)].copy()
    excluded['reason'] = 'WGBS full processing unavailable; Sample4 mate2 checksum unresolved'
    return paired,primary,excluded


def prepare(out,platform):
    if (out/'preparation.json').exists():
        raise FileExistsError('Analysis preparation already exists')
    out.mkdir(parents=True, exist_ok=True)
    meta = SOURCE/'metadata'/NS
    paired_source = pd.read_csv(meta/'paired_samples.tsv', sep='\t')
    runs = pd.read_csv(meta/'primary_wgbs_runs.tsv', sep='\t').query("study == 'gse128734'").copy()
    candidates = pd.read_csv(meta/'sources/candidate_pairing.tsv',sep='\t')
    titles = candidates.loc[candidates.cohort.eq('GSE128734'),['array_gsm','array_title']].rename(columns={'array_gsm':'array_id'})
    if len(runs)!=8 or runs.sample_id.duplicated().any():
        raise ValueError('Unexpected cohort keys')
    completed = set()
    for row in runs.itertuples(index=False):
        status = SOURCE/'logs'/NS/'processing'/f'{row.run_accession}.full.json'
        if status.exists():
            state = json.loads(status.read_text())
            if state['state']=='complete' and state['wgbs_id']==row.wgbs_id:
                completed.add(row.run_accession)
    if len(completed)!=7 or 'SRR9888319' in completed:
        raise ValueError('Completed cohort changed; review the frozen seven-sample branch')
    paired,primary,excluded = select_pairs(paired_source,titles,runs,completed,platform)
    if len(primary)!=(3 if platform=='450K' else 4):
        raise ValueError('Unexpected platform primary sample count')
    primary['participant_id'] = primary.sample_id
    primary['tissue_label'] = primary.tissue.str.removeprefix('tissue: ')
    excluded.to_csv(out/'excluded_pairs.tsv',sep='\t',index=False)
    paired.to_csv(out/'all_array_pairs.tsv',sep='\t',index=False)
    primary.to_csv(out/'primary_pairs.tsv',sep='\t',index=False)
    idat = pd.read_csv(meta/'array_idat_manifest.tsv',sep='\t')
    green = idat.loc[idat.study.eq('gse128734') & idat.platform.eq(platform) & idat.channel.eq('Grn')]
    names = green.local_path.map(lambda s: Path(s).name.removesuffix('_Grn.idat.gz'))
    rename = dict(zip(names,green.array_id,strict=True))
    beta_path = SOURCE/'results'/NS/f'array/gse128734_{platform}.beta.tsv'
    beta = pd.read_csv(beta_path,sep='\t',index_col=0).rename(columns=rename)
    if set(beta.columns)!=set(paired.array_id) or beta.index.has_duplicates:
        raise ValueError('Array IDAT-to-beta sample mapping mismatch')
    if not np.isfinite(beta.to_numpy()).all() or not ((beta>=0)&(beta<=1)).all().all():
        raise ValueError('Invalid array beta values')
    mapping = epic_mapping(pd.read_csv(EPIC,sep='\t',usecols=['probeID','CpG_chrm','CpG_beg']),
        pd.read_csv(MODEL/'A2H_cpg_probe.hg38.tsv',sep='\t',usecols=['loci.hg38','EPIC.hg38','HM450.hg38']),
        allow_conflicts=True)
    coef = pd.read_pickle(MODEL/'A2H_coef.pkl')
    info = pd.read_pickle(MODEL/'A2H_cpg_info.pkl')
    if coef.featureID.duplicated().any() or info.index.has_duplicates:
        raise ValueError('Ambiguous deployed locus coefficients/info')
    mapping['in_array'] = mapping.probe_id.isin(beta.index)
    mapping['in_450k'] = mapping.HM450_locus.notna()
    mapping['has_model'] = mapping.locus.isin(coef.featureID) & mapping.locus.isin(info.index)
    platform_gate = mapping.in_450k if platform=='450K' else True
    mapping['eligible'] = mapping.eligible_mapping & mapping.in_array & platform_gate & mapping.has_model
    mapping.to_csv(out/'mapping_audit.tsv',sep='\t',index=False)
    eligible = mapping.loc[mapping.eligible,['probe_id','locus']].sort_values('probe_id').reset_index(drop=True)
    if eligible.probe_id.duplicated().any() or len(eligible)<100:
        raise ValueError('Invalid or empty compatible mapping')
    eligible.to_csv(out/'mapping.tsv',sep='\t',index=False)
    beta = beta.loc[eligible.probe_id,paired.array_id].astype(np.float64)
    beta.index.name = 'Probe_ID'
    beta.to_csv(out/'array_beta.tsv',sep='\t')
    statuses = {}
    for row in primary.itertuples(index=False):
        path = SOURCE/'logs'/NS/'processing'/f'{row.run_accession}.full.json'
        s = json.loads(path.read_text())
        if s['state']!='complete' or s['wgbs_id']!=row.wgbs_id:
            raise ValueError('WGBS sample not fully completed')
        statuses[row.run_accession] = record(path)
    inputs = {'pairs':meta/'paired_samples.tsv','runs':meta/'primary_wgbs_runs.tsv',
              'idat':meta/'array_idat_manifest.tsv','beta':beta_path,
              'model_script':MODEL/'A2H.py','coefficients':MODEL/'A2H_coef.pkl',
              'model_info':MODEL/'A2H_cpg_info.pkl','probe_map':MODEL/'A2H_cpg_probe.hg38.tsv',
              'epic':EPIC,'control_provenance':CONTROL/'provenance.json',
              'transfer_preparation':TRANSFER/'preparation.json',
              'analysis_script':Path(__file__), 'legacy_metrics':HIST/'src/evaluate_pgpuk.py',
              'source_titles':meta/'sources/candidate_pairing.tsv',
              'source_workflow':PROJECT/'reviewer_revision_a2h/src/gse110821_external.py',
              'legacy_grouping':HIST/'src/run_pgpuk_disjoint_complete10.py',
              'legacy_mapping':HIST/'src/tissue_only_transfer.py'}
    for name in ('primary_pairs.tsv','mapping.tsv','array_beta.tsv','all_array_pairs.tsv','excluded_pairs.tsv'):
        inputs[name] = out/name
    pgp_provenance=json.loads((HIST/'results/pgpuk_depth10_calibrated/provenance.json').read_text())
    for target in TARGETS:
        p=CONTROL/f'{target}_controls.parquet'
        if file_hash(p)!=pgp_provenance['file_sha256'][f'control_{target}']:
            raise ValueError('Reference controls differ from the verified PGP-UK inputs')
        inputs[f'{target}_controls']=p
    save(out/'preparation.json', {'created_utc':utc(),'state':'prepared','frozen_inputs':{k:record(p) for k,p in inputs.items()},
         'completed_wgbs_states':statuses,'compatible_probes':len(eligible),
         'coordinate_conflicts_in_array':int((mapping.in_array & ~mapping.eligible_mapping).sum()),
         'platform':platform,'prefix':f'gse128734_{platform}',
         'primary_replicate':'source array title rep1; all arrays retained for QC; curated replicate labels preserved separately',
         'primary_samples':len(primary),'biological_units':'source sample labels; donor relationships not inferred',
         'training_depth':4,'external_primary_depth':10,
         'support_cutoff':.05,'grouping_complete_samples':len(primary),'group_size':10,'mean_block_size':100,
         'variance_ddof':0,'new_support_refilter':False,'no_external_training':True})
    print('Prepared',platform,len(eligible),'compatible probes,',len(primary),'primary samples,',len(paired),'arrays',flush=True)


def read_truth(path, mapping):
    loci = set(mapping.locus)
    parts=[]
    for chunk in pd.read_csv(path,sep='\t',usecols=['chr','pos','context','ratio','CT_count'],chunksize=300000):
        chunk = chunk.loc[chunk.context.eq('CG')].copy()
        chunk['locus'] = chunk.chr.astype(str)+'_'+chunk.pos.astype(str)
        keep = chunk.loc[chunk.locus.isin(loci),['locus','ratio','CT_count']]
        if len(keep):
            parts.append(keep)
    if not parts:
        raise ValueError('No matched WGBS CpGs')
    table = pd.concat(parts,ignore_index=True)
    if table.locus.duplicated().any() or not table.CT_count.ge(4).all():
        raise ValueError('Duplicate locus or incorrect historical depth gate')
    return table.set_index('locus')


def truth(out,index):
    prepared(out)
    pairs = pd.read_csv(out/'primary_pairs.tsv',sep='\t')
    if not 0<=index<len(pairs):
        raise ValueError('Invalid explicit sample task')
    row = pairs.iloc[index]
    dest = out/'truth'/f'{row.array_id}.parquet'
    if dest.exists():
        raise FileExistsError('Truth output already exists')
    mapping = pd.read_csv(out/'mapping.tsv',sep='\t')
    state = json.loads((SOURCE/'logs'/NS/'processing'/f'{row.run_accession}.full.json').read_text())
    tables,inputs = {},{}
    for key,stage in [('beta','beta'),('CHALM','chalm')]:
        a = state['steps'][stage]['artifacts'][0]
        path = Path(a['path'])
        if file_hash(path)!=a['sha256']:
            raise ValueError('Completed WGBS output changed')
        inputs[key] = a
        tables[key] = read_truth(path,mapping).reindex(mapping.locus)
    result = mapping.copy()
    for key in ('beta','CHALM'):
        result[key] = tables[key].ratio.to_numpy(dtype=np.float64)
        result[key+'_depth'] = tables[key].CT_count.to_numpy(dtype=np.float64)
    result['CAMDA'] = result.CHALM-result.beta
    if result.loc[result.CAMDA.notna(),'CAMDA'].lt(-.0011).any():
        raise ValueError('Direct CHALM-beta identity/range requires review')
    dest.parent.mkdir(exist_ok=True)
    result.to_parquet(dest,index=False)
    save(dest.with_suffix('.json'),{'state':'complete','run':row.run_accession,'array':row.array_id,
         'inputs':inputs,'output':record(dest),'finished_utc':utc(),
         'depth10_joint_sites':int((result.beta_depth.ge(10)&result.CHALM_depth.ge(10)).sum()),
         'CAMDA_definition':'aligned CHALM ratio minus traditional beta; both count thresholds required'})
    print('Truth complete',row.array_id,flush=True)


def transform(out,pilot=False,validate_existing=False):
    prep=prepared(out)
    prefix=prep['prefix']
    platform=prep['platform']
    root = out/('pilot_transform' if pilot else 'transformation')
    if not validate_existing:
        root.mkdir(exist_ok=False)
    path = out/'array_beta.tsv'
    if pilot:
        beta = pd.read_csv(path,sep='\t',index_col=0)
        idx = np.linspace(0,len(beta)-1,200,dtype=int)
        path = root/'input.tsv'
        beta.iloc[idx].to_csv(path,sep='\t')
    cpus = int(os.environ.get('SLURM_CPUS_PER_TASK','2'))
    available = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0,set(available[:cpus]))
    command = [PYTHON,str(MODEL/'A2H.py'),'-i',str(path),'-o',str(root),'--prefix',prefix,
               '--ref','hg38','--beadchip',('450k' if platform=='450K' else 'EPIC'),'-t',str(cpus),'--probe','--chalm','--camda',
               '--filter','0.05','--cpg-info',str(MODEL/'A2H_cpg_info.pkl'),'-c',str(MODEL/'A2H_coef.pkl'),'--verbose','0']
    if not validate_existing:
        subprocess.run(command,check=True)
    beta = pd.read_csv(path,sep='\t',index_col=0)
    mapping = pd.read_csv(out/'mapping.tsv',sep='\t').set_index('probe_id').loc[beta.index]
    coef = pd.read_pickle(MODEL/'A2H_coef.pkl').set_index('featureID')
    from scipy.stats import betabinom
    info = pd.read_pickle(MODEL/'A2H_cpg_info.pkl').loc[mapping.locus]
    platform_column = 'HM450.hg38' if platform=='450K' else 'EPIC.hg38'
    deployed = pd.read_csv(MODEL/'A2H_cpg_probe.hg38.tsv',sep='\t',usecols=['loci.hg38',platform_column]).dropna().set_index(platform_column).drop_duplicates()
    available = beta.index.isin(deployed.index)
    support_raw = pd.read_csv(root/f'{prefix}.confidence.tsv',sep='\t',index_col=0)
    if set(support_raw.index)!=set(beta.index[available]) or set(support_raw.columns)!=set(beta.columns):
        raise ValueError('Deployed output keys differ beyond documented duplicate-locus collapse')
    pd.DataFrame({'probe_id':beta.index[~available],
                  'reason':'deployed_CLI_probe_map_drop_duplicates_collapses_shared_locus'}).to_csv(root/'deployed_mapping_exclusions.tsv',sep='\t',index=False)
    support = support_raw.reindex(index=beta.index,columns=beta.columns)
    # Replicate deployed support precisely: floor beta*100, CDF, round 4 digits.
    expected = (1-2*np.abs(.5-betabinom.cdf((beta.to_numpy()*100).astype(int),100,
                   info.betabinom_a.to_numpy()[:,None],info.betabinom_b.to_numpy()[:,None]))).round(4)
    np.testing.assert_allclose(support.to_numpy()[available],expected[available],atol=1e-12,rtol=0,equal_nan=True)
    for target,column in [('CHALM','chalm_coef'),('CAMDA','camda_coef')]:
        raw = pd.read_csv(root/f'{prefix}.{target.lower()}.tsv',sep='\t',index_col=0)
        if set(raw.index)!=set(beta.index[available]): raise ValueError('Target output probe mismatch')
        observed = raw.reindex(index=beta.index,columns=beta.columns)
        positions=np.flatnonzero(available)
        indices = positions[np.linspace(0,len(positions)-1,min(200,len(positions)),dtype=int)]
        for i in indices:
            c = coef.loc[mapping.locus.iloc[i],column]
            prediction = np.full(len(beta.columns),c) if np.isscalar(c) else np.polyval(c,beta.iloc[i].to_numpy())
            np.testing.assert_allclose(observed.iloc[i],prediction,atol=1e-12,rtol=1e-12,equal_nan=True)
        filtered = pd.read_csv(root/f'{prefix}.{target.lower()}.filtered.tsv',sep='\t',index_col=0).reindex(index=beta.index,columns=beta.columns)
        np.testing.assert_allclose(filtered,observed.where(support>=.05),atol=1e-12,rtol=1e-12,equal_nan=True)
    outputs={p.name:record(p) for p in root.glob(f'{prefix}.*.tsv')}
    save(root/'completion.json',{'state':'complete','pilot':pilot,'command':command,'outputs':outputs,
         'support_reproduced':True,'polynomial_reproduced':True,'filter_reproduced':True,
         'deployed_unavailable_probes':beta.index[~available].tolist(),
         'validated_existing_outputs':validate_existing,'finished_utc':utc()})


def matrices(out):
    prefix=json.loads((out/'preparation.json').read_text())['prefix']
    pairs = pd.read_csv(out/'primary_pairs.tsv',sep='\t')
    mapping = pd.read_csv(out/'mapping.tsv',sep='\t')
    beta = pd.read_csv(out/'array_beta.tsv',sep='\t',index_col=0).loc[mapping.probe_id,pairs.array_id]
    roots = out/'transformation'
    completion=json.loads((roots/'completion.json').read_text())
    if completion['state']!='complete':
        raise ValueError('Transformation did not complete')
    for info in completion['outputs'].values():
        if file_hash(Path(info['path']))!=info['sha256']:
            raise ValueError('Transformation output changed')
    predictions={t:pd.read_csv(roots/f'{prefix}.{t.lower()}.filtered.tsv',sep='\t',index_col=0).reindex(index=mapping.probe_id,columns=pairs.array_id).to_numpy() for t in TARGETS}
    truth_arrays={k:[] for k in ('beta','CHALM','CAMDA','beta_depth','CHALM_depth')}
    for row in pairs.itertuples(index=False):
        p=out/'truth'/f'{row.array_id}.parquet'
        provenance=json.loads(p.with_suffix('.json').read_text())
        if file_hash(p)!=provenance['output']['sha256']:
            raise ValueError('Truth matrix changed')
        frame=pd.read_parquet(p)
        if not frame.probe_id.equals(mapping.probe_id) or not frame.locus.equals(mapping.locus):
            raise ValueError('Truth CpG mapping/order mismatch')
        for k in truth_arrays:
            truth_arrays[k].append(frame[k].to_numpy())
    return pairs,mapping,beta.to_numpy(),predictions,{k:np.stack(v,axis=1) for k,v in truth_arrays.items()}


def summarize(y,pred):
    result=scalar_metrics(y,pred)
    valid=np.isfinite(y)&np.isfinite(pred)
    result['mae']=float(np.mean(np.abs(y[valid]-pred[valid]))) if valid.any() else np.nan
    return result


def controls(mapping,beta,target):
    meta=json.loads((CONTROL/'provenance.json').read_text())
    f=CONTROL/f'{target}_controls.parquet'
    pgp=json.loads((HIST/'results/pgpuk_depth10_calibrated/provenance.json').read_text())
    if file_hash(f)!=pgp['file_sha256'][f'control_{target}']:
        raise ValueError('Reference-control hash mismatch')
    return reference_control_predictions(beta,mapping.locus,pd.read_parquet(f),
        meta[target]['universal_coefficients_intercept_linear_quadratic'])


def exclusion_predictions(mapping,beta,target,model):
    parts=[]
    provenance=[]
    for directory in sorted((TRANSFER/'fits'/model).glob('shard_*')):
        complete=json.loads((directory/'completion.json').read_text())
        info=complete['outputs'][f'{target}_coefficients']
        path=Path(info['path'])
        if file_hash(path)!=info['sha256']:
            raise ValueError('Frozen transfer coefficients changed')
        frame=pd.read_parquet(path)
        parts.append(frame.loc[frame.locus.isin(set(mapping.locus))])
        provenance.append(info)
    if len(parts)!=52:
        raise ValueError('Missing tissue-only coefficient shards')
    frame=pd.concat(parts,ignore_index=True)
    if frame.locus.duplicated().any():
        raise ValueError('Duplicate exclusion-model locus')
    frame=frame.set_index('locus').reindex(mapping.locus)
    prediction=frame.a2h_intercept.to_numpy()[:,None]+frame.a2h_linear.to_numpy()[:,None]*beta+frame.a2h_quadratic.to_numpy()[:,None]*beta**2
    return prediction,provenance


def figures(out,pairs,mapping,beta,preds,truth,group_metrics,site,methods):
    folder=out/'figures'
    folder.mkdir(exist_ok=True)
    for target in TARGETS:
        y=truth[target]
        gate=(truth['CHALM_depth']>=10)&np.isfinite(preds[target])
        if target=='CAMDA': gate &= truth['beta_depth']>=10
        for values in methods[target].values(): gate &= np.isfinite(values)
        nrows=(len(pairs)+2)//3
        fig,axes=plt.subplots(nrows,3,figsize=(12,3.5*nrows),squeeze=False)
        for ax in axes.flat[len(pairs):]: ax.set_visible(False)
        for j,(ax,row) in enumerate(zip(axes.flat[:len(pairs)],pairs.itertuples(index=False),strict=True)):
            mask=gate[:,j]&np.isfinite(y[:,j])
            ax.hexbin(y[mask,j],preds[target][mask,j],gridsize=50,mincnt=1,bins='log',cmap='viridis',rasterized=True)
            metrics=summarize(y[mask,j],preds[target][mask,j])
            ax.set_title(f'{row.sample_id}\nr={metrics["pearson_r"]:.3f}; RMSE={metrics["rmse"]:.3f}',fontsize=10)
            ax.set_xlabel(f'Observed {target}'); ax.set_ylabel(f'A2H {target}')
        fig.tight_layout()
        for ext in ('png','pdf'): fig.savefig(folder/f'{target}_per_sample.{ext}',dpi=180)
        plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,8))
    for j,target in enumerate(TARGETS):
        for i,metric in enumerate(('pearson_r','rmse')):
            ax=axes[i,j]
            for method,color in [('A2H','#1b6ca8'),('m2','#d95f02'),('m3','#2ca25f')]:
                v=group_metrics.loc[group_metrics.target.eq(target)&group_metrics.method.eq(method),metric].dropna().sort_values().to_numpy()
                if len(v): ax.plot(v,np.arange(1,len(v)+1)/len(v),label=method,color=color)
            ax.set(xlabel=metric,ylabel='Cumulative fraction',title=target); ax.legend()
    fig.tight_layout()
    for ext in ('png','pdf'): fig.savefig(folder/f'R1-4_group_distributions.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,target in zip(axes,TARGETS,strict=True):
        v=site.loc[site.target.eq(target)&site.method.eq('A2H'),'rmse'].dropna().sort_values().to_numpy()
        ax.plot(v,np.arange(1,len(v)+1)/len(v),color='#1b6ca8')
        ax.set(xlabel='CpG RMSE',ylabel='Cumulative fraction',title=target)
    fig.tight_layout()
    for ext in ('png','pdf'): fig.savefig(folder/f'R2-3_site_errors.{ext}',dpi=180)
    plt.close(fig)


def evaluate(out):
    prep=prepared(out)
    if (out/'analysis_completion.json').exists(): raise FileExistsError('Analysis already completed')
    pairs,mapping,beta,preds,truth=matrices(out)
    n_samples=len(pairs)
    beta_agreement=[]
    for depth in (4,10):
        measured=np.where(truth['beta_depth']>=depth,truth['beta'],np.nan)
        beta_agreement.append({'depth':depth,'scope':'pooled','sample_id':'all',**summarize(measured,beta)})
        for j,row in enumerate(pairs.itertuples(index=False)):
            beta_agreement.append({'depth':depth,'scope':'sample','sample_id':row.sample_id,
                                   'tissue_label':row.tissue_label,**summarize(measured[:,j],beta[:,j])})
    pd.DataFrame(beta_agreement).to_csv(out/'beta_array_wgbs_agreement.tsv',sep='\t',index=False)
    metrics,sites,flow,transfer_rows=[],[],[],[]
    arrays={}
    common_complete=np.ones(len(mapping),dtype=bool)
    input_records={}
    for target in TARGETS:
        methods={'A2H':preds[target],'m':beta,'m2':beta**2,'m3':beta**3,**controls(mapping,beta,target)}
        mask=np.isfinite(truth[target])&np.isfinite(beta)
        for values in methods.values(): mask &= np.isfinite(values)
        arrays[target]=methods
        for depth in (4,10):
            gate=mask&(truth['CHALM_depth']>=depth)
            if target=='CAMDA': gate &= truth['beta_depth']>=depth
            y=np.where(gate,truth[target],np.nan)
            if depth==10: common_complete &= gate.all(axis=1)
            flow.append({'target':target,'depth':depth,'probes':len(mapping),'observations':int(gate.sum()),
                         'probes_any':int(gate.any(axis=1).sum()),'probes_complete_samples':int(gate.all(axis=1).sum())})
            for method,values in methods.items():
                prediction=np.where(gate,values,np.nan)
                metrics.append({'target':target,'depth':depth,'method':method,'scope':'pooled','sample_id':'all',**summarize(y,prediction)})
                for j,row in enumerate(pairs.itertuples(index=False)):
                    metrics.append({'target':target,'depth':depth,'method':method,'scope':'sample','sample_id':row.sample_id,
                                    'participant_id':row.participant_id,'tissue_label':row.tissue_label,**summarize(y[:,j],prediction[:,j])})
                for tissue in pairs.tissue_label.unique():
                    select=pairs.tissue_label.eq(tissue).to_numpy()
                    metrics.append({'target':target,'depth':depth,'method':method,'scope':'tissue','sample_id':tissue,
                                    'tissue_label':tissue,**summarize(y[:,select],prediction[:,select])})
                stats=row_stats(y,prediction)
                sites.append(pd.DataFrame({'probe_id':mapping.probe_id,'locus':mapping.locus,'target':target,
                                          'method':method,'depth':depth,'n':stats['n'],'rmse':stats['rmse'],'bias':stats['bias']}))
            if depth==10:
                idx,col=np.where(gate)
                obs=pd.DataFrame({'probe_id':mapping.probe_id.to_numpy()[idx],'locus':mapping.locus.to_numpy()[idx],
                                  'sample_id':pairs.array_id.to_numpy()[col],'biological_sample_id':pairs.sample_id.to_numpy()[col],
                                  'observed':truth[target][idx,col],'m':beta[idx,col],
                                  'WGBS_beta':truth['beta'][idx,col],**{k:v[idx,col] for k,v in methods.items()}})
                obs.to_parquet(out/f'{target}_observations.parquet',index=False)
        exclusions={}
        for model in ('exclude_blood_immune','exclude_epithelial'):
            values,records=exclusion_predictions(mapping,beta,target,model)
            exclusions[model]=values
            input_records[f'{target}_{model}']=records
        transfer_gate=np.isfinite(preds[target])&np.isfinite(truth[target])&(truth['CHALM_depth']>=10)
        if target=='CAMDA': transfer_gate &= truth['beta_depth']>=10
        for values in exclusions.values(): transfer_gate &= np.isfinite(values)
        models={'full_reference':preds[target],**exclusions}
        ti,tj=np.where(transfer_gate)
        pd.DataFrame({'probe_id':mapping.probe_id.to_numpy()[ti], 'sample_id':pairs.array_id.to_numpy()[tj],
                      'observed':truth[target][ti,tj],**{k:v[ti,tj] for k,v in models.items()}}).to_parquet(out/f'{target}_transfer_predictions.parquet',index=False)
        for model,values in models.items():
            y=np.where(transfer_gate,truth[target],np.nan)
            prediction=np.where(transfer_gate,values,np.nan)
            transfer_rows.append({'target':target,'model':model,'scope':'pooled','sample_id':'all',**summarize(y,prediction)})
            for tissue in pairs.tissue_label.unique():
                select=pairs.tissue_label.eq(tissue).to_numpy()
                transfer_rows.append({'target':target,'model':model,'scope':'tissue','sample_id':tissue,**summarize(y[:,select],prediction[:,select])})
            for j,row in enumerate(pairs.itertuples(index=False)):
                transfer_rows.append({'target':target,'model':model,'scope':'sample','sample_id':row.sample_id,**summarize(y[:,j],prediction[:,j])})
    selected=mapping.loc[common_complete,'probe_id']
    if len(selected)<10: raise ValueError('Fewer than 10 complete eligible CpGs')
    stats=pd.DataFrame({'probe_id':selected,'mean_m':beta[common_complete].mean(axis=1),
                        'var_m':beta[common_complete].var(axis=1,ddof=0),'n_samples':n_samples})
    groups,remainder=make_groups(stats)
    if groups.probe_id.duplicated().any() or not groups.groupby('group_id').size().eq(10).all():
        raise ValueError('Overlapping/incorrect disjoint groups')
    positions=pd.Index(mapping.probe_id).get_indexer(groups.probe_id)
    group_metrics=[]
    for target in TARGETS:
        y=truth[target][positions].reshape(-1,10*n_samples)
        for method in ('A2H','m2','m3'):
            v=arrays[target][method][positions].reshape(-1,10*n_samples)
            if not np.isfinite(y).all() or not np.isfinite(v).all(): raise ValueError('Incomplete group')
            s=row_stats(y,v)
            if not (s['n']==10*n_samples).all(): raise ValueError('Incorrect group denominator')
            group_metrics.append(pd.DataFrame({'group_id':groups.group_id.to_numpy()[::10],'target':target,
                'method':method,'n_cpgs':10,'n_samples':n_samples,'n':s['n'],'pearson_r':s['pearson_r'],'rmse':s['rmse']}))
    quality=groups.groupby('group_id').agg(n_cpgs=('probe_id','size'),mean_cpg_mean_beta=('mean_m','mean'),
        mean_cpg_variance_beta=('var_m','mean'),min_cpg_mean_beta=('mean_m','min'),max_cpg_mean_beta=('mean_m','max'),
        min_cpg_variance_beta=('var_m','min'),max_cpg_variance_beta=('var_m','max')).reset_index()
    quality['mean_range']=quality.max_cpg_mean_beta-quality.min_cpg_mean_beta
    quality['variance_range']=quality.max_cpg_variance_beta-quality.min_cpg_variance_beta
    pooled=beta[positions].reshape(-1,10*n_samples)
    quality['group_mean_beta']=pooled.mean(axis=1)
    quality['group_variance_beta']=pooled.var(axis=1,ddof=0)
    grouped=pd.concat(group_metrics,ignore_index=True)
    site=pd.concat(sites,ignore_index=True)
    metric=pd.DataFrame(metrics)
    transfer=pd.DataFrame(transfer_rows)
    for name,frame in [('metrics.tsv',metric),('site_metrics.tsv',site),('flow.tsv',pd.DataFrame(flow)),
                       ('transfer_metrics.tsv',transfer),('cpg_beta_stats.tsv',stats),('groups.tsv',groups),
                       ('group_quality.tsv',quality),('group_metrics.tsv',grouped),('ungrouped_remainder.tsv',remainder)]:
        frame.to_csv(out/name,sep='\t',index=False)
    coef=pd.read_pickle(MODEL/'A2H_coef.pkl').set_index('featureID').loc[mapping.locus]
    constant=[]
    for target,col in [('CHALM','chalm_coef'),('CAMDA','camda_coef')]:
        scalar=coef[col].map(np.isscalar).to_numpy()
        n=(np.isfinite(preds[target])&np.isfinite(truth[target])&(truth['CHALM_depth']>=10))
        if target=='CAMDA': n &= truth['beta_depth']>=10
        constant.append({'target':target,'compatible_constant_fallback_probes':int(scalar.sum()),
                         'transformed_observations':int(np.isfinite(preds[target][scalar]).sum()),
                         'depth10_evaluable_observations':int(n[scalar].sum())})
    pd.DataFrame(constant).to_csv(out/'constant_fallback_flow.tsv',sep='\t',index=False)
    qc_replicates=[]
    allbeta=pd.read_csv(out/'array_beta.tsv',sep='\t',index_col=0)
    allpairs=pd.read_csv(out/'all_array_pairs.tsv',sep='\t')
    for wgbs,frame in allpairs.groupby('wgbs_id'):
        if len(frame)<2: continue
        first=frame.loc[frame.replicate.eq(1)].iloc[0]
        for row in frame.loc[frame.array_id.ne(first.array_id)].itertuples(index=False):
            qc_replicates.append({'wgbs_id':wgbs,'primary_array':first.array_id,'replicate_array':row.array_id,
                                 'metadata_status':row.metadata_status,**summarize(allbeta[first.array_id].to_numpy(),allbeta[row.array_id].to_numpy())})
    pd.DataFrame(qc_replicates).to_csv(out/'array_replicate_qc.tsv',sep='\t',index=False)
    figures(out,pairs,mapping,beta,preds,truth,grouped,site.loc[site.depth.eq(10)],arrays)
    report(out,metric,transfer,grouped,quality,prep)
    outputs={p.name:record(p) for p in out.glob('*.tsv')}
    save(out/'analysis_completion.json',{'state':'complete','finished_utc':utc(),'inputs':input_records,
         'outputs':outputs,'report':record(out/'REPORT.md'),'groups_disjoint':True,'groups_complete_samples':n_samples,
         'report_definitions':'No model fitted on external truth; no added support filter; no median within-CpG correlation endpoint'})


def report(out,metrics,transfer,groups,quality,prep):
    primary=metrics.loc[metrics.depth.eq(10)&metrics.scope.eq('pooled')]
    lines=[f'# GSE128734 {prep["platform"]} External A2H Validation Report','',
           f'Generated {utc()}. {prep["primary_samples"]} completed source samples; explicit source-title replicate 1 is primary.',
           f'Historical model training depth >=4; primary external truth >=10. {prep["compatible_probes"]:,} coordinate-compatible EPIC-covered probes available before truth/support missingness.',
           '','## R1-1: Baseline comparisons','',
           'Methods use identical eligible observations within each target. RMSE below is sqrt(mean(squared error)) over individual pairs; it differs from the mean CpG RMSE.',
           'NRMSE divides RMSE by max(observed)-min(observed) for the given panel; zero observed range is undefined.','',
           '| Target | Method | Observations | Pearson r | RMSE | NRMSE |','|---|---|---:|---:|---:|---:|']
    for row in primary.itertuples(index=False):
        lines.append(f'| {row.target} | {row.method} | {row.n:,} | {row.pearson_r:.5f} | {row.rmse:.5f} | {row.nrmse:.5f} |')
    lines += ['','## R1-4: Disjoint matched-mean/variance groups','',
        f'{len(quality):,} disjoint groups of 10 CpGs; complete data across all {prep["primary_samples"]} matched samples, {10*prep["primary_samples"]} observations per group/method.',
        'Groups use external array-beta mean sorting into blocks of 100, then variance sorting into groups of 10 (population variance). Group means, variances and ranges are recorded in group_quality.tsv.',
        'Groups are identical across CHALM/CAMDA and A2H/beta squared/beta cubed. Undefined correlations remain missing; error estimates are retained.','',
        '| Target | Method | Defined correlations | Median group r | Median group RMSE |','|---|---|---:|---:|---:|']
    for (target,method),frame in groups.groupby(['target','method']):
        lines.append(f'| {target} | {method} | {frame.pearson_r.notna().sum():,} | {frame.pearson_r.median():.5f} | {frame.rmse.median():.5f} |')
    lines += ['','## R1-5 and R2-2/R2-4: Reference composition','',
        'Frozen tissue-only exclusion models were trained on 161 reference samples after excluding blood/immune or 122 after excluding epithelial samples. Tissue labels determine membership; external truth is not used for training.',
        'All three models below use a common observation mask. Tissue/sample breakdowns and raw predictions are included.','',
        '| Target | Reference | Observations | Pearson r | RMSE |','|---|---|---:|---:|---:|']
    for row in transfer.loc[transfer.scope.eq('pooled')].itertuples(index=False):
        lines.append(f'| {row.target} | {row.model} | {row.n:,} | {row.pearson_r:.5f} | {row.rmse:.5f} |')
    lines += ['','## R2-3/R2-5: Site error and modeling unit','',
        'site_metrics.tsv records CpG-specific RMSE, bias and available sample counts at depth >=4 and >=10 for each method. These are external errors, not a calibrated prediction interval.',
        'Historical atlas CV/site-quality artifacts remain the source for training uncertainty. R1-6 atlas CV/functional holdout and R2-6 atlas mean-variance diagnostics are reused from the revision package; they are not rerun here.',
        '','## R3-S4/R3-S5: Sample panels and normalized error','',
        'figures/ contains one panel per matched tissue sample for each target, with sample Pearson r and RMSE. metrics.tsv contains each sample NRMSE and tissue summaries.',
        '','## R1-12 and supporting depth sensitivity','',
        'constant_fallback_flow.tsv documents retained scalar/KDE fallback probes. metrics.tsv/flow.tsv compare external depth >=4 and >=10. Array replicate QC is in array_replicate_qc.tsv; duplicates are not independent observations.',
        '','## Interpretation and scope','',
        '450K covers three completed whole-blood source samples; EPIC covers four CD4/neutrophil source samples. These platforms and tissues are confounded, so differences do not identify a platform effect. Donor relationships across source sample labels are not inferred.',
        'All use Swift/NovaSeq WGBS, with the previously approved Swift-specific R1 clip18. Sample4/SRR9888319 is unavailable due to unresolved mate2 checksum validation and is not imputed or replaced.',
        'Pooled and grouped correlations are descriptive transformation diagnostics. They do not establish disease-tissue validity, recovery of unobserved read configurations, or within-CpG participant ranking.',
        'A2H support is applied once by the deployed code at 0.05. Mapping conflicts are excluded without remapping, unsupported/missing values remain missing, and no median within-CpG correlation is a primary endpoint.',
        '','## Artifacts','',
        'Raw observations: CHALM_observations.parquet and CAMDA_observations.parquet. Transfer raw predictions: *_transfer_predictions.parquet. Frozen model/input hashes: preparation.json and analysis_completion.json.',
        'Matched pairs and exclusions: primary_pairs.tsv, all_array_pairs.tsv, mapping_audit.tsv, flow.tsv.','']
    (out/'REPORT.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','truth','transform','evaluate'])
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--index',type=int)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--validate-existing',action='store_true')
    parser.add_argument('--platform',choices=['450K','EPIC'])
    args=parser.parse_args()
    if args.action=='prepare':
        if not args.platform: parser.error('prepare requires an explicit --platform')
        prepare(args.output,args.platform)
    elif args.action=='truth': truth(args.output,args.index)
    elif args.action=='transform': transform(args.output,args.pilot,args.validate_existing)
    else: evaluate(args.output)


if __name__=='__main__': main()
