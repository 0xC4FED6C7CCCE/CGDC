import pandas as pd
import numpy as np
from itertools import product
import os
from cfg import naming_scheme as proj_naming_scheme
from cfg import main_acc as proj_main_acc
from cfg import firth_reg_col, crn_cols, scale_percent
from cfg import prop_cols as proj_prop_cols


def add_missing_dflt_vals(df, def_dict):
    for colname, defval in def_dict.items():
        if colname in df.columns:
            df[colname].fillna(defval, inplace=True)
        else:
            df[colname] = defval
    return df


def default_str_maker(flt_mean, flt_ci=None):
    if (flt_ci is not None) and not():
        pm = 'PM'
        out_str = f'%0.2f {pm} %.2f' % (scale_percent*flt_mean, scale_percent*flt_ci)
    else:
        out_str = f'%0.2f' % (scale_percent*flt_mean)
    return out_str + '%'


def gen_table(summary_df, row_tree, col_tree, y_col,
              y_col_ci=None, str_maker=None, is_bold_maker=None):
    if str_maker is None:
        str_maker = default_str_maker
    x_cols = row_tree + col_tree
    my_summary_df = summary_df.copy(deep=False)

    row_tree_uniques = [my_summary_df[col].unique().tolist() for col in row_tree]
    col_tree_uniques = [my_summary_df[col].unique().tolist() for col in col_tree]
    x_col_uniques = row_tree_uniques + col_tree_uniques

    ndarr_shape = tuple(len(a) for a in x_col_uniques)
    np_ndarr = np.full(ndarr_shape, np.nan).reshape(-1)
    np_ndarr_ci = np.empty(ndarr_shape, dtype=object).reshape(-1)
    np_ndarr_raveled = np.empty_like(np_ndarr, dtype=object)
    for i, x_tup in enumerate(product(*x_col_uniques)):
        df = my_summary_df
        for x_col, x_val in zip(x_cols, x_tup):
            df = df[df[x_col] == x_val]
        if len(df) > 1:
            os.makedirs('../trash', exist_ok=True)
            df.to_csv('../trash/dbg.csv')
        assert len(df) == 1, f'The combination {x_cols}={x_tup} has {len(df)} rows in it instead of 1: \n{df}'

        entry = df[y_col].values.item()
        if y_col_ci is not None:
            entry_ci = df[y_col_ci].values.item()
        else:
            entry_ci = None
        np_ndarr[i] = entry
        np_ndarr_ci[i] = entry_ci

    if is_bold_maker is not None:
        bold_indic = is_bold_maker(np_ndarr).reshape(-1)
    else:
        bold_indic = np.full(ndarr_shape, False).reshape(-1)

    for i, x_tup in enumerate(product(*x_col_uniques)):
        entry, entry_ci, is_bold = np_ndarr[i], np_ndarr_ci[i], bold_indic[i]
        np_ndarr_raveled[i] = str_maker(entry, entry_ci, is_bold=is_bold)

    np_ndarr = np_ndarr_raveled.reshape(*ndarr_shape)
    nrows = np.prod(tuple(len(a) for a in row_tree_uniques))
    ncols = np.prod(tuple(len(a) for a in col_tree_uniques))
    out_df = pd.DataFrame(np_ndarr.reshape(nrows, ncols),
                          columns=pd.MultiIndex.from_product(col_tree_uniques),
                          index=pd.MultiIndex.from_product(row_tree_uniques, names= row_tree))
    return out_df


# getting the files in each folder
def get_csvfiles(fldr_mini, results_dir):
    fldr_mini_files = []
    for fldr in fldr_mini:
        if not os.path.exists(f'{results_dir}/{fldr}'):
            continue
        for file in os.listdir(f'{results_dir}/{fldr}'):
            if file.endswith(".csv"):
                fldr_mini_files.append(f'{results_dir}/{fldr}/{file}')
    return fldr_mini_files


def conditioner(df_full, condition_var_dict=None):
    condition_var_dict = condition_var_dict or dict()
    df_full = df_full.copy(deep=True)
    for var, val in condition_var_dict.items():
        df_full = df_full[df_full[var] == val]
    return df_full


def beststat_summarizer(df, prop_cols=None, y_col=None, reg_column=None,
                        reg_sources=None, sync_rng=True, naming_scheme=None):

    if reg_sources is None:
        reg_sources = ['val']
    if prop_cols is None:
        prop_cols = proj_prop_cols
    if naming_scheme is None:
        naming_scheme = proj_naming_scheme
    assert naming_scheme in ('s2m2rf', 'firth')
    if y_col is None:
        y_col = proj_main_acc

    if naming_scheme == 's2m2rf':
        assert 'split' in df.columns, 'Are you sure you are using the right naming scheme?'
        assert 'rng_seed' in df.columns, 'Are you sure you are using the right naming scheme?'
        assert 'task_id' in df.columns, 'Are you sure you are using the right naming scheme?'
    elif naming_scheme == 'firth':
        assert 'data_type' in df.columns, 'Are you sure you are using the right naming scheme?'
        assert 'rng_seed' in df.columns, 'Are you sure you are using the right naming scheme?'

    if reg_column is None:
        reg_column = firth_reg_col

    if naming_scheme == 's2m2rf':
        df_val = df[df['split'] == 'val']
        df_test = df[df['split'] == 'novel']
    elif naming_scheme == 'firth':
        df_val = df[df['data_type'] == 'val']
        df_test = df[df['data_type'] == 'test']
    
    dfval_exists = len(df_val) > 0
    out_dict = dict()
    if not sync_rng:
        if dfval_exists:
            stats_val = df_val.groupby(reg_column)[y_col].agg(['mean', 'count', 'std'])
            df_val_mean = df_val.groupby([reg_column]).mean()
            df_val_mean[f'{y_col}_ci'] = 1. * stats_val['std'] / np.sqrt(stats_val['count'])
            df_val_mean.reset_index(inplace=True)
            out_dict['val'] = df_val_mean
            
        stats_test = df_test.groupby([reg_column])[y_col].agg(['mean', 'count', 'std'])
        df_test_mean = df_test.groupby([reg_column]).mean()
        df_test_mean[f'{y_col}_ci'] = 1. * stats_test['std'] / np.sqrt(stats_test['count'])
        df_test_mean.reset_index(inplace=True)
        out_dict['test'] = df_test_mean
    else:
        msg_ = 'A prop col may be missing in narrowing down the data. its not safe to keep going.'
        if dfval_exists:
            assert len(df_val.groupby(prop_cols)) == 1, msg_
            out_df_list = []
            for name, df_grp in df_val.groupby(crn_cols):
                if not (df_grp[reg_column] == 0.).any():
                    continue
                base_acc = df_grp.loc[df_grp[reg_column] == 0., y_col]
                if len(base_acc) > 1:
                    assert np.allclose(base_acc.values, base_acc.values[0]), df_grp.loc[df_grp[reg_column] == 0., :]
                base_acc_item = base_acc.values[0].item()
                df_grp[f'delta_{y_col}'] = df_grp[y_col] - base_acc_item
                out_df_list.append(df_grp)
            out_df = pd.concat(out_df_list, axis=0, ignore_index=True)
            df_val_mean = out_df.groupby(prop_cols + [reg_column]).mean()
            for ci_col in [y_col, f'delta_{y_col}']:
                stats_val = out_df.groupby(prop_cols + [reg_column])[ci_col].agg(['mean', 'count', 'std'])
                df_val_mean[f'{ci_col}_ci'] = 1.96 * stats_val['std'] / np.sqrt(stats_val['count'])
            df_val_mean.reset_index(inplace=True)
            out_dict['val'] = df_val_mean

        assert len(df_test.groupby(prop_cols)) == 1, msg_
        out_df_list = []
        for name, df_grp in df_test.groupby(crn_cols):
            if not (df_grp[reg_column] == 0.).any():
                continue
            base_acc = df_grp.loc[df_grp[reg_column] == 0., y_col]
            if len(base_acc) > 1:
                assert np.allclose(base_acc.values, base_acc.values[0]), df_grp.loc[df_grp[reg_column] == 0., :]
            base_acc_item = base_acc.values[0].item()
            df_grp[f'delta_{y_col}'] = df_grp[y_col] - base_acc_item
            out_df_list.append(df_grp)
        out_df = pd.concat(out_df_list, axis=0, ignore_index=True)
        df_test_mean = out_df.groupby(prop_cols + [reg_column]).mean()
        for ci_col in [y_col, f'delta_{y_col}']:
            stats_val = out_df.groupby(prop_cols + [reg_column])[ci_col].agg(['mean', 'count', 'std'])
            df_test_mean[f'{ci_col}_ci'] = 1.96 * stats_val['std'] / np.sqrt(stats_val['count'])
        df_test_mean.reset_index(inplace=True)
        out_dict['test'] = df_test_mean

    
    for reg_src in reg_sources:
        if reg_src == 'val':
            best_val_firth_coeff = df_val_mean.loc[df_val_mean[y_col].idxmax()][reg_column]
        elif reg_src == 'test':
            best_val_firth_coeff = df_test_mean.loc[df_test_mean[y_col].idxmax()][reg_column]
        else:
            raise Exception(f'Unknown reg_source: {reg_src}')

        base_y = df_test_mean[df_test_mean[reg_column] == 0.][y_col].values.item()
        base_y_ci = df_test_mean[df_test_mean[reg_column] == 0.][f'{y_col}_ci'].values.item()
        row_df = df_test_mean[df_test_mean[reg_column] == best_val_firth_coeff]
        row_df[f'delta_{y_col}'] = row_df[y_col] - base_y
        if not sync_rng:
            row_df[f'delta_{y_col}_ci'] = 2 * 1.96 * row_df[f'{y_col}_ci']

        ##########
        base_test_row = df_test_mean[df_test_mean[reg_column] == 0.0]
        assert len(base_test_row) == 1, f'found multiple rows in base_test_row. ' \
                                        f'The conditioning is incomplete:\n{base_test_row}'
        row_df[f'base_{y_col}'] = base_y
        row_df[f'base_{y_col}_ci'] = base_y_ci

        row_df.reset_index(drop=True)
        out_dict[f'{reg_src}2test'] = row_df

    return out_dict
