import os
import pandas as pd
import shutil
import time
from collections import defaultdict
from pandas.errors import EmptyDataError
from summ_utils import get_csvfiles, beststat_summarizer, add_missing_dflt_vals
from pathlib import Path
from cfg import main_acc, firth_reg_col, reg_sources, summ_cond_vars, dfltvals_dict
from cfg import results_csv_dir, smry_tbls_dir, prop_cols, deprecated_cols
from cfg import crn_cols, specific_csv_fldrs

try:
    from mpi4py import MPI
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()
    use_mpi = True
except ImportError:
    mpi_rank, mpi_size, mpi_comm, use_mpi = 0, 1, None, False

pd.options.mode.chained_assignment = None

y_col = main_acc
reg_column = firth_reg_col
cond_variables = summ_cond_vars

input_csv_dir = results_csv_dir
output_dir = smry_tbls_dir

# Recreating the output directory in case it doesn't exist
Path(output_dir).mkdir(parents=True, exist_ok=True)
tmp_dir = f'{output_dir}/temp'
Path(tmp_dir).mkdir(parents=True, exist_ok=True)


tmp_postfix_maker = lambda rank: f'_r{rank}'

# Collecting all csv file names to be concatenated
all_files = get_csvfiles(specific_csv_fldrs, input_csv_dir)

# Loading the csv files
if mpi_rank == 0:
    print(f'--> Collecting the input csv files...', flush=True)
df_lst = []
for filename in all_files:
    try:
        df = pd.read_csv(filename, sep=',')
    except EmptyDataError:
        if mpi_rank == 0:
            print(f'WARNING: {filename} seems to be empty. I will ignore it and move on', flush=True)
        continue
    df = add_missing_dflt_vals(df, dfltvals_dict)
    is_NaN = df.isnull()
    row_has_NaN = is_NaN.any(axis=1)
    rows_with_NaN = df[row_has_NaN]
    msg = f'You still have some missing values in your df at {filename} after adding default columns: '
    msg = msg + '\n{rows_with_NaN}\n'
    msg = msg + 'Did you set all default values for new columns in the cfg file?'
    assert len(rows_with_NaN) == 0, msg
    df_lst.append(df)
df_full = pd.concat(df_lst, axis=0, ignore_index=True)

df_summ_dict = defaultdict(list)
if mpi_rank == 0:
    print(f'--> Starting to summarize...', flush=True)
cntr = -1
for _, (cond, df_cond) in enumerate(df_full.groupby(cond_variables)):
    cntr += 1
    if cntr % mpi_size != mpi_rank:
        continue
        
    df_collection = beststat_summarizer(df_cond.reset_index(), prop_cols=prop_cols,
                                        y_col=y_col, reg_column=reg_column,
                                        reg_sources=reg_sources, sync_rng=True)

    for key, val in df_collection.items():
        df_summ_dict[key].append(val)

    print('.', end='', flush=True)

for key, df_list in df_summ_dict.items():
    df_summ = pd.concat(df_list, axis=0, ignore_index=True)
    tmp_postfix = tmp_postfix_maker(mpi_rank)
    df_summ.to_csv(f'{tmp_dir}/{key}_{tmp_postfix}.csv', index=False)

# Just waiting until everyone is done!
if use_mpi:
    mpi_comm.Barrier()
if mpi_rank == 0:
    print('\nEvery rank seems done. Root will start aggregating the csv files...', flush=True)
time.sleep(1)

if mpi_rank == 0:
    for key, df_list in df_summ_dict.items():
        df_cat_list = []
        for rank in range(mpi_size):
            tmp_postfix = tmp_postfix_maker(rank)
            csv_file = f'{tmp_dir}/{key}_{tmp_postfix}.csv'
            if not os.path.exists(csv_file):
                print(f'WARNING: {csv_file} does not exist!', flush=True)
                print(f'         This could be due to: ', flush=True)
                print(f'           (1) possibly there was nothing to do for that rank, or', flush=True)
                print(f'           (2) that particular rank got killed or ran into an issue, or', flush=True)
                print(f'           (3) the file has not yet been published to the disk. ', flush=True)
                print(f'         Either way, I will ignore it and move on!', flush=True)
                continue
            disk_df = pd.read_csv(csv_file)
            df_cat_list.append(disk_df)
        agg_df = pd.concat(df_cat_list, axis=0, ignore_index=True)
        
        def drop_deprecated_or_extra_cols(df):
            for colname in deprecated_cols:
                if colname in df.columns:
                    df = df[df[colname] == dfltvals_dict[colname]]
            for col in ['index'] + crn_cols + deprecated_cols:
                if col in df.columns:
                    df = df.drop(columns=[col])
            return df
        
        agg_df = drop_deprecated_or_extra_cols(agg_df)
        agg_df.to_csv(f'{output_dir}/{key}.csv', index=False)

    print('Wiping out the temperory directory...', flush=True)
    shutil.rmtree(tmp_dir)
