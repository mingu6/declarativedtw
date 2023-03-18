import argparse
import os
import random

import pandas as pd
import numpy as np


random.seed(2020)
np.random.seed(0)

# train reference sequences can be created from following
train_db_ids = ['2015-08-17-13-30-19', '2015-03-03-11-31-36', '2014-06-26-09-53-12', '2015-07-14-16-17-39', '2014-11-28-12-07-13', '2015-02-10-11-58-05']
val_query_ids = ['2015-02-03-08-45-10', '2014-11-18-13-20-12', '2015-08-14-14-54-57']
val_db_id = '2014-12-02-15-30-08'


def accumulate_distances(xy):
    accum_dists = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    accum_dists = np.concatenate(([0.], accum_dists)).cumsum()
    return accum_dists


def subsample_scalar(scalars, interval, start_ind=0, noise=0., seq_len=None):
    inds = [start_ind]
    curr = scalars[start_ind]
    for i, s in enumerate(scalars[start_ind:], start=start_ind):
        if s > curr + random.uniform(interval - noise, interval + noise):
            inds += [i]
            curr = s
        if seq_len is not None and len(inds) == seq_len:
            break
    return inds


def match_endpoints_sequence(xy_q, xy_db, no_match_threshold=5.):
    """
    Find points along query sequence (if any) that align with endpoints of (sliced) database sequence.
    """
    dists_to_start = np.linalg.norm(xy_db[None, 0, :] - xy_q, axis=1)
    dists_to_end = np.linalg.norm(xy_db[None, -1, :] - xy_q, axis=1)
    if not np.any(dists_to_start < no_match_threshold):
        return None
    start_ind = np.argmax(dists_to_start < no_match_threshold)
    if not np.any(dists_to_end[start_ind:] < no_match_threshold):
        return None
    end_ind = np.argmax(dists_to_end[start_ind:] < no_match_threshold)
    return start_ind, start_ind + end_ind


def match_query_along_db(ts_db, xy_db, ts_q, xy_q):
    """
    For each query, find nearest match along database trajectory assuming linear interpolation between observations
    """
    eps = 1e-10
    xy_lo = xy_db[:-1, :]
    xy_up = xy_db[1:, :]
    xy_diff = xy_up - xy_lo
    # for each segment b/w two observations in db, find point where query best matches
    t = np.einsum('nmd,md->nm', xy_q[:, None, :] - xy_lo[None, ...], xy_diff)
    t /= (np.linalg.norm(xy_diff, axis=1)[None, :] ** 2 + eps)
    np.clip(t, 0., 1., out=t)
    # compute distance at best match point for each segment
    dists = np.linalg.norm(xy_lo[None, ...] + t[..., None] * xy_diff[None, ...] - xy_q[:, None, :], axis=2)
    nearest_segment = np.argmin(dists, axis=1)  # find closest segment
    nearest_t = t[np.arange(t.shape[0]), nearest_segment]

    ts_interp = ts_db[nearest_segment] + nearest_t * (ts_db[nearest_segment + 1] - ts_db[nearest_segment])
    xy_interp = xy_lo[nearest_segment] + nearest_t[..., None] * xy_diff[nearest_segment]
    return ts_interp, xy_interp


def max_err(xy1, xy2):
    dists = np.linalg.norm(xy1 - xy2, axis=1)
    return dists.max()


def preprocess_df(df, spacing=1., noise=0.):
    xy = df[['northing', 'easting']].to_numpy()
    yaw = df['yaw'].to_numpy()
    fname = df['fname'].to_numpy()
    ts = np.array([float(fname[:-4]) / 1e6 for fname in df['fname'].tolist()])
    accum_dists = accumulate_distances(xy)
    start_inds = subsample_scalar(accum_dists, interval=spacing, noise=noise)
    return {'xy': xy, 'yaw': yaw, 'ts': ts, 'fname': fname, 'accum_dists': accum_dists, 'start_inds': start_inds}


def build_sequences(traverse_data, scalar='accum_dists', spacing=1., seq_len=10):
    seqs = []
    for t_id, data in traverse_data.items():
        for start_ind in data['start_inds']:
            seq_inds = subsample_scalar(data[scalar], interval=spacing, start_ind=start_ind, seq_len=seq_len)
            if len(seq_inds) == seq_len:
                seqs += [(t_id, seq_inds)]
    return seqs


def compile_seq_data(traverse_data, seq_inds):
    xy_seq = traverse_data['xy'][seq_inds]
    yaw_seq = traverse_data['yaw'][seq_inds]
    ts_seq = traverse_data['ts'][seq_inds]
    ts_seq -= ts_seq[0]
    fnames_seq = traverse_data['fname'][seq_inds]
    return xy_seq, yaw_seq, ts_seq, fnames_seq


def paired_sequences(df_query, df_db, db_spacing=5., query_spacing=1., spacing_noise=0.1, spacing_bw_dbs=20., q_len=10,
                     db_len=25, q_per_db=5):
    db_ids = set(df_db['traverse'].tolist())
    query_ids = set(df_query['traverse'].tolist())
    db_traverse_data = {
        t_id: preprocess_df(df_db.loc[df_db['traverse'] == t_id], spacing=spacing_bw_dbs, noise=spacing_noise) for t_id
        in db_ids}
    query_traverse_data = {
        t_id: preprocess_df(df_query.loc[df_query['traverse'] == t_id], spacing=spacing_bw_dbs, noise=spacing_noise) for
        t_id in query_ids}
    db_seqs = build_sequences(db_traverse_data, scalar='accum_dists', spacing=db_spacing, seq_len=db_len)
    query_seqs = build_sequences(query_traverse_data, scalar='ts', spacing=query_spacing, seq_len=q_len)
    # for each short database sequence, find aligned endpoints in queries to identify relevant short query seqs
    endpoint_lookups = []
    for db_id, seq_inds in db_seqs:
        lookup = {}
        for q_id in query_ids:
            endpoints = match_endpoints_sequence(query_traverse_data[q_id]['xy'],
                                                 db_traverse_data[db_id]['xy'][seq_inds, :])
            if endpoints is not None:
                lookup[q_id] = endpoints
        endpoint_lookups.append(lookup)
    # for each database sequence, compile all relevant query sequences and sample subset
    matched_queries_per_db = []
    for (db_id, db_seq_ind), lookup in zip(db_seqs, endpoint_lookups):
        matched_queries = []
        for q_id, q_seq_ind in query_seqs:
            if q_id in lookup:
                start, end = lookup[q_id]
                if q_seq_ind[0] >= start and q_seq_ind[-1] <= end:
                    matched_queries += [(q_id, q_seq_ind)]
        matched_queries = random.sample(matched_queries, min(len(matched_queries), q_per_db))
        matched_queries_per_db.append(matched_queries)
    # compile sequence indices into actual data
    paired_sequences = []
    for (db_id, db_seq_inds), matched_queries in zip(db_seqs, matched_queries_per_db):
        xy_db, yaw_db, ts_db, fnames_db = compile_seq_data(db_traverse_data[db_id], db_seq_inds)
        if not matched_queries:
            continue
        for q_id, q_seq_inds in matched_queries:
            xy_q, yaw_q, ts_q, fnames_q = compile_seq_data(query_traverse_data[q_id], q_seq_inds)
            ts_best, xy_best = match_query_along_db(ts_db, xy_db, ts_q, xy_q)
            max_e = max_err(xy_best, xy_q)
            data_dict = {'traverse_query': q_id, 'fnames_query': fnames_q, 'ts_query': ts_q, 'yaw_query': yaw_q,
                         'xy_query': xy_q,
                         'traverse_db': db_id, 'fnames_db': fnames_db, 'ts_db': ts_db, 'yaw_db': yaw_db, 'xy_db': xy_db,
                         'ts_best': ts_best, 'xy_best': xy_best, 'max_err': max_e}
            paired_sequences.append(data_dict)
    return paired_sequences


def save_paired_sequences(paired_sequences, fname):
    len_q = len(paired_sequences[0]['ts_query'].tolist())
    len_db = len(paired_sequences[0]['ts_db'].tolist())
    column_names = [f'traverse_query', 'traverse_db'] + [f'fname_query_{t}' for t in range(len_q)] + [f'fname_db_{t}'
                                                                                                      for t in
                                                                                                      range(len_db)] + \
                   [f'ts_query_{t}' for t in range(len_q)] + [f'ts_db_{t}' for t in range(len_db)] + \
                   [f'ts_best_{t}' for t in range(len_q)] + \
                   [f'best_northing_db_{t}' for t in range(len_q)] + [f'best_easting_db_{t}' for t in range(len_q)] + \
                   [f'northing_query_{t}' for t in range(len_q)] + [f'easting_query_{t}' for t in range(len_q)] + \
                   [f'northing_db_{t}' for t in range(len_db)] + [f'easting_db_{t}' for t in range(len_db)] + [
                       'max_err']
    df = pd.DataFrame(columns=column_names)

    for obs in paired_sequences:
        if obs is not None:
            query_traverse = [obs['traverse_query']]
            db_traverse = [obs['traverse_db']]
            query_fnames = obs['fnames_query'].tolist()
            db_fnames = obs['fnames_db'].tolist()
            query_ts = obs['ts_query'].tolist()
            db_ts = obs['ts_db'].tolist()
            best_ts = obs['ts_best'].tolist()
            best_northing = obs['xy_best'][:, 0].tolist()
            best_easting = obs['xy_best'][:, 1].tolist()
            query_northing = obs['xy_query'][:, 0].tolist()
            query_easting = obs['xy_query'][:, 1].tolist()
            db_northing = obs['xy_db'][:, 0].tolist()
            db_easting = obs['xy_db'][:, 1].tolist()
            max_e = [obs['max_err']]
            df.loc[len(df.index)] = query_traverse + db_traverse + query_fnames + db_fnames + query_ts + db_ts + \
                                    best_ts + best_northing + best_easting + query_northing + query_easting + db_northing + db_easting + max_e

    df.to_csv(fname, index=False)


def filter_pair(paired_sequence):
    # ensure sequences are sufficiently nearby
    if paired_sequence['max_err'] > 5.:
        return True

    # ensure optimal time warps are monotonic
    accum_dist_best = accumulate_distances(paired_sequence['xy_best'])
    if np.any(np.diff(paired_sequence['ts_best']) < 0.):
        return True

    # ensure optimal alignment is not stationary the whole time
    if np.any(np.all(np.diff(accum_dist_best) <= 0.1)):
        return True

    # ensure yaw after alignment is valid (removes some sketchy corners and opposite directions)
    yaw_query = paired_sequence['yaw_query']
    yaw_db_align = paired_sequence['yaw_db'][np.searchsorted(paired_sequence['ts_db'], paired_sequence['ts_best'])]
    yaw_diff = np.arctan2(np.sin(yaw_db_align - yaw_query), np.cos(yaw_db_align - yaw_query)) * 180. / np.pi
    if (np.abs(yaw_diff) > 30.).sum() > 0.2 * len(yaw_db_align):
        return True

    # ensures query is evenly sampled
    if np.any(np.diff(paired_sequence['ts_query'])) > 2. * np.median(np.diff(paired_sequence['ts_query'])):
        return True

    # ensures reference sequence does not jump too much in time
    if np.any(np.diff(paired_sequence['ts_db']) > 5 * np.median(np.diff(paired_sequence['ts_db']))):
        return True
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="create paired sequences for VPR DTW fine-tuning. run interpolate_ins.py first on files in thoma_lists directory",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gps', type=str, required=True, help='path to gps interpolated to image timestamps data directory')
    parser.add_argument('--out', type=str, default='', help='path to save paired sequences')
    parser.add_argument('--q-len', type=int, default=10, help='size of query sequence')
    parser.add_argument('--db-len', type=int, default=25, help='size of database sequence')
    parser.add_argument('--db-spacing', type=float, default=5., help='spacing between database frames (meters)')
    parser.add_argument('--q-spacing', type=float, default=1., help='spacing between query frames (Hz)')
    args = parser.parse_args()

    df_train = pd.read_csv(os.path.join(args.gps, 'train.csv'))
    df_train_db = df_train.loc[df_train['traverse'].isin(train_db_ids)]
    df_train_q = df_train.loc[~df_train['traverse'].isin(train_db_ids + [val_db_id])]
    df_val = pd.read_csv(os.path.join(args.gps, 'validation_query.csv'))
    df_val_db = df_train.loc[df_train['traverse'] == val_db_id]
    df_val_q = df_val.loc[df_val['traverse'].isin(val_query_ids)]

    df_test_db = pd.read_csv(os.path.join(args.gps, 'oxford_reference.csv'))
    df_test_queries = {t_id: pd.read_csv(os.path.join(args.gps, f'oxford_{t_id}_query.csv')) for t_id in ['overcast', 'snow', 'sunny']}

    train_pairs = paired_sequences(df_train_q, df_train_db, db_spacing=args.db_spacing, query_spacing=args.q_spacing,
                                   spacing_bw_dbs=20., spacing_noise=0.2 * args.db_spacing, q_len=args.q_len,
                                   db_len=args.db_len, q_per_db=24)
    train_pairs = [pair for pair in train_pairs if not filter_pair(pair)]
    print(f'{len(train_pairs)} sequence pairs generated for training set')
    save_paired_sequences(train_pairs, os.path.join(args.out, f'sequence_pairs_train.csv'))

    val_pairs = paired_sequences(df_val_q, df_val_db, db_spacing=args.db_spacing, query_spacing=args.q_spacing,
                                 spacing_bw_dbs=12., spacing_noise=0.2 * args.db_spacing, q_len=args.q_len,
                                 db_len=args.db_len, q_per_db=20)
    val_pairs = [pair for pair in val_pairs if not filter_pair(pair)]
    print(f'{len(val_pairs)} sequence pairs generated for validation set')
    save_paired_sequences(val_pairs, os.path.join(args.out, f'sequence_pairs_validation.csv'))

    for t_id, test_query in df_test_queries.items():
        test_pairs = paired_sequences(test_query, df_test_db, db_spacing=args.db_spacing, query_spacing=args.q_spacing,
                                      spacing_bw_dbs=10., spacing_noise=0.2 * args.db_spacing, q_len=args.q_len,
                                      db_len=args.db_len, q_per_db=5)
        test_pairs = [pair for pair in test_pairs if not filter_pair(pair)]
        print(f'{len(test_pairs)} sequence pairs generated for test {t_id} set')
        save_paired_sequences(test_pairs, os.path.join(args.out, f'sequence_pairs_test_{t_id}.csv'))