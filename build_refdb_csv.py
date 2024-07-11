################################################################################
# This script takes a directory of JWST calints files as input and builds a    #
# database of reference info for each file, which can then be read via the     #
# pipeline script.                                                             #
#                                                                              #
# Basically it will hold the target-specific info needed by the pipeline, as   #
# well as info needed to choose reference observations for a given science     #
# target.                                                                      #
#                                                                              #
# Written 2024-07-10 by Ellis Bogat                                            #
################################################################################

# TODO: 
# - set up pip installability
# - make a readme describing each column & its units
# - add toggle to ingest uncal or calints files
# - write tests!
#       - nonexistent input directory
#       - empty input directory 
#       - no calints files in input directory
#       - header kw missing
#       - duplicate science target with different program names
#       - calints vs uncal toggle (warning if no calints present but uncals are
#               'did you mean to set uncal = True?')

# imports
import os
import glob
import mocapy
import pandas as pd
from astropy.io import fits

from astroquery.simbad import Simbad
import numpy as np

def isnone(series):
    try:
        return np.isnan(series)
    except:
        return (series == '') | (series == None) 

def build_refdb(idir,odir=None,uncal=False):
    """
    Constructs a database of target-specific reference info for each
    calints file in the input directory.

    Args:
        idir (path): Path to pre-processed (stage 2) JWST images to be added 
         to the database
        odir (path, optional): Location to save the database. If odir is None,
         the database will not be saved. Defaults to None.
        uncal (bool, optional): Toggle using uncal files as inputs instead of
         calints. Defaults to None.

    Raises:
        Exception: _description_
        Exception: _description_
        Exception: _description_

    Returns:
        pandas.DataFrame: database containing target- and observation-specific 
         info for each file.
    """
    
    # Read in uncal files 
    print('Reading input files...')
    suffix = 'uncal' if uncal else 'calints'
    fpaths = sorted(glob.glob(os.path.join(idir,f"*_{suffix}.fits")))

    # Start a dataframe with the header info we want from each file
    csv_list = []
    fits_cols = [
        'TARGPROP',
        'TARGNAME', # Save 2MASS ID also
        'FILENAME',
        'DATE-OBS',
        'TIME-OBS',
        'DURATION', # Total exposure time 
        'TARG_RA',
        'TARG_DEC',
        'TARGURA', # RA uncertainty
        'TARGUDEC', # Dec uncertainty
        'MU_RA', # Proper motion
        'MU_DEC',                
        'MU_EPOCH',
        'INSTRUME',
        'DETECTOR',
        'MODULE',
        'CHANNEL',
        'FILTER',
        'PUPIL',
    ]
    for fpath in fpaths:
        row = []
        hdr = fits.getheader(fpath)
        for col in fits_cols:
            row.append(hdr[col])
        csv_list.append(row)

    df = pd.DataFrame(csv_list,columns=fits_cols)

    # Make a df with only one entry for each unique target
    targnames = np.unique(df['TARGNAME']) 
    df_unique = pd.DataFrame(np.transpose([targnames]),columns=['TARGNAME'])

    # Get 2MASS IDs
    print('Collecting 2MASS IDs...')
    twomass_ids = []
    for targname in targnames:
        result_table = Simbad.query_objectids(targname)
        if result_table is None:
            raise Exception(f'No SIMBAD object found for targname {targname}.')    
        tmids_found = []
        for name in list(result_table['ID']):
            if name[:6] == '2MASS ':
                twomass_ids.append(name)
                tmids_found.append(name) 
        if len(tmids_found) < 1:
            raise Exception(f'No 2MASS ID found for targname {targname}.')
        elif len(tmids_found) > 1:
            raise Exception(f'Multiple 2MASS ID found for targname {targname}: {tmids_found}')
    df_unique['2MASS_ID'] = twomass_ids
    df_unique.set_index('2MASS_ID',inplace=True)

    # Query SIMBAD
    print('Querying SIMBAD...')

    customSimbad = Simbad()
    customSimbad.add_votable_fields('sptype', 
                                    'flux(K)', 'flux_error(K)', 
                                    'plx', 'plx_error')
    simbad_list = list(df_unique.index)
    scistar_simbad_table = customSimbad.query_objects(simbad_list)

    # Convert to pandas df and make 2MASS IDs the index
    df_simbad = scistar_simbad_table.to_pandas()
    df_simbad['2MASS_ID'] = simbad_list
    df_simbad.set_index('2MASS_ID',inplace=True)

    # Rename some columns
    simbad_cols = {
        'SPTYPE': 'SP_TYPE', # maybe use 'simple_spt' or 'complete_spt'?
        'KMAG': 'FLUX_K', # 'kmag'
        'KMAG_ERR': 'FLUX_ERROR_K', # 'ekmag'
        'PLX': 'PLX_VALUE', # 'plx'
        'PLX_ERR': 'PLX_ERROR', # 'eplx'
        }
    for col,simbad_col in simbad_cols.items():
        df_simbad[col] = list(df_simbad[simbad_col])

    # Add the values we want to df_unique
    df_unique = pd.concat([df_unique,df_simbad.loc[:,simbad_cols.keys()]],axis=1)

    # Query mocadb.ca for extra info
    print('Querying MOCADB (this may take a minute)...')
    names_df = pd.DataFrame(list(df_unique.index),columns=['designation'])
    moca = mocapy.MocaEngine()
    mdf = moca.query("SELECT tt.designation AS input_designation, sam.* FROM tmp_table AS tt LEFT JOIN mechanics_all_designations AS mad ON(mad.designation LIKE tt.designation) LEFT JOIN summary_all_objects AS sam ON(sam.moca_oid=mad.moca_oid)", tmp_table=names_df)
    mdf.set_index('input_designation',inplace=True)

    moca_cols = {
        'SPTYPE': 'spt', # maybe use 'simple_spt' or 'complete_spt'?
        'PLX': 'plx', # 'plx'
        'PLX_ERR': 'eplx', # 'eplx'
        'AGE': 'age', # 'age'
        'AGE_ERR': 'eage', # 'eage'
    }

    # Update the column names for consistency
    for col,moca_col in moca_cols.items():
        # print(col, list(mdf[moca_col]))
        mdf[col] = list(mdf[moca_col])

    # Fill in values missing from SIMBAD with MOCA

    df_unique['COMMENTS'] = ''

    # Sort all the dfs by index so they match up
    df_unique.sort_index(inplace=True)   
    df_simbad.sort_index(inplace=True)   
    mdf.sort_index(inplace=True)   

    # for col in ['SPTYPE','KMAG','KMAG_ERR']:
    #     df_unique[isnone(df_simbad[col]) & ~isnone(mdf[col]),col] += f"{col} adopted from MOCA. "

    for col in ['SPTYPE','PLX','PLX_ERR']:
        df_unique.loc[df_unique[col].index[isnone(df_simbad[col]) & ~isnone(mdf[col])].to_list(),'COMMENTS'] += f"{col} adopted from MOCA. "

    for col in ['AGE','AGE_ERR']:
        df_unique[col] = mdf[col]
        df_unique.loc[df_unique[col].index[~isnone(mdf[col])].to_list(),'COMMENTS'] += f"{col} adopted from MOCA. "


    df_unique_replaced = df_unique.loc[:,simbad_cols.keys()].combine_first(df_simbad.loc[:,simbad_cols.keys()])
    df_unique.loc[:,simbad_cols.keys()] = df_unique_replaced.loc[:,simbad_cols.keys()]

    # Calculate distances from plx in mas
    df_unique['DIST'] = 1. / (df_unique['PLX'] / 1000)
    df_unique['DIST_ERR'] = df_unique['PLX_ERR'] / 1000 / ((df_unique['PLX'] / 1000)**2)

    # Add empty columns
    empty_cols = [
        'FLAGS',
        'HAS_DISK',
        'HAS_CANDS']
    for col in empty_cols:
        df_unique[col] = ''

    # Apply dataframe of unique targets to the original file list
    df.set_index('TARGNAME',inplace=True)
    df_unique.reset_index(inplace=True)
    df_unique.set_index('TARGNAME',inplace=True)
    df_unique = df_unique.reindex(df.index)
    df_out = pd.concat([df,df_unique],axis=1)

    if not odir is None:
        outpath = os.path.join(odir,'ref_lib.csv')
        df_out.to_csv(outpath)

        print(f'Database saved to {outpath}')
    print(f'Done.')

    return df_out