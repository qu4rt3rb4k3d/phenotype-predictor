import subprocess
import os

import hail
import pandas
import numpy

import training

def hail_init(spark_max_mem):
    hail.init(spark_conf={
        'spark.jars': 'hadoop-aws-3.0.0.jar,aws-java-sdk-bundle-1.11.199.jar',
        'spark.hadoop.fs.s3a.impl': 'org.apache.hadoop.fs.s3a.S3AFileSystem',
        'spark.hadoop.fs.s3a.aws.credentials.provider': 'com.amazonaws.auth.InstanceProfileCredentialsProvider',
        'spark.driver.memory': spark_max_mem
    })

def make_working_dir(working_dir):
    os.makedirs(working_dir, exist_ok=True)

def get_sumstats_df(phenocode, pop, pval_threshold, working_dir):
    def get_phenofile_path():
        return working_dir + '/continuous-'+phenocode+'-both_sexes-irnt.tsv'
    def get_sumstats_path():
        return working_dir + '/ss_' + phenocode + '_' + pop + '_' + str(pval_threshold) + '.csv'
    def fetch_phenotype_file():
        def get_phenofile_url():
            return 'https://pan-ukb-us-east-1.s3.amazonaws.com/sumstats_flat_files/continuous-'+phenocode+'-both_sexes-irnt.tsv.bgz'
        subprocess.run(['wget', '-O', get_phenofile_path() + '.bgz', get_phenofile_url()])
        subprocess.run(['gzip', '-d', '-S', '.bgz', get_phenofile_path() + '.bgz'])
    if not os.path.exists(get_sumstats_path()):
        if not os.path.exists(get_phenofile_path()):
            fetch_phenotype_file()
        def get_pval_column():
            return 'neglog10_pval_' + pop
        pval_filtered_chunks = []
        for chunk in pandas.read_csv(get_phenofile_path(), sep='\t', chunksize=1000000, low_memory=False):
            pval_filtered_chunks.append(chunk[chunk[get_pval_column()] >= pval_threshold])
        sumstats_df = pandas.concat(pval_filtered_chunks)
        def row_to_variant(row):
            return str(row['chr']) + ':' + str(row['pos']) + ':' + str(row['ref']) + ':' + str(row['alt'])
        sumstats_df['variant'] = sumstats_df.apply(row_to_variant, axis=1).to_list()
        sumstats_df.to_csv(get_sumstats_path(), sep=' ', index=False)
        return sumstats_df
    else:
        return pandas.read_csv(get_sumstats_path(), sep=' ', low_memory=False)

def get_ld_matrix(sumstats_df, pop, phenocode, pval_threshold, spark_max_mem, working_dir):
    def get_ld_matrix_hail(pop):
        return hail.linalg.BlockMatrix.read('s3a://pan-ukb-us-east-1/ld_release/UKBB.'+pop+'.ldadj.bm')
    def get_idx_table_hail(pop):
        return hail.read_table('s3a://pan-ukb-us-east-1/ld_release/UKBB.'+pop+'.ldadj.variant.ht')
    def get_variant_df(ref_genome=None):
        if not ref_genome:
            ref_genome = hail.get_reference('GRCh37')
        def row_to_variant(row):
            locus_str = str(row['chr']) + ':' + str(row['pos'])
            locus = hail.Locus.parse(locus_str, ref_genome)
            alleles = [row['ref'], row['alt']]
            return (locus, alleles)
        var_list = sumstats_df.apply(row_to_variant, axis=1).to_list()
        var_df = pandas.DataFrame(var_list, columns=['locus', 'alleles'])
        return var_df
    def get_filtered_idx_df(idx_table, var_table):
        idx_table = idx_table.key_by('locus', 'alleles')
        var_table = var_table.key_by('locus', 'alleles')
        return idx_table.join(var_table, 'inner').to_pandas()
    def get_filtered_ss_df(idx_df, var_df):
        left_key_df = idx_df[['locus', 'alleles']].copy()
        right_key_df = var_df.copy()
        right_key_df['locus'] = right_key_df['locus'].astype(str)
        left_key_df['alleles'] = left_key_df['alleles'].apply(lambda a: str(a))
        right_key_df['alleles'] = right_key_df['alleles'].apply(lambda a: str(a))
        left_keyed_ss_df = pandas.concat([sumstats_df, left_key_df], axis=1)
        filtered_ss_df = pandas.merge(left_keyed_ss_df, right_key_df, on=['locus', 'alleles'], how='inner')
        return filtered_ss_df.drop(['locus', 'alleles'], axis=1)
    def get_ld_matrix_path():
        return working_dir + '/ld_' + phenocode + '_' + pop + '_' + str(pval_threshold) + '.npy'
    def get_idx_path():
        return working_dir + '/idx_' + phenocode + '_' + pop + '_' + str(pval_threshold) + '.csv'

    hail_init(spark_max_mem)
    var_df = get_variant_df()

    if os.path.exists(get_idx_path()):
        idx_df = pandas.read_csv(get_idx_path(), sep=' ')
    else:
        ld_matrix = get_ld_matrix_hail(pop)
        idx_table = get_idx_table_hail(pop)
        var_table = hail.Table.from_pandas(var_df)
        idx_df = get_filtered_idx_df(idx_table, var_table)
        idx_df.to_csv(get_idx_path(), sep=' ', index=False)
    
    if os.path.exists(get_ld_matrix_path()):
        ld_matrix = numpy.load(get_ld_matrix_path())
    else:
        idxs = idx_df['idx'].to_list()
        ld_matrix = ld_matrix.filter(idxs, idxs).to_numpy()
        ld_matrix = ld_matrix + ld_matrix.T - numpy.diag(numpy.diag(ld_matrix))
        numpy.save(get_ld_matrix_path(), ld_matrix)

    filtered_ss_df = get_filtered_ss_df(idx_df, var_df)
    hail.stop()
    return ld_matrix, filtered_ss_df

def get_results_df(ss_df, b):
    results_df = pandas.DataFrame()
    results_df['variant'] = ss_df['variant']
    results_df['beta'] = b
    return results_df

def run_flow(phenocode, pop, num_samples, pval_threshold=6, lambda_ = 1, spark_max_mem='16g', working_dir='./working_dir'):
    make_working_dir(working_dir)

    sumstats_df = get_sumstats_df(phenocode, pop, pval_threshold, working_dir)
    ld, sumstats_df = get_ld_matrix(sumstats_df, pop, phenocode, pval_threshold, spark_max_mem, working_dir)

    af = numpy.array(sumstats_df['af_' + pop])
    se = numpy.array(sumstats_df['se_' + pop])
    beta = numpy.array(sumstats_df['beta_' + pop])
    b = training.train_predictor_lasso(ld, af, se, beta, num_samples, lambda_)

    results_df = get_results_df(sumstats_df, b)
    def get_results_path():
        return working_dir + '/results_' + phenocode + '_' + pop + '_' + str(pval_threshold) + '_' + str(lambda_) + '.csv'
    results_df.to_csv(get_results_path(), sep=' ', index=False)
    return results_df
