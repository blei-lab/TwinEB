// "
//     Compare a factor model using held-out predictive likelihoods.
//     Set the correlation to 1 to not remove any genes. Also remove the PMF family of models. 
//     Runs ../src/driver.py
//
//     # 1. parse csv with the input data
//     # 2. build code
//     # 3. create the heldout data
//     # 4. run the model
//     # 5. gather loglikelihoods
// "

nextflow.enable.dsl=1

// ------------ Functions ------------

import java.time.LocalDateTime
import java.text.SimpleDateFormat

def randomFileName(rndLen = 10) {
  // Generate a timestamped random file name
  def rndStr = org.apache.commons.lang.RandomStringUtils.random(rndLen, true, true).toString()
  def formatter = new SimpleDateFormat("yyyy-MMM-dd-HH-mm-ss")
  def strDate = formatter.format(new Date())
  return strDate + '-' + rndStr
}

def parse_csv_to_dict_list(infile) {
    // define a function that reads a csv file and converts it to a list of dictionaries
    def header = infile.readLines()[0].split(',')
    content = infile.readLines()[1..-1]
    dictList = content.collect { line ->
        def fields = line.split(',')
        def dict = [:]
        for (int i = 0; i < header.size(); i++) {
            dict[header[i]] = fields[i]
        }
        return dict
    }
    return dictList
}

def dictToString(dict) {
    // Convert dictionary to a string "--key value" pairs
    // if testing is true, first run dict through set_default_dict
    // Make a copy of the dict
    dict = dict.clone()
    if (params.is_testing) {
        dict = set_default_dict(dict)
    }
    
    // remove the id key
    dict.remove('id')
    def str = ""
    for (key in dict.keySet()) {
        str += "--${key} ${dict[key]} "
    }
    return str
}

def get_default_dict() {
    // When testing, set run params to low
    return [
        'max_steps': 100,
        'batch_size': 20,
        'num_pseudo_obs': 2,
        'num_pseudo_obs_global': 2,
        'n_llhood_samples': 3,
        'num_samples': 1
    ]
}

def set_default_dict(source_dict) {
    // define a function that given a source dict, will go over the keys in the default dict, and sets the values in the source dict to the default dict
    def default_dict = get_default_dict()
    for (key in default_dict.keySet()) {
        source_dict[key] = default_dict[key]
    }
    return source_dict
}

// ------------ Parameters ------------

params.is_testing = false
params.rna_paths = './batches/local_test/paths_local.csv'
params.sweep_path = './batches/local_test/grid_search.csv.gz'
params.cache_data_dir = (new File('../data/')).getCanonicalPath() 
params.suffix = randomFileName()
// For the createHeldout process
params.tolerance = 10
params.corlimit = 1
params.ignore_cache = true


// ------------ Variables ------------
deliverableDir = '../../results/deliverables/' + workflow.scriptName.replace('.nf','')
// Convert to absolute path 
deliverableDir = (new File(deliverableDir)).getCanonicalPath()
batchName = params.sweep_path.split('/')[-2]
deliverableDir = deliverableDir + '/' + batchName + '/' + params.suffix

// Read in the paramter sweep
dictList = parse_csv_to_dict_list(file(params.sweep_path))


// ------------ Channels ------------
// Create a channel that goes from 1 to the length of the dictList
sweeps = Channel.from(1..dictList.size())

if (params.is_testing == true) {
    sweeps = sweeps.take(2)
    params.ignore_cache = false
} else {
    params.ignore_cache = false
    // pass
}


sweeps.into { sweeps0; sweeps1}


// Read in the dataset paths
rnaFiles = Channel
    .fromPath(params.rna_paths)
    .splitCsv(header:true)
    .map( row -> [row.sample_id, file(row.paths)] )

if (params.is_testing == true) {
    rnaFiles = rnaFiles.take(3)
}

rnaFiles.into {rna0; rna1}

holdout_proportion = Channel.of(.2)


// ------------ Processes ------------

// Build the code
process buildCode {
  cache false
  executor 'local'
  output:
    file 'code' into codeR

  script:
  """
  echo $PWD
  mkdir -p code
  ln -sf `ls $PWD/../../../src/*.py` code
  """  
}

codeR.into {
    codeR0
    codeR1
    codeR2
    codeR3
    codeR4
}

// Create holdout data given the holdout proportions
process createHeldout {
  errorStrategy { (task.exitStatus == 140 && task.attempt <= maxRetries)  ? 'retry' : 'ignore' }
  memory {5.GB + 5.GB*task.attempt}
  cpus 1
  maxRetries 1
  time {0.h + 1.h*task.attempt}
  
  input:
    file codeR0
    tuple val(sampleId), val(rnaPath) from rna1
    each holdOutProp from holdout_proportion
  output:
    set sampleId, file('*.pkl') into outs_heldout
  publishDir "${deliverableDir}/${sampleId}", mode: 'copy', overwrite: true

  script:
  """ 
  python3 code/driver.py setup_data  \
        -i "$rnaPath" \
        -o . \
        -p "$holdOutProp" \
        -f $params.ignore_cache \
        -c $params.cache_data_dir \
        -l $params.corlimit
  """

  stub:
  """
  touch heldout.pkl
  """
}

outs_heldout.into {heldout0; heldout1}

// A process that runs driver.py with parameters from dictList
process runModel {
    errorStrategy { ((task.exitStatus == 140 || task.exitStatus == 137 || task.exitStatus == 130) && task.attempt <= maxRetries)  ? 'retry' : 'ignore' }
    memory {5.GB + 5.GB*task.attempt}
    time {0.h + 1.h*task.attempt}
    cpus 1
    maxRetries 3
    // Add your GPU support if available
    // clusterOptions '-R A100 -gpu "num=1:gmem=8:j_exclusive=no:mps=yes" -q gpuqueue'
  
    input:
        file codeR1
        tuple val(sampleId), val(heldoutPath) from heldout0
        each idct_indx from sweeps1

    output:
        set sampleId, idct_indx, path('out/config.yaml') into outs_loglikelihood

    publishDir "${deliverableDir}/${sampleId}/${dictList[idct_indx - 1]['id']}", mode: 'copy', overwrite: true
 
  script:
  """
  echo $PWD
  python code/driver.py run_model \
    ${dictToString(dictList[idct_indx - 1])} \
    --picklePath ${heldoutPath} \
    --save_dir . \
    --tolerance $params.tolerance \
    --stddv_datapoints 1 \
    --annealing_factor 1 \
    --log_cadence 10 \
    --optimizer Adam \
    --scheduler ReduceLROnPlateau \
  # ensure that config.yaml is copied to the publishDir
  # ensure that publishDir is created
  mkdir -p ${deliverableDir}/${sampleId}/${dictList[idct_indx - 1]['id']}
  cp out/config.yaml ${deliverableDir}/${sampleId}/${dictList[idct_indx - 1]['id']}/
  # Remove all non hidden files and non yaml files config.yaml in the out directory 
  find out -type f -not -name "config.yaml" -not -name "sumamry.csv.gz"  -exec rm {} +
  """
}


outs_loglikelihood.into {loglikelihood0; loglikelihood1; loglikelihood2}

// Collect to ensure running after all runs are finished and cast the channel to a csv friendly format
outs_loglikelihood0 = loglikelihood2.collect{it.join(',')}.flatten().collate(3).collect()
outs_loglikelihood0.into { outs_loglikelihood1; outs_loglikelihood2 }

// Summarise the pipeline
process gatherResults {
  if (params.is_testing == true) {
    debug true
    executor 'local'
  } else {
    errorStrategy { (task.exitStatus == 140 && task.attemptf <= maxRetries)  ? 'retry' : 'ignore' }
  }
  
  maxRetries 2
  memory {5.GB + 5.GB*task.attempt}
  time '1h'
  input:
      file codeR4
      val(all_res) from outs_loglikelihood2
  output:
      file 'merged.csv'
  
  publishDir "${deliverableDir}/merge/", mode: 'mv', overwrite: true
  
  script:
  """
  python code/gather_configs.py -b ${deliverableDir}
  """
}

// Write pipeline info to a file
process summarizePipeline {
    executor 'local'
    cache false
    output:
        file 'pipeline-info.txt'   
    publishDir deliverableDir, mode: 'copy', overwrite: true
    """
    echo 'scriptName: $workflow.scriptName' >> pipeline-info.txt
    echo 'start: $workflow.start' >> pipeline-info.txt
    echo 'runName: $workflow.runName' >> pipeline-info.txt
    echo 'nextflow.version: $workflow.nextflow.version' >> pipeline-info.txt
    """
}