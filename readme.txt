Version: CHALM-array v0.3.1

Environment requirement:
    - python >= 3.7
    - pandas
    - numpy
    - joblib

    
==========CHALM-array v0.3.1==========

usage: CHALM_array.py [-h] [-i INPUT] [-o OUTPUT] [--prefix PREFIX] [-c COEF] [--cpg-info CPG_INFO_PATH] [-t THREAD] [--ref {hg38,hg19}] [--beadchip {450k,EPIC,HM450,850k}] [--cl] [--probe]
                      [--verbose VERBOSE] [--camda] [--filter FILTER] [--replace REPLACE]

options:
  -h, --help            show this help message and exit
  -i INPUT              Input methyaltion file with first column as probe ID and first row as sample ID.
  -o OUTPUT             Output directory for CHALM-array transformation.
  --prefix PREFIX       Prefix for CHALM-array transformation output.
  -c COEF               CHALM-array coefficients provided in .pkl format.
  --cpg-info CPG_INFO_PATH
                        CpG loci and their infomation in training data.
  -t THREAD
  --ref {hg38,hg19}
  --beadchip {450k,EPIC,HM450,850k}
  --cl                  Output confidence level for each CpG.
  --probe               Use probe ID for output index, otherwise use CpG loci.
  --verbose VERBOSE     Print CHALM-array transformation progress.
  --camda               Output CAMDA.
  --filter FILTER       The minimum confidence level to keep the transformed CHALM-array, if assigned. Overwriting --cl.
  --replace REPLACE     If assigned, replace the filtered and missing value with the assigned value. Otherwise, replace the values with NaNs.

Example:
    1. Perform CHALM-array transformation for EPIC beadchip data aligned with hg38
    python ./CHALM_array.py -i ./test_methylation.tsv -o ./ --ref hg38 --beadchip EPIC
    2. Perform CHALM-array transformation for EPIC beadchip data aligned with hg38. Use "CA_test" as prefix for output files.
    python ./CHALM_array.py -i ./test_methylation.tsv -o ./ --ref hg38 --beadchip EPIC --prefix CA_test
    3. Perform CHALM-array transformation for EPIC beadchip data aligned with hg38. Plus, output probeID instead of CpG loci.
    python ./CHALM_array.py -i ./test_methylation.tsv -o ./ --ref hg38 --beadchip EPIC --probe
    4. Perform CHALM-array transformation for EPIC beadchip data aligned with hg38. Plus, output CAMDA.
    python ./CHALM_array.py -i ./test_methylation.tsv -o ./ --ref hg38 --beadchip EPIC --camda
    5. Perform CHALM-array transformation for EPIC beadchip data aligned with hg38. Plus, output confidence levels for the transformation.
    python ./CHALM_array.py -i ./test_methylation.tsv -o ./ --ref hg38 --beadchip EPIC --cl
    6. Perform CHALM-array transformation for EPIC beadchip data aligned with hg38. Plus, filter transformation results based on confidence levels(minimun confidence level=0.1).
    python ./CHALM_array.py -i ./test_methylation.tsv -o ./ --ref hg38 --beadchip EPIC --filter 0.1
    7. Perform CHALM-array transformation for EPIC beadchip data aligned with hg38. Plus, filter transformation results based on confidence levels and replace filtered and missing values with "-1".
    python ./CHALM_array.py -i ./test_methylation.tsv -o ./ --ref hg38 --beadchip EPIC --filter 0.1 --replace -1


- Replace "test_methylation.tsv" as your methylation matrix .tsv file, with first row as sample names and first columns as probe locations.