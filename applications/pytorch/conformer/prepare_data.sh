# Install espnet dependency package
sudo apt-get install cmake sox libsndfile1-dev ffmpeg flac libfreetype6-dev libpng-dev bc;
git clone -b v.0.10.1 --depth 1 https://github.com/espnet/espnet.git;
# If 2-10 lines are redundant, delete
sed -i '2,10d' ./espnet/egs2/aishell/asr1/local/path.sh
export CONFORMER_ROOT=`pwd`
# Enter the ‘espnet/tools/’ directory folder, then run setup_anaconda.sh to setup and activate the virtual environment of espnet's anaconda 
cd espnet/tools/ && ./setup_anaconda.sh anaconda espnet 3.8 && make CPU_ONLY=0;
# Enter the ‘aishell’ directory folder, then execut run.sh to generate the aishell's feature, target and other intermediate result file data; when the file is generated, that is, step 10 in the script, exit the running script at this time
cd $CONFORMER_ROOT/espnet/egs2/aishell/asr1/ && ./run.sh --use_ngram false --use_lm false --stop_stage 10 --asr_args "--write_collected_feats true"
cd $CONFORMER_ROOT
