python test_ctc_loss.py --input-size 6  --target-size 3 --batch-size 1 --reduction-type sum
python test_ctc_loss.py --input-size 6  --target-size 3 --batch-size 1 --reduction-type mean
python test_ctc_loss.py --input-size 9  --target-size 5 --batch-size 1 --reduction-type sum
python test_ctc_loss.py --input-size 9  --target-size 5 --batch-size 1 --reduction-type mean
python test_ctc_loss.py --input-size 12 --target-size 9 --batch-size 1 --reduction-type mean
python test_ctc_loss.py --input-size 12 --target-size 9 --batch-size 1 --reduction-type sum

python test_ctc_loss.py --input-size 6  --target-size 3 --batch-size 2 --reduction-type sum
python test_ctc_loss.py --input-size 6  --target-size 3 --batch-size 2 --reduction-type mean
python test_ctc_loss.py --input-size 12 --target-size 9 --batch-size 2 --reduction-type mean
python test_ctc_loss.py --input-size 12 --target-size 9 --batch-size 2 --reduction-type sum

python test_ctc_loss.py --input-size 6  --target-size 3 --batch-size 3 --reduction-type sum
python test_ctc_loss.py --input-size 6  --target-size 3 --batch-size 3 --reduction-type mean
python test_ctc_loss.py --input-size 12 --target-size 9 --batch-size 3 --reduction-type mean
python test_ctc_loss.py --input-size 12 --target-size 9 --batch-size 3 --reduction-type sum
