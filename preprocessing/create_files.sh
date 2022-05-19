
./grab_sequence.sh splice_table_Human.txt ../GRCh38.primary_assembly.genome.fa canonical_sequence_Human.txt
./grab_sequence.sh splice_table_Human.test.txt ../GRCh38.primary_assembly.genome.fa canonical_sequence_Human.test.txt
./grab_sequence.sh splice_table_Mouse.txt ../GRCm38.primary_assembly.genome.fa canonical_sequence_Mouse.txt
./grab_sequence.sh splice_table_Macaque.txt ../Macaca_mulatta.Mmul_10.dna.toplevel.fa canonical_sequence_Macaque.txt
./grab_sequence.sh splice_table_Rat.txt ../Rattus_norvegicus.Rnor_6.0.dna.toplevel.fa canonical_sequence_Rat.txt

# TRAINING
python create_datafile.py train all 1 Human
python create_datafile.py train all 2 Human
python create_datafile.py train all 3 Human
python create_datafile.py train all 4 Human
python create_datafile.py train all 5 Human
python create_datafile.py train all 6 Human

python create_dataset_multi.py train all Human 0 1
python create_dataset_multi.py train all Human 1 2
python create_dataset_multi.py train all Human 1 3
python create_dataset_multi.py train all Human 1 4
python create_dataset_multi.py train all Human 1 5
python create_dataset_multi.py train all Human 1 6

rm datafile*

python create_datafile.py train all 1 Mouse
python create_datafile.py train all 2 Mouse
python create_datafile.py train all 3 Mouse
python create_datafile.py train all 4 Mouse
python create_datafile.py train all 5 Mouse
python create_datafile.py train all 6 Mouse

python create_dataset_multi.py train all Mouse 1 1
python create_dataset_multi.py train all Mouse 1 2
python create_dataset_multi.py train all Mouse 1 3
python create_dataset_multi.py train all Mouse 1 4
python create_dataset_multi.py train all Mouse 1 5
python create_dataset_multi.py train all Mouse 1 6

rm datafile*

python create_datafile.py train all 1 Macaque
python create_datafile.py train all 2 Macaque
python create_datafile.py train all 3 Macaque
python create_datafile.py train all 4 Macaque
python create_datafile.py train all 5 Macaque
python create_datafile.py train all 6 Macaque

python create_dataset_multi.py train all Macaque 1 1
python create_dataset_multi.py train all Macaque 1 2
python create_dataset_multi.py train all Macaque 1 3
python create_dataset_multi.py train all Macaque 1 4
python create_dataset_multi.py train all Macaque 1 5
python create_dataset_multi.py train all Macaque 1 6

rm datafile*

python create_datafile.py train all 1 Rat
python create_datafile.py train all 2 Rat
python create_datafile.py train all 3 Rat
python create_datafile.py train all 4 Rat
python create_datafile.py train all 5 Rat
python create_datafile.py train all 6 Rat

python create_dataset_multi.py train all Rat 1 2
python create_dataset_multi.py train all Rat 1 3
python create_dataset_multi.py train all Rat 1 4
python create_dataset_multi.py train all Rat 1 5
python create_dataset_multi.py train all Rat 1 6

rm datafile*

# TESTING
python create_datafile.py test all 1 Human.test
python create_datafile.py test all 2 Human.test
python create_datafile.py test all 3 Human.test
python create_datafile.py test all 4 Human.test

python create_dataset_multi.py test 1 Human.test 0 1
python create_dataset_multi.py test 1 Human.test 1 2
python create_dataset_multi.py test 1 Human.test 1 3
python create_dataset_multi.py test 1 Human.test 1 4

rm datafile*
