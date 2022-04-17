
chroms = {}
for l in open("Macaca_mulatta.Mmul_10.dna.toplevel.fa.chrom_sizes"):
    chrom, length = l.split()
    chroms[chrom] = int(length)

lines = []
for l in open("splice_table_Macaque.txt"):
    l = l.split()
    chrom, start, end = l[2], int(l[4]), int(l[5])
    if start-5001 < 0 or end+5000 > chroms[chrom]:
        continue
    lines.append('\t'.join(l))

fout = open("splice_table_Macaque.txt", 'w')
for l in lines:
    fout.write(l+'\n')

chroms = {}
for l in open("Rattus_norvegicus.Rnor_6.0.dna.toplevel.fa.chrom_sizes"):
    chrom, length = l.split()
    chroms[chrom] = int(length)

lines = []
for l in open("splice_table_Rat.txt"):
    l = l.split()
    chrom, start, end = l[2], int(l[4]), int(l[5])
    if start-5001 < 0 or end+5000 > chroms[chrom]:
        continue
    lines.append('\t'.join(l))

fout = open("splice_table_Rat.txt", 'w')
for l in lines:
    fout.write(l+'\n')
