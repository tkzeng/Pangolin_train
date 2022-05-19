import sys

fin = sys.argv[1]
fout = sys.argv[2]

all_sites = {}
sites_to_keep = {}

for l in open(fin):
    l = l.split('\t')
    try:
        chrom, site, partners = l[0], int(l[1]), l[-2]
    except:
        continue # is a header line
    #nreads = int(l[4]) + int(l[5]) # alpha_count + beta1_count
    #if nreads < 10:
    #    continue

    greater = []
    for x in partners.split(','):
        pos, counts = x.split(':')
        pos, counts = int(pos.strip("{}: ")), int(counts.strip("{}: "))
        if pos > site:
            try: all_sites["%s:%s" % (chrom,pos)][0] += counts
            except KeyError: all_sites["%s:%s" % (chrom,pos)] = [counts, 0]
        elif pos < site:
            try: all_sites["%s:%s" % (chrom,pos)][1] += counts
            except KeyError: all_sites["%s:%s" % (chrom,pos)] = [0, counts]
        assert pos != site

fout = open(fout, 'w')
for site in all_sites:
    ct1, ct2 = all_sites[site]
    assert ct1 > 0 or ct2 > 0
    if ct1 == 0 or ct2/ct1 >= 10:
        sites_to_keep[site] = 'r'
        fout.write("%s\t%s\n" % (site, sites_to_keep[site]))
    elif ct2 == 0 or ct1/ct2 >= 10:
        sites_to_keep[site] = 'l'
        fout.write("%s\t%s\n" % (site, sites_to_keep[site]))
    fout.flush()

print("to keep", len(sites_to_keep))
print("all", len(all_sites))
