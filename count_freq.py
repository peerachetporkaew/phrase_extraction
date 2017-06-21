import nltk,sys

if __name__ == "__main__":
    fname = sys.argv[1]
    output = sys.argv[2]
    fp = open(fname,"r")
    fo = open(output,"w")
    line = fp.readline()
    final = nltk.FreqDist()
    i = 0
    while line:
        print i
        i += 1
        item = line.strip().split("\t")
        #print item
        if len(item) > 1:
            final.update([(item[0],item[1])])
        line = fp.readline()
    for k in final:
        fo.writelines(k[0] + "\t" + k[1] + "\t" + str(final[k]) + "\n")
    fo.close()
