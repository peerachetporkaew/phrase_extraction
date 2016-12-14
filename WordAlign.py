#-*- coding:utf-8 -*-
# Python Word Alignment for Lattice Combination by Peerachet Porkaew
# 2016-09-30

#Process
#1. Create e2e         : LoadWordTranslationProbability, CreateWordAlignmentProbability
#2. Forward Alignment  : ForwardWordAlignment
#3. Backward Alignment : BackwardWordAlignment
#4. GlowDiag             ==> Port from C++
#5. MultiLattice       : Generate Phrase Table from multiple hypothesis
#6. Gen Feature        : Generate Score for each phrase pair
#7. Decoding           : Decode
#8. Hyperparameter Tunning
#9. Configuration
#10. Document          : Follow the Standard Template

#TODO : Add Distortion Model to Word Alignment, IHMM Implementation

"""

Changes :

2016-12-14 : Fix Phrase Extraction BUGS

"""

EPSILON = 0.000001 #a very small number for smoothing the probability (in case of unk or oov)
DEBUG_LEVEL = 1

def LoadWordTranslationProbability(filename): #PC01
    """
    Input : word translation prob. file (lex.e2f or lex.f2e)
    Output : table ,table[x][y] = P(x|y)
    """
    table = {}

    fp = open(filename,"r").readlines()
    for line in fp:
        x = line.strip()
        if len(x) == 0: #Check empty line
            continue;
        item = line.strip().split(" ")
        if table.has_key(item[0]):
            table[item[0]][item[1]] = float(item[2])
        else:
            table[item[0]] = {item[1] : float(item[2])}
    return table

def CreateWordAlignmentProbability(pe2f,pf2e): #PC02
    """
    Objective : Create F2F for word alignment between backbone and hypothesis (same language)
    Input     : p(e|f),p(f|e)
    Output    : Table of P(f1->f2) ==> table[f1][f2] ==> Semantic Similarity ??
    Process   : Using 2-nested for loop for each f
              : !!! MEMORY PROBLEM !!!
    """
    #Initialize table
    table = {}
    allWord = pf2e.keys()
    _iteration = 0
    for word1 in allWord: #Loop over target word (f)
        _iteration += 1
        if DEBUG_LEVEL == 1 and _iteration%100 == 0:
            print "Iteration : ",_iteration
        table[word1] = {} #Init table
        allword_e = pf2e[word1].keys() #Find all possible e from f->e table
        for word2 in allword_e: #Loop over source word (e)
            allword_f = pe2f[word2].keys() #Find all possible f from e->f table
            for word3 in allword_f: #Loop over allword_f
                if table[word1].has_key(word3):
                    table[word1][word3] += pf2e[word1][word2]*pe2f[word2][word3]
                else:
                    table[word1][word3] = pf2e[word1][word2]*pe2f[word2][word3]

    return table

def CreateWordAlignmentProbability_Filtered(pe2f,pf2e,sent_F1,sent_F2):
    """
    Objective : Same as CreateWordAlignmentProbability but only for one sentence pair
    Input     : p(e|f),p(f|e) , sentence F1 and F2 (F1 is the backbone sentence)
    Output    : Table of P(f1->f2)
    Process   : Keep only f1 in sent_F1, and f2 in sent_F2
    """
    f1_wordlist = set(sent_F1.split(" "))
    #print f1_wordlist
    f2_wordlist = set(sent_F2.split(" "))

    #Initialize table
    table = {}
    allWord = list(f1_wordlist)
    _iteration = 0
    for word1 in allWord: #Loop over target word (f)
        _iteration += 1

        if DEBUG_LEVEL == 1 and _iteration%100 == 0:
            print "Iteration : ",_iteration

        table[word1] = {} #Init table

        if not pf2e.has_key(word1):
            continue

        allword_e = pf2e[word1].keys() #Find all possible e from f->e table
        for word2 in allword_e: #Loop over source word (e)
            allword_f = pe2f[word2].keys() #Find all possible f from e->f table
            for word3 in allword_f: #Loop over allword_f
                if word3 in f2_wordlist:
                    if table[word1].has_key(word3):
                        table[word1][word3] += pf2e[word1][word2]*pe2f[word2][word3]
                    else:
                        table[word1][word3] = pf2e[word1][word2]*pe2f[word2][word3]
    return table

def CreateWordAlignmentProbability_BatchFiltered(pe2f,pf2e,backbone_wordList,hypo_allWordList):
    """
    TODO : Implement Batch Filter
    Objective : Create Word Translation Prob. for backbone and all hypotheses.
    Input : backbone_wordList, hypo_allwordList (set of word in hypotheses side).
    Output : Forward probability P(f1->f2)
    """



    return None

def ForwardWordAlignment(forwardProb,backwardProb,backbone,hypo):
    backbone_wordList = backbone.strip().split(" ")
    hypo_wordList = hypo.strip().split(" ")
    alignment = [] #2D zero-one alignment matrix
    for i in range(len(backbone_wordList)): # backbone_wordList[i] = the i-th word of backbone

        if not forwardProb.has_key(backbone_wordList[i]): #backbone_wordList[i] is <unk>.
            alignment += [[0 for i in range(len(hypo_wordList))]]
            continue;

        max_prob = 0.0
        max_index = 0
        align_temp = [0] * len(hypo_wordList)
        align_temp[0] = 1
        for j in range(0,len(hypo_wordList)):
            if forwardProb[backbone_wordList[i]].has_key(hypo_wordList[j]):
                if forwardProb[backbone_wordList[i]][hypo_wordList[j]] > max_prob:
                    align_temp[max_index] = 0
                    align_temp[j] = 1
                    max_prob = forwardProb[backbone_wordList[i]][hypo_wordList[j]]
                    max_index = j

        alignment += [align_temp]

    return alignment

def ForwardWordAlignment_withDistortion(forwardProb,backbone,hypo):
    """
    TODO: Add Distortion Model. Why Distortion Model is needed ??
    """
    return None

def BackwardWordAlignment(forwardProb,backwardProb,backbone,hypo):
    backbone_wordList = backbone.strip().split(" ")
    hypo_wordList = hypo.strip().split(" ")

    alignment = [[0 for i in range(len(hypo_wordList))] for j in range(len(backbone_wordList))]
    for i in range(len(hypo_wordList)):
        if not backwardProb.has_key(hypo_wordList[i]): #hypo_wordList[i] is <unk>.
            continue;

        max_prob = 0.0
        max_index = 0

        for j in range(len(backbone_wordList)):
            #print i,j
            if backwardProb[hypo_wordList[i]].has_key(backbone_wordList[j]):
                if backwardProb[hypo_wordList[i]][backbone_wordList[j]] > max_prob:
                    max_prob = backwardProb[hypo_wordList[i]][backbone_wordList[j]]
                    max_index = j

        alignment[max_index][i] = 1
    return alignment

def IntersecWordAlignment(alignment1,alignment2):
    """
    Word Alignment Intersection
    TODO : Implement dimension validation
    """
    nrow = len(alignment1)
    ncol = len(alignment1[0])

    alignment = [[0 for i in range(0,ncol)] for j in range(0,nrow)]
    for r in range(0,nrow):
        for c in range(0,ncol):
            alignment[r][c] = alignment1[r][c] * alignment2[r][c] #For better performance
    return alignment

def UnionWordAlignment(alignment1,alignment2):
    """
    Word Alignment Union
    TODO : Implment dimension validation
    """
    nrow = len(alignment1)
    ncol = len(alignment1[0])

    alignment = [[0 for i in range(0,ncol)] for j in range(0,nrow)]
    for r in range(0,nrow):
        for c in range(0,ncol):
            if alignment1[r][c] == 1 or alignment2[r][c] == 1:
                alignment[r][c] = 1
    return alignment

def PrintAlignment(alignment):
    """
    Show the alignment in 0,1 fashion
    """
    for i in range(len(alignment)):
        for j in range(len(alignment[i])):
            print str(alignment[i][j]) + " ",
        print ""

def GrowDiagFinal(alnIntersec,alnUnion):
    """
    Implement Grow Algorithm port from C++
    TODO : Implment dimension validation
    """

    nrow = len(alnIntersec)
    ncol = len(alnIntersec[0])

    alignment = alnIntersec[:][:]

    for i in range(nrow):
        for j in range(ncol):
            if alnIntersec[i][j] == 0:
                continue
            if i >= 1:
                if alignment[i-1][j] != 1 and alnUnion[i-1][j] == 1:
                    alignment[i-1][j] = 1

                if j >= 1 and alignment[i-1][j-1] != 1 and alnUnion[i-1][j] == 1:
                    alignment[i-1][j-1] = 1

                if j < ncol-1 and alignment[i-1][j+1] != 1 and alnUnion[i-1][j+1] == 1:
                    alignment[i-1][j+1] = 1

            if j >= 1 and alignment[i][j-1] != 1 and alnUnion[i][j-1] == 1:
                alignment[i][j-1] = 1

            if j < ncol-1 and alignment[i][j+1] != 1 and alnUnion[i][j+1] == 1:
                alignment[i][j+1] = 1

            if i < nrow-1:
                if alignment[i+1][j] != 1 and alnUnion[i+1][j] == 1:
                    alignment[i+1][j] = 1
                if j > 1 and alignment[i+1][j-1] != 1 and alnUnion[i+1][j-1] == 1:
                    alignment[i+1][j-1] = 1
                if j < ncol-1 and alignment[i+1][j+1] != 1 and alnUnion[i+1][j+1] == 1:
                    alignment[i+1][j+1] = 1

    return alignment

def test_PhraseExtraction():

    backbone = "1 2 3 4 5 6 7".split()
    hypo = "A B C D E F G H I".split()
    alignment = [[1,0,0,0,0,0,0,0,0],
                 [0,1,0,0,0,0,0,0,0],
                 [0,0,0,0,0,1,1,0,0],
                 [0,0,1,1,1,0,0,0,0],
                 [0,0,0,0,0,0,0,0,1],
                 [0,0,0,0,0,0,0,1,0],
                 [0,0,0,0,0,0,0,1,0]]

    output = PhraseExtraction(backbone,hypo,alignment)
    for phrase in output:
        print phrase


def PhraseExtraction(backbone,hypo,alignment):
    """
    2016-10-09  Extract Phrase Pair
    Written by Peerachet Porkaew
    Process from Top to Bottom and Left to Right with Double-Stack Approach
    Start from top-left corner
    Pseudocode ::

    Alignment ==>

     0 1 2 3
0    1 0 1 0
1    0 1 0 0
2    0 0 0 1
3    0 0 0 1

    curr = 0
    curc = 0

    get rec_x1, rec_x2 = 0,2

    col_stack = range(rec_x1,rec_x2)

    col_stack = [0,1,2] - Initialize col_stack
    row_stack = []

    visited_col = [] - store visited column
    visited_row = [] - store visited row
    Do Loop Until all alignment are processed
        while len(col_stack) != 0 and len(row_stack) != 0:
            foreach c in col_stack:
                max_r = get_last_alignment along that column (return row index)
                for x in range(curr,max_r+1):
                    if x not in visited_row:
                        row_stack += [x]
                visited_col += [c]
                col_stack.remove(c)

            foreach r in row_stack:
                max_c = get_last_alignment point along that row (return col index)
                for x in range(curc,max_c+1):
                    if x not in visited_col:
                        col_stack += [x]
                visited_row += [r]
                row_stack.remove(r)

        rec_x2 = max(visited_col)
        rec_y2 = max(visited_row)

        update curr, curc

    """

    phrase_pair = []

    nrow = len(alignment)
    ncol = len(alignment[0])

    curr = 0 #current row
    curc = 0 #current column

    Gvisited_col = []
    Gvisited_row = []


    while curr < nrow and curc < ncol:
        #Get new pair from current position (row,col)
        rec_x1 = curc
        rec_x2 = curc

        rec_y1 = curr
        rec_y2 = curr

        #find left alignment point
        while alignment[curr][rec_x1] == 0:
            rec_x1 += 1
            if rec_x1 >= ncol:
                break

        if rec_x1 >= ncol: #this backbone word is not aligned with any hypothesis word
            """
            Continue next row
            """
            #print "NO ALIGNMENT AT THIS ROW !"
            curr += 1
            continue

        rec_x2 = rec_x1+1

        #find right alignment point
        rec_x2 = ncol-1

        while alignment[curr][rec_x2] == 0:
            rec_x2 -= 1
            if rec_x2 == rec_x1:
                break

        col_stack = range(rec_x1,rec_x2+1)

        """
        if rec_y1 == 2:
            print "COL STACK"
            print rec_x2
            print col_stack
        """

        row_stack = []

        visited_col = []
        visited_row = []

        while len(col_stack) != 0 or len(row_stack) != 0:
            for c in col_stack:
                #max_r = get_last_alignment along that column
                max_r = curr
                r_temp = curr
                while r_temp < nrow:
                    if alignment[r_temp][c] == 1:
                        max_r = r_temp
                    r_temp += 1

                """
                if rec_y1 == 2:
                    print "C,max_R"
                    print c,max_r
                """

                for x in range(curr,max_r+1):
                    if x not in visited_row:
                        row_stack += [x]

                visited_col += [c]
                col_stack.remove(c)

            """
            if rec_y1 == 2:
                print "ROW STACK"
                print row_stack
            """

            for r in row_stack:
                #max_c = get_last_alignment point along that row
                max_c = rec_x1
                c_temp = rec_x1
                while c_temp < ncol:
                    if alignment[r][c_temp] == 1:
                        max_c = c_temp
                    c_temp += 1

                for x in range(rec_x1,max_c+1):
                    if x not in visited_col:
                        col_stack += [x]
                visited_row += [r]
                row_stack.remove(r)
            """
            if rec_y1 == 2:
                print "COL STACK"
                print col_stack
            """
        rec_x2 = max(visited_col)
        rec_y2 = max(visited_row)

        min_x1 = min(visited_col)
        min_y1 = min(visited_row)

        #print curr,curc,"-",rec_y2,rec_x2
        phrase_pair += [((min_y1,min_x1),(rec_y2,rec_x2))]

        """
        if rec_y1 == 2:
            print "LINE 2"
            print ((min_y1,min_x1),(rec_y2,rec_x2))
        """

        Gvisited_row += visited_row
        Gvisited_col += visited_col

        curr = curr + 1
        curc = 0
        while curr in Gvisited_row or curc in Gvisited_col:
            while curr in Gvisited_row:
                curr += 1
            curc = 0
            while curc in Gvisited_col:
                curc += 1

    output = []
    for pair in phrase_pair:
        bone = [backbone[x] for x in range(pair[0][0],pair[1][0]+1)]
        hypo1 = [hypo[x] for x in range(pair[0][1],pair[1][1]+1)]
        output += [(" ".join(bone)," ".join(hypo1))]

    return output


def MainFunc():
    PE2F = LoadWordTranslationProbability("lex.e2f") #PFE = P(F|E)
    PF2E = LoadWordTranslationProbability("lex.f2e") #PFE = P(E|F)

    Backbone = "keep in mind"
    Hypo1    = "kept at your head"
    PF2F_Forward  = CreateWordAlignmentProbability_Filtered(PE2F,PF2E,Backbone,Hypo1)
    PF2F_Backward = CreateWordAlignmentProbability_Filtered(PE2F,PF2E,Hypo1,Backbone)

    print "Forward  : ",PF2F_Forward
    print "Backward : ",PF2F_Backward

    aln = ForwardWordAlignment(PF2F_Forward,PF2F_Backward,Backbone,Hypo1)
    aln2 = BackwardWordAlignment(PF2F_Forward,PF2F_Backward,Backbone,Hypo1)

    print "FORWARD"
    PrintAlignment(aln)

    print "BACKWARD"
    PrintAlignment(aln2)

    print "INTERSECTION"
    aln3 = IntersecWordAlignment(aln,aln2)
    PrintAlignment(aln3)

    print "UNION"
    aln4 = UnionWordAlignment(aln,aln2)
    PrintAlignment(aln4)

    print "GROW"
    aln5 = GrowDiagFinal(aln3,aln4)
    PrintAlignment(aln5)

    output = PhraseExtraction(Backbone.split(" "),Hypo1.split(" "), aln5)

    print aln5

    print output

    print "Finished"
    print ""
    print ""


def GenMultiLattice(PE2F,PF2E,BackBone,AllHypo):
    """
    Generate Multi-Lattice by Peerachet Porkaew 2016-10-12
    PE2F : Source-to-Target Word Translation Probability
    PF2E : Target-to-Source Word Translation Probability

    Backbone : String of backbone
    AllHypo  : [ [("system1 hyp1 . .", 0.8),
                   "systen1 hyp2 . .", 0.74)],

                 [("system2 hyp1 . .", 0.81),
                   "systen2 hyp2 . .", 0.75)]
                ]

                AllHypo is an array which stores all hypothesis of every system with score.

                AllHypo[sysid][hypoid][0] => hypothesis (target sentence)
                AllHypo[sysid][hypoid][1] => score


    Output : Phrase Pair with score (feature)

    Pseudocode :

        phrase_table = []
        nsys = number of system

        1. Create F2F word Translation Table (pf2f_forward, pf2f_backward)

        2. For each system :
                sid = system id
                For each Hypothesis => (hypo,score)
                    k = rank of current hypo (in n-best list)
                    rank_bias = 1 / (k + 1) ==> as k increase , rank_bias decrease.
                    alignment = GetFinalAlignment(backbone,hypo,pf2f_forward,pf2f_backward)
                    phrase_pair = PhraseExtraction(backbone,hypo,alignment)

                    feature_score = GenScore(sid,nsys,score,rank_bias)

                    foreach phrase_pair => pair
                        phrase_table.add(pair,feature_score)

        3. Return phrase_table

    """

    phrase_table = [] #phrase_table = [(backbone , target , [0.8 0 0])]  the first of 3 systems

    #[0.8 0 0] These three number is corresponding to each system, 0.8 is the confident score of system #1.

    nsys = len(AllHypo)

    """
    Create F2F word Translation Table (pf2f_forward, pf2f_backward)
    Pseudocode :
        1. List all word in AllHypo
        2. Create PF2F (backbone and AllHypo), pf2f_forward, pf2f_backward
    """

    #Concat all hypothesis
    allhypo_temp = ""
    for system in AllHypo:
        for h in system:
            allhypo_temp += " " + h[0]

    #Backbone and allhypo_temp both are string (without splited)
    PF2F_Forward  = CreateWordAlignmentProbability_Filtered(PE2F,PF2E,Backbone,allhypo_temp)
    PF2F_Backward = CreateWordAlignmentProbability_Filtered(PE2F,PF2E,allhypo_temp,Backbone)

    sid = 0
    for system in AllHypo:
        k = 0.0
        for h in system:
            Hypo1 = h[0]
            score = h[1]
            alnF = ForwardWordAlignment(PF2F_Forward,PF2F_Backward,Backbone,Hypo1)
            alnB = BackwardWordAlignment(PF2F_Forward,PF2F_Backward,Backbone,Hypo1)
            alnI = IntersecWordAlignment(alnF,alnB)
            alnU = UnionWordAlignment(alnF,alnB)
            alnG = GrowDiagFinal(alnI,alnU)
            print "ALIGNMENT : "
            print alnG
            output = PhraseExtraction(Backbone.split(" "),Hypo1.split(" "), alnG)
            feature_score = GenScore(sid,nsys,score,k)
            print "PAIR"
            for pair in output:
                print pair
                phrase_table += [(pair[0],pair[1],feature_score)]
            k += 1.0
        sid += 1

    #print phrase_table
    return phrase_table

def GenScore(sid,nsys,score,k):
    feature_score = [] #log linear score
    rank_bias = 1.0 / (k + 1.0)
    for i in range(0,nsys):
        if i != sid:
            feature_score += [0]
        else:
            feature_score += [score + rank_bias]

    return feature_score

if __name__ == "__main__":

    #MainFunc()

    test_PhraseExtraction()

    exit()

    allhypo = [ [ ("keep in your mind", 0.8),
                  ("kept in your head", 0.74)],

                [ ("remember it", 0.81),
                  ("keep it in your mind", 0.75)]
              ]

    PE2F = LoadWordTranslationProbability("lex.e2f") #PFE = P(F|E)
    PF2E = LoadWordTranslationProbability("lex.f2e") #PFE = P(E|F)

    Backbone = "keep in your mind"
    phrase_table = GenMultiLattice(PE2F,PF2E,Backbone,allhypo)

    for phrase in phrase_table:
        print phrase
