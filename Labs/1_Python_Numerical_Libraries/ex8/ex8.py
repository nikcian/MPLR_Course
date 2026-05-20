import sys
import numpy

class Competitor:
    def __init__(self, name, surname, country):
        self.name = name
        self.surname = surname
        self.country = country

def load_competitors(fname):

    f=open(fname)
    n = int(f.readline())
    lScores = []
    lCompetitors = []
    for idx, line in enumerate(f):
        name, surname, country = line.split()[0:3]
        scores = line.split()[3:]
        scores = [float(i) for i in scores]
        lScores.append(scores)
        lCompetitors.append(Competitor(name, surname, country))
    scoreMatrix = numpy.array(lScores)
    # numpy.array converts a list of lists to a 2D array
    # Alternatively, we can create 1D arrays for each competitor, lScores.append(numpy.array(scores)) in the for loop, and then create the scoreMatrix as scoreMatrix = numpy.hstack(lScores)
    # Another alternative would be to create a zero-matrix at the beginning of the funcion, scoreMatrix = numpy.zeros((n, n)), and then fill it inside the loop
    f.close()
    return lCompetitors, scoreMatrix


if __name__ == '__main__':

    lCompetitors, mScores = load_competitors(sys.argv[1])

    # Compute the score of each competitor. We make use of reduction operators to sum the matrix scores column-wise and to find the max and min of each competitor

    totScoreArray = mScores.sum(1) # We sum over axis 1 -> the result is the sum of the columns of the matrix as a 1D array
    maxScoreArray = mScores.max(1) # We look for the maximum over axis 1 -> we find the maximum of each row (maximum is computed over columns). We also get a 1D array
    minScoreArray = mScores.min(1)

    # Since the three arrays are all 1D, we can simply compute the competitor scores using vector arithmetic
    compScoresArray = totScoreArray - maxScoreArray - minScoreArray # 1D array with scores for each competitor

    # Alternatively, we could sort the elements of the matrix column-wise, i.e., sort each row by using the sort function with axis = 1
    # We first create a copy of the array, since sort happens in-place
    alternativeSolution = numpy.array(mScores)
    alternativeSolution.sort(axis = 1)
    alternativeCompScoresArray = alternativeSolution[:, 1:-1] # We remove the first and last column of the partially matrix
    alternativeCompScoresArray = alternativeCompScoresArray.sum(1) # We sum the remaining values

    # Print the two solutions
    print ("Debug - compScoresArray vs alternativeCompScoresArray:")
    print (compScoresArray)
    print (alternativeCompScoresArray)
    print ()
    
    # To find the best three we can do as in ex1 with a for loop, but we can also exploit numpy functionalities. The argsort function provides the indices that would sort the array:
    # idx = numpy.argsort(compScoresArray) -> compScoresArray[idx] corresponds to the sorted array
    idx = numpy.argsort(compScoresArray)
    lBest = idx[-3:][::-1] # We take the last 3 (sort is in ascending order) and we reverse the order (the highest score is the last element of the sorted array)
    
    print ("Final ranking:")
    for i in lBest:
        print ('%d: %s %s - Score: %.1f' % (i+1, lCompetitors[i].name, lCompetitors[i].surname, compScoresArray[i]))
    print()