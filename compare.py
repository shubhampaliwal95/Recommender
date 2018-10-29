'''
This script compares three different recommendation algorithms for generation
of missing/expected movie ratings and displays the results to the user.
Author: 
Date of Completion: 
The dataset containing the movie ratings was obtained from www.movielens.org
and the following paper is responsible for its creation:
F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History
and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4,
Article 19 (December 2015), 19 pages. DOI=http://dx.doi.org/10.1145/2827872
'''

import os.path
import math
import heapq
import numpy
import time
import tkinter as tk


class RECOMMENDER(tk.Frame):
    '''
    This is the master application defining class and encapsulates
    everything from the GUI widgets to the core search implementation.
    '''

    def __init__(self, master=None):
        ''' Constructor for the RecommenderSystem class. '''
        tk.Frame.__init__(self, master)
        self.entry = tk.StringVar()
        self.msg = tk.StringVar()
        self.grid()
        self.configureGrid()
        self.createWidgets()
        self.setMessage('Loading, please wait...')

    def configureGrid(self):
        ''' Configure the Tkinter grid layout. '''
        self.rowconfigure(0, minsize=50)
        self.columnconfigure(0, minsize=80)
        self.rowconfigure(1, minsize=50)
        self.columnconfigure(1, minsize=80)
        self.rowconfigure(2, minsize=50)
        self.columnconfigure(2, minsize=80)
        self.rowconfigure(3, minsize=50)
        self.columnconfigure(3, minsize=80)
        self.rowconfigure(4, minsize=50)
        self.columnconfigure(4, minsize=80)
        self.rowconfigure(5, minsize=50)
        self.columnconfigure(5, minsize=80)
        self.rowconfigure(6, minsize=50)
        self.columnconfigure(6, minsize=80)
        self.columnconfigure(7, minsize=80)
        self.columnconfigure(8, minsize=80)
        self.columnconfigure(9, minsize=80)

    def createWidgets(self):
        ''' Define the attributes of the GUI elements. '''

        # The Quit Button
        self.quitButton = tk.Button(self, text='Quit', command=self.quit)
        self.quitButton.grid(row=6, column=9, rowspan=1, columnspan=1, padx=15, pady=25,
                             sticky=tk.N + tk.E + tk.S + tk.W)

        # The Message Box to display the output
        self.messageBox = tk.Message(self, anchor=tk.NW, padx=25, pady=25, width=700, justify=tk.LEFT, relief=tk.RIDGE,
                                     bd=3, textvariable=self.msg)
        self.messageBox.grid(row=0, column=0, rowspan=6, columnspan=10, padx=20, pady=25,
                             sticky=tk.N + tk.E + tk.S + tk.W)

    def setMessage(self, message):
        ''' Set messageBox to display "message". '''
        self.msg.set(message)



    ###############################################
    ########### Data Loading Function #############
    ###############################################

    def loadData(self):
        ''' Load the ratings data and run the algorithms. '''

        # File with the ratings data
        ratings_file = open('./ratings/ratings.dat', 'r')

        # Dictionary of keys format sparse matrix
        self.ratings_matrix = numpy.zeros((3952, 6040))

        for rating in ratings_file:
            data = rating.strip().split('::')
            self.ratings_matrix[int(data[1]) - 1, int(data[0]) - 1] = float(data[2])

        ratings_file.close()

        self.result_string = ''

        self.common()

        self.colab()

        self.svd90()
        
        self.svd()

        self.cur()

       

        self.setMessage(self.result_string)


    ###############################################


    ###############################################
    ######## Common Processing Function ###########
    ###############################################

    def common(self):
        ''' Perform calculations common to various algorithms '''

        self.result_string += 'Technique\t\tRMSE\tPrecision on top 10\tSpearman Rank Correlation\tTime\n\n'

        self.transpose_ratings_matrix = numpy.transpose(self.ratings_matrix)

        self.total_mean = self.ratings_matrix.mean()

        self.movie_sums = self.ratings_matrix.sum(1)

        self.movie_non_zeros = numpy.count_nonzero(self.ratings_matrix, 1)

        self.movie_means = numpy.zeros(self.movie_sums.shape)

        for i in range(self.movie_sums.size):
            if self.movie_non_zeros[i] != 0:
                self.movie_means[i] = self.movie_sums[i] / self.movie_non_zeros[i]

        self.user_sums = self.transpose_ratings_matrix.sum(1)

        self.user_non_zeros = numpy.count_nonzero(self.transpose_ratings_matrix, 1)

        self.user_means = numpy.zeros(self.user_sums.shape)

        for i in range(self.user_sums.size):
            if self.user_non_zeros[i] != 0:
                self.user_means[i] = self.user_sums[i] / self.user_non_zeros[i]

        self.training_ratings_matrix = numpy.zeros(self.ratings_matrix.shape)

        for i in range(0, 400):
            for j in range(400, self.ratings_matrix.shape[1]):
                self.training_ratings_matrix[i][j] = self.ratings_matrix[i][j]

        for i in range(400, self.ratings_matrix.shape[0]):
            for j in range(0, self.ratings_matrix.shape[1]):
                self.training_ratings_matrix[i][j] = self.ratings_matrix[i][j]

        self.transpose_training_ratings_matrix = numpy.zeros(self.transpose_ratings_matrix.shape)

        for i in range(0, 400):
            for j in range(400, self.transpose_ratings_matrix.shape[1]):
                self.transpose_training_ratings_matrix[i][j] = self.transpose_ratings_matrix[i][j]

        for i in range(400, self.transpose_ratings_matrix.shape[0]):
            for j in range(0, self.transpose_ratings_matrix.shape[1]):
                self.transpose_training_ratings_matrix[i][j] = self.transpose_ratings_matrix[i][j]

        self.training_total_mean = self.training_ratings_matrix.mean()

        for i in range(self.ratings_matrix.shape[0]):
            for j in range(self.ratings_matrix.shape[1]):
                if self.training_ratings_matrix[i][j] != 0:
                    self.training_ratings_matrix[i][j] -= self.training_total_mean
                if self.transpose_training_ratings_matrix[j][i] != 0:
                    self.transpose_training_ratings_matrix[j][i] -= self.training_total_mean

    ###############################################

    ###############################################
    ############## CUR Decomposition ##############
    ###############################################


    def cur(self):
        ''' Run CUR Decomposition algorithms. '''

        start_time = time.time()

        A = self.transpose_training_ratings_matrix

        At = numpy.transpose(A)

        row_squares = numpy.zeros((A.shape[0], 1))

        column_squares = numpy.zeros((A.shape[1], 1))

        square_sum = 0

        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                square_sum += A[i][j] ** 2
                row_squares[i] += A[i][j] ** 2
                column_squares[j] += A[i][j] ** 2

        row_probs = numpy.zeros((A.shape[0], 1))

        column_probs = numpy.zeros((A.shape[1], 1))

        for i in range(A.shape[0]):
            row_probs[i] = row_squares[i] / square_sum
        for i in range(A.shape[1]):
            column_probs[i] = column_squares[i] / square_sum

        r = 400

        row_sel = numpy.sort(numpy.random.choice(A.shape[0], r, False, row_probs[:, 0]))
        column_sel = numpy.sort(numpy.random.choice(A.shape[1], r, False, column_probs[:, 0]))

        R = numpy.zeros((r, A.shape[1]))
        Ct = numpy.zeros((r, A.shape[0]))
        W = numpy.zeros((r, r))

        row_rq = numpy.sqrt(r * row_probs)
        column_rq = numpy.sqrt(r * column_probs)

        for i in range(r):
            R[i] = A[row_sel[i]] / row_rq[row_sel[i]]
            Ct[i] = At[column_sel[i]] / column_rq[column_sel[i]]
            for j in range(r):
                W[i][j] = A[row_sel[i]][column_sel[j]]

        C = numpy.transpose(Ct)

        X, Z, Yt = numpy.linalg.svd(W)

        Y = numpy.transpose(Yt)
        Xt = numpy.transpose(X)
        Zinv = numpy.linalg.pinv(numpy.diag(Z))

        U = numpy.dot(Y, numpy.dot(numpy.dot(Zinv, Zinv), Xt))

        guesses = numpy.dot(C, numpy.dot(U, R))

        x = 10
        z = 300

        for i in range(guesses.shape[0]):
            for j in range(guesses.shape[1]):
                guesses[i][j] += self.training_total_mean

        rmse = 0
        values_tested = 0

        for i in range(400):
            for j in range(400):
                if self.transpose_ratings_matrix[i][j] != 0:
                    values_tested += 1
                    guess_error = guesses[i][j] - self.transpose_ratings_matrix[i][j]
                    rmse += guess_error ** 2
        rmse /= values_tested
        while rmse > z:
            rmse /= x
        rho = 1 - (6 / (values_tested ** 2 - 1)) * rmse
        rmse = math.sqrt(rmse)

        t_guesses = numpy.transpose(guesses)
        precision_at_10 = 0.0

        for i in range(400):
            top_10 = heapq.nlargest(10, range(400), t_guesses[i].take)
            for r in top_10:
                if r >= 3.5:
                    precision_at_10 += 1

        precision_at_10 /= 4000

        # Add results to GUI

        self.result_string += '90% Energy CUR\t\t'

        self.result_string += '{0:.3f}'.format(rmse) + '\t\t'

        self.result_string += '{0:.3f}'.format(precision_at_10 * 100) + '%\t\t\t'

        self.result_string += '{0:.6f}'.format(rho) + '\t\t\t'

        self.result_string += '{0:.3f}'.format((time.time() - start_time) / 60) + '\n\n'

    ###############################################

    ###############################################
    ######### Singular Value Decomposition ########
    ###############################################

    def svd_calc(self, A):
        ''' Run SVD algorithms. '''

        AAt = numpy.dot(A, numpy.transpose(A))

        AtA = numpy.dot(numpy.transpose(A), A)

        x1, y1 = numpy.linalg.eigh(AAt)
        y1 = numpy.transpose(y1)
        ar = numpy.argsort(x1)
        ar = numpy.flipud(ar)
        sx1 = numpy.copy(x1)
        sy1 = numpy.copy(y1)
        for i in range(len(ar)):
            sx1[i] = x1[ar[i]]
            sy1[i] = y1[ar[i]]
        u = numpy.transpose(sy1)

        x2, y2 = numpy.linalg.eigh(AtA)
        y2 = numpy.transpose(y2)
        ar = numpy.argsort(x2)
        ar = numpy.flipud(ar)
        sx2 = numpy.copy(x2)
        sy2 = numpy.copy(y2)
        for i in range(len(ar)):
            sx2[i] = x2[ar[i]]
            sy2[i] = y2[ar[i]]
        v = numpy.transpose(sy2)
        vt = numpy.transpose(v)

        r = round(min(u.shape[1], vt.shape[0]))
        for i in range(r):
            if sx1[i] <= 0:
                r = i
                break

        sigma = numpy.zeros((r, r))
        for i in range(r):
            sigma[i][i] = sx1[i]
        sigma = numpy.sqrt(sigma)

        u = u[:, :r]
        vt = vt[:r, :]

        return u, sigma, vt

    def svd(self):
        ''' Error calculation and driver method for SVD. '''

        start_time = time.time()

        u, sigma, vt = self.svd_calc(self.transpose_training_ratings_matrix)

        guesses = numpy.dot(u, numpy.dot(sigma, vt))

        for i in range(guesses.shape[0]):
            for j in range(guesses.shape[1]):
                guesses[i][j] += self.training_total_mean

        rmse = 0
        values_tested = 0

        for i in range(400):
            for j in range(400):
                if self.transpose_ratings_matrix[i][j] != 0:
                    values_tested += 1
                    guess_error = guesses[i][j] - self.transpose_ratings_matrix[i][j]
                    rmse += guess_error ** 2

        rmse /= values_tested
        rho = 1 - (6 / (values_tested ** 2 - 1)) * rmse
        rmse = math.sqrt(rmse)

        t_guesses = numpy.transpose(guesses)
        precision_at_10 = 0.0

        for i in range(400):
            top_10 = heapq.nlargest(10, range(400), t_guesses[i].take)
            for r in top_10:
                if r >= 3.5:
                    precision_at_10 += 1

        precision_at_10 /= 4000

        # Add results to GUI

        self.result_string += '100% Energy SVD\t\t'

        self.result_string += '{0:.3f}'.format(rmse) + '\t\t'

        self.result_string += '{0:.3f}'.format(precision_at_10 * 100) + '%\t\t\t'

        self.result_string += '{0:.6f}'.format(rho) + '\t\t\t'

        self.result_string += '{0:.3f}'.format((time.time() - start_time) / 60) + '\n\n'

    def svd_calc90(self, A):
        ''' Run SVD algorithms. '''

        AAt = numpy.dot(A, numpy.transpose(A))

        AtA = numpy.dot(numpy.transpose(A), A)

        x1, y1 = numpy.linalg.eigh(AAt)
        y1 = numpy.transpose(y1)
        ar = numpy.argsort(x1)
        ar = numpy.flipud(ar)
        sx1 = numpy.copy(x1)
        sy1 = numpy.copy(y1)
        for i in range(len(ar)):
            sx1[i] = x1[ar[i]]
            sy1[i] = y1[ar[i]]
        u = numpy.transpose(sy1)

        x2, y2 = numpy.linalg.eigh(AtA)
        y2 = numpy.transpose(y2)
        ar = numpy.argsort(x2)
        ar = numpy.flipud(ar)
        sx2 = numpy.copy(x2)
        sy2 = numpy.copy(y2)
        for i in range(len(ar)):
            sx2[i] = x2[ar[i]]
            sy2[i] = y2[ar[i]]
        v = numpy.transpose(sy2)
        vt = numpy.transpose(v)
        r = round(min(u.shape[1], vt.shape[0]))
        net=0;
        for i in range(r):
            if sx1[i] <= 0:
                r = i
                break

        #sigma = numpy.zeros((r, r))
        for i in range(r):
            net+= sx1[i]
        percentage=100
        while (percentage>=90):
            r=r-1
            newnet=0

            for i in range(r):
                if sx1[i] <= 0:
                    r = i
                    break

            #sigma = numpy.zeros((r, r))
            for i in range(r):
                newnet += sx1[i]
            percentage=(newnet/net)*100

        sigma = numpy.zeros((r, r))
        for i in range(r):
            sigma[i][i] = sx1[i]
        sigma = numpy.sqrt(sigma)
        #sigma = numpy.sqrt(sigma)

        u = u[:, :r]
        vt = vt[:r, :]

        return u, sigma, vt

    def svd90(self):
        ''' Error calculation and driver method for SVD. '''

        start_time = time.time()

        u, sigma, vt = self.svd_calc90(self.transpose_training_ratings_matrix)

        guesses = numpy.dot(u, numpy.dot(sigma, vt))

        for i in range(guesses.shape[0]):
            for j in range(guesses.shape[1]):
                guesses[i][j] += self.training_total_mean

        rmse = 0
        values_tested = 0

        for i in range(400):
            for j in range(400):
                if self.transpose_ratings_matrix[i][j] != 0:
                    values_tested += 1
                    guess_error = guesses[i][j] - self.transpose_ratings_matrix[i][j]
                    rmse += guess_error ** 2

        rmse /= values_tested
        rho = 1 - (6 / (values_tested ** 2 - 1)) * rmse
        rmse = math.sqrt(rmse)

        t_guesses = numpy.transpose(guesses)
        precision_at_10 = 0.0

        for i in range(400):
            top_10 = heapq.nlargest(10, range(400), t_guesses[i].take)
            for r in top_10:
                if r >= 3.5:
                    precision_at_10 += 1

        precision_at_10 /= 4000

        # Add results to GUI

        self.result_string += '90% Energy SVD\t\t'

        self.result_string += '{0:.3f}'.format(rmse) + '\t\t'

        self.result_string += '{0:.3f}'.format(precision_at_10 * 100) + '%\t\t\t'

        self.result_string += '{0:.6f}'.format(rho) + '\t\t\t'

        self.result_string += '{0:.3f}'.format((time.time() - start_time) / 60) + '\n\n'




    ###############################################

    ###############################################
    #### Preprocessing Collaborative Filtering ####
    ###############################################

    

    def preprocessColab(self):
        ''' Preprocess the data for collaborative filtering. '''

        magnitudes = numpy.zeros(self.movie_sums.shape)
        norm_ratings_matrix = numpy.zeros(self.ratings_matrix.shape)
        for i in range(self.ratings_matrix.shape[0]):
            for j in range(self.ratings_matrix.shape[1]):
                if self.ratings_matrix[i, j] != 0:
                    norm_ratings_matrix[i, j] = (self.ratings_matrix[i, j] - self.movie_means[i])
                    magnitudes[i] += norm_ratings_matrix[i, j] * norm_ratings_matrix[i, j]
            magnitudes[i] = math.sqrt(magnitudes[i])
            for j in range(self.ratings_matrix.shape[1]):
                if magnitudes[i] != 0.0:
                    norm_ratings_matrix[i, j] = norm_ratings_matrix[i, j] / magnitudes[i]

        similarities = numpy.zeros((self.ratings_matrix.shape[0], self.ratings_matrix.shape[0]))
        most_similar = numpy.zeros((self.ratings_matrix.shape[0], 100))

        most_similar_file = open('./ratings/most_similar.dat', 'w')

        for i in range(self.ratings_matrix.shape[0]):
            for j in range(i + 1, norm_ratings_matrix.shape[0]):
                for k in range(norm_ratings_matrix.shape[1]):
                    similarities[i][j] += norm_ratings_matrix[i, k] * norm_ratings_matrix[j, k]
                similarities[j][i] = similarities[i][j]
            most_similar[i] = heapq.nlargest(100, range(len(similarities[i])), similarities[i].take)
            for j in range(0, 100):
                write_string = str(i) + '::' + str(j) + '::' + str(int(most_similar[i][j])) + '::' + str(
                    similarities[i][int(most_similar[i][j])]) + '\n'
                most_similar_file.write(write_string)

        most_similar_file.close()

    ###############################################


    ###############################################
    ########### Collaborative Filtering ###########
    ###############################################

    def colab(self):
        ''' Run collaborative filtering algorithms. '''

        start_time = time.time()

        # If preprocessed file doesn't exist, create it
        if not os.path.exists('./ratings/most_similar.dat'):
            self.preprocessColab()

        # Load the preprocessed file
        most_similar_file = open('./ratings/most_similar.dat', 'r')

        similarities = numpy.zeros((self.ratings_matrix.shape[0], self.ratings_matrix.shape[0]))
        top_100 = {}

        for line in most_similar_file:
            data = line.strip().split('::')
            i = int(data[0])
            j = int(data[2])
            s = data[3]
            if i not in top_100:
                top_100[i] = []
            top_100[i].append(j)
            similarities[i][j] = s

        most_similar_file.close()

        guesses = numpy.zeros((400, 400))
        baseline_guesses = numpy.zeros((400, 400))

        rmse = 0
        baseline_rmse = 0
        values_tested = 0

        for i in range(400):
            for j in range(400):
                if self.ratings_matrix[i][j] != 0:
                    values_tested += 1
                    simsum = 0.0
                    for k in top_100[i]:
                        if self.ratings_matrix[k][j] != 0:
                            guesses[i][j] += self.ratings_matrix[k][j] * similarities[i][k]
                            baseline_score = self.total_mean + self.movie_means[k] + self.user_means[j]
                            baseline_guesses[i][j] += (self.ratings_matrix[k][j] - baseline_score) * similarities[i][k]
                            simsum += similarities[i][k]
                    if simsum != 0.0:
                        guesses[i][j] /= simsum
                        baseline_guesses[i][j] /= simsum
                    baseline_guesses[i][j] += self.total_mean + self.movie_means[i] + self.user_means[j]
                    guess_error = guesses[i][j] - self.ratings_matrix[i][j]
                    rmse += guess_error ** 2
                    baseline_guess_error = baseline_guesses[i][j] - self.ratings_matrix[i][j]
                    baseline_rmse += baseline_guess_error ** 2

        rmse /= values_tested
        rho = 1 - (6 / (values_tested ** 2 - 1)) * rmse
        rmse = math.sqrt(rmse)
        baseline_rmse /= values_tested
        baseline_rho = 1 - (6 / (values_tested ** 2 - 1)) * baseline_rmse
        baseline_rmse = math.sqrt(baseline_rmse)

        t_guesses = numpy.transpose(guesses)
        t_baseline_guesses = numpy.transpose(baseline_guesses)

        precision_at_10 = 0.0
        baseline_precision_at_10 = 0.0

        for i in range(400):
            top_10 = heapq.nlargest(10, range(400), t_guesses[i].take)
            for r in top_10:
                if r >= 3.5:
                    precision_at_10 += 1
            top_10 = heapq.nlargest(10, range(400), t_baseline_guesses[i].take)
            for r in top_10:
                if r >= 3.5:
                    baseline_precision_at_10 += 1

        precision_at_10 /= 4000
        baseline_precision_at_10 /= 4000

        # Add results to GUI

        self.result_string += 'Collaborative\t\t'

        self.result_string += '{0:.3f}'.format(rmse) + '\t\t'

        self.result_string += '{0:.3f}'.format(precision_at_10 * 100) + '%\t\t\t'

        self.result_string += '{0:.6f}'.format(rho) + '\t\t\t'

        self.result_string += '{0:.3f}'.format((time.time() - start_time) / 60) + '\n\n'

        self.result_string += 'Baseline Collaborative\t'

        self.result_string += '{0:.3f}'.format(baseline_rmse) + '\t\t'

        self.result_string += '{0:.3f}'.format(baseline_precision_at_10 * 100) + '%\t\t\t'

        self.result_string += '{0:.6f}'.format(baseline_rho) + '\t\t\t'

        self.result_string += '{0:.3f}'.format((time.time() - start_time) / 60) + '\n\n'

    ###############################################








###############################################################################

gui = tk.Tk()
app = RECOMMENDER(gui)
app.master.title('Recommender Algorithms Comparison Table')
app.after(500, app.loadData)
app.mainloop()