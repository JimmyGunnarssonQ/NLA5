import numpy as np 
import numpy.linalg as nl 
import matplotlib.pyplot as plt 
import numpy.random as nr 
from scipy.linalg import hessenberg 
A = np.array([[5,0,0,-1],
               [1,0,-1,1],
               [-1.5, 1, -2, 1],
               [-1,1,3,-3]])


sign = lambda x: -1 if x<0 else 1 #sign function 

class Eigenval:
    def __init__(self, matrix):
        self.matrix = matrix
        self.mval, self.nval = matrix.shape


    def parameter(self,p):
        '''defines the parameter space '''
        if p<0 or p>1:
            return print("False Value:" + f" {p} needs to be in range [0,1]")
        else:

            diagA = np.zeros(self.matrix.shape)
            for i in range(self.mval):
                diagA[i,i] = self.matrix[i,i]
        
            newmatrix = np.multiply(self.matrix, p) + np.multiply(diagA, 1-p)

            return newmatrix 
    
    def plots(self,p):
        '''Plots Gershgorin circles for some parameter p '''
        Ap = self.parameter(p) #our modified A(p)
        n = self.nval #n value
        cenlist = np.diag(Ap) #list of centres of disks
        fig, ax = plt.subplots() #set up plot 
        nu = [0 for i in cenlist] #axis  
        eigenvalues, _ = nl.eig(Ap) #finds eigenvalues 
        for i in range(n):
            cen = Ap[i,i] #centre 
            rad = 0 #radius 

            for j in range(n): #defines radius 
                if j == i:
                    pass 
                else:
                    rad+= abs(Ap[j,i]) 

            disk = plt.Circle((cen.real, cen.imag), rad, alpha=0.5) #defines disk
            ax.add_patch(disk) #plots disk 
        '''The rest sets up the plotting enviroment'''
        ax.set_aspect('equal', adjustable='datalim')
        ax.plot()
        plt.plot(cenlist, nu, "o", color = "red", label = "Diagonal elements")
        plt.plot(eigenvalues.real, eigenvalues.imag, "x", color = "black", label="Eigenvalues")
        plt.legend(title="legend") 
        plt.ylabel(f"$\Im(z)$" + " part")
        plt.xlabel(f"$\Re(z)$" + " part")
        plt.grid()

    def householder(self, input):
        '''Householder method, used to generate an upper triangular matrix R '''
        dummymatrix = np.array(input)
        R = np.array(input) #we fill it up with Householder reflected submatricies 

        m,n = input.shape 

        Qfull = np.eye(m, m)
        for i in range(n):
            a = dummymatrix[:,0] #vector to be modified for reflection
            if abs(a[0])<1e-15:
                sign = 1
            else:
                sign = a[0]/abs(a[0])
            mpart, npart = dummymatrix.shape
            v = a + np.multiply(np.array([nl.norm(a)] + (mpart-1)*[0]), sign) #vector used for reflection axis
            if nl.norm(v)<1e-15: #if already zeroed 
                pass 
            else:
                v/=nl.norm(v) #normalisation 


                Q = np.eye(mpart) - np.multiply(np.outer(v,v),2) #reflection matrix 
                dummymatrix = Q@dummymatrix #performing the Household reflection
                R[i:,i:] = dummymatrix #updating values

                dummymatrix = dummymatrix[1:, 1:] #new itteration matrix submatrix with dim m-1 x n-1 of previous run
                ident = np.eye(m, n)
                for j in range(mpart):
                    for k in range(mpart):
                        ident[i+j:, i+k:] = Q[j,k]
                Qfull = Qfull@ident 

        return R,Qfull  

    def isdiagonal(self, inputmatrix,tol):
            a=0 #non-zero off-diagonal elements 
            i=0 #checking i
            j=0 #checking j
            mm, nn = inputmatrix.shape


            while i<mm and a==0: #loop to detect non-zero off-diagonal elements 
                while j<nn and a==0:

                    if i == j: #igonore diagonal
                        pass 


                    else:
                        if abs(inputmatrix[i,j])>tol: #if off-diagonal bigger than tolerance
                            a+=1 #breaks the loops 

                        else:
                            pass #ignores if off-diagonal is smaller than tolerance 
                    j+=1
                i+=1
            return a 
    
    def diagonal(self, input=None, cat='self', tol=1e-8,counts=1000):
        if cat == 'ns':
            dummymatrix = np.array(input) #generate initial matrix for 'not self'
        else:
            dummymatrix= np.array(self.matrix) #if cat is self 
        mm,nn=dummymatrix.shape #shape 
        count = 0 #itteration counter 


        while count<counts and not (mm==1 and nn==1): #highest bound of iterations
            a = self.isdiagonal(inputmatrix = dummymatrix, tol = tol)


            if not a==0:
                count+=1 #performs itteration
                r,q = Eigenval(dummymatrix).householder(input=dummymatrix) #q,r decomposition 
                dummymatrix = r@q #new itteration 

            else:
                break #breaks if off-diagonal elements are smaller than tolerance


        for i in range(mm): #cleans the off-diagonal elements 
            for j in range(nn):
                if i==j:
                    pass 
                else:
                    dummymatrix[i,j] = 0


        return np.array(dummymatrix), count #returns the diagonal matrix, and the count of itterations 
     
    def rayleigh(self,vec, input):
        '''Defines the rayleigh shift '''
        num = nl.norm(vec)**2
        den = np.dot(vec,input@vec)

        return den/num 

    def pracqr(self):
        R,Qfull = self.householder(input=self.matrix) #qr 
        Anull = Qfull@self.matrix@Qfull.T #initial value
        a=0 #param to find off diagonal elements 
        counttot = 0 
        Qnew = Qfull #initial value 
        Anew = Anull #initial value 
        for k in range(1000): 
            x = Qnew[0] #rayleigh vector 
            mu = self.rayleigh(vec=x, input = Anew) #shift 
            ident = np.eye(self.nval, self.mval)
            muident = mu*ident #shift times identity 


            Rnew, Qnew = self.householder(input=Anull - muident) #shifted qr 
            Anew = Rnew@Qnew + muident #recompose 

            '''The next method seeks if we can make submatrices '''
            for i in range(self.mval): 
                for j in range(self.nval):
                    if abs(Anew[i,j])<1e-8:
                        Anew[i,j] = 0.
                        a+=1
                    else:
                        pass
            if a != 0: 
                for i in range(self.mval):
                    row = Anew[i] #checks the row space
                    counter = 0 
                    for j in range(i+1,self.nval):
                        value = row[j]
                        if abs(value)>1e-8:
                            counter+=1 #def dimensionality of the new submatrix
                        else:
                            break

                        
                    if counter == 0:
                        pass 
                    if counter !=0: 
                        submat = Anew[i:i+counter,i:i+counter] #submatrix 
                        inp = Eigenval(submat)
                        if submat.shape == (1,1):
                            pass 
                        else:
                            mat, co = inp.diagonal()
                            Anew[i:i+counter,i:i+counter]=mat
                            counttot+=co 


            nullmat = np.eye(self.mval, self.nval)
            for q in range(self.mval):
                nullmat[q,q] = Anew[q,q] #diagonal 

            if np.allclose(nullmat, Anew, rtol=1e-8):
                break #if close, break 
            else:  
                Anull = Anew #else new itteration 



        return Anew, k+counttot 


    def bisection(self, a, b):
        '''Bisection form to seek eigenvalues '''
        hessenform = hessenberg(self.matrix) #hessenberg matrix 
        for i in range(self.mval):
            for j in range(self.nval):
                if abs(hessenform[i,j])<1e-8:
                    hessenform[i,j] = 0 #fixes float 
                else:
                    pass 
        
        '''Checks change of sign for upper-left minors'''
        L= [nl.det(hessenform[:i, :i]) for i in range(self.mval+1)] #Sturm sequence 
        change=0
        for i in range(1,len(L)):
            if L[i-1] == 0:
                change+=1
            else:
                if L[i]/L[i-1]<0:
                    change+=1
                else:
                    pass 



        def seqpoly(x):
            '''Sequent polynomial for the characteristic polynomial'''
            pmone = 0
            pzero = 1
            values = []
            for j in range(self.mval):
                if j == 0:
                    pnew = (hessenform[j,j] - x)*pzero 
                else:
                    pnew = (hessenform[j,j] - x)*pzero - hessenform[j,j-1]**2*pmone
                values.append(pnew)
                pzero, pmone = pnew, pzero
            return values  

        for j in range(200):
            '''Seeks interval [a,b] for where the root exists'''
            f0 = seqpoly(a)[-1]
            f1 = seqpoly(b)[-1]
            if sign(f0) != sign(f1):
                break 
            else:
                a-=.2
                b+=.2

        

        for i in range(1000):
            '''Bisection method'''
            
            c = (a+b)/2 
            fnew = seqpoly(c)[-1]

            if abs(fnew)<1e-8 or abs(a-b)/2<1e-17:
                break 
            
            else:
                if sign(f0) == sign(fnew):
                    a=c 
                else:
                    b=c



        if abs(a-b)/2<1e-17:
            return "failed due to too small interval to seek solution."
        elif j == 199:
            return "failed due to sign error."
        else:
            return c, f"itteration count: {i}"

        





                    






A=nr.rand(4,4)            
S = 1/2*(A + A.T)
#S = A
#S = np.array([[0,1],[1,0]])
eigen, _ = nl.eig(S)
print(eigen)
classitem = Eigenval(S)
a = classitem.pracqr()
print(a)
