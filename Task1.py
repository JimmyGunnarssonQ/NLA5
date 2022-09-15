import numpy as np 
import numpy.linalg as nl 
import matplotlib.pyplot as plt 
import numpy.random as nr 
A = np.array([[5,0,0,-1],
               [1,0,-1,1],
               [-1.5, 1, -2, 1],
               [-1,1,3,-3]])



class Eigenval:
    def __init__(self, matrix):
        self.matrix = matrix
        self.mval, self.nval = matrix.shape


    def parameter(self,p):
        if p<0 or p>1:
            return print("False Value:" + f" {p} needs to be in range [0,1]")
        else:

            diagA = np.zeros(self.matrix.shape)
            for i in range(self.mval):
                diagA[i,i] = self.matrix[i,i]
        
            newmatrix = np.multiply(self.matrix, p) + np.multiply(diagA, 1-p)

            return newmatrix 
    
    def plots(self,p):
        Ap = self.parameter(p)
        n = self.nval 
        cenlist = np.diag(Ap)
        fig, ax = plt.subplots()
        nu = [0 for i in cenlist]
        eigenvalues, _ = nl.eig(Ap)
        for i in range(n):
            cen = Ap[i,i]
            rad = 0

            for j in range(n):
                if j == i:
                    pass 
                else:
                    rad+= abs(Ap[j,i])

            disk = plt.Circle((cen.real, cen.imag), rad, alpha=0.5)
            ax.add_patch(disk)

        ax.set_aspect('equal', adjustable='datalim')
        ax.plot()
        plt.plot(cenlist, nu, "o", color = "red", label = "Diagonal elements")
        plt.plot(eigenvalues.real, eigenvalues.imag, "x", color = "black", label="Eigenvalues")
        plt.legend(title="legend") 
        plt.ylabel(f"$\Im(z)$" + " part")
        plt.xlabel(f"$\Re(z)$" + " part")
        plt.grid()

    def householder(self):
        '''Householder method, used to generate an upper triangular matrix R '''
        dummymatrix = self.matrix #dummy matrix to be used for Householder reflections
        R = self.matrix #we fill it up with Householder reflected submatricies 

        Qfull = np.eye(self.mval, self.mval)
        for i in range(self.nval):
            a = dummymatrix[:,0] #vector to be modified for reflection

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
                ident = np.eye(self.mval, self.mval)
                for j in range(mpart):
                    for k in range(mpart):
                        ident[i+j:, i+k:] = Q[j,k]
                Qfull = Qfull@ident 

        return R,Qfull  
    
    
    def diagonal(self, tol,counts=1000):
        dummymatrix = self.matrix #generate initial matrix 
        count = 0 #itteration counter 


        while count<counts: #highest bound of iterations 
            a=0 #non-zero off-diagonal elements 
            i=0 #checking i
            j=0 #checking j


            while i<self.mval and a==0: #loop to detect non-zero off-diagonal elements 
                while j<self.nval and a==0:

                    if i == j: #igonore diagonal
                        pass 


                    else:
                        if abs(dummymatrix[i,j])>tol: #if off-diagonal bigger than tolerance
                            a+=1 #breaks the loops 

                        else:
                            pass #ignores if off-diagonal is smaller than tolerance 
                    j+=1
                i+=1 


            if not a==0:
                count+=1 #performs itteration
                r,q = Eigenval(dummymatrix).householder() #q,r decomposition 
                dummymatrix = r@q #new itteration 

            else:
                break #breaks if off-diagonal elements are smaller than tolerance


        for i in range(self.mval): #cleans the off-diagonal elements 
            for j in range(self.nval):
                if i==j:
                    pass 
                else:
                    dummymatrix[i,j] = 0


        return np.array([dummymatrix]), f"itt count: {count}" #returns the diagonal matrix, and the count of itterations 



                    






            
#S = 1/2*(A + A.T)
S = A
#print(nl.eig(S)[0])
classitem = Eigenval(S)
a = classitem.plots(1)
plt.show()