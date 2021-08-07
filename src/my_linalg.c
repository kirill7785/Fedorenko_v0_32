// ���� my_linalg.c
// ���������������� ���������� ��������� ������� �������� �������.

#include <stdio.h> // ��� ������� getchar
#include <stdlib.h> // ��� ������� exit, atoi, atof
#include <math.h> // �������������� ������� sqrt, fabs
#include <omp.h> // OpenMP
#include "my_linalg.h" // ���������� ������� �������� �������


#define Real double // ������ �������������� �����


//const Real dterminatedTResudual = 1e-40; // ��� ��� Congruate Gradients

/*  ������ ������� ��������� ��� ����������
 *  �������������� ������� ������������� A
 *        A*x=b;
 *  ��� A �������� nodesxnodes. ��������� 
 *  ��������� ����� ���������� � ����.
 *  ��������� ������������ ����� ����� ������
 *  ��� ������ �������� �������� � ��� �����
 *  ������������� �������.
 *  A � b �� �����������. 
*/
void eqsolve_simple_gauss(Real **A, int nodes, Real *b, Real *x) {
   int i=0, j=0, k=0; // �������� ����� for
   const Real epsilon = 1e-100;
   Real M, sum, akk;

   omp_set_num_threads(inumcore); // ��������� ����� �������

   // ���������� � ������������ ����:
   for(k=0; k<nodes; k++){
	   akk=A[k][k];
       if(fabs(akk)<epsilon){
		  // ������� �� ����� ���� ��������, �.�.
		  // �� ��������� ��������� ����.
	      printf("\nSolution is not exist! Gauss divizion by zero...\n");
	      getchar();
	      exit(0);
	   }
       #pragma omp parallel for shared(k, nodes, A, b) private(i,j,M) firstprivate(akk)
       for(i=k+1; i<nodes; i++) {
	      
	      M = A[i][k] / akk;
	      for(j=k; j<nodes; j++){
	          A[i][j] -= M * A[k][j];
	      }
	      b[i] -= M*b[k];
       }
   }
   // ������� ��������� ����������
   x[nodes-1]=b[nodes-1]/A[nodes-1][nodes-1];
   for(i=nodes-2; i>=0; i--){
       sum = 0.0;
       #pragma omp parallel for shared(A,x,i,nodes) private(j) reduction (+: sum)
       for(j = i+1; j<nodes; j++){
	       sum+= A[i][j]*x[j];
       }
       x[i] = (b[i] - sum) / A[i][i];
   }
} // eqsolve_simple_gauss

/*  ������ ������� ��������� ��� ����������
 *  ������������ ������������ �����������
 *  (� ������������ �������������) �������
 *  ������������� �:
 *        A*x=b;
 *  ��� A �������� nodesxnodes. ������� �
 *  �������������� �� �����������. ��������� 
 *  ��������� ����� ���������� � ����.
 *  ��������� ������������ ����� ���������� ����������:
 *        A=L*transpose(L),
 *  ����� �������� ����������� ������ ���������� � 
 *  �������� �����������. A � b �� �����������. 
*/
void eqsolv_simple_holesskii(Real **A, int nodes, Real *b, Real *x) {
	// ���������� ����������: ������ A ������� � ������ 
	// ������������ �����������.
	A[0][0]=sqrt(A[0][0]);
	A[1][0]/=A[0][0];
	A[0][1]=A[1][0];
	A[1][1]=sqrt(A[1][1]-A[1][0]*A[1][0]);

	omp_set_num_threads(inumcore); // ��������� ����� �������

	int irow,irow1;
	int icol, icol1;
	Real sum;
	int k;
	for (irow=2; irow<nodes; irow++) {
		irow1=irow-1;
		A[irow][0]/=A[0][0];
        A[0][irow]=A[irow][0];
        #pragma omp parallel for shared(irow1,A) private(icol, icol1, sum, k)
		for (icol=1; icol<=irow1; icol++) {
			icol1=icol-1;
            sum=0.0;   
            for (k=0; k<=icol1; k++) sum+=A[irow][k]*A[icol][k];
			A[irow][icol]=(A[irow][icol]-sum)/A[icol][icol];
			A[icol][irow]=A[irow][icol];
		}
		sum=0.0;
		#pragma omp parallel for shared(A,irow,irow1) private(k) reduction (+: sum)
		for (k=0; k<=irow1; k++) sum+=A[irow][k]*A[irow][k];
		A[irow][irow]=sqrt(A[irow][irow]-sum);
	}
    
	// ������ ����������. ���������� ���������� ������ �����
	b[0]/=A[0][0];

	for (irow=1; irow<nodes; irow++) {
		irow1=irow-1;
		sum=0.0;
		#pragma omp parallel for shared(A,b,irow,irow1) private(icol) reduction (+: sum)
		for (icol=0; icol<=irow1; icol++) sum+=A[irow][icol]*b[icol];
        b[irow]=(b[irow]-sum)/A[irow][irow];
	}

	// �������� ����������� ������������ ������� ����������� ���������
	x[nodes-1]=b[nodes-1]/A[nodes-1][nodes-1];
	for (k=1; k<=nodes; k++) {
		irow=nodes+1-k-1;
		irow1=irow+1;
		sum=0.0;
        #pragma omp parallel for shared(A,x,irow,irow1,nodes) private(icol) reduction (+: sum)
		for (icol=irow1; icol<nodes; icol++) sum+=A[irow][icol]*x[icol];
		x[irow]=(b[irow]-sum)/A[irow][irow];
	}

} // eqsolv_simple_holesskii

/* ������� �������� ������� ��� 
*  ���������� ������� A nodes*nodes � 
*  ���������� ���������� �� ������� ���������.
*  ������� ������������ ���� ������ ����������
*  ������, � ������ ����� nodes ����. 
*          A*inv=e
*  ����������  � ������������ ���� ��������
*  ������ ���� ���.
* ���� flag==true, �� ������� ��� ��������� � ������������������ ����.
*/
void inverse_matrix_simple(Real** &A, int nodes, bool flag) {

    const Real epsilon = 1e-100;

	Real **e; // ��������� ������� ������ ������.
	Real **inv; // ������� �������� �������

	Real **acopy; // ����� ������� �
	if (nodes==3) {
		acopy = new Real*[nodes];
        for (int i1=0; i1<nodes; i1++) acopy[i1]=new Real[nodes]; 
	}

	int i1=0, j1=0, k1=0;
	e = new Real* [nodes];
    for (i1=0; i1<nodes; i1++) e[i1]=new Real[nodes]; 
	inv = new Real* [nodes];
    for (i1=0; i1<nodes; i1++) inv[i1]=new Real[nodes];
    
	// �������������
	for (i1=0; i1<nodes; i1++) for (j1=0; j1<nodes; j1++) {
		inv[i1][j1]=0.0; // �������� �������
		e[i1][j1]=0.0; // ������ �����
		if (nodes==3) {
			acopy[i1][j1]=A[i1][j1];
		}
	}
	for (i1=0; i1<nodes; i1++) e[i1][i1]=1.0;

	

    
	if (!flag) { // ���� ������� ��� �� ��������� � ������������������ ����
        Real M;
		// ���������� � ������ ������������ ����:
        for(k1=0; k1<nodes; k1++){
           for(i1=k1+1; i1<nodes; i1++){
		       // ���� �� ��������� ����:
		       if (fabs(A[k1][k1])<epsilon) {
			      // ������� �� ����� ���� ��������, �.�.
			      // �� ��������� ��������� ����.
				  printf("\n inverse matrix simple ERROR !!! may be diagonal value is zero...\n");
	              printf("\nSolution is not exist.\n");
				  for (int irow=0; irow<nodes; irow++) {
					  for (int icol=0; icol<nodes; icol++) {
						  if (nodes==3) {
							  printf("%1.4e ",acopy[irow][icol]);
						  }
						  else {
							  printf("%1.4e ",A[irow][icol]);
						  }
					  }
					  printf("\n");
				  }
	              getchar();
		          exit(0);
		       }
	           M = A[i1][k1] / A[k1][k1];
	           for(j1=k1; j1<nodes; j1++){
	              A[i1][j1] -= M * A[k1][j1];
	           }
		       // �������������� ������ ������:
              for(j1=0; j1<nodes; j1++) e[i1][j1] -= M*e[k1][j1];
           }
        }
	}
	Real *sum=new Real[nodes];

   // ������� ��������� ����������
   for(i1=nodes-1; i1>=0; i1--){
	   // �������������
       for (k1=0; k1<nodes; k1++) sum[k1] = 0.0;

       for(j1 = i1+1; j1<nodes; j1++){
		   for (k1=0; k1<nodes; k1++) {
	           sum[k1]+= A[i1][j1]*inv[j1][k1];
		   }
       }
       for (k1=0; k1<nodes; k1++) {
	        inv[i1][k1]=(e[i1][k1] - sum[k1])/A[i1][i1];
	   }
   }
   for(i1=nodes-1; i1>=0; i1--) delete[] e[i1];
   delete[] e;

   for(k1=0; k1<nodes; k1++){
       for(i1=0; i1<nodes; i1++){
		   A[k1][i1]=inv[k1][i1];
	   }
   }
   for(i1=nodes-1; i1>=0; i1--) delete[] inv[i1];
   delete[] inv;
   if (nodes==3) {
      for(i1=nodes-1; i1>=0; i1--) delete[] acopy[i1];
      delete[] acopy;
   }
   delete[] sum;
} // inverse_matrix_simple
 
/* ������� ������������ ���� ����������
* ������ A � B ��������� nodesxnodes 
*             C=A*B. 
* ���������  ������������ � ������� B.
*/
void multiply_matrix_simple(Real **A1, Real **A2, int nodes) {
	int i1=0, j1=0, k1=0; // �������� ����� for
	
	Real **c;
	c = new Real* [nodes];
    for (i1=0; i1<nodes; i1++) c[i1]=new Real[nodes];

	for (i1=0; i1<nodes; i1++) for (j1=0; j1<nodes; j1++) c[i1][j1]=0.0; // �������������

	// ��������� C=A1*A2:
    for (i1=0; i1 < nodes; i1++)
        for (k1=0; k1 < nodes; k1++)
            for (j1=0; j1 < nodes; j1++)
                c[i1][k1]+=(A1[i1][j1])*(A2[j1][k1]);

	// ����������� ���������� � A2:
    for (i1=0; i1<nodes; i1++) for (j1=0; j1<nodes; j1++) A2[i1][j1]=c[i1][j1];

	delete c;
} // multiply_matrix_simple


// ��������� ��������� ������� (�����, �� ������
// ���������������� �� ������������, � ��������� �������� ����� �������). 
// ������������ ��� ��������������� ��� �������
// ������ �������� ����������� ��������:

/* 1. ��������� ���������� ������ ������� nxn:
*                t=m*p.
* ��������� ���������� � ����.
* �� ��������� ������ ��������� �������� � ������� t.
*/
void multi_m(Real **m, Real **p, Real **t, int n) {
    for (int i = 0; i < n; i++)
       for (int j = 0; j < n; j++) {
           Real s = 0;
           for (int l = 0; l < n; l++)
               s += m[i][l]*p[l][j];
           t[i][j] = s;
    }
} // multi_m 

/* 2. ���������������� ���������� ������� m
*  �������� nxn. �� ��������� ������ � ������� 
*  m �������� ��������� ����������������.
*/
void tr_m(Real **m, int n) {
    for (int i = 1; i < n; i++)
        for (int j = 0; j < i; j++) {
            Real buf = m[i][j];
            m[i][j] = m[j][i];
            m[j][i] = buf;
        }
} // tr_m

/* 3. ���������� ������������ ���������������
* ������� ��� ������������ ������� A �������� 
* nxn. ������� ������������� �������� A[f][g].
* ��� ��������� ����������, �.�. ��� �� ����������
* ���������� � ���������� ������� �������������
* �������� � ������� �.
*/
Real max_el(Real **A, int n, int& f, int& g) {
   Real max = A[0][1];
   f=0; g=1; // ��������� ��������
   for (int j = 1; j < n; j++)
      for (int i = 0; i < j; i++) {
        if (A[i][j] > max) {
            max = A[i][j];
            f = i; g = j;
        }
    }
    return max;
 } // max_el

/* 4. �������� ������ ������� � ������: A=B.
* ������� ���������� �������� nxn
*/
void matr_copy(Real **A1, Real **A2, int n) {
   for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
		  A1[i][j]=A2[i][j];
}

/* 5. ������� ��������� ���� ���������� ������ ������������ 
* ���� ������� nxn (����� ���������):
*                A=hi�*A.
* ����� hic -  �������������� ����������������� ������� ��������:
* hic[f][f] = cosfi;
* hic[g][g] = cosfi;
* hic[f][g] = +sinfi;
* hic[g][f] = -sinfi;
* ����� f � g ������� ��������� ���������.
* ��������� ���������� � ����.
* �� ��������� ������ ��������� �������� � �������� ������� A.
* ������ ������� hi� ��������� ������ ��� ���� ������ ������ ��������
* ��� ��������� ����������� ��������� ������ � ��������������.
*/
void multi_m_left(Real **A, Real **rab, int n, int f, int g, Real cosfi, Real sinfi) {
	/* ���������� ������������� �� �������� ���:
    for (int i = 0; i < n; i++)
       for (int j = 0; j < n; j++) {
		   if ((i!=f) && (i!=g)) {
			   t[i][j]=A[i][j];
		   }
		   else if (i==f) {
			   //t[i][j]=hic[f][f]*A[f][j]+hic[f][g]*A[g][j];
               t[i][j]=cosfi*A[f][j]+sinfi*A[g][j];
		   }
		   else if (i==g) {
			   //t[i][j]=hic[g][f]*A[f][j]+hic[g][g]*A[g][j];
			   t[i][j]=-sinfi*A[f][j]+cosfi*A[g][j];
		   }
    }
	*/
    
	// ������ ��������� ��������� ������������ ����� � ������� �
	// � �������� �������� ������������ ������ rab �����������
	// 2xn. ����������� �������� ����� 4*n ���������.
	for (int j = 0; j < n; j++) {
	   rab[0][j]=cosfi*A[f][j]+sinfi*A[g][j];
	   rab[1][j]=-sinfi*A[f][j]+cosfi*A[g][j];
	}
    for (int j = 0; j < n; j++) {
	   A[f][j]=rab[0][j];
	   A[g][j]=rab[1][j];
	}

} // multi_m_left 

/* 6. ������� ��������� ���� ���������� ������ ������������ 
* ���� ������� nxn (������ ���������):
*                A=A*hi.
* ����� hi - �������������� ������� ��������:
* hi[f][f] = cosfi;
* hi[g][g] = cosfi;
* hi[f][g] = -sinfi;
* hi[g][f] = +sinfi;
* ����� f � g ������� ��������� ���������.
* ��������� ���������� � ����.
* �� ��������� ������ ��������� �������� � �������� ������� A.
* ������ ������� hi ��������� ������ ��� ���� ������ ������ ��������
* ��� ��������� ����������� ��������� ������ � ��������������.
*/
void multi_m_right(Real **A, Real **rab, int n, int f, int g, Real cosfi, Real sinfi) {
	/* ������������� ���������
    for (int i = 0; i < n; i++)
       for (int j = 0; j < n; j++) {
		   if ((j!=f) && (j!=g)) {
			   t[i][j]=A[i][j];
		   }
		   else if (j==f) {
			   //t[i][j]=A[i][f]*hi[f][f]+A[i][g]*hi[g][f];
               t[i][j]=A[i][f]*cosfi+A[i][g]*sinfi;
		   }
		   else if (j==g) {
			   //t[i][j]=A[i][f]*hi[f][g]+A[i][g]*hi[g][g];
			   t[i][j]=-A[i][f]*sinfi+A[i][g]*cosfi;
		   }
    }
	*/

	// ������ ��������� ��������� ������������ ����� � ������� �
	// � �������� �������� ������������ ������ rab �����������
	// 2xn. ����������� �������� ����� 4*n ���������.
	for (int i = 0; i < n; i++) {
	   rab[0][i]=A[i][f]*cosfi+A[i][g]*sinfi; // f
	   rab[1][i]=-A[i][f]*sinfi+A[i][g]*cosfi; // g
	}
    for (int i = 0; i < n; i++) {
		A[i][f]=rab[0][i];
		A[i][g]=rab[1][i];
	}

} // multi_m_right 


/* ������������ �������� ����� �� 1846 ����. 
* ������ ������ �������� ����������� �������� � �����
*              A-lambda_scal*E=0
*  ������� ��������. ��. ���������� ������ �����.
*  ������������ ������������ ����������� ������� A 
*  �������� nodesxnodes. ������� A � ���������� ������  
*  �������� ( �� ��������� � � ������������ �������� ����� ��).
*  � ���������� ������� U �� �������� �������� 
*  ����������� �������. � ������� lambda ��������� ������
*  ����������� ��������.
*  ������� ���������� �������� � �� �������� ������������,
*  ��� �������� ��������������� ��������� epsilon.
*  �� ������ �������� �������� 12xnodes ��������� ���������.
*  �������������� ������ ����� 2xnodes.
*  EIGEM - ����� �����.
*/
void jacobi_matrix_simple(Real **A, Real **U, Real *lambda, int nodes, Real epsilon) {

	// �������� ���� ���������� ����� ��������� �����������.
    const Real eps=1e-10; // ��������  � ������� �������� ����������� �� ���������,
	
	int i,j; // �������� ����� for
    int im , jm; // ������� ������������� ��������
    int p = 1; // ����� ��������
	Real maxij; // ������������ �������
	Real fi; // �������� ����
	Real cosfi, sinfi; // �������� �������� � ������ ���� fi

	/* ��� ������� ��������� ������ ���� ��� �� ���������
	// �������  ��������
	Real **hi=new Real*[nodes];
    for (i = 0; i < nodes; i++) hi[i]=new Real[nodes];

	// ������������� ���� ������� ����������� ���� ���:
    for (i = 0; i < nodes; i++)
         for (j = 0; j < nodes; j++) {
            if (i == j)
                hi[i][j] = 1.0;
            else hi[i][j] = 0.0;
    }

	// ��������������� ������� �������� (�����)
	Real **hic=new Real*[nodes];
    for (i = 0; i < nodes; i++) hic[i]=new Real[nodes];

	// ������������� ��������������� ������� ���� ���:
	matr_copy(hic,hi,nodes); // ���� ��������� ������� 
	*/

	// ��������������� ������� ��� ���������
    // ���������� ������������� �� ������ �������.
    //Real **b=new Real*[nodes];
    //for (i = 0; i < nodes; i++) b[i]=new Real[nodes];

	// ������� ������.
	Real **rab=new Real*[2];
    for (i = 0; i < 2; i++) rab[i]=new Real[nodes];

    maxij = max_el(A, nodes,im,jm);
    
	// ������ �������� �������� 12xnodes ���������.
    while (fabs(maxij) > epsilon) {
       
       
	   // ���������� ����:
	   if (fabs(A[im][im]-A[jm][jm])<eps) {
		   // ������ ������ �������� �����
		   fi=3.141/4.0;
	   }
	   else fi= atan(2*maxij/(A[im][im]-A[jm][jm]))/2;
       
       // ���������� ������������������ �������
	   // �� ���� fi:
       cosfi = cos(fi);
	   sinfi = sin(fi);
 
	   /* ��� ������� ��������� ���� ����������������� ��� �� ������������.
	   // ������� �������� �� �������� ������������:
	   // ������������� ������� ��������
       hi[im][im] = cosfi;
       hi[jm][jm] = cosfi;
       hi[im][jm] = -sinfi;
       hi[jm][im] = sinfi;
	   // ����������������� �������: 
       hic[im][im] = cosfi;
       hic[jm][jm] = cosfi;
       hic[im][jm] = +sinfi; // ����������������.
       hic[jm][im] = -sinfi;
	   */
 
       //  ������������� ������� �� ������� �������� �� ��������
	   if (p==1) {
		   //matr_copy(U,hi,nodes);
           for (i = 0; i < nodes; i++)
               for (j = 0; j < nodes; j++) {
                   if (i == j)
                      U[i][j] = 1.0;
                   else U[i][j] = 0.0;
               }
		   U[im][im] = cosfi;
           U[jm][jm] = cosfi;
           U[im][jm] = -sinfi;
           U[jm][im] = sinfi;

	   } else {
            //multi_m(U,hi,b, nodes);
            multi_m_right(U, rab, nodes, im, jm, cosfi, sinfi); // ������� ���������
			//matr_copy(U,b,nodes); // ������ ������ ������������� ������� b ������������ 
			// ����������� ������� rab. ������������� �������� 4xnodes ���������.
	   }
      
       //multi_m(hic, A, b, nodes); // b=transpose(H)*A
	   multi_m_left(A, rab, nodes, im, jm, cosfi, sinfi); // ������� ���������: 4xnodes �������� ���������
       //multi_m(b, hi, A, nodes); // A=b*H.
	   multi_m_right(A, rab, nodes, im, jm, cosfi, sinfi); // ������� ���������: 4xnodes �������� ���������
 
	   /* ��� ������� ��������� ���� ����������������� ��� �� ������������.
	   // �������������� ������ ��������:
       hi[im][im] = 1.0;
       hi[jm][jm] = 1.0;
       hi[im][jm] = 0.0;
       hi[jm][im] = 0.0;
	   // �������������� ����� ������� ��������:
       hic[im][im] = 1.0;
       hic[jm][jm] = 1.0;
       hic[im][jm] = 0.0;
       hic[jm][im] = 0.0;
	   */

	   maxij = max_el(A, nodes,im,jm); // ����������� ��������� ����������� ��������.
       p++; // ������� � ��������� ��������

    } // while

    for (i = 0; i < nodes; i++) lambda[i]=A[i][i]; //  ��

} // jacobi_matrix_simple

/* ��������� ��������� ������� ����������� ��� GSEP
*  � ����� �������������� �� ����������� ������ 
*  ����������� ��������.
*/

// ����������� ����������.
void BubbleSortGSEP1(Real *a, int *mask, int n) {
   int i=0, j=0, k=0;
   Real x;

   for (i=1; i<n; i++) {
	   for (j=n-1; j>=i; j--) {
		   if (a[j-1] > a[j]) {
			   // swap
			   x=a[j-1];
			   a[j-1]=a[j];
			   a[j]=x;
               k=mask[j-1];
			   mask[j-1]=mask[j];
			   mask[j]=k;
		   }
	   }
   }
} // BubbleSortGSEP1


/* ������ ���������� ������������ �������� ����������� ��������
*   GSEP1:  A*x-lambda_scal*B*x=0;
*   ���������� ����������: B=L*transpose(L);
*   L - ������ �����������, transpose(L) - ������� �����������.
*/
void GSEP1(Real **A1, Real **A2, Real **U, Real *lambda, int *mask, int nodes, Real epsilon) {

	// ���������� ����������: ������ B ������� � ������ 
	// ������������ �����������.
	A2[0][0]=sqrt(A2[0][0]);
	A2[1][0]/=A2[0][0];
	A2[0][1]=A2[1][0];
	A2[1][1]=sqrt(A2[1][1]-A2[1][0]*A2[1][0]);

	int irow,irow1;
	int icol, icol1;
	Real sum;
	int k;
	for (irow=2; irow<nodes; irow++) {
		irow1=irow-1;
		A2[irow][0]/=A2[0][0];
        A2[0][irow]=A2[irow][0];
		for (icol=1; icol<=irow1; icol++) {
			icol1=icol-1;
            sum=0.0;
            for (k=0; k<=icol1; k++) sum+=A2[irow][k]*A2[icol][k];
			A2[irow][icol]=(A2[irow][icol]-sum)/A2[icol][icol];
			A2[icol][irow]=A2[irow][icol];
		}
		sum=0.0;
		for (k=0; k<=irow1; k++) sum+=A2[irow][k]*A2[irow][k];
		A2[irow][irow]=sqrt(A2[irow][irow]-sum);
	}

	printf("L*LT 10...\n");
   // TODO: ������ � �� ����� ������� ��� �����������
	// ������������� ����� ����:

    int i=0, j=0;

	for (i=0; i<nodes; i++) mask[i]=i;

	// ������ ����������� �������
    Real **L=new Real*[nodes];
    for (i = 0; i < nodes; i++) L[i]=new Real[nodes];

	/* ���� ����������������� ����� ���� ��������� � ��������� ����������:
	// ���� ������������ ��������� ���������� �� ��� ���� ����������������.
	// ������������� ���� ������� ����������� ���� ���:
    for (i = 0; i < nodes; i++)
         for (j = 0; j < nodes; j++) {
            if (j > i)
                L[i][j] = 0.0;
            else L[i][j] = B[i][j];
    }
	*/

    /*
    // ������� ����������� �������
    Real **LT=new Real*[nodes];
    for (i = 0; i < nodes; i++) LT[i]=new Real[nodes];

	// ������������� ���� ������� ����������� ���� ���:
    for (i = 0; i < nodes; i++)
         for (j = 0; j < nodes; j++) {
            if (j < i)
                LT[i][j] = 0.0;
            else LT[i][j] = B[i][j];
    }
	*/

    // ��������������� ������� ��� ���������
    Real **b=new Real*[nodes];
    for (i = 0; i < nodes; i++) b[i]=new Real[nodes];

	// Ac ����� ������� �
    Real **Ac=new Real*[nodes];
    for (i = 0; i < nodes; i++) Ac[i]=new Real[nodes];
	matr_copy(Ac,A1,nodes); // ���������� � TODO �������� ����� �������

	// ��������� ����������
    //inverse_matrix_simple(L,nodes); // ���������� L^(-1)
	//multi_m(L,A,b,nodes); // b=(L^(-1))*A;
	//matr_copy(A,b,nodes); // A=(L^(-1))*A;

	// ����� ������� ����������
    // A=(L^(-1))*A;
	for (i=0; i < nodes; i++) {
		A1[0][i]/=A2[0][0];

	    for (irow=1; irow<nodes; irow++) {
		    irow1=irow-1;
		    sum=0.0;
		    for (icol=0; icol<=irow1; icol++) sum+=A2[irow][icol]*A1[icol][i];
            A1[irow][i]=(A1[irow][i]-sum)/A2[irow][irow];
	    }
	}

    printf("(L^(-1))*A 20...\n");
	//matr_copy(L,LT,nodes); // L=transpose(L); �.�. ������� L ������ �� �����:
	// ������ ����� ��� ������ L ������������ transpose(L).
    
    // L=LT: L=transpose(L);
	// ������ L ������� ����������� �������
    for (i = 0; i < nodes; i++)
         for (j = 0; j < nodes; j++) {
            if (j < i)
                L[i][j] = 0.0;
            else L[i][j] = A2[i][j];
    }

    // ��� ������� ��� ��������� � ������������������ ����,
    // ������� ������ ��� � � ������ ���� ��������� �� ������������ ������� true.
    inverse_matrix_simple(L,nodes,true); // ���������� (transpose(L))^(-1)
	 
	multi_m(A1,L,b,nodes); // b=(L^(-1))*A*(transpose(L))^(-1).
    matr_copy(A1,b,nodes); // A=(L^(-1))*A*(transpose(L))^(-1).

	printf("C 30...\n");

	jacobi_matrix_simple(A1,U,lambda,nodes,epsilon); // ���������� �� � �� � �������� ���������.

    printf("C 90...\n");

	BubbleSortGSEP1(lambda,mask,nodes); // �������������� ����������� ��������.
	multi_m(L,U,b,nodes); // b=((transpose(L))^(-1))*U
    matr_copy(U,b,nodes); // ����������� �������.

	/* �������� ��������� ����������� ��������.
    multi_m(Ac,U,b,nodes); // b=A1*U
    matr_copy(L,U,nodes); // L=U
	tr_m(L,nodes);
	multi_m(L,b,Ac,nodes); // Ac=transpose(U)*A1*U

	Real *test=new Real[nodes];
    for (int i=0; i<nodes; i++) test[i]=Ac[i][i];
    BubbleSortGSEP1(test,mask,nodes); 
    for (int i=0; i<8; i++) printf("%.2f ",test[i]/3.141/3.141); // ����������� ��������
	printf("\n");
	*/

	delete L;  delete b; //delete LT;
} // GSEP1

/* ����� ������ ��� ��������� ������� A ��������
*              nodes x 2*icolx+1, ���
*   2*icolx+1 - ������ �����. ��� ��� ��� �������
*  A ��������� ���������� �� ��� ��������� ��������
*  ������� ���������� ������ ������ �����.
*  b - ������ ������ ����� ����, x - ������ �������.
*  ��������� ��������� ���������� � ����.
*  ��� ������������ ����������� �������� ��������������
*  ������ �, ������� �������� ����� ������.
*  ����� ���� ������� 1777-1855.
*  � ���������� ������ ������� � ��������.
*/
void eqsolve_lenta_gauss(Real **A, int nodes, int icolx, Real *b, Real *x) {

	const Real eps=1e-300; // ��� ��������� � ����
	Real dCik, dSum=0.0;
	int max;

	int *move=new int[nodes]; // ������ �������.
	int i=0, j=0, k=0; // �������� ����� for
	for (i=0; i<nodes; i++) move[i]=icolx-i; // ������������� ������� �������

	for (i=0; i<nodes; i++) x[i]=0.0; // �������������

	// ������ ��� ������ ������
	// ���������� � �������� ������������ ����:

	// �� ���� �������� ����� �������
	for (k=0; k<nodes; k++) {
        max=min(k+icolx,nodes-1);
		// ���� �� ���� ������� ���� ������ � ������� k
		for (i=k+1; i<=max; i++) {
			// ����������� ������ � ��� ������
			// ���� ������� ���������
			// ��� ������ ��������� �������� ����.
			if (fabs(A[i][k+move[i]]) > eps) {
               
                if(fabs(A[k][k+move[k]])<eps){
			          // ������� �� ����� ���� ��������, �.�.
			          // �� ��������� ��������� ����.
	                  printf("\nSolution is not exist! divizion by zero...\n");
	                  getchar();
		              exit(0);
	            }

                // ��������� ������������� ������ � ������� i
				dCik=A[i][k+move[i]]/A[k][k+move[k]];
				// ��������������� ������� � ������������������ ����:
				for (j=k; j<=max; j++) A[i][j+move[i]] -= dCik*A[k][j+move[k]];
				b[i]-= dCik*b[k]; // �������������� ������ �����
			}
		}
	}

    // ������ ����� ������� ��������� � ������������������ ����
	// ����� ��������� �������� ��� ������ ������:
	for (k=nodes-1; k>=0; k--) {
        dSum=0.0; // ��������� ���������
		max=min(k+icolx,nodes-1);
		for (i=k+1; i<=max; i++) dSum+= A[k][i+move[k]]*x[i];
		x[k]=(b[k]-dSum)/A[k][k+move[k]];
	}

}  // eqsolve_lenta_gauss

// ����� (�����) ������-�������
// ��� ������� ���� � �������� � nxn
// �������� ��������������, �� � ������������ 
// �������������. ������� � ��������������
// ��������� ���������� (�������������).
// b - ������ �����, x - ���������� �������, 
// eps - �������� ����������� �������.
// omega - ����������� ����������� �������� ����������.
void Seidel(Real **A, Real *b, Real *x, int n, Real eps, Real omega) {
	int i,j;
	Real s1, s2, s, v, m;
	bool bdiag=true;

	// ��������� ����������
	for (i=0; i<n; i++) {
		s=0.0;
		for (j=0; j<n; j++) {
			if (j!=i) s+=fabs(A[i][j]);
		}
		if (s>=fabs(A[i][i])) {
			bdiag=false;
		}
	}
	if (!bdiag) {
		printf("net diagonalnogo preobladaniq...");
		getchar();
	}

	do {
		m=0.0;
		for (i=0; i<n; i++) {
			// ��������� �����
			s1=s2=0.0; 
			for (j=0; j<=i-1; j++) s1+=A[i][j]*x[j];
			for (j=i+1; j<n; j++) s2+=A[i][j]*x[j];
			// ��������� ����� ����������� � �����������
			v=x[i];
			x[i]=omega*(b[i]-s1-s2)/A[i][i]+(1-omega)*x[i];

			if (fabs(v-x[i])>m) m=fabs(v-x[i]);
		}

	} while (m > eps);

} // Seidel

// ���������� ������������ �� ���� 
// ������������ �����.
/*
Real fmax(Real fA, Real fB) {
	Real r=fB;
	if (fA > fB) r=fA;
	return r;
} // fmax 
*/

// ����������� ��� ��������� �������� �������� 
// � ������ ����� �� ���� ������� ����� ������� �������.
void SOR(equation* &sl, Real* &x, int n) {
	Real rURF=1.855; // �������� ������� ����������
	// ��������� �������� �������
	Real eps = 1e-3;
	Real ptilda;
	Real sE,sW,sN, sS;
	int i=0,j=0, kend=3000; // ������� ����� for
	Real dmax=1.0;
	while ((dmax>eps) && (j<kend)) {
		dmax=0.0;
	    for (i=0; i<n; i++) {
            if (sl[i].iE>-1) sE=sl[i].ae*x[sl[i].iE]; else sE=0.0;
		    if (sl[i].iW>-1) sW=sl[i].aw*x[sl[i].iW]; else sW=0.0;
		    if (sl[i].iN>-1) sN=sl[i].an*x[sl[i].iN]; else sN=0.0;
		    if (sl[i].iS>-1) sS=sl[i].as*x[sl[i].iS]; else sS=0.0;
		    ptilda=(sE+sW+sN+sS+sl[i].b)/sl[i].ap;
		    //dmax=fmax(dmax,sl[i].ap*(ptilda-x[sl[i].iP]));
			dmax+=fabs(sl[i].ap*(ptilda-x[sl[i].iP]));
		    x[sl[i].iP]=x[sl[i].iP]+rURF*(ptilda-x[sl[i].iP]);
	    }
		dmax/=n;
		printf("%e \n",dmax);
		j++;
	}

} // SOR

/*
// ����� ����� ��������� ������� �� ������� ������� ������������
// � ����������� ��������� ANSYS fluent. ���� ������� ����������� �� 
// ������� Fluent ����� ����� ���������� � ��� ��������� � ��������� ��������������
// ��������� AliceFlow_v0_07.
// ������ ������� ������������ ��� ���� ������� ������� ����� �������� ��������.
// ���������� � ������� �� ������� ����������� ������� ����� �� icepak user Guide. Threory chapter.
Real fluent_residual_for_x(equation3D* &sl, equation3D_bon* &slb, Real* &x, int maxelm, int maxbound) {
	Real r=0.0;

	Real fsum1=0.0, fsum2=0.0;

	// ���������� ����������� ������.
	for (int i=0; i<maxelm; i++) {
		// ���������
		Real sE,sW,sN,sS,sT,sB;
		if (sl[i].iE>-1) sE=sl[i].ae*x[sl[i].iE]; else sE=0.0;
		if (sl[i].iW>-1) sW=sl[i].aw*x[sl[i].iW]; else sW=0.0;
		if (sl[i].iN>-1) sN=sl[i].an*x[sl[i].iN]; else sN=0.0;
		if (sl[i].iS>-1) sS=sl[i].as*x[sl[i].iS]; else sS=0.0;
        if (sl[i].iT>-1) sT=sl[i].at*x[sl[i].iT]; else sT=0.0;
		if (sl[i].iB>-1) sB=sl[i].ab*x[sl[i].iB]; else sB=0.0;
		fsum1+=fabs(sE+sW+sN+sS+sT+sB+sl[i].b-sl[i].ap*x[sl[i].iP]);
		fsum2+=fabs(sl[i].ap*x[sl[i].iP]); // �����������.
	}

	
	// ��������� ����������� ������.
	for (int i=0; i<maxbound; i++) {
		// ���������
		Real sI;
		if (slb[i].iI>-1) sI=slb[i].ai*x[slb[i].iI]; else sI=0.0;
		fsum1+=fabs(sI+slb[i].b-slb[i].aw*x[slb[i].iW]);
		// �����������
		fsum2+=fabs(slb[i].aw*x[slb[i].iW]);
	}
	
	if (fsum2<1e-41) {
		r=0.0;
	} 
	else {
		r=fsum1/fsum2;
	}
	return r;
} // fluent_residual_for_x
*/

// ���������� ����� ������������� ����� (������������������ ��������� �����)
// � ��������� ��� �������� �������� � ����� ��������������� ��������� ANSYS fluent.
// �� ����� �������� ����� ������ � ���������� ���� ������� ��������� �����-������.
// ���������� � �������� �������� �� ����������� � ��������� icepak.
Real no_balance_mass_flux_fluent(Real* &b, Real operating_value_b, int n) {
	// b - ������ ������������������ ���������� �����.
	// n - ����������� ����� �������.

	Real r=0.0;

	for (int i=0; i<n; i++) {
		r+=fabs(b[i]);
	}

	r=r/operating_value_b;
	return r;
} // no_balance_mass_flux_fluent

/*
// ����������� ��� ��������� �������� �������� 
// � ������ ����� �� ���� ������� ����� ������� �������.
// �������� ���������� ����� � ����� ���������,
// ������� ���� ����� ������������ �������.
// ����������� �� ����� ��������� ��������� ���� ����� ����� ��������� ��� ������������.
void SOR3D(equation3D* &sl, equation3D_bon* &slb, Real* &x, int maxelm, int maxbound, int iVar) {
    Real rURF=1.0; // �������� ������� ����������
	switch (iVar) {
		case PAM: rURF=1.0; //1.855; 
			      break;
		case VX : rURF=1.0;  // ��������� �� �������� ��������� � �� ����� ������ ����������. ����� �������� ������������.
			      break; // ��� ������ ���������� ������������ ��� ������������ ������� ����.
		case VY : rURF=1.0;
			      break;
		case VZ : rURF=1.0;
			      break;
		default : rURF=1.0; break; // � ��������� �������.
	}
	// ��������� �������� �������
	Real eps = 1e-25;
	Real ptilda;
	Real sE,sW,sN,sS,sT,sB,sI;
	int i=0,j=0, kend=4000;//100; // ��� ����� ���� ����������� ������ ������� 40 ��������.
	Real dmax=1.0;
	while ((dmax>eps) && (j<kend)) {
		dmax=0.0;
        #pragma omp parallel for private(i,ptilda,sE,sW,sN,sS,sT,sB) shared(maxelm,x,rURF,sl) schedule (guided)
	    for (i=0; i<maxelm; i++) {
            if (sl[i].iE>-1) sE=sl[i].ae*x[sl[i].iE]; else sE=0.0;
		    if (sl[i].iW>-1) sW=sl[i].aw*x[sl[i].iW]; else sW=0.0;
		    if (sl[i].iN>-1) sN=sl[i].an*x[sl[i].iN]; else sN=0.0;
		    if (sl[i].iS>-1) sS=sl[i].as*x[sl[i].iS]; else sS=0.0;
            if (sl[i].iT>-1) sT=sl[i].at*x[sl[i].iT]; else sT=0.0;
		    if (sl[i].iB>-1) sB=sl[i].ab*x[sl[i].iB]; else sB=0.0;
		    ptilda=(sE+sW+sN+sS+sT+sB+sl[i].b)/sl[i].ap;
		    //dmax=fmax(dmax,sl[i].ap*(ptilda-x[sl[i].iP]));
			dmax=fmax(dmax,fabs(sl[i].ap*(ptilda-x[sl[i].iP])));
		    x[sl[i].iP]=x[sl[i].iP]+rURF*(ptilda-x[sl[i].iP]);
			
			/*
			if (0&&j==0) {// debug
				printf("ae=%e, aw=%e, an=%e, as=%e, at=%e, ab=%e, ap=%e, b=%e\n",sl[i].ae,sl[i].aw,sl[i].an,sl[i].as,sl[i].at,sl[i].ab,sl[i].ap,sl[i].b);
				getchar();
			}
			*//*
	    }
		#pragma omp parallel for private(i,ptilda,sI) shared(maxbound,x,rURF,slb) schedule (guided)
		for (i=0; i<maxbound; i++) {
			if (slb[i].iI>-1) sI=slb[i].ai*x[slb[i].iI]; else sI=0.0;
			ptilda=(sI+slb[i].b)/slb[i].aw;
			dmax=fmax(dmax,fabs(slb[i].aw*(ptilda-x[slb[i].iW])));
			//x[slb[i].iW]=x[slb[i].iW]+rURF*(ptilda-x[slb[i].iW]);
            if (slb[i].iI==-1) x[slb[i].iW]=(ptilda);
			else x[slb[i].iW]=x[slb[i].iW]+rURF*(ptilda-x[slb[i].iW]);
		}
		//dmax/=maxelm;
		//printf("%d %e \n", j+1, dmax);
		j++;
	}

} // SOR3D
*/

/* ����� ���������� ����������
*  ��� ����� ������������� ������� ����.
*/

// ��������� ������� �� ������
Real* MatrixByVector(Real** H,Real* V,int n){
	Real* tmp=new Real[n];
	Real sum=0.0;
	for (int i=0;i<n;++i){
		for (int j=0;j<n;++j)
			sum+=V[j]*H[i][j];
		tmp[i]=sum;
		sum=0.0;}
	return tmp;
} // MatrixByVector



// ��������� ����� �������
// ���������� �������.
Real NormaVdebug(Real *V, int n){
	Real norma;
	Real s=0.0;
	//#pragma omp parallel for shared(V) schedule (guided) reduction (+:s)
	for (int i=0;i<n;i++) {
		s+=V[i]*V[i];
		if (!(V[i]==V[i])) {
			printf("bitji vector i=%d\n",i);
			printf("%e ",V[i]);
			getchar();
		}
		//if (i%200==0) getchar();
	}
	printf("%e\n",s);
	if (!(s==s)) {
		// ��� NaN (Not a Number)
		// ���������������� NaN ��������� ��� ������������ ���� ����� �����
		// ����� ��� ���������� ��������� ����� ����� ��� ��� �� ��������� � ������ ������������ �����.
		// ��� ���������� ��������������.
		norma=0.0;
	}
	else {
		norma=sqrt(s);
	}
	printf("%e\n",norma);
	return norma;
} // NormaV

// ����� �������� ��� ����������� ��������� � ������� ��������
Real NormaChebyshev(Real *V, int n){
	Real norma=-1.0;
	int i=0;
	for (i=0; i<n; i++) if (fabs(V[i])>norma) norma=fabs(V[i]);
	return norma;
} // NormaChebyshev 

// ��������� ������������ ���� ��������
Real Scal(Real *v1, Real *v2, int n){
	Real s=0.0;
	int i; // ������� ����� for

    omp_set_num_threads(inumcore);

    #pragma omp parallel for shared(v1, v2, n) private(i) reduction (+: s) schedule (guided)
	for ( i=0; i<n; i++)
	{ 
		s+=v1[i]*v2[i];
	}
	return s;
} // Scal 

//----------����� ����������� ����������---------------
/* ������� ���������:
*  A - ������������� ������� ����,
*  dV - ������ ������ �����, 
*  x - ��������� ����������� � ������� ��� NULL.
*  n - ����������� ���� Anxn.
*  ������� A ���������� ������������ ����������� � 
*  ������������ (������������ ������������ ������������).
*  ���������� �������� ���������� 1000, �.�. ��������������,
*  ��� ���� ������� �� ������� �� 1000 �������� �� ��� � �� �������.
*  �������� ������ �� ������� ������� � ���������� ���������:
*  dterminatedTResudual.
*/
Real *SoprGrad(Real **A, Real *dV, Real *x, int n){
	printf("Reshenie metodom sopryjennyh gradientov:\n");
	int k=0;
	int i; // �������
	Real *ap=new Real[n],
		 *z=new Real[n], *p=new Real[n];
	Real a, b, nz;

	// ��� 1.1
	//X0==
	if (x==NULL) {
        x=new Real[n];
		for(i=0;i<n;i++) x[i] = 0.0;
	}

	// ��������� �������� �������
	Real e = dterminatedTResudual;
	
	// ��� 1.2
    // ���������� z - ������� ���������� �����������
	ap=MatrixByVector(A,x,n);
	for (i=0; i<n; i++) z[i]=dV[i]-ap[i];

	if (Scal(z,z,n)!=0){
		// ��� 1.3
	   for (i=0; i<n; i++)	p[i]=z[i];
	   nz=1000.;
	   while ((nz>e) && (k<1000)) {
		   // ��� 2.1
	 	  ap=MatrixByVector(A,p,n);
		  // ��� 2.2
		  //a=Scal(z,p,n)/Scal(z,ap,n);
		  a=Scal(z,p,n)/Scal(ap,p,n); // ������� ���������
		  // ��� 2.3 � 2.4
		  for (i=0; i<n; i++) {
		      x[i]+=a*p[i]; // ��������� �����������
			  z[i]-=a*ap[i]; // ������� k+1-�� �����������
		  }
		  // ��� 2.5
		  nz=NormaV(z,n);
		  if (k%10==0) printf("iter residual\n");
		  printf(" %d %e\n", k, nz);
		  // ��� 3.1
		  b=Scal(z,ap,n)/Scal(p,ap,n);
		  // ��� 3.2
		  for (i=0; i<n; i++) {
		     p[i]=z[i]-b*p[i]; // ����� ����������� �����������
		  }
          // ��� 3.3 
		  k++;
	   } // while

	   // ������������ ������
        delete ap;
		delete z; delete p;

	   return x;
	}
	else {
		// ������������ ������
		delete ap;
		delete z; delete p;

		return x;
	}
} // SoprGrad

/* �������� ��������� �������� CRS:
*  1. val - ��������� �������� ��������� ������� ���������������
*  �� ������� ����� (��������� ���������� � ����).
*  2. col_ind - ��������������� ��������� �� val ������ ��������.
*  3. row_ptr - ������������ ��� ����������� ������ ��������� ������.
*  ������:
*
*  9.0   0.0   0.0   3.0   1.0   0.0   1.0    
*  0.0   11.0   2.0   1.0   0.0   0.0   2.0    
*  0.0   2.0   10.0   2.0   0.0   0.0   0.0    
*  3.0   1.0   2.0   9.0   1.0   0.0   0.0    
*  1.0   0.0   0.0   1.0   12.0   0.0   1.0    
*  0.0   0.0   0.0   0.0   0.0   8.0   0.0    
*  1.0   2.0   0.0   0.0   1.0   0.0   8.0    
*
*------------- ����������� ������� ------------ 
* ������ ��������: CRS  
* val:      9.0 3.0 1.0 1.0 11.0 2.0 1.0 2.0 10.0 2.0 9.0 1.0 12.0 1.0 8.0 8.0 
* col_ind:  0 3 4 6 1 2 3 6 2 3 3 4 4 6 5 6 
* row_ptr:  0 4 8 10 12 14 15 16 
*------------------------------------------------------
*/


// ��������� ������� �� ������
// ������������ ������ �������� CRS
// ����������� ������� A (val, col_ind, row_ptr) ���������� �������� nxn.
// ����� ��������� ����� ����� ����������� � ����� n.
void MatrixCRSByVector(Real* val, int* col_ind, int* row_ptr, Real* V, Real* &tmp, int n)
{
	int i,j; // �������� �����

    // ������ tmp ������������� ������� � ���� ��� �� ��� � ������ V
	for (i=0; i<n; i++) tmp[i]=0.0;
	/*
	// � ����� ���������� �������������� 
	// ��� ����������� ������ ���������� �������.
	if (tmp == NULL)
	{
		printf("malloc: out of memory for vector tmp in MatrixCRSByVector\n"); // �������� ������
		getchar();
		exit(0);  // ���������� ���������
	}*/
	
	Real sum;
	int rowend, rowbeg;
    
	omp_set_num_threads(inumcore);

    #pragma omp parallel for shared(row_ptr, val, col_ind, V, tmp) private(sum, rowend, rowbeg, i, j) schedule (guided)
	for (i=0; i<n; i++) {
	    sum = 0.0;
		rowend=row_ptr[i+1];
		rowbeg=row_ptr[i];
	    for (j = rowbeg; j<rowend; j++)
		{
		    	sum += val[j]*V[col_ind[j]];
		}
		tmp[i] = sum;
	}
	
	//return tmp;
} // MatrixCRSByVector

// ��������� ������� �� ������ (���������� ������� ��� ������ ������).
// ������������ ������ �������� CRS
// ����������� ������� A (val, col_ind, row_ptr) ���������� �������� nxn.
// ����� ��������� ����� ����� ����������� � ����� n.
void MatrixCRSByVectordebug(Real* val, int* col_ind, int* row_ptr, Real* V, Real* &tmp, int n)
{
	int i,j; // �������� �����

    // ������ tmp ������������� ������� � ���� ��� �� ��� � ������ V
	for (i=0; i<n; i++) tmp[i]=0.0;
	/*
	// � ����� ���������� �������������� 
	// ��� ����������� ������ ���������� �������.
	if (tmp == NULL)
	{
		printf("malloc: out of memory for vector tmp in MatrixCRSByVector\n"); // �������� ������
		getchar();
		exit(0);  // ���������� ���������
	}*/
	
	Real sum;
	int rowend, rowbeg;
    
	omp_set_num_threads(inumcore);

    //#pragma omp parallel for shared(row_ptr, val, col_ind, V, tmp) private(sum, rowend, rowbeg, i, j) schedule (guided)
	for (i=0; i<n; i++) {
	    sum = 0.0;
		if (i>23897) {
			printf("diagnostic message node %i\n",i);
			printf("start=%i, end=%i\n",row_ptr[i],row_ptr[i+1]);
			for (j = rowbeg; j<rowend; j++)
		    {
		 	    printf("val=%e, V=%e, col_ind=%d, j=%i\n",val[j],V[col_ind[j]],col_ind[j],j);
			}
			printf("diagnostic message node %i\n",i);
			getchar();
		}
		rowend=row_ptr[i+1];
		rowbeg=row_ptr[i];
	    for (j = rowbeg; j<rowend; j++)
		{
		    	sum += val[j]*V[col_ind[j]];
		}
		tmp[i] = sum;
	}
	
	//return tmp;
} // MatrixCRSByVectordebug

// ��������� ����������������� ������� �� ������
// (������������, ��������, � ������ BiCG - ������������ ����������)
// ��� �������� (�� ����������������� �������) ������������ ������ �������� CRS
// ����������� ������� A (val, col_ind, row_ptr) ���������� �������� nxn.
// ����� ��������� ����� ����� ����������� � ����� n.
Real* MatrixTransposeCRSByVector(Real* val, int* col_ind, int* row_ptr, Real* V, int n)
{
	
	Real* tmp=new double[n]; // ������ ������������� ������� � ���� ��� �� ��� � ������ V
	if (tmp == NULL)
	{
		printf("malloc: out of memory for vector tmp in MatrixTransposeCRSByVector\n"); // �������� ������
		getchar();
		exit(0);
		return NULL; // ���������� ���������
	}
	
	
    int i,j; // �������� �����
	int rowend, rowbeg;
    
	for (i=0; i<n; i++) tmp[i]=0.0;

	for (j=0; j<n; j++) {
		rowend=row_ptr[j+1];
		rowbeg=row_ptr[j];
	    for (i = rowbeg; i<rowend; i++)
		{
		    	tmp[col_ind[i]] += val[i]*V[j];
		}
	}
	
	return tmp;
} // MatrixTransposeCRSByVector


/* ����� ���������� ���������� �������� � ������� [1952]
*  ������� ���������:
*  val, col_ind, row_ptr - ����������� ������� ���� � ������� CRS,
*  dV - ������ ������ �����, 
*  x - ��������� ����������� � ������� ��� NULL.
*  n - ����������� ���� Anxn.
*  ����������� ������� A (val, col_ind, row_ptr) ���������� �������� nxn.
*  ����� ��������� ����� ����� ����������� � ����� n.
*  ������� A ���������� ������������ ����������� � 
*  ������������ (������������ ������������ ������������).
*  ���������� �������� ���������� 1000, �.�. ��������������,
*  ��� ���� ������� �� ������� �� 1000 �������� �� ��� � �� �������.
*  �������� ������ �� ������� ������� � ���������� ���������:
*  dterminatedTResudual.
*/
Real *SoprGradCRS(Real *val, int* col_ind, int* row_ptr, Real *dV, Real *x, int n){
	printf("Conjugate Gradients Method...:\n");
	int k=0;
	int i; // �������
	Real *ap=new Real[n],
		 *z=new Real[n], *p=new Real[n];
	Real a, b, nz;

    omp_set_num_threads(inumcore);

	// ��� 1.1
	//X0==
	if (x==NULL) {
        x=new Real[n];
		for(i=0;i<n;i++) x[i] = 0.0;
	}

	// ��������� �������� �������
	Real e = dterminatedTResudual;
	
	// ��� 1.2
    // ���������� z - ������� ���������� �����������
	MatrixCRSByVector(val,col_ind,row_ptr,x,ap,n);
	
    #pragma omp parallel for shared(z,dV,ap) private(i) schedule (guided)
	for (i=0; i<n; i++) z[i]=dV[i]-ap[i];

	if (Scal(z,z,n)!=0){
		// ��� 1.3
       #pragma omp parallel for shared(p,z) private(i) schedule (guided)
	   for (i=0; i<n; i++)	p[i]=z[i];

	   nz=1000.;
	   while ((nz>e) && (k<2*n)) {
		   // ��� 2.1
		  // ����� �������� ������ ������
	 	  MatrixCRSByVector(val,col_ind,row_ptr,p,ap,n);
		  // ��� 2.2
		  //a=Scal(z,p,n)/Scal(z,ap,n);
		  a=Scal(z,p,n)/Scal(ap,p,n); // ������� ���������
		  // ��� 2.3 � 2.4
		  #pragma omp parallel for shared(x,z,p,ap,a) private(i) schedule (guided)
		  for (i=0; i<n; i++) {
		      x[i]+=a*p[i]; // ��������� �����������
			  z[i]-=a*ap[i]; // ������� k+1-�� �����������
		  }
		  // ��� 2.5
		  nz=NormaV(z,n);
		  if (k%10==0) printf("iter residual\n");
		  printf(" %d %e\n", k, nz);
		  // ��� 3.1
		  b=Scal(z,ap,n)/Scal(p,ap,n);
		  // ��� 3.2
		  #pragma omp parallel for shared(p,z,b) private(i) schedule (guided)
		  for (i=0; i<n; i++) {
		     p[i]=z[i]-b*p[i]; // ����� ����������� �����������
		  }
          // ��� 3.3 
		  k++;
	   } // while

	   // ������������ ������
        delete ap;
		delete z; delete p;

	   return x;
	}
	else {
		// ������������ ������
		delete ap;
		delete z; delete p;

		return x;
	}
} // SoprGradCRS

// ����� ������������ ����������
// ��� �������� �������������� ������� � (val, col_ind, row_ptr).
// ����������������� �� ������ ��������, ������ : "������
// ������� ���� ������� �����������".
// dV - ������ ����� ����,
// x - ��������� ����������� � ������� ��� NULL.
// n - ����������� � nxn.
// ���������� �������� ���������� 2000.
// �������� ������ �� ������� ������� � ���������� ���������:
//  dterminatedTResudual.
void BiSoprGradCRS(Real *val, int* col_ind, int* row_ptr, Real *dV, Real* &x, int n, int maxit){
	printf("BiConjugate Gradients Method...:\n");

	Real *r=new Real[n], *r_tilda=new Real[n];
	Real *p=new Real[n], *p_tilda=new Real[n];
	Real nz; // �������
	Real *ap=new Real[n];
	Real a,b,dold, dnew;

	int i; // ������� ����� for
	int k=0; // ����� ��������.

	// ��������� �����������:
    //X0==
	if (x==NULL) {
        x=new Real[n];
		for(i=0;i<n;i++) x[i] = 0.0;
	}

	// ��������� �������� �������
	Real e = dterminatedTResudual;

	MatrixCRSByVector(val,col_ind,row_ptr,x,ap,n);
	for (i=0; i<n; i++) {
		r[i]=dV[i]-ap[i];
		r_tilda[i]=r[i];
		p[i]=r[i];
		p_tilda[i]=r_tilda[i];
	}

	nz=NormaV(r,n); // ��������� �������� �������
	dold=Scal(r,r_tilda,n);

    while ((nz>e) && (k<maxit)) {
		MatrixCRSByVector(val,col_ind,row_ptr,p,ap,n);

		a=dold/Scal(ap,p_tilda,n);
		for (i=0; i<n; i++) {
           x[i]+=a*p[i];
		   r[i]-=a*ap[i];
		}
		delete ap;
		ap=MatrixTransposeCRSByVector(val,col_ind,row_ptr,p_tilda,n);
        for (i=0; i<n; i++) {
			r_tilda[i]-=a*ap[i];
		}
		dnew=Scal(r,r_tilda,n);
		b=dnew/dold;
		dold=dnew;
		// ���������� �������.
        nz=NormaV(r,n);
		if (k%10==0) printf("iter residual\n");
		printf(" %d %e\n", k, nz);

		if (fabs(b) < 1e-270) {
			printf("\nBiCG divergence detected...\n");
            getchar();
			exit(0); // ����� �� ����������.
			break; // ����� �� ����� while
		}

        for (i=0; i<n; i++) {
			p[i]=r[i]+b*p[i];
			p_tilda[i]=r_tilda[i]+b*p_tilda[i];
		}

		k++; // ������� � ��������� ��������.
	}

	// ������������ ������
	delete r; delete r_tilda; 
	delete p; delete p_tilda;
	delete ap;

	//return x;

} // BiSoprGradCRS

// ������ ��� �� ����������� ���������������� ������� L.
// ������������ ������������ ����������� �������
// ���� A ������������ �������� ����������� ��������� 
// A~=L*transpose(L); L - ������ ����������� �������.
// L - �������� � ��������� ����:
// 1. ldiag - ������������ �������� L.
// 2. lltr - ��������������� �������� � �������� �������,
// �.�. �������� ����������. 
// 3. jptr - �������������� ������ �������� ��� lltr, 
// 4. iptr - ���������� � ������ ��������� ������ ��� lltr.
// f - ������ ������ ����� �������� nodes.
// ���������� ������ z=inverse(L)*f;
// ������ f ��������.
// ������ (CSIR - ������):
//  L = 
//  9.0   0.0   0.0   0.0   0.0   0.0   0.0   
//  0.0   11.0   0.0   0.0   0.0   0.0   0.0   
//  0.0   2.0   10.0   0.0   0.0   0.0   0.0   
//  3.0   1.0   2.0   9.0   0.0   0.0   0.0   
//  1.0   0.0   0.0   1.0   12.0   0.0   0.0   
//  0.0   0.0   0.0   0.0   0.0   8.0   0.0   
//  1.0   2.0   0.0   0.0   1.0   0.0   8.0   
// ------------------------------------------
// ldiag: 9.0 11.0 10.0 9.0 12.0 8.0 8.0
// lltr: 2.0 3.0 1.0 2.0 1.0 1.0 1.0 2.0 1.0
// jptr: 1 0 1 2 0 3 0 1 4
// iptr: 0 0 0 1 4 6 6 9
//-------------------------------------------
Real* inverseL(Real* f, Real* ldiag, Real* lltr, int* jptr, int* iptr, int n) {
	Real *z=new Real[n];

    if (z == NULL)
	{
		printf("malloc: out of memory for vector z in inverse(L)*f \n"); // �������� ������
		getchar();
		exit(0);
		return NULL; // ���������� ���������
	}

	int i,j;
	for (i=0; i<n; i++) {
		for (j=iptr[i]; j<iptr[i+1]; j++) {
			f[i]-=z[jptr[j]]*lltr[j];
		}
		z[i]=f[i]/ldiag[i];
	}
	return z;
}//inverseL

// ������ ��� �� ����������� ���������������� ������� L.
// ������������ ������������ ����������� �������
// ���� A ������������ �������� ����������� ��������� 
// A~=L*transpose(L); L - ������ ����������� �������.
// L - �������� � ��������� ����:
// 1. val - ������������ � ��������������� �������� L.
// � ���������� �������. 
// 3. indx - �������������� ������ ����� ��� val, 
// 4. pntr - ���������� � ������ ���������� �������.
// f - ������ ������ ����� �������� nodes.
// ���������� ������ z=inverse(L)*f;
// ������ f ��������.
// ������ (CSIR - ������):
//  L = 
//  9.0   0.0   0.0   0.0   0.0   0.0   0.0   
//  0.0   11.0   0.0   0.0   0.0   0.0   0.0   
//  0.0   2.0   10.0   0.0   0.0   0.0   0.0   
//  3.0   1.0   2.0   9.0   0.0   0.0   0.0   
//  1.0   0.0   0.0   1.0   12.0   0.0   0.0   
//  0.0   0.0   0.0   0.0   0.0   8.0   0.0   
//  1.0   2.0   0.0   0.0   1.0   0.0   8.0   
// ------------------------------------------
// val: 9.0 3.0 1.0 1.0 11.0 2.0 1.0 2.0 10.0 2.0 9.0 1.0 12.0 1.0 8.0 8.0
// indx: 0 3 4 6 1 2 3 6 2 3 3 4 4 6 5 6
// pntr: 0 4 8 10 12 14 15 16
//-------------------------------------------
void inverseL_ITL(Real* f, Real* val, int* indx, int* pntr, Real* &z, int n) {
	

    if (z == NULL)
	{
		Real *z=new Real[n];
		if (z==NULL) {
			printf("malloc: out of memory for vector z in inverse(L)*f \n"); // �������� ������
		    getchar();
		    exit(0); // ���������� ���������
		}
	}

	int i,j;
	for (i=0; i<n; i++) {
        z[i]=f[i]/val[pntr[i]];
		// ��������� i-�� �������
		// ��� ����� �� �������� �����������������.
        //#pragma omp parallel for shared(f, indx, z, val, i, pntr) private(j)
		for (j=pntr[i]+1; j<pntr[i+1]; j++) {
			f[indx[j]]-=z[i]*val[j];
		}
		
	}

}//inverseL_ITL

// �������� ��� �� ����������� ����������������� ������� U.
// ������������ ������������ ����������� �������
// ���� A ������������ �������� ����������� ��������� 
// A~=L*transpose(L); L - ������ ����������� �������.
// U=transpose(L);
// U - �������� � ��������� ����:
// 1. udiag - ������������ �������� U.
// 2. uutr - ��������������� �������� � ���������� �������,
// �.�. �������� ������������. 
// ��� ������� �����������, ��:
// 3. jptr - �������������� ������ �������� ��� lltr, 
// 4. iptr - ���������� � ������ ��������� ������ ��� lltr.
// f - ������ ������ ����� �������� nodes.
// ���������� ������ z=inverse(U)*f;
// ������ f ��������.
// ������ (CSIR - ������):
//  U=transpose(L) = 
//  9.0   0.0   0.0   3.0   1.0   0.0   1.0   
//  0.0   11.0   2.0   1.0   0.0   0.0   2.0   
//  0.0   0.0   10.0   2.0   0.0   0.0   0.0   
//  0.0   0.0   0.0   9.0   1.0   0.0   0.0   
//  0.0   0.0   0.0   0.0   12.0   0.0   1.0   
//  0.0   0.0   0.0   0.0   0.0   8.0   0.0   
//  0.0   0.0   0.0   0.0   0.0   0.0   8.0 
// ------------------------------------------
// udiag==ldiag: 9.0 11.0 10.0 9.0 12.0 8.0 8.0
// uutr==lltr: 2.0 3.0 1.0 2.0 1.0 1.0 1.0 2.0 1.0
// jptr: 1 0 1 2 0 3 0 1 4
// iptr: 0 0 0 1 4 6 6 9
//-------------------------------------------
Real* inverseU(Real* f, Real* udiag, Real* uutr, int* jptr, int* iptr, int n) {
	Real *z=new Real[n];

    if (z == NULL)
	{
		printf("malloc: out of memory for vector z in inverse(U)*f \n"); // �������� ������
		getchar();
		exit(0);
		return NULL; // ���������� ���������
	}

	int i,j;
	for (i=(n-1); i>=0; i--) {
        z[i]=f[i]/udiag[i];
		// ��������� i-�� ������� ��� ����������:
		for (j=iptr[i]; j<iptr[i+1]; j++) {
			f[jptr[j]]-=z[i]*uutr[j];
		}
		
	}
	return z;
}//inverseU

// �������� ��� �� ����������� ����������������� ������� U.
// ������������ ������������ ����������� �������
// ���� A ������������ �������� ����������� ��������� 
// A~=L*transpose(L); L - ������ ����������� �������.
// U=transpose(L); - ������� ����������� �������.
// U - �������� � ��������� ����:
// 1. val - ������������ � ��������������� �������� U (� ��������� �������).
// 2. indx - �������������� ������ ��������, 
// 3. pntr - ���������� � ������ ��������� ������ ��� val.
// f - ������ ������ ����� �������� nodes.
// ���������� ������ z=inverse(U)*f;
// ������ f ��������.
// ������ (CSIR_ITL - ������):
//  U=transpose(L) = 
//  9.0   0.0   0.0   3.0   1.0   0.0   1.0   
//  0.0   11.0   2.0   1.0   0.0   0.0   2.0   
//  0.0   0.0   10.0   2.0   0.0   0.0   0.0   
//  0.0   0.0   0.0   9.0   1.0   0.0   0.0   
//  0.0   0.0   0.0   0.0   12.0   0.0   1.0   
//  0.0   0.0   0.0   0.0   0.0   8.0   0.0   
//  0.0   0.0   0.0   0.0   0.0   0.0   8.0 
// ------------------------------------------
// val: 9.0 3.0 1.0 1.0 11.0 2.0 1.0 2.0 10.0 2.0 9.0 1.0 12.0 1.0 8.0 8.0
// indx: 0 3 4 6 1 2 3 6 2 3 3 4 4 6 5 6
// pntr: 0 4 8 10 12 14 15 16
//-------------------------------------------
void inverseU_ITL(Real* f, Real* val, int* indx, int* pntr, Real* &z, int n) {

    if (z == NULL)
	{
		z = new Real[n];
		if (z==NULL) {
			printf("malloc: out of memory for vector z in inverse(U)*f \n"); // �������� ������
		    getchar();
		    exit(0); // ���������� ���������
		}
	}

	int i,j;

	for (i=(n-1); i>=0; i--) {
        
		// ��������� i-�� ������:
		// ��� ����� �� �������� �����������������.
		//#pragma omp parallel for shared(f, indx, z, val, i, pntr) private(j)
		for (j=pntr[i]+1; j<pntr[i+1]; j++) {
			f[i]-=z[indx[j]]*val[j];
		}
		// ����� �� ������������ �������:
        z[i]=f[i]/val[pntr[i]];
		
	}
	
}//inverseU_ITL



// ��������� �� ������� CSIR � ������ CSIR_ITL
// �������:
// CSIR: ldiag, lltr, jptr, iptr
// CSIR_ITL: val, indx, pntr
// ������:
// A = 
// 9.0   0.0   0.0   3.0   1.0   0.0   1.0    
// 0.0   11.0   2.0   1.0   0.0   0.0   2.0    
// 0.0   2.0   10.0   2.0   0.0   0.0   0.0    
// 3.0   1.0   2.0   9.0   1.0   0.0   0.0    
// 1.0   0.0   0.0   1.0   12.0   0.0   1.0    
// 0.0   0.0   0.0   0.0   0.0   8.0   0.0    
// 1.0   2.0   0.0   0.0   1.0   0.0   8.0 
// ------------------------------------------
// ������ CSIR:
// ldiag: 9.0 11.0 10.0 9.0 12.0 8.0 8.0
// lltr: 2.0 3.0 1.0 2.0 1.0 1.0 1.0 2.0 1.0
// jptr: 1 0 1 2 0 3 0 1 4
// iptr: 0 0 0 1 4 6 6 9
//-------------------------------------------
//��������� ����������� ������ CSIR_ITL
//val : 9.0 3.0 1.0 1.0 11.0 2.0 1.0 2.0 10.0 2.0 9.0 1.0 12.0 1.0 8.0 8.0 
//indx: 0 3 4 6 1 2 3 6 2 3 3 4 4 6 5 6 
//pntr: 0 4 8 10 12 14 15 16 
//--------------------------------------------
void convertCSIRtoCSIR_ITL(Real *ldiag, Real *lltr, int *jptr, int *iptr, int n, int nz, Real* &val, int* &indx, int* &pntr, int nnz) {
	int i,j,k;
	//nnz=n+nz; // ������ �������� val � indx
	// ��������� ����������� ������:
	val = new Real[nnz];
	indx = new int[nnz];
	pntr = new int[n+1];
	for (i=0; i<=n; i++) pntr[i]=nnz;

	if ((val == NULL) || (indx == NULL) || (pntr == NULL))
	{
		printf("malloc: out of memory in convertCSIRtoCSIR_ITL \n"); // �������� ������
		getchar();
		exit(0); // ���������� ���������
	}

	// �������� :
	// �� ������� ��� ���� �������� ������� CSIR_ITL
	int ic=0; // ������� ��������� ���������
	for (k=0; k<n; k++) {
		// ���������� ������������� �������� k - �� ������
		val[ic]=ldiag[k];
		indx[ic]=k;
		pntr[k]=min(ic,pntr[k]);
		ic++;

		// ���������� ��������� ��������� k-�� �������
		// ������������ ������� � CSIR �������:
		for (i=1; i<n; i++) {
			for (j=iptr[i]; j<iptr[i+1]; j++)
				if (jptr[j] == k) {
					// ���������� �������� � k-�� �������
					val[ic]=lltr[j];
					indx[ic]=i;
                    pntr[k]=min(ic,pntr[k]);
					ic++;
				}
		}

	}

} // convertCSIRtoCSIR_ITL

// �������� ���������� ���������
// ��� ������������ ����������� ������������
// ������� � �������� nxn.
// n - ����������� ������� ����
// ������� val ���������� � � ��� ������������
// �������� ���������� ��������� IC(0):
// val == U ������� ����������� �������
// A = transpose(U)*U=L*transpose(L);
// L=transpose(U);
// ������:
// A = 
// 9.0   0.0   0.0   3.0   1.0   0.0   1.0    
// 0.0   11.0   2.0   1.0   0.0   0.0   2.0    
// 0.0   2.0   10.0   2.0   0.0   0.0   0.0    
// 3.0   1.0   2.0   9.0   1.0   0.0   0.0    
// 1.0   0.0   0.0   1.0   12.0   0.0   1.0    
// 0.0   0.0   0.0   0.0   0.0   8.0   0.0    
// 1.0   2.0   0.0   0.0   1.0   0.0   8.0 
//������ CSIR_ITL (������� ����������� �������� ���������).
// val : 9.0 3.0 1.0 1.0 11.0 2.0 1.0 2.0 10.0 2.0 9.0 1.0 12.0 1.0 8.0 8.0 
// indx: 0 3 4 6 1 2 3 6 2 3 3 4 4 6 5 6 
// pntr: 0 4 8 10 12 14 15 16 
//--------------------------------------------
// ��������� ������������ ��� ����������:
// ��������� ������ val (indx � pntr �������� ��� ���������):
// val (factorization)= 
// 3.0
// 1.0
// 0.3333333333333333
// 0.3333333333333333
// 3.3166247903554
// 0.6030226891555273
// 0.30151134457776363
// 0.6030226891555273
// 3.1622776601683795
// 0.6324555320336759
// 2.932575659723036
// 0.34099716973523675
// 3.4472773213410837
// 0.2578524458667825
// 2.8284271247461903
// 2.7310738989293286
//-------------------------------------------
void IC0Factor_ITL(Real* val, int* indx, int* pntr, int n)
{
  int d, g, h, i, j, k;
  Real z;

  for (k = 0; k < n - 1; k++) {
    d = pntr[k];
    z = val[d] = sqrt(val[d]);

    for (i = d + 1; i < pntr[k+1]; i++)
      val[i] /= z;

    for (i = d + 1; i < pntr[k+1]; i++) {
      z = val[i];
      h = indx[i];
      g = i;

      for (j = pntr[h] ; j < pntr[h+1]; j++)
        for ( ; g < pntr[k+1] && indx[g+1] <= indx[j]; g++)
          if (indx[g] == indx[j])
             val[j] -= z * val[g];
    }
  }
  d = pntr[n-1];
  val[d] = sqrt(val[d]);
} // IC0Factor_ITL

// ���������������� �������� ���������� ���������.
void IC0FactorModify_ITL(Real* val, int* indx, int* pntr, int n)
{
  int d, g, h, i, j, k;
  Real z, accumulate_fill_in;

  for (k = 0; k < n - 1; k++) {
    d = pntr[k];
    z = val[d] = sqrt(val[d]);

    for (i = d + 1; i < pntr[k+1]; i++)
      val[i] /= z;

    for (i = d + 1; i < pntr[k+1]; i++) {
      z = val[i];
      h = indx[i];
      g = i;

      accumulate_fill_in = 0.0;

      for (j = pntr[h] ; j < pntr[h+1]; j++)
        for ( ; g < pntr[k+1] && indx[g+1] <= indx[j]; g++)
          if (indx[g] == indx[j]) // ������ �������� �����
             val[j] -= z * val[g];
	  else //index does not match accumulate the fill-in value
		  accumulate_fill_in += z * val[g];

	  val[pntr[h]] -= accumulate_fill_in;

    }
  }
  d = pntr[n-1];
  val[d] = sqrt(val[d]);
} // IC0FactorModify_ITL

// ��������� �� ������� CSIR_ITL � ������ CSIR (�������� ��������������)
// ������ ��� ��� ������� �������������� ���������� �������!!!
// �������:
// CSIR_ITL: val, indx, pntr
// CSIR: ldiag, lltr, jptr, iptr
// ������:
// A = 
// 9.0   0.0   0.0   3.0   1.0   0.0   1.0    
// 0.0   11.0   2.0   1.0   0.0   0.0   2.0    
// 0.0   2.0   10.0   2.0   0.0   0.0   0.0    
// 3.0   1.0   2.0   9.0   1.0   0.0   0.0    
// 1.0   0.0   0.0   1.0   12.0   0.0   1.0    
// 0.0   0.0   0.0   0.0   0.0   8.0   0.0    
// 1.0   2.0   0.0   0.0   1.0   0.0   8.0 
// ------------------------------------------
//��������� ����������� ������ CSIR_ITL
//val : 9.0 3.0 1.0 1.0 11.0 2.0 1.0 2.0 10.0 2.0 9.0 1.0 12.0 1.0 8.0 8.0 
//indx: 0 3 4 6 1 2 3 6 2 3 3 4 4 6 5 6 
//pntr: 0 4 8 10 12 14 15 16 
//--------------------------------------------
// ������ CSIR:
// ldiag: 9.0 11.0 10.0 9.0 12.0 8.0 8.0
// lltr: 2.0 3.0 1.0 2.0 1.0 1.0 1.0 2.0 1.0
// jptr: 1 0 1 2 0 3 0 1 4
// iptr: 0 0 0 1 4 6 6 9
//-------------------------------------------
void convertCSIR_ITLtoCSIR(Real* ldiag, Real* lltr, int* jptr, int* iptr, int n, int nz, Real* val, int* indx, int* pntr, int nnz) {
	int i,j,k;//,k1;
	int imin=1;
	//nz=nnz-n; // ������ �������� lltr � jptr
	// ������ �������������� ���������� �������!!!
	// jptr � iptr ���������� �� �����
	for (i=0; i<n; i++) ldiag[i]=0.0;
	for (i=0; i<nz; i++) {
		lltr[i]=0.0;
		//jptr[i]=0;
	}
	//for (i=0; i<=n; i++) iptr[i]=nz;


	// �������� :
	// �� ������� ��� ���� ����� ������� CSIR
	int ic=0; // ������� ��������� ���������
	for (k=0; k<n; k++) {
		// ���������� ������������� �������� k - �� ������
		ldiag[k]=val[pntr[k]];

		// ���������� ��������� ��������� k-�� ������
		// ������������ ������� � CSIR_ITL �������:
		for (i=0; i<n-1; i++) {
			for (j=pntr[i]+1; j<pntr[i+1]; j++)
				if (indx[j] == k) {
					// ���������� �������� � k-�� ������
					lltr[ic]=val[j];
					//jptr[ic]=i;
					//imin=min(ic,iptr[k]);
                    //iptr[k]=imin;
					//if (imin==0) {
					//	for (k1=0; k1<k; k1++) iptr[k1]=0;
					//}
					ic++;
				}
		}

	}

} // convertCSIR_ITLtoCSIR

// �������� ���������� ��������� IC(0).
// ������� ������ ������ ����������� ������������ ������� � ������� CSIR.
// ������ ��������� ���� �������������� � ������� CSIR_ITL ���������� �������� ITL.
void ICFactor0(Real* ldiag, Real* lltr, int* jptr, int* iptr, int n, int nz) {
    
	Real *val;
	int *indx, *pntr;

	// ������ ���������� ��������� ������
	// �������������� (������ � ��������) ����������� �������� ��� ������� ������,
	// ������� �� �� ����� ����������.
	convertCSIRtoCSIR_ITL(ldiag, lltr, jptr, iptr, n, nz, val, indx, pntr, n+nz);
	printf("Incoplete Cholesky 49.9%%...\n");
	IC0Factor_ITL(val, indx, pntr, n);
	printf("Incoplete Cholesky 50%%...\n");
    convertCSIR_ITLtoCSIR(ldiag, lltr, jptr, iptr, n, nz, val, indx, pntr, n+nz);
	printf("Incoplete Cholesky 100%%...\n");

	// ������������ ������
	delete val; delete indx; delete pntr;
} // ICFactor0


// ��������� ������������ ������������ �����������  ������� �� ������ 
// ������������ ������ �������� CSIR. � ���� ��������� �������� ������ ��������������� �������� altr. 
// ����������� SPD ������� A (adiag, altr, jptr, iptr) ���������� �������� nxn.
// ����� ��������� ����� ����� ����������� � ����� n.
// ������:
// A = 
// 9.0   0.0   0.0   3.0   1.0   0.0   1.0    
// 0.0   11.0   2.0   1.0   0.0   0.0   2.0    
// 0.0   2.0   10.0   2.0   0.0   0.0   0.0    
// 3.0   1.0   2.0   9.0   1.0   0.0   0.0    
// 1.0   0.0   0.0   1.0   12.0   0.0   1.0    
// 0.0   0.0   0.0   0.0   0.0   8.0   0.0    
// 1.0   2.0   0.0   0.0   1.0   0.0   8.0 
// ------------------------------------------
// ������ CSIR:
// adiag: 9.0 11.0 10.0 9.0 12.0 8.0 8.0
// altr: 2.0 3.0 1.0 2.0 1.0 1.0 1.0 2.0 1.0
// jptr: 1 0 1 2 0 3 0 1 4
// iptr: 0 0 0 1 4 6 6 9
//-------------------------------------------
void  SPDMatrixCSIRByVector(Real* adiag, Real* altr, int* jptr, int* iptr, Real* V, Real* &tmp, int n)
{
	
	// ������ tmp ������������� ������� � ���� ��� �� ��� � ������ V
	if (tmp == NULL)
	{
		printf("in SPDMatrixCSIRByVector tmp==NULL\n");
		getchar();
		tmp =new Real[n];
		if (tmp==NULL) {
			printf("malloc: out of memory for vector tmp in SPDMatrixCSIRByVector\n"); // �������� ������
		    getchar();
		    exit(0); // ���������� ���������
		}
	}
	
	
    int i,j; // �������� �����
    

	omp_set_num_threads(inumcore);

    #pragma omp parallel for shared(tmp, V, adiag) private(i) schedule (guided)
	for (i=0; i<n; i++) tmp[i]=V[i]*adiag[i];

    // ���������������� ������
	/*
	for (i=0; i<n; i++) {
	    for (j = iptr[i]; j<iptr[i+1]; j++)
		{
		    tmp[i] += V[jptr[j]]*altr[j];
		    tmp[jptr[j]] += V[i]*altr[j];
		}
	}
	*/

	// ����� ������ �� ����.
	#pragma omp parallel for shared(tmp, V, altr, iptr, jptr,n) private(i,j) schedule (guided)
    for (i=0; i<n; i++) {
	    for (j = iptr[i]; j<iptr[i+1]; j++)
		{
		    tmp[i] += V[jptr[j]]*altr[j];
		}
	}

	// ������ ����� �� �������� �����������������
    for (i=0; i<n; i++) {

		// ��� ����� �� �������� �����������������.
        //#pragma omp parallel for shared(tmp, V, altr, i, iptr, jptr) private(j)
	    for (j = iptr[i]; j<iptr[i+1]; j++)
		{
			tmp[jptr[j]] += V[i]*altr[j];
		}
	}

} // SPDMatrixCSIRByVector

// ��������� �������������� ������������ �����������  ������� �� ������ 
// ������������ ������ �������� CSIR.  
// ����������� ������� A (adiag, altr, autr, jptr, iptr) ���������� �������� nxn.
// ����� ��������� ����� ����� ����������� � ����� n.
// ��������� adiag �������� ��������. ������ ����������� altr �������� ���������.
// ������� ����������� �������� �� �������� autr. ������� ������� (������� ��������� 
// ��������� ) �������������� ������������. ������ jptr - ������ �������� ��� ������� 
// ������������, ������ iptr - ���������� ��� ���������� ����� ������ ��� ������� ������������.
// ������:
// A = 
// 9.0   0.0   0.0   3.0   1.0   0.0   1.0    
// 0.0   11.0   2.0   1.0   0.0   0.0   2.0    
// 0.0   1.0   10.0   2.0   0.0   0.0   0.0    
// 2.0   1.0   2.0   9.0   1.0   0.0   0.0    
// 1.0   0.0   0.0   1.0   12.0   0.0   1.0    
// 0.0   0.0   0.0   0.0   0.0   8.0   0.0    
// 2.0   2.0   0.0   0.0   3.0   0.0   8.0 
// ------------------------------------------
// ������ CSIR:
// adiag: 9.0 11.0 10.0 9.0 12.0 8.0 8.0
// altr: 1.0  2.0 1.0 2.0  1.0 1.0  2.0 2.0 3.0
// autr: 2.0 3.0 1.0 2.0 1.0 1.0 1.0 2.0
// jptr: 1 0 1 2 0 3 0 1 4
// iptr: 0 0 0 1 4 6 6 9
//-------------------------------------------
Real* MatrixCSIRByVector(Real* adiag, Real* altr, Real* autr, int* jptr, int* iptr, Real* V, int n)
{
	
	Real* tmp=new double[n]; // ������ ������������� ������� � ���� ��� �� ��� � ������ V
	if (tmp == NULL)
	{
		printf("malloc: out of memory for vector tmp in SPDMatrixCSIRByVector\n"); // �������� ������
		getchar();
		exit(0);
		return NULL; // ���������� ���������
	}
	
	
    int i,j; // �������� �����

	for (i=0; i<n; i++) tmp[i]=V[i]*adiag[i];

    
	for (i=0; i<n; i++) {
	    for (j = iptr[i]; j<iptr[i+1]; j++)
		{
		    	tmp[i] += V[jptr[j]]*altr[j];
		        tmp[jptr[j]] += V[i]*autr[j];
		}
	}
	
	return tmp;
} // MatrixCSIRByVector

// ��������� ����������������� �������������� ������������ �����������  ������� �� ������ 
// ������������ ������ �������� CSIR.  
// ����������� ������� A (adiag, altr, autr, jptr, iptr) ���������� �������� nxn. �������� 
// ������ �������� �������, � ���������� � ����������������� �������.
// ����� ��������� ����� ����� ����������� � ����� n.
// ��������� adiag �������� ��������. ������ ����������� altr �������� ���������.
// ������� ����������� �������� �� �������� autr. ������� ������� (������� ��������� 
// ��������� ) �������������� ������������. ������ jptr - ������ �������� ��� ������� 
// ������������, ������ iptr - ���������� ��� ���������� ����� ������ ��� ������� ������������.
// ������:
// A = 
// 9.0   0.0   0.0   3.0   1.0   0.0   1.0    
// 0.0   11.0   2.0   1.0   0.0   0.0   2.0    
// 0.0   1.0   10.0   2.0   0.0   0.0   0.0    
// 2.0   1.0   2.0   9.0   1.0   0.0   0.0    
// 1.0   0.0   0.0   1.0   12.0   0.0   1.0    
// 0.0   0.0   0.0   0.0   0.0   8.0   0.0    
// 2.0   2.0   0.0   0.0   3.0   0.0   8.0 
// ------------------------------------------
// ������ CSIR:
// adiag: 9.0 11.0 10.0 9.0 12.0 8.0 8.0
// altr: 1.0  2.0 1.0 2.0  1.0 1.0  2.0 2.0 3.0
// autr: 2.0 3.0 1.0 2.0 1.0 1.0 1.0 2.0
// jptr: 1 0 1 2 0 3 0 1 4
// iptr: 0 0 0 1 4 6 6 9
//-------------------------------------------
Real* MatrixTransposeCSIRByVector(Real* adiag, Real* altr, Real* autr, int* jptr, int* iptr, Real* V, int n)
{
	
	Real* tmp=new double[n]; // ������ ������������� ������� � ���� ��� �� ��� � ������ V
	if (tmp == NULL)
	{
		printf("malloc: out of memory for vector tmp in SPDMatrixCSIRByVector\n"); // �������� ������
		getchar();
		exit(0);
		return NULL; // ���������� ���������
	}
	
	
    int i,j; // �������� �����

	for (i=0; i<n; i++) tmp[i]=V[i]*adiag[i];

    
	for (i=0; i<n; i++) {
	    for (j = iptr[i]; j<iptr[i+1]; j++)
		{
		    	tmp[i] += V[jptr[j]]*autr[j];
		        tmp[jptr[j]] += V[i]*altr[j];
		}
	}
	
	return tmp;
} // MatrixTransposeCSIRByVector


/* ����� ���������� ���������� �������� � ������� [1952]
*  ������� ���������:
*  adiag, altr, jptr, iptr - ����������� ������� ���� � ������� CSIR,
*  dV - ������ ������ �����, 
*  x - ��������� ����������� � ������� ��� NULL.
*  n - ����������� ���� Anxn.
*  nz - ����������� �������� altr, jptr.
*  ����������� ������� A (adiag, altr, jptr, iptr) ���������� �������� nxn.
*  ����� ��������� ����� ����� ����������� � ����� n.
*  ������� A ���������� ������������ ����������� � 
*  ������������ (������������ ������������ ������������).
*  �������� ������ ������ ����������� � ���������� altr � adiag.
*  ���������� �������� ���������� 1000, �.�. ��������������,
*  ��� ���� ������� �� ������� �� 1000 �������� �� ��� � �� �������.
*  �������� ������ �� ������� ������� � ���������� ���������:
*  dterminatedTResudual.
*  � �������� ������������������� �������� �������� ���������� ���������:
*  M^(-1)==transpose(L)^(-1)*L^(-1); // ���������� �������������������.
*  
*/
Real *SoprGradCSIR(Real* adiag, Real* altr, int* jptr, int* iptr, Real *dV, Real *x, int n, int nz){

	printf("Reshenie metodom sopryjennyh gradientov:\n");
	int k=0;
	int i; // �������
	Real *ap=new Real[n], *vcopy=new Real[n],
		 *z=new Real[n], *p=new Real[n];
    Real a, b, res;
	
	// ��� ��������� ���������� ���������:
	Real  *ldiag=new Real[n], *lltr=new Real[nz];
	int *jptrsort=new int[nz];
	Real *f=new Real[n];

	Real dold, dnew;
	

	
	// �������������
	for (i=0; i<n; i++) ldiag[i]=adiag[i];
	for (i=0; i<nz; i++) lltr[i]=altr[i];
	// �������� ���������� ���������:
	// ���������� ����� ������ ����������� �����������.
	printf("Incoplete Cholesky decomposition beginig...:\n");
    ICFactor0(ldiag, lltr, jptr, iptr, n, nz);
	printf("Incoplete Cholesky decomposition finish...:\n");//*/

    
	for (i=0; i<nz; i++) jptrsort[i]=jptr[i];
	for (i=0; i<n; i++) QuickSort(jptrsort, iptr[i], iptr[i+1]-1);
    //printf("jptrsort...\n");
	//for (i=0; i<nz; i++) printf("%d ",jptrsort[i]); getchar();



	// ��� 1.1
	//X0==
	if (x==NULL) {
        x=new Real[n];
		for(i=0;i<n;i++) x[i] = 0.0;
	}

	// ��������� �������� �������
	Real e = dterminatedTResudual;
	
	// ��� 1.2
    // ���������� z - ������� ���������� �����������
	SPDMatrixCSIRByVector(adiag, altr, jptr, iptr, x, ap, n);
	for (i=0; i<n; i++) z[i]=dV[i]-ap[i];
	for (i=0; i<n; i++) vcopy[i]=z[i];
    f=inverseL(vcopy, ldiag, lltr, jptrsort, iptr, n);
    for (i=0; i<n; i++) vcopy[i]=f[i]; delete f; 
	f=inverseU(vcopy, ldiag, lltr, jptrsort, iptr, n);
    dnew=Scal(z,f,n);

	if (fabs(dnew)>1e-100){
		// ��� 1.3
	   for (i=0; i<n; i++)	p[i]=f[i];
	   res=1000.;
	   while ((fabs(res)>e) && (k<1000)) {
		   // ��� 2.1
		  SPDMatrixCSIRByVector(adiag, altr, jptr, iptr, p, ap, n);

		  // ��� 2.2
		  a=dnew/Scal(p,ap,n);// ������� ���������
		  // ��� 2.3 � 2.4
		  for (i=0; i<n; i++) {
		      x[i]+=a*p[i]; // ��������� ����������� 
              z[i]-=a*ap[i];// ������� k+1-�� �����������
		  }
          for (i=0; i<n; i++) vcopy[i]=z[i]; delete f; 
          f=inverseL(vcopy, ldiag, lltr, jptrsort, iptr, n);
          for (i=0; i<n; i++) vcopy[i]=f[i]; delete f; 
	      f=inverseU(vcopy, ldiag, lltr, jptrsort, iptr, n);
		  // ��� 2.5
          dold=dnew;
		  dnew=Scal(z,f,n);

		  
		  res=dnew;
		  if (k%10==0) printf("iter residual\n");
		  printf(" %d %e\n", k, res);
		  // ��� 3.1
		  b=dnew/dold;
		  // ��� 3.2
		  for (i=0; i<n; i++) {
		     p[i]=f[i]+b*p[i]; // ����� ����������� �����������
		  }
          // ��� 3.3
		  k++;
	   } // while

	   // ������������ ������
        delete ap; delete vcopy;
		delete z; delete p; delete f;

	   return x;
	}
	else {
		// ������������ ������
		delete ap; delete vcopy;
		delete z; delete p; delete f;

		return x;
	}
} // SoprGradCSIR


// ������� ���������� ���� ������������� ������� ���� �.
// ������� ���� � ������� � CSIR ������� : adiag, altr, jptr, iptr.
// �������� ���������� ��������� ��� � ������������ � ���������� � ����:
// A = L*transpose(L); � ������� �����������. ������� jptr �  iptr �������� ���� ��.
// ����� ������� : A~=inverse(L)*A*inverse(transpose(L)) ���� ����������� � ������������ ����������.
// ������ ����� ��������������� ������� ����� ���: dV~=inverse(L)*dV.
// ������� ���� ����� ����� A~*x~=dV~; => x~=transpose(L)*x; => x=inverse(transpose(L))*x~;
// ������������������ �������� ������������ ��������� ��������� ���������� �������� ��� ������� ����,
// �������� ������������ �������������� ������� ����.
Real *SoprGradCSIR2(Real* adiag, Real* altr, int* jptr, int* iptr, Real *dV, Real *x, int n, int nz0){
	printf("Reshenie metodom sopryjennyh gradientov:\n");
	int k=0;
	int i; // �������
	Real *ap, *vcopy=new Real[n],
		 *z=new Real[n], *p=new Real[n];
	Real a, b, nz;

    // ��� ��������� ���������� ���������:
	Real  *ldiag=new Real[n], *lltr=new Real[nz0];
	int *jptrsort=new int[nz0];


    // �������������
	for (i=0; i<n; i++) ldiag[i]=adiag[i];
	for (i=0; i<nz0; i++) lltr[i]=altr[i];
	// �������� ���������� ���������:
	// ���������� ����� ������ ����������� �����������.
	printf("Incoplete Cholesky decomposition beginig...:\n");
    ICFactor0(ldiag, lltr, jptr, iptr, n, nz0);
	printf("Incoplete Cholesky decomposition finish...:\n");//*/
   


   /*
	ldiag[0]=1.0; ldiag[1]=1.0;  ldiag[2]=1.838477; ldiag[3]=2.00055;
    ldiag[4]=0.590477; ldiag[5]=1.0;  ldiag[6]=1.0;
	lltr[0]=-1.22383866; lltr[1]=-0.5439282932;  lltr[2]=-1.33247070; //*/
    
    /* // ������������ ��������
	ldiag[0]=1.0; ldiag[1]=1.0;  ldiag[2]=1.838477; ldiag[3]=2.00055;
    ldiag[4]=0.590477; ldiag[5]=1.465913;  ldiag[6]=0.37585673;
	lltr[0]=-1.22383866; lltr[1]=-1.33247070;  lltr[2]=-0.5439282932; lltr[3]=-0.1457305633;
    lltr[4]=-0.4998613742; lltr[5]=-1.401073265;  lltr[6]=-0.06498197865;//*/

	for (i=0; i<nz0; i++) jptrsort[i]=jptr[i];
	for (i=0; i<n; i++) QuickSort(jptrsort, iptr[i], iptr[i+1]-1);

	// ��� 1.1
	//X0==
	if (x==NULL) {
        x=new Real[n];
		for(i=0;i<n;i++) x[i] = 0.0;
	}

	// ��������� �������� �������
	Real e = dterminatedTResudual;
	
	// ��� 1.2
    // ���������� z - ������� ���������� �����������
	//ap=SPDMatrixCSIRByVector(adiag, altr, jptr, iptr, x, n);
	//for (i=0; i<n; i++) z[i]=dV[i]-ap[i];

	for(i=0;i<n;i++) vcopy[i]=x[i]; 
    ap=inverseU(vcopy, ldiag, lltr, jptrsort, iptr, n);
	for(i=0;i<n;i++) vcopy[i]=ap[i]; 
    SPDMatrixCSIRByVector(adiag, altr, jptr, iptr, vcopy, ap, n);
    for(i=0;i<n;i++) vcopy[i]=ap[i]; delete ap;
    ap=inverseL(vcopy, ldiag, lltr, jptrsort, iptr, n);
    for(i=0;i<n;i++) vcopy[i]=dV[i]; delete dV;
	dV=inverseL(vcopy, ldiag, lltr, jptrsort, iptr, n);
    
    for (i=0; i<n; i++) z[i]=dV[i]-ap[i];

	if (Scal(z,z,n)!=0){
		// ��� 1.3
	   for (i=0; i<n; i++)	p[i]=z[i];
	   nz=1000.;
	   while ((nz>e) && (k<1000)) {
		   // ��� 2.1
	 	  //ap=SPDMatrixCSIRByVector(adiag, altr, jptr, iptr, p, n);

		   delete ap; // ������������ ������
           for(i=0;i<n;i++) vcopy[i]=p[i];
          ap=inverseU(vcopy, ldiag, lltr, jptrsort, iptr, n);
           for(i=0;i<n;i++) vcopy[i]=ap[i]; 
          SPDMatrixCSIRByVector(adiag, altr, jptr, iptr, vcopy, ap, n);
          for(i=0;i<n;i++) vcopy[i]=ap[i]; delete ap;
          ap=inverseL(vcopy, ldiag, lltr, jptrsort, iptr, n);

		  // ��� 2.2
		  //a=Scal(z,p,n)/Scal(z,ap,n);
		  a=Scal(z,p,n)/Scal(ap,p,n); // ������� ���������
		  // ��� 2.3 � 2.4
		  for (i=0; i<n; i++) {
		      x[i]+=a*p[i]; // ��������� �����������
			  z[i]-=a*ap[i]; // ������� k+1-�� �����������
		  }
		  // ��� 2.5
		  nz=NormaV(z,n);
		  if (k%10==0) printf("iter residual\n");
		  printf(" %d %e\n", k, nz);
		  // ��� 3.1
		  b=Scal(z,ap,n)/Scal(p,ap,n);
		  // ��� 3.2
		  for (i=0; i<n; i++) {
		     p[i]=z[i]-b*p[i]; // ����� ����������� �����������
		  }
          // ��� 3.3 
		  k++;
	   } // while

	   // ������������ ������
        delete ap; delete vcopy;
		delete z; delete p;

		for(i=0;i<n;i++) vcopy[i]=x[i]; delete x;
		x=inverseU(vcopy, ldiag, lltr, jptrsort, iptr, n);
	   return x;
	}
	else {
		// ������������ ������
		delete ap; delete vcopy;
		delete z; delete p;

		return x;
	}
} // SoprGradCSIR2

/* ����� ���������� ���������� �������� � ������� [1952]
*  ������� ���������:
*  M - ����������� ������� ���� � ������� SIMPLESPARSE,
*  dV - ������ ������ �����, 
*  x - ��������� ����������� � ������� ��� NULL.
*  n - ����������� ���� Anxn.
*
*  ����������� ������� M ���������� �������� nxn.
*  ����� ��������� ����� ����� ����������� � ����� n.
*  ������� M �������������� ������������ ����������� � 
*  ������������ (������������ ������������ ������������).
*  �������� ������ ��������� ��������. 
*  ���������� �������� ���������� 1000, �.�. ��������������,
*  ��� ���� ������� �� ������� �� 1000 �������� �� ��� � �� �������.
*  �������� ������ �� ������� ������� � ���������� ���������:
*  dterminatedTResudual.
*  � �������� ������������������� �������� �������� ���������� ���������:
*  K^(-1)==transpose(L)^(-1)*L^(-1); // ���������� �������������������.
* 
*  ������� �������� ��������������, ��� ��� 4 ������� ��������� �������� �� 54%
*  � ���������, ��������� �������� �� ��������� �����������������.
*/
void ICCG(SIMPLESPARSE &M, Real *dV, Real* &x, int n, bool bprintmessage, bool bdistwall, int maxiter)
{

	// ���� bdistwall==true �� �������� ���� ��� ���������� ����������� ���������� �� ������.

	if (bprintmessage) {
		printf("Reshenie metodom sopryjennyh gradientov:\n");
		fprintf(fp_log,"Reshenie metodom sopryjennyh gradientov:\n");
	}
    // ������� ����
	// � ������� CSIR:
	Real *adiag=NULL, *altr=NULL;
	int *jptr=NULL, *iptr=NULL;

	// �������������������:
	// �������� ����������� ���������� �
	// ������� CSIR_ITL:
	Real *val=NULL;
	int *indx=NULL, *pntr=NULL;
	
	
	
	
	// �������������
	// ������ ���������� ������:
	simplesparsetoCSIR(M, adiag, altr, jptr, iptr, n);
	simplesparsetoCSIR_ITLSPD(M, val, indx, pntr, n);
	//printf("max memory fics 2...\n"); // debug
	//getchar();
	simplesparsefree(M, n);

	int k=0;
	int i; // �������
	Real *ap=new Real[n], *vcopy=new Real[n], *f=new Real[n],
		 *z=new Real[n], *p=new Real[n];
    Real a, b, res, dbuf;
	

	Real dold, dnew;

	// �������� ���������� ���������:
	// ���������� ����� ������ ����������� �����������.
	if (bprintmessage) {
		printf("Incoplete Cholesky decomposition beginig...:\n");
		fprintf(fp_log,"Incoplete Cholesky decomposition beginig...:\n");
	}
	//IC0Factor_ITL(val, indx, pntr, n);
	IC0FactorModify_ITL(val, indx, pntr, n);
	if (bprintmessage) {
		printf("Incoplete Cholesky decomposition finish...:\n");//*/
		fprintf(fp_log,"Incoplete Cholesky decomposition finish...:\n");
	}


	// ��� 1.1
	//X0==
	if (x==NULL) {
        x=new Real[n];
		for(i=0;i<n;i++) x[i] = 0.0;
	}

	// ��������� �������� �������
	Real e = dterminatedTResudual;
	
	// ��� 1.2
    // ���������� z - ������� ���������� �����������
	SPDMatrixCSIRByVector(adiag, altr, jptr, iptr, x, ap, n);
	for (i=0; i<n; i++) z[i]=dV[i]-ap[i];
	// �������� ���������� � ��������� ������ ������� � ��������� ����:
	if (!bdistwall) fprintf(fp_statistic_convergence,"%+.16f ",NormaV(z,n));
	for (i=0; i<n; i++) vcopy[i]=z[i]; 
    inverseL_ITL(vcopy, val, indx, pntr, f, n);
    for (i=0; i<n; i++) vcopy[i]=f[i];  
	inverseU_ITL(vcopy, val, indx, pntr, f, n);
    dnew=Scal(z,f,n);

	
	// ����������� ������� ������ �� �������� ������������� ������ ��������� �������.
	if (e*dnew<e) e*=dnew;
	dterminatedTResudual=e;
	
	
	
	if (fabs(dnew) > e) {
		// ��� 1.3
	   for (i=0; i<n; i++)	p[i]=f[i];
	   res=1000.;
	   while ((fabs(res)>e) && (k<maxiter)) {
		   // ��� 2.1
		  SPDMatrixCSIRByVector(adiag, altr, jptr, iptr, p, ap, n);

		  // ��� 2.2
		  a=dnew/Scal(p,ap,n);// ������� ���������
		  // ��� 2.3 � 2.4
          #pragma omp parallel for shared(x,z,p,ap,a,n) private(i) schedule (guided)
		  for (i=0; i<n; i++) {
		      x[i]+=a*p[i]; // ��������� ����������� 
              z[i]-=a*ap[i];// ������� k+1-�� �����������
		  }
          #pragma omp parallel for shared(vcopy,z,n) private(i) schedule (guided)
          for (i=0; i<n; i++) vcopy[i]=z[i];  
          inverseL_ITL(vcopy, val, indx, pntr, f, n);
          #pragma omp parallel for shared(vcopy,f,n) private(i) schedule (guided)
          for (i=0; i<n; i++) vcopy[i]=f[i]; 
	      inverseU_ITL(vcopy, val, indx, pntr, f, n);
		  // ��� 2.5
          dold=dnew;
		  dnew=Scal(z,f,n);

		  
		  res=dnew;
		  if (bprintmessage) {
			  if (k%10==0) {
				  printf("iter residual\n");
				  fprintf(fp_log,"iter residual\n");
			  }
		      printf(" %d %e\n", k, res);
			  fprintf(fp_log," %d %e\n", k, res);
		  }
		  // ��� 3.1
		  b=dnew/dold;
		  // ��� 3.2

          #pragma omp parallel for shared(p,f,b,n) private(i,dbuf) schedule (guided)
		  for (i=0; i<n; i++) {
			 dbuf=p[i];
		     p[i]=f[i]+b*dbuf; // ����� ����������� �����������
		  }
          // ��� 3.3
		  k++;
	   } // while

	   // � ���� ���� ������� ���������� �� ���������� ������� ����:
       //fprintf(fp_statistic_convergence, " ICCG finish residual=%e \n",res);
       //fprintf(fp_statistic_convergence,"%e ",res); // ��� ������ �������� �������� ������� ��� ��� ��� ������ �������������

	   // ������������ ������
        delete ap; delete vcopy;
		delete z; delete p; delete f;  
	}
	else {
		// ������������ ������
		printf("ICCG inform: residual of the initial approximation is too small...\n");
		fprintf(fp_log,"ICCG inform: residual of the initial approximation is too small...\n");
		//fprintf(fp_statistic_convergence, " ICCG no solve start residual < %e \n",e);
		//fprintf(fp_statistic_convergence,"%e ",e); // ��� ������ �������� �������� ������� ��� ��� ��� ������ �������������
		delete ap; delete vcopy;
		delete z; delete p; delete f;		
	}

	// ������������ ������
	delete adiag; 
	delete altr;
	delete jptr;
	delete iptr;

	delete val; 
	delete indx;
	delete pntr;
	
} // ICCG

// �������� �.�. ����������� [1993]
// ��� �������� �������������� ������.
// ���������������� �� ����������
// "��������� ������ ������� ������ ���������" [2004]
// �������������� ���������������� ������������ ������������.
Real* SoloveichikAlgCSIR_SPD(int isize, // ������ ���������� �������
						Real* adiag, Real* altr, int* jptr, int* iptr, // ������� ����
                         Real *dV,  // ������ ������ �����
                         const Real *dX0, // ������ ���������� �����������
                         bool bconsole_message) // �������� �� �������� ������� �� ������� ?
{

     int i,k; // �������� ����� for
     Real *dx, *dax, *dr, *dz, *dp, *dar1, *dres;
     Real dar, dbr, dnz, dscalp;
	 Real kend=1000; // ����������� �� ������������ ����� ��������
	 Real epsilon=dterminatedTResudual;  // �������� ����������
	 bool bweShouldContinue=true;


    // ��������� ������ ��� ������������ �������
    dx=new Real[isize]; dax=new Real[isize]; dr= new Real[isize];
    dz=new Real[isize]; dp=new Real[isize]; dar1=new Real[isize];
	dres=new Real[isize]; // ������ ����������
   

   // ��������� �����������
   // X0 ==
   // ��� X0 ���������� ������ ���� ���������� � �������.
   if (dX0==NULL) {
	   for (i=0; i<isize; i++) dx[i]=0.0;
   }
   else {
	   for (i=0; i<isize; i++) dx[i]=dX0[i];
   }

   SPDMatrixCSIRByVector(adiag, altr, jptr, iptr, dx, dax, isize); // ��������� ������ �  dax
   for (i=0; i<isize; i++) dr[i]= dV[i] - dax[i];  // ��������� �������
   dnz=Scal(dr,dr,isize); // ��������� �������� �������
   for (i=0; i<isize; i++) dz[i]=dr[i];  // ������ ������ (���������� ����������� ������).
   SPDMatrixCSIRByVector(adiag, altr, jptr, iptr, dz, dp, isize); // ��������� ������ � dp

   if (fabs(Scal( dp, dp, isize))>1e-270) 
   {
      k=1; // �������� ���������� ������ � 1
      // ��������� �������� ������� ��������� ����
      while ((bweShouldContinue) && (k <= kend) && (dnz > epsilon))
	  {
         dscalp=1.0/Scal( dp, dp, isize);
         dar=Scal(dp, dr,isize)*dscalp;
         for (i=0; i<isize; i++)
		 {
            dx[i]=dx[i]+dar*dz[i];
            dr[i]=dr[i]-dar*dp[i];
		 }
         dnz=dnz-dar*dar/dscalp; // ����� �������
         
         if (bconsole_message) 
		 {
            // ������ ������� �� �������
            if ((k % 10) == 0)  printf("iter  residual\n");
            printf("%d %e \n",k,dnz);
		 } 
		 SPDMatrixCSIRByVector(adiag, altr, jptr, iptr, dr, dar1, isize);// ��������� ������ � dar1=A*dr
         dbr=-Scal(dp,dar1,isize)*dscalp;
         for (i=0; i<isize; i++)
		 {
            dz[i]=dr[i]+dbr*dz[i];
            dp[i]=dar1[i]+dbr*dp[i];
		 }
         k++;
         // ���� ������� ���������� �� ��� ���� ����������
         if (dnz > 1e7) 
		 {
            // �������������� ���������� �����������
            for (i=0; i<isize; i++) if (dX0==NULL) dx[i]=0.0; else dx[i]=dX0[i];
            printf("\n divergence Soloveichik solver \n");
            bweShouldContinue=false;
            break; // ����� �� ����� while
		 }
 
	  } // while
      // ����������� ����������
      for (i=0; i<isize; i++) dres[i]=dx[i];
   }
   else
   {
      // ���������� ��������� �����������
	  for (i=0; i<isize; i++) dres[i]=dX0[i];
   }

   // ������������ ������ ���������� ��� ������������ �������
   delete dx; delete dax; delete dr;
   delete dz; delete dp; delete dar1;

   return dres; 

} // SoloveichikAlgCSIR_SPD

// �������� �.�. ����������� [1993]
// ��� �������� �������������� ������.
// ���������������� �� ����������
// "��������� ������ ������� ������ ���������" [2004]
// �������������� ���������������� ������������ ������������.
Real* SoloveichikAlgCSIR_SPDgood(int isize, int nz0,// ������ ���������� �������
						Real* adiag, Real* altr, int* jptr, int* iptr, // ������� ����
                         Real *dV,  // ������ ������ �����
                         const Real *dX0, // ������ ���������� �����������
                         bool bconsole_message) // �������� �� �������� ������� �� ������� ?
{

     int i,k; // �������� ����� for
     Real *dx, *dax, *dr, *dz, *dp, *dar1, *dres, *df, *vcopy;
     Real dar, dbr, dnz, dscalp;
	 Real kend=1000; // ����������� �� ������������ ����� ��������
	 Real epsilon=dterminatedTResudual;  // �������� ����������
	 bool bweShouldContinue=true;


    // ��������� ������ ��� ������������ �������
    dx=new Real[isize]; dr= new Real[isize];
    dz=new Real[isize]; dp=new Real[isize]; dar1=new Real[isize];
	dres=new Real[isize]; vcopy=new Real[isize]; // ������ ����������
	df=new Real[isize];
   


	// ��� ��������� ���������� ���������:
	Real  *ldiag=new Real[isize], *lltr=new Real[nz0];
	int *jptrsort=new int[nz0];


    // �������������
	for (i=0; i<isize; i++) ldiag[i]=adiag[i];
	for (i=0; i<nz0; i++) lltr[i]=altr[i];
	// �������� ���������� ���������:
	// ���������� ����� ������ ����������� �����������.
	printf("Incoplete Cholesky decomposition beginig...:\n");
    ICFactor0(ldiag, lltr, jptr, iptr, isize, nz0);
	printf("Incoplete Cholesky decomposition finish...:\n");
    

   /*
	ldiag[0]=1.0; ldiag[1]=1.0;  ldiag[2]=1.838477; ldiag[3]=2.00055;
    ldiag[4]=0.590477; ldiag[5]=1.465913;  ldiag[6]=0.37585673;
	lltr[0]=-1.22383866; lltr[1]=-0.5439282932;  lltr[2]=-1.33247070; lltr[3]=-0.4998613742;
    lltr[4]=-0.1457305633; lltr[5]=-0.06498197865;  lltr[6]=-1.401073265;//*/

    //lltr[0]=-1.22383866; lltr[1]=-1.33247070;  lltr[2]=-0.5439282932; lltr[3]=-0.1457305633;
    //lltr[4]=-0.4998613742; lltr[5]=-1.401073265;  lltr[6]=-0.06498197865;

    for (i=0; i<nz0; i++) jptrsort[i]=jptr[i];
	//for (i=0; i<isize; i++) QuickSort(jptrsort, iptr[i], iptr[i+1]-1);

   // ��������� �����������
   // X0 ==
   // ��� X0 ���������� ������ ���� ���������� � �������.
   if (dX0==NULL) {
	   for (i=0; i<isize; i++) dx[i]=0.0;
   }
   else {
	   for (i=0; i<isize; i++) dx[i]=dX0[i];
   }

   //dax=SPDMatrixCSIRByVector(adiag, altr, jptr, iptr, dx, isize); // ��������� ������ �  dax
   for (i=0; i<isize; i++) vcopy[i]=dx[i];
   dax=inverseU(vcopy, ldiag, lltr, jptrsort, iptr, isize);
   for (i=0; i<isize; i++) vcopy[i]=dax[i];
   SPDMatrixCSIRByVector(adiag, altr, jptr, iptr, vcopy, dax, isize);
   for (i=0; i<isize; i++) vcopy[i]=dax[i]; delete dax;
   dax=inverseL(vcopy, ldiag, lltr, jptrsort, iptr, isize);

   for (i=0; i<isize; i++) vcopy[i]=dV[i]; delete dV;
   dV=inverseL(vcopy, ldiag, lltr, jptrsort, iptr, isize);

   for (i=0; i<isize; i++) dr[i]= dV[i] - dax[i];  // ��������� �������
   dnz=Scal(dr,dr,isize); // ��������� �������� �������
   for (i=0; i<isize; i++) dz[i]=dr[i];  // ������ ������ (���������� ����������� ������).
   //dp=SPDMatrixCSIRByVector(adiag, altr, jptr, iptr, dz, isize); // ��������� ������ � dp
   for (i=0; i<isize; i++) vcopy[i]=dz[i]; 
   dp=inverseU(vcopy, ldiag, lltr, jptrsort, iptr, isize);
   for (i=0; i<isize; i++) vcopy[i]=dp[i]; 
   SPDMatrixCSIRByVector(adiag, altr, jptr, iptr, vcopy, dp, isize);
   for (i=0; i<isize; i++) vcopy[i]=dp[i]; delete dp;
   dp=inverseL(vcopy, ldiag, lltr, jptrsort, iptr, isize);

   if (fabs(Scal( dp, dp, isize))>1e-270) 
   {
      k=1; // �������� ���������� ������ � 1
      // ��������� �������� ������� ��������� ����
      while ((bweShouldContinue) && (k <= kend) && (fabs(dnz) > epsilon))
	  {
         dscalp=1.0/Scal( dp, dp, isize);
         dar=Scal(dp, dr,isize)*dscalp;
         for (i=0; i<isize; i++)
		 {
            dx[i]=dx[i]+dar*dz[i];
            dr[i]=dr[i]-dar*dp[i];
		 }
         //dnz=dnz-dar*dar/dscalp; // ����� �������
		 dnz=Scal( dr, dr, isize);
         
         if (bconsole_message) 
		 {
            // ������ ������� �� �������
            if ((k % 10) == 0)  printf("iter  residual\n");
            printf("%d %e \n",k,dnz);
		 } 
		 //dar1=SPDMatrixCSIRByVector(adiag, altr, jptr, iptr, dr, isize);// ��������� ������ � dar1=A*dr
          
         for (i=0; i<isize; i++) vcopy[i]=dr[i];
         dar1=inverseU(vcopy, ldiag, lltr, jptrsort, iptr, isize);
		 for (i=0; i<isize; i++) vcopy[i]=dar1[i]; 
         SPDMatrixCSIRByVector(adiag, altr, jptr, iptr, vcopy, dar1, isize); 
		 for (i=0; i<isize; i++) vcopy[i]=dar1[i]; delete dar1;
         dar1=inverseL(vcopy, ldiag, lltr, jptrsort, iptr, isize);


         dbr=-Scal(dp,dar1,isize)*dscalp;
         for (i=0; i<isize; i++)
		 {
            dz[i]=dr[i]+dbr*dz[i];
            dp[i]=dar1[i]+dbr*dp[i];
		 }
         k++;
         // ���� ������� ���������� �� ��� ���� ����������
         if (dnz > 1e7) 
		 {
            // �������������� ���������� �����������
            for (i=0; i<isize; i++) if (dX0==NULL) dx[i]=0.0; else dx[i]=dX0[i];
            printf("\n divergence Soloveichik solver \n");
            bweShouldContinue=false;
            break; // ����� �� ����� while
		 }
 
	  } // while
      // ����������� ����������
      //for (i=0; i<isize; i++) dres[i]=dx[i];
	  dres=inverseU(dx, ldiag, lltr, jptrsort, iptr, isize);
   }
   else
   {
      // ���������� ��������� �����������
	  for (i=0; i<isize; i++) dres[i]=dX0[i];
   }

   // ������������ ������ ���������� ��� ������������ �������
   delete dx; delete dax; delete dr;
   delete dz; delete dp; delete dar1;
   delete vcopy;

   return dres; 

} // SoloveichikAlgCSIR_SPDgood

// �������� �.�. ����������� [1993]
// ��� �������� �������������� ������.
// ���������������� �� ����������
// "��������� ������ ������� ������ ���������" [2004]
// �������������� ���������������� ������������ ������������.
void SoloveichikAlgCRS(int isize, // ������ ���������� �������
						 Real *val, int* col_ind, int* row_ptr, // ������� ����
                         Real *dV,  // ������ ������ �����
                         Real* &dX0, // ������ ���������� �����������
                         bool bconsole_message, int maxit) // �������� �� �������� ������� �� ������� ?
{

     int i,k; // �������� ����� for
     Real *dx, *dax, *dr, *dz, *dp, *dar1, *dres, *dstart;
     Real dar, dbr, dnz, dscalp;
	 Real kend=maxit; // ����������� �� ������������ ����� ��������
	 Real epsilon=dterminatedTResudual;  // �������� ����������
	 bool bweShouldContinue=true;


    // ��������� ������ ��� ������������ �������
    dx=new Real[isize]; dax=new Real[isize]; dr= new Real[isize];
    dz=new Real[isize]; dp=new Real[isize]; dar1=new Real[isize];
	dres=new Real[isize], dstart=new Real[isize]; // ������ ����������
   

   // ��������� �����������
   // X0 ==
   // ��� X0 ���������� ������ ���� ���������� � �������.
   if (dX0==NULL) {
	   for (i=0; i<isize; i++) { 
		   dx[i]=0.0;
		   dstart[i]=0.0;
	   }
	   dX0=new Real[isize];

   }
   else {
	   for (i=0; i<isize; i++) {
		   dx[i]=dX0[i];
           dstart[i]=dX0[i];
	   }
   }

   
   MatrixCRSByVector(val,col_ind,row_ptr,dx, dax,isize); // ��������� ������ �  dax
   //printf("dax=%e, rthsd=%e, %d\n",Scal(dax,dax,isize),Scal(dV,dV,isize), isize); // debug
   for (i=0; i<isize; i++) dr[i]= dV[i] - dax[i];  // ��������� �������
   dnz=Scal(dr,dr,isize); // ��������� �������� �������
   //printf("%e\n",dnz); // debug
   for (i=0; i<isize; i++) dz[i]=dr[i];  // ������ ������ (���������� ����������� ������).
   MatrixCRSByVector(val,col_ind,row_ptr,dz, dp, isize);// ��������� ������ � dp

   if (fabs(Scal( dp, dp, isize))>1e-270) 
   {
      k=1; // �������� ���������� ������ � 1
      // ��������� �������� ������� ��������� ����

      if ((k==1) && (fabs(dnz) < epsilon)) {
		  printf("residual on a first iteration == %e zero...\n",dnz);
		  //getchar();
	  }

      while ((bweShouldContinue) && (k <= kend) && (fabs(dnz) > epsilon))
	  {
		  
         dscalp=1.0/Scal( dp, dp, isize);
         dar=Scal(dp, dr,isize)*dscalp;
         #pragma omp parallel for shared(dx,dr,dz,dp,dar,isize) private(i) schedule (guided)
         for (i=0; i<isize; i++)
		 {
            dx[i]=dx[i]+dar*dz[i];
            dr[i]=dr[i]-dar*dp[i];
		 }
         dnz=dnz-dar*dar/dscalp; // ����� �������
         
         if (bconsole_message) 
		 {
            // ������ ������� �� �������
            if ((k % 10) == 0)  printf("iter  residual\n");
            printf("%d %e \n",k,dnz);
			//printf("%d %e \n",k, NormaChebyshev(dr, isize));
		 } 
		 
		 MatrixCRSByVector(val,col_ind,row_ptr,dr, dar1, isize);// ��������� ������ � dar1=A*dr
         dbr=-Scal(dp,dar1,isize)*dscalp;
         #pragma omp parallel for shared(isize,dz,dp,dr,dar1,dbr) private(i) schedule (guided)
         for (i=0; i<isize; i++)
		 {
            dz[i]=dr[i]+dbr*dz[i];
            dp[i]=dar1[i]+dbr*dp[i];
		 }
         k++;
         // ���� ������� ���������� �� ��� ���� ����������
         if (dnz > 1e14) 
		 {
            // �������������� ���������� �����������
            for (i=0; i<isize; i++) if (dX0==NULL) dx[i]=0.0; else dx[i]=dstart[i];
            printf("\n divergence Soloveichik solver \n");
			// � ���� ���� ������� ���������� �� ���������� ������� ����:
			//fprintf(fp_statistic_convergence, " Soloveichik solver finish residual: dnz=%e, dr=%e. divergence solution \n",dnz,NormaChebyshev(dr, isize));
			fprintf(fp_statistic_convergence,"%e ",fabs(dnz));
            bweShouldContinue=false;
            break; // ����� �� ����� while
		 }
 
	  } // while

	  if (bweShouldContinue) {
		  //fprintf(fp_statistic_convergence, " Soloveichik solver finish residual=%e \n",dnz);
		  fprintf(fp_statistic_convergence,"%e ",fabs(dnz));
		  //fprintf(fp_statistic_convergence, " Soloveichik solver finish residual=%e \n",NormaChebyshev(dr, isize));  
	  }
	  
      // ����������� ����������
      for (i=0; i<isize; i++) dres[i]=dx[i];
   }
   else
   {
      // ���������� ��������� �����������
	  for (i=0; i<isize; i++) dres[i]=dstart[i];
	  printf(" (fabs(Scal( dp, dp, isize))>1e-270)==false\n");
	  //fprintf(fp_statistic_convergence, " Soloveichik solver start residual < 1e-270 \n");
	  fprintf(fp_statistic_convergence,"%e ",0.0);
	  //getchar();
   }

   // ������������ ������ ���������� ��� ������������ �������
   delete dx; delete dax; delete dr;
   delete dz; delete dp; delete dar1;

   //return dres;
   for (i=0; i<isize; i++) dX0[i]=dres[i];
   delete dres; delete dstart;

} // SoloveichikAlgCRS


/* ���������� �� ������������� �������
// �������������� ����������� �������
void initsimplesparse(SIMPLESPARSE &M) {
	M.a=NULL;
	M.n=0;
	M.incCLUSTER_SIZE=10;
	M.POOL_SIZE=0;
} // initsimplesparse
*/

// ���������� �� ������� ������
// �������������� ����������� �������
void initsimplesparse(SIMPLESPARSE &M, int nodes) {
	M.n=0; // ���������� ��� �������� ������� 
	M.root=new NONZEROELEM*[nodes];
	int i; // ����� ������, ����� ��������� � ����
	for (i=0; i<nodes; i++) M.root[i]=NULL; 
} // initsimplesparse

/* ���������� �� �������.
// ��������� ��������� ������� �
// ���������� ����������� ������� M
void addelmsimplesparse(SIMPLESPARSE &M, Real aij, int i, int j, bool bset) {
	if (M.n==0) {
		// ������ �������
		M.POOL_SIZE+=M.incCLUSTER_SIZE;
		M.n++;
		M.a=new NONZEROELEM[M.POOL_SIZE];
		M.a[0].aij=aij;
		M.a[0].i=i;
		M.a[0].j=j;
	}
	else if (M.n<M.POOL_SIZE) 
	{
		bool flag=false; // ������� �� ������
		int i1; // �������
		for (i1=0; i1<M.n; i1++) if ((M.a[i1].i==i) && (M.a[i1].j==j)) {
           flag=true;
           if (bset) M.a[i1].aij=aij;  // ���������
		   else M.a[i1].aij+=aij; // ����������
		}
		if (!flag) {
			M.a[M.n].aij=aij;
		    M.a[M.n].i=i;
		    M.a[M.n].j=j;
            M.n++;
		} 
	}
	else // M.n==M.POOL_SIZE
	{
        bool flag=false; // ������� �� ������
		int i1; // �������
		for (i1=0; i1<M.n; i1++) if ((M.a[i1].i==i) && (M.a[i1].j==j)) {
           flag=true;
           if (bset) M.a[i1].aij=aij;  // ���������
		   else M.a[i1].aij+=aij; // ����������
		}
		if (!flag) {
           NONZEROELEM* list=new NONZEROELEM[M.POOL_SIZE];
		   for (i1=0; i1<M.n; i1++) list[i1]=M.a[i1]; // �����������
		   delete M.a;
		   M.POOL_SIZE+=M.incCLUSTER_SIZE;
		   M.a=new NONZEROELEM[M.POOL_SIZE];
           for (i1=0; i1<M.n; i1++) M.a[i1]=list[i1]; // �������� �����������
           M.a[M.n].aij=aij;
		   M.a[M.n].i=i;
		   M.a[M.n].j=j;
		   M.n++;

		}
	}
} // addelmsimplesparse
*/

// ���������� �� ������� ������
// ��������� ��������� ������� �
// ���������� ����������� ������� M
// �������� �� ��������� ������������ �������� ���� ���, �������
// ����� �������� � ������� �������.
void addelmsimplesparse(SIMPLESPARSE &M, Real aij, int i, int j, bool bset) {
    NONZEROELEM* p;
	p=M.root[i];
	// �������� ����� �������� � ������ key
	while ((p!=NULL) && (p->key!=j)) p=p->next;
	if (p!=NULL) {
		// ������� ������
		if (bset) p->aij=aij; // ���������
		else p->aij+=aij; // ����������
	}
	else 
	{
		// ���� ������ �������� ��� � ������
		// �� ���������� �������� � ������ ������.
        NONZEROELEM* q=new NONZEROELEM;
		q->aij=aij;
		q->key=j;
		q->next=M.root[i];
		M.root[i]=q;
		q=NULL;
		M.n++; // ���������� ��������� ��������� ����������� �� 1. 
	}
} // addelmsimplesparse

// ������������ ������ ��� ������� SIMPLESPARSE
void simplesparsefree(SIMPLESPARSE &M, int nodes) {
	int i; // ������� ����� for
	for (i=0; i<nodes; i++) {
        NONZEROELEM* p9, *q9;
        p9=M.root[i]; q9=p9;
		M.root[i]=NULL;
		while (p9!=NULL) {
			p9=p9->next;
			q9->next=NULL;
			delete q9;
			q9=p9;
		}
	}
	delete M.root;
} // simplesparsefree 

/*
// ��� ��������� ������� ���� ��������� � ������ ����������
// �� ������������ �������� ������������������ ���������:
// ����������. ����� ����� ����������� ������� ����������.
// ������ �������� � ����� ����� "The C programming language".
// swap: ����� ������� v[i] � v[j]
void swap(NONZEROELEM* &v, int i, int j)
{
        NONZEROELEM temp;

		// change v[i] <-> v[j]
		temp = v[i];
		v[i] = v[j];
		v[j] = temp;
} // swap

// ��� �������� PivotList
int PivotList(NONZEROELEM* &list, int first, int last) {
	// list �������������� ������
	// first ����� ������� ��������
	// last ����� ���������� ��������

	int PivotValue = list[first].key;
	int PivotPoint = first;

	for (int index=(first+1); index<=last; index++) {
		if (list[index].key<PivotValue) {
			PivotPoint++;
			swap(list, PivotPoint, index);
		}
	}

	swap(list, first, PivotPoint);

	return PivotPoint;
} // PivotList


// ������� ���������� �����.
// ����������������� � �������������� ��. ��������� ������ ����������
// ���. 106.
void QuickSort(NONZEROELEM* &list, int first, int last) {
	// list ��������������� ������ ���������
	// first ����� ������� �������� � ����������� ����� ������
	// last ����� ���������� �������� � ����������� ����� ������

	int pivot;

	if (first < last) {
        pivot = PivotList(list, first, last);
        QuickSort(list, first, pivot-1);
		QuickSort(list, pivot+1, last);
	}
} // QuickSort
*/
// ��� ��������� ������� ���� ��������� � ������ ����������
// �� ������������ �������� ������������������ ���������:
// ����������. ����� ����� ����������� ������� ����������.
// ������ �������� � ����� ����� "The C programming language".
// swap: ����� ������� v[i] � v[j]
void swap(int* &v, int i, int j)
{
        int temp;

		// change v[i] <-> v[j]
		temp = v[i];
		v[i] = v[j];
		v[j] = temp;
} // swap

// ��� �������� PivotList
int PivotList(int* &list, int first, int last) {
	// list �������������� ������
	// first ����� ������� ��������
	// last ����� ���������� ��������

	int PivotValue = list[first];
	int PivotPoint = first;

	for (int index=(first+1); index<=last; index++) {
		if (list[index]<PivotValue) {
			PivotPoint++;
			swap(list, PivotPoint, index);
		}
	}

	swap(list, first, PivotPoint);

	return PivotPoint;
} // PivotList


// ������� ���������� �����.
// ����������������� � �������������� ��. ��������� ������ ����������
// ���. 106.
void QuickSort(int* &list, int first, int last) {
	// list ��������������� ������ ���������
	// first ����� ������� �������� � ����������� ����� ������
	// last ����� ���������� �������� � ����������� ����� ������

	int pivot;

	if (first < last) {
        pivot = PivotList(list, first, last);
        QuickSort(list, first, pivot-1);
		QuickSort(list, pivot+1, last);
	}
} // QuickSort

// ��� ��������� ������� ���� ��������� � ������ ����������
// �� ������������ �������� ������������������ ���������:
// ����������. ����� ����� ����������� ������� ����������.
// ������ �������� � ����� ����� "The C programming language".
// swap: ����� ������� v[i] � v[j]
void swapCSIR(int* &v, Real* &dr, int i, int j)
{
        int tempi;
		Real tempr;

		// change v[i] <-> v[j]
		tempi = v[i];
		v[i] = v[j];
		v[j] = tempi;
		// change dr[i] <-> dr[j]
		tempr = dr[i];
		dr[i] = dr[j];
		dr[j] = tempr;

} // swap

// ��� �������� PivotList
int PivotListCSIR(int* &jptr, Real* &altr, int first, int last) {
	// list==jptr and altr �������������� ������
	// first ����� ������� ��������
	// last ����� ���������� ��������

	int PivotValue = jptr[first];
	int PivotPoint = first;

	for (int index=(first+1); index<=last; index++) {
		if (jptr[index]<PivotValue) {
			PivotPoint++;
			swapCSIR(jptr, altr, PivotPoint, index);
		}
	}

	swapCSIR(jptr, altr, first, PivotPoint);

	return PivotPoint;
} // PivotList


// ������� ���������� �����.
// ����������������� � �������������� ��. ��������� ������ ����������
// ���. 106.
void QuickSortCSIR(int* &jptr, Real* &altr, int first, int last) {
	// list ��������������� ������ ���������
	// first ����� ������� �������� � ����������� ����� ������
	// last ����� ���������� �������� � ����������� ����� ������

	int pivot;

	if (first < last) {
        pivot = PivotListCSIR(jptr, altr, first, last);
        QuickSortCSIR(jptr, altr, first, pivot-1);
		QuickSortCSIR(jptr, altr, pivot+1, last);
	}
} // QuickSortCSIR

/* ���������� �� ������������ �������.
// ����������� ���������� ������ �������� ����������� �������
// � ������ CRS. ����� nodes - ���������.
void simplesparsetoCRS(SIMPLESPARSE &M, Real* &val, int* &col_ind, int* &row_ptr, int nodes) {
	if (M.n!=0) {
		val = new Real[M.n];
		col_ind = new int[M.n];
		row_ptr = new int[nodes+1];

		int k; // �������
		// �������������
        for (k=0; k<(M.n); k++) {
		   val[k]=0.0;
		   col_ind[k]=0;
	    }
        for (k=0; k<=nodes; k++) {
		    row_ptr[k]=M.n; // ����������� ���������� ��������� ��������� ���� 1 � ������ ���� ��� ��������� ������� ���������� � 0
	    }

        // ������� ���������� �����.
		// �������������� �� �������
		QuickSort(M.a, 0, M.n-1);

		// ���������� ����������� �������
		for (k=0; k<M.n; k++) {
			val[k]=M.a[k].aij;
            col_ind[k]=M.a[k].j;
            row_ptr[M.a[k].i]=min(k,row_ptr[M.a[k].i]);
		}
	}
} // simplesparsetoCRS
*/

// ���������� �� ������� ������.
// ����������� ���������� ������ �������� ����������� �������
// � ������ CRS. ����� nodes - ���������.
void simplesparsetoCRS(SIMPLESPARSE &M, Real* &val, int* &col_ind, int* &row_ptr, int nodes) {
	bool flag=true;
    int k; // �������
	for (k=0; k<nodes; k++) if (M.root[k]==NULL) {
		flag=false; break;
	}

	if (flag) {
		val = new Real[M.n];
		col_ind = new int[M.n];
		row_ptr = new int[nodes+1];

		
		// �������������
        for (k=0; k<(M.n); k++) {
		   val[k]=0.0;
		   col_ind[k]=0;
	    }
        for (k=0; k<=nodes; k++) {
		    row_ptr[k]=M.n; // ����������� ���������� ��������� ��������� ���� 1 � ������ ���� ��� ��������� ������� ���������� � 0
	    }

        // ������� ���������� �����.
		// �������������� �� �������
		//QuickSort(...); �� ���������,
		// �.�. ���� ��������� �������� 
		// ������������� �������������� �� �������.

		/*
		// ���������� ����������� �������
		for (k=0; k<M.n; k++) {
			val[k]=M.a[k].aij;
            col_ind[k]=M.a[k].j;
            row_ptr[M.a[k].i]=min(k,row_ptr[M.a[k].i]);
		}
		*/
		int ik=0; // ������� ��������� ��������� ����
		NONZEROELEM* p;
        for (k=0; k<nodes; k++) {
			p=M.root[k];
			while (p!=NULL) {
				val[ik]=p->aij;
				col_ind[ik]=p->key;
                row_ptr[k]=min(ik,row_ptr[k]);
				ik++;
				p=p->next;
			}
		}

		// � ������ ������ �������� ������������� �� ������� ��������:
        for (k=0; k<nodes; k++) QuickSortCSIR(col_ind, val, row_ptr[k], row_ptr[k+1]-1); 

	}
} // simplesparsetoCRS

// ����������� equation3D  ������ �������� � CRS ������.
// ���� ��������� ����� ���������������: �������� ����������� ������ ����������.
// �.�. ������ SIMPLESPARSE ������� ������� ����� ������.
void equation3DtoCRS(equation3D* &sl, equation3D_bon* &slb, Real* &val, int* &col_ind, int* &row_ptr, int maxelm, int maxbound, Real alpharelax) {
	
	bool flag=true;
    int k; // �������
	int n=0; // ����� ��������� ���������

    const Real nonzeroEPS=1e-37; // ��� ��������� ������������� ����

	// ������� ���������� ��������� ���������
	// �� ���������� ������ ��������� �������.
	for (k=0; k<maxelm; k++) {
		
		if (fabs(sl[k].ap)> nonzeroEPS) n++; // ������������ �������
		else flag=false;

        if ((sl[k].iE>-1) && (fabs(sl[k].ae) > nonzeroEPS)) n++;
        if ((sl[k].iN>-1) && (fabs(sl[k].an) > nonzeroEPS)) n++;
        if ((sl[k].iT>-1) && (fabs(sl[k].at) > nonzeroEPS))	n++;		
        if ((sl[k].iS>-1) && (fabs(sl[k].as) > nonzeroEPS)) n++;
        if ((sl[k].iW>-1) && (fabs(sl[k].aw) > nonzeroEPS)) n++;
        if ((sl[k].iB>-1) && (fabs(sl[k].ab) > nonzeroEPS)) n++;
	}

	// ������� ���������� ��������� ���������
    // ��� ��������� ����� ��������� �������.
	for (k=0; k<maxbound; k++) {
		if (fabs(slb[k].aw)>nonzeroEPS) n++; // ������������ �������
		else flag=false;

		if ((slb[k].iI>-1) && (fabs(slb[k].ai) > nonzeroEPS)) n++;
	}

	if (flag) {
		// memory +15N
		val = new Real[n];
		col_ind = new int[n];
		row_ptr = new int[(maxelm+maxbound)+1];

		
		// �������������
        for (k=0; k<(n); k++) {
		   val[k]=0.0;
		   col_ind[k]=-1;
	    }
        for (k=0; k<=(maxelm+maxbound); k++) {
		    row_ptr[k]=n; // ����������� ���������� ��������� ��������� ���� 1 � ������ ���� ��� ��������� ������� ���������� � 0
	    }

        // ������� ���������� �����.
		// �������������� �� �������
		//QuickSort(...); �� ���������,
		// �.�. ���� ��������� �������� 
		// ������������� �������������� �� �������.

		/*
		// ���������� ����������� �������
		for (k=0; k<M.n; k++) {
			val[k]=M.a[k].aij;
            col_ind[k]=M.a[k].j;
            row_ptr[M.a[k].i]=min(k,row_ptr[M.a[k].i]);
		}
		*/
		int ik=0; // ������� ��������� ��������� ����
		
		// ��� ���������� ����� ��������� �������:
        for (k=0; k<maxelm; k++) {

			if (fabs(sl[k].ap) > nonzeroEPS) {
                val[ik]=sl[k].ap/alpharelax;
				col_ind[ik]=sl[k].iP;
                row_ptr[k]=min(ik,row_ptr[k]);
				ik++;
			}
			if ((sl[k].iE>-1) && (fabs(sl[k].ae) > nonzeroEPS)) {
                val[ik]=-sl[k].ae;
				col_ind[ik]=sl[k].iE;
                row_ptr[k]=min(ik,row_ptr[k]);
				ik++;
			}
			if ((sl[k].iN>-1) && (fabs(sl[k].an) > nonzeroEPS)) {
                val[ik]=-sl[k].an;
				col_ind[ik]=sl[k].iN;
                row_ptr[k]=min(ik,row_ptr[k]);
				ik++;
			}
			if ((sl[k].iT>-1) && (fabs(sl[k].at) > nonzeroEPS)) {
                val[ik]=-sl[k].at;
				col_ind[ik]=sl[k].iT;
                row_ptr[k]=min(ik,row_ptr[k]);
				ik++;
			}		
			if ((sl[k].iS>-1) && (fabs(sl[k].as) > nonzeroEPS)) {
                val[ik]=-sl[k].as;
				col_ind[ik]=sl[k].iS;
                row_ptr[k]=min(ik,row_ptr[k]);
				ik++;
			}
			if ((sl[k].iW>-1) && (fabs(sl[k].aw) > nonzeroEPS)) {
				val[ik]=-sl[k].aw;
				col_ind[ik]=sl[k].iW;
                row_ptr[k]=min(ik,row_ptr[k]);
				ik++;
			}
			if ((sl[k].iB>-1) && (fabs(sl[k].ab) > nonzeroEPS)) {
				val[ik]=-sl[k].ab;
				col_ind[ik]=sl[k].iB;
                row_ptr[k]=min(ik,row_ptr[k]);
				ik++;
			}


		}

		// ��� ���������� ����� ��������� �������:
        for (k=0; k<maxbound; k++) {
			if (fabs(slb[k].aw) > nonzeroEPS) {
                val[ik]=slb[k].aw/alpharelax;
				col_ind[ik]=slb[k].iW;
                row_ptr[maxelm+k]=min(ik,row_ptr[maxelm+k]);
				ik++;
			}
			if ((slb[k].iI>-1) && (fabs(slb[k].ai) > nonzeroEPS)) {
				val[ik]=-slb[k].ai;
				col_ind[ik]=slb[k].iI;
                row_ptr[maxelm+k]=min(ik,row_ptr[maxelm+k]);
				ik++;
			}

		}

		// � ������ ������ �������� ������������� �� ������� ��������:
        for (k=0; k<(maxelm+maxbound); k++) QuickSortCSIR(col_ind, val, row_ptr[k], row_ptr[k+1]-1); 
	}

	if (!flag) {
		printf("Error: zero diagonal element...\n");
		getchar();
	}

	for (k=0; k<n; k++) if (col_ind[k]==(-1)) {
		printf("Error equation3D to CRS.\n");
		getchar();
	}

	
} // equation3DtoCRS

// ���������� �� ������� ������.
// ����������� ���������� ������ �������� ����������� �������
// � ������ CSIR. ����� nodes - ���������.
// ��� �������� ������ ��� SPD ������.
// ������������ ������������ ����������� ������,
// �������� ������ ������ �����������.
void simplesparsetoCSIR(SIMPLESPARSE &M, Real* &adiag, Real* &altr, int* &jptr, int* &iptr, int nodes) {
	bool flag=true;
    int k; // �������
	for (k=0; k<nodes; k++) if (M.root[k]==NULL) {
		flag=false; break;
	}

	if (flag) {
		// ��������������� �������� � altr �������� ���������
		int nz=(int)(M.n-nodes)/2; // ����� ��������� ���������
		adiag = new Real[nodes]; // ������������ ��������
		altr = new Real[nz]; // ��������������� ��������
		jptr = new int[nz]; // ������ ������� ��� ������� ������������
		iptr = new int[nodes+1]; // ��������� �� ��������� ������

		
		// �������������
		for (k=0; k<nodes; k++) adiag[k]=0.0;
        for (k=0; k<(nz); k++) {
		   altr[k]=0.0;
		   jptr[k]=0;
	    }
        for (k=0; k<=nodes; k++) {
		    iptr[k]=nz; // ����������� ���������� ��������� ��������� ���� 1 � ������ ���� ��� ��������� ������� ���������� � 0
	    }

        // ������� ���������� �����.
		// �������������� �� �������
		//QuickSort(...); �� ���������,
		// �.�. ���� ��������� �������� 
		// ������������� �������������� �� �������.

		/*
		// ���������� ����������� �������
		for (k=0; k<M.n; k++) {
			val[k]=M.a[k].aij;
            col_ind[k]=M.a[k].j;
            row_ptr[M.a[k].i]=min(k,row_ptr[M.a[k].i]);
		}
		*/
		/*
		int ik=0; // ������� ��������� ��������� ����
		NONZEROELEM* p;
        for (k=0; k<nodes; k++) {
			p=M.root[k];
			while (p!=NULL) {
				val[ik]=p->aij;
				col_ind[ik]=p->key;
                row_ptr[k]=min(ik,row_ptr[k]);
				ik++;
				p=p->next;
			}
		}
		*/

		int ik=0, imin=1,k1; // ������� ��������� ��������������� ��������� ����
		bool bvisit;
		NONZEROELEM* p;
        for (k=0; k<nodes; k++) {
			bvisit=false;
			p=M.root[k];
			while (p!=NULL) {
				if (p->key==k) {
					adiag[k]=p->aij;
				}
				else if (p->key<k) {
					if (ik<(nz)) {
						altr[ik]=p->aij; // ��������� ��������
					    jptr[ik]=p->key; // ����� �������
					}
					else {
						printf("non simmetric matrix ICCG. simplesparsetoCSIR\n");
						getchar();
					}
					bvisit=true;			   
				}
				imin=min(ik,iptr[k]);
				//printf("imin=%d\n",imin);
                iptr[k]=imin;
                if (imin==0) for (k1=0; k1<k; k1++) iptr[k1]=0;	
				if (bvisit) { 
					ik++;
					bvisit=false;
				}
				p=p->next;
			}
		}


		for (k=0; k<nodes; k++) QuickSortCSIR(jptr, altr, iptr[k], iptr[k+1]-1);

	}
} // simplesparsetoCSIR


// ������ ������� � �������
void printM_and_CSIR(SIMPLESPARSE &sparseM, int  n) {

	FILE *fp;
    errno_t err;

	if ((err = fopen_s( &fp, "matrix.txt", "w")) != 0) {
		printf("Create File temp Error\n");
		getchar();

	}
	else {

	   int i;
	   // ������ ���������� ����� ����������� �������.
	   for (i=0; i<n; i++) {
           NONZEROELEM* pelm=sparseM.root[i];
		   while (pelm!=NULL) {
		    	fprintf(fp,"a[%d][%d]=%e  ",i,pelm->key,pelm->aij);
			    pelm=pelm->next;
		   }
		   fprintf(fp,"\n");
	   }//*/
	   fclose(fp); // �������� �����

	}
	getchar();
}

// ���������� �� ������� ������.
// ����������� ���������� ������ �������� ����������� �������
// � ������ CSIR_ITL. ����� nodes - ���������.
// ��� �������� ������ ��� SPD ������.
// ������������ ������������ ����������� ������,
// �������� ������ ������� �����������.
// ������ ���������� ������ ������.
void simplesparsetoCSIR_ITLSPD(SIMPLESPARSE &M, Real* &val, int* &indx, int* &pntr, int nodes) {
	bool flag=true;
    int k; // �������
	for (k=0; k<nodes; k++) if (M.root[k]==NULL) {
		flag=false; break;
	}

	if (flag) {
 
		//printM_and_CSIR(M, nodes); // debug

		// ��������������� �������� � altr �������� ���������
		int nz=(int)((M.n-nodes)/2 + nodes); // ����� ��������� ���������
		val = new Real[nz]; // ������������ �������� � ��������������� ��������
		indx = new int[nz]; // ������ ������� ��� ������� ������������
		pntr = new int[nodes+1]; // ��������� �� ��������� ������

		
		// �������������
        for (k=0; k<(nz); k++) {
		   val[k]=0.0;
		   indx[k]=0;
	    }
        for (k=0; k<=nodes; k++) {
		    pntr[k]=nz; // ����������� ���������� ��������� ��������� ���� 1 � ������ ���� ��� ��������� ������� ���������� � 0
	    }

        

		int ik=0; // ������� ��������� ��������������� ��������� ����
		NONZEROELEM* p;
        for (k=0; k<nodes; k++) {
			
			p=M.root[k];
			while (p!=NULL) {

				// k - ����� ������������� ��������
				if (p->key>=k) {
					if (ik<(nz)) {
						val[ik]=p->aij; // ��������� ��������
					    indx[ik]=p->key; // ����� �������	
					}
					else {
						printf(" Error non simmetric matrix ICCG. simplesparsetoCSIR_ITLSPD\n");
						getchar();
					}
					pntr[k]=min(ik,pntr[k]);

					ik++;
				}

				p=p->next;
			}

		}

		for (k=0; k<nodes; k++) QuickSortCSIR(indx, val, pntr[k], pntr[k+1]-1);

		/* // debug
		for (k=0; k<=nodes; k++) {
		    printf("%d ",pntr[k]);
		}
        printf("\n");
		printf("nz==%d\n", nz);
		getchar();
		//*/

	}
} // simplesparsetoCSIR_ITLSPD

/* �������� LU ���������� ��� �������������� ������
*  ������ � nxn=
*    9.0 0.0 0.0 3.0 1.0 0.0 1.0
*    0.0 11.0 2.0 1.0 0.0 0.0 2.0 
*    0.0 1.0 10.0 2.0 0.0 0.0 0.0 
*    2.0 1.0 2.0 9.0 1.0 0.0 0.0 
*    1.0 0.0 0.0 1.0 12.0 0.0 1.0 
*    0.0 0.0 0.0 0.0 0.0 8.0 0.0
*    2.0  2.0 0.0 0.0 3.0 0.0 8.0
*-----------------------------------------
*  ������������� (� ���� ���� ������ ��������� �� ���� ���������)
*  ������ �������������� ���������� ������� :
*  ������� ����������� ������� �������� ���������, � ������ ������
*  �������� ������������� �� �������� ������� ��������.
*  U_val :   1.0, 1.0, 3.0, 9.0,   2.0, 1.0, 2.0, 11.0,   2.0, 10.0, 1.0, 9.0, 1.0,12.0, 8.0, 8.0
*  U_ind :   6, 4, 3, 0,  6, 3, 2, 1,  3,2, 4,3, 6,4, 5, 6
*  U_ptr :   0, 4, 8, 10, 12, 14, 15, 16
*  ������ ����������� ������� �������� �����������, � ������ �������
*  �������� ������������� �� �������� ������� �����.
*  L_val :  2.0, 1.0, 2.0, 9.0,    2.0, 1.0, 1.0, 11.0,  2.0, 10.0, 1.0, 9.0,  3.0, 12.0, 8.0, 8.0
*  L_ind :  6, 4, 3, 0,  6, 3, 2, 1,   3, 2,  4,3,  6, 4, 5, 6
*  L_ptr :  0, 4, 8, 10, 12, 14, 15, 16
*----------------------------------------------
*  ��������� ILU ����������:
*  U_val : 1.0, 1.0, 3.0, 9.0, 2.0, 1.0, 2.0, 11.0, 2.0, 10.0, 1.0, 9.0, 1.0, 12.0, 8.0, 8.0.
*  L_val : 0.222, 0.111, 0.222, 1.0, -1.273, 0.091, 0.091, 1.0, 0.2, 1.0, 0.111, 1.0, -0.417, 1.0, 1.0, 1.0.
*/
void ILU0_Decomp_ITL(Real* &U_val, int* &U_ind, int* &U_ptr, Real* &L_val, int* &L_ind, int* &L_ptr, int n)
{
	/*
	// ��������� ������
	int n=7;
	//Real U_val[16] = { 3.0, 1.0, 1.0, 9.0,  2.0, 1.0, 2.0, 11.0, 2.0, 10.0, 1.0, 9.0, 1.0,12.0, 8.0, 8.0};
	//int U_ind[16] = { 3, 4, 6, 0,  2, 3, 6, 1,  3,2, 4,3, 6,4, 5, 6};
	//int U_ptr[8] = {0, 4, 8, 10, 12, 14, 15, 16};

	// ������������� � ������� �������� �� ��������.

	// verno
	Real U_val[16] = {  1.0, 1.0, 3.0, 9.0,     2.0, 1.0, 2.0, 11.0,   2.0, 10.0,   1.0, 9.0, 1.0,12.0, 8.0, 8.0};
	int U_ind[16] = { 6, 4, 3, 0,  6, 3, 2, 1,  3,2, 4,3, 6,4, 5, 6};
	int U_ptr[8] = {0, 4, 8, 10, 12, 14, 15, 16};

	//Real U_val[16] = {  9.0, 11.0, 2.0, 10.0, 3.0, 1.0, 2.0, 9.0, 1.0, 1.0, 12.0, 8.0, 1.0, 2.0, 1.0, 8.0 };
	//int U_ind[16] = { 0, 1, 1, 2, 0, 1, 2, 3, 0, 3, 4, 5, 0, 1, 4, 6};
	//int U_ptr[8] = {0, 1, 2, 4, 8, 11, 15, 16};

	//Real L_val[16] = {2.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0,  3.0};
	//int L_ind[16] = { 3, 4, 6, 2, 3, 6, 3, 4, 6};
	//int L_ptr[8] = {0, 3, 6, 7, 8, 8, 8, 8};

	// verno
	Real L_val[16] = {2.0, 1.0, 2.0, 9.0,    2.0, 1.0, 1.0, 11.0,  2.0, 10.0, 1.0, 9.0,  3.0, 12.0, 8.0, 8.0};
	int L_ind[16] = { 6, 4, 3, 0,  6, 3, 2, 1,   3, 2,  4,3,  6, 4, 5, 6};
	int L_ptr[8] = {0, 4, 8, 10, 12, 14, 15, 16};

     //Real L_val[16] = {9.0, 11.0, 1.0, 10.0, 2.0, 1.0, 2.0, 9.0, 1.0, 1.0, 12.0, 8.0, 2.0, 2.0, 3.0, 8.0};
	//int L_ind[16] = { 0, 1, 1, 2, 0, 1, 2, 3, 0, 3, 4, 5, 0, 1, 4, 6};
	//int L_ptr[8] = {0, 1, 2, 4, 8, 11, 12, 16};
	*/

	// �������
	int i, j, qn, pn, rn; 
      for (i = 0; i < n - 1; i++) {
	     Real multiplier = U_val[U_ptr[i+1]-1];
    
	     for (j = L_ptr[i]; j < L_ptr[i+1]; j++)
	          L_val[j] /= multiplier;
    
	     for (j = U_ptr[i+1]; j < U_ptr[i+2]-1; j++) {
	         multiplier = U_val[j];
	         qn = j + 1;
	         rn = L_ptr[i+1];
	         for (pn = L_ptr[U_ind[j]]; L_ind[pn] <= i + 1 && pn < L_ptr[U_ind[j]+1]; pn++) {
	              while (U_ind[qn] < L_ind[pn] && qn < U_ptr[i+2]) qn++;

	              if (L_ind[pn] == U_ind[qn] && qn < U_ptr[i+2])
	                     U_val[qn] -= multiplier * L_val[pn];
	         }
	         for (; pn < L_ptr[U_ind[j]+1]; pn++) {
	             while (L_ind[rn] < L_ind[pn] && rn < L_ptr[i+2])  rn++;

	             if (L_ind[pn] == L_ind[rn] && rn < L_ptr[i+2])
	                    L_val[rn] -= multiplier * L_val[pn];
	         }
	      }
      }
	  L_val[L_ptr[n-1]]=1.0;

	  // ���������� �� �����������
	  for (i = 0; i < n; i++) {
          QuickSortCSIR(U_ind, U_val, U_ptr[i], U_ptr[i+1]-1);
          QuickSortCSIR(L_ind, L_val, L_ptr[i], L_ptr[i+1]-1);
	  }

	/*
	printf("Uval : ");
	for (i=0; i<16; i++) printf("%.3f, ",U_val[i]);
	printf("\n\n Lval: ");
	for (i=0; i<16; i++) printf("%.3f, ",L_val[i]);
	getchar();
	exit(0);
	*/
} // ILU0_Decomp_ITL

// �������� LU ���������� � ������� ����������� �� ����� �. �����
// Iterative Methods for Sparse linear systems.
// ������ ��� ������ � ������������ ���������.
// �� ���� ������� ������� � � CRS �������.
// �� ������ ������� luval � CRS ������� � ������� L �� ��������� 1.0,
// uptr - ��������� �� ������������ ��������.
void ilu0_Saad(int n, Real* a, int* ja, int* ia, Real* &luval, int* &uptr, int &icode) {
//void ilu0_Saadtest() {
	//int n=5; // ����� ����� � �������.
	//Real a[12] = {1.0, 2.0,   3.0, 4.0, 5.0,   6.0, 7.0, 8.0, 9.0,  10.0, 11.0,   12.0};
	//int ja[12] = {0, 3,   0, 1, 3,   0, 2, 3, 4,   2, 3,   4};
	//int ia[6] = {0, 2, 5, 9, 11, 12};

	//int n=7; // ����� ����� � �������.
	//Real a[25] = {9.0, 3.0, 1.0, 1.0,   11.0, 2.0, 1.0, 2.0,   1.0, 10.0, 2.0,   2.0, 1.0, 2.0, 9.0, 1.0,   1.0, 1.0, 12.0, 1.0,   8.0,   2.0, 2.0, 3.0, 8.0};
	//int ja[25] = {0, 3, 4, 6,   1, 2, 3, 6,   1, 2, 3,   0, 1, 2, 3, 4,   0, 3, 4, 6,   5,  0, 1, 4, 6};
	//int ia[8] = {0, 4, 8, 11, 16, 20, 21, 25};

	//Real *luval;
	//int *uptr;
	int *iw; 

	//int icode;

    // *******

	// INPUT:
	// n - dimension of matrix
	// a, ja, ia - sparse matrix in CRS format
	// iw - integer work array of length n
	// OUTPUT:
	// luval - L/U matrices stored together. On return luval,
	//         ja, ia is the combined CSR data structure for 
	//         the LU factors.
	// uptr  - pointer to the diagonal elements in the CRS
	//        data structure luval, ja, ia.
	// icode - integer indicating error code on return
	//         icode == -1: normal return
	//         icode == k: encountered a zero pivot at step k

	luval = new Real[ia[n]];
	iw = new int[n];
	uptr = new int[n];

    icode=-1; // Normal return

	int i=0;
	// initialize work array iw to zero and luval array to a
	for (i=0; i<ia[n]; i++) luval[i]=a[i];

	for (i=0; i<n; i++) iw[i]=-1;

	// Main loop
	int k=0;
	int j1, j2;
	int j=0;
	int jrow;
	Real t1;
	int jj, jw;
	bool bcont=true;

	k=0;
	while ((icode==-1) && (k<n)) {
		
		j1=ia[k]; 
		j2=ia[k+1]-1;
		for (j=j1; j<=j2; j++) iw[ja[j]]=j;
		j=j1;
        jrow=ja[j];

		do {
	
			bcont=true;
		    if (jrow >= k) {// Exit if diagonal element is reached
			    // Store pointer to diagonal element
			    uptr[k]=j;
			    if ((jrow != k) || (fabs(luval[j])<1e-37)) {
				   icode=k; // Error: zero pivot
			    }
			    else {
				   luval[j]=1.0/luval[j];
			    }
				bcont=false; // ����� �� ����� do
		    }
		    else {
			   // Compute the multiplier for jrow
			   t1=luval[j]*luval[uptr[jrow]];
			   luval[j]=t1;

			   for (jj=uptr[jrow]+1; jj<ia[jrow+1]; jj++) {
			    	jw=iw[ja[jj]];
				    if (jw!=(-1)) luval[jw]-=t1*luval[jj];
			   }
			   j++;
               jrow=ja[j];
		    }
		}
		while ((bcont)&&(j <= j2));

		if (icode==(-1)) {
			// Refresh all entries of iw to zero.
			for (i=j1; i<=j2; i++) iw[ja[i]]=-1;
		    k++;
		}
	}

	delete iw;

    //********
	//for (i=0; i<ia[n]; i++) printf("%e ",luval[i]);
	//printf("\n");
    //for (i=0; i<n; i++) printf("%d ",uptr[i]);
	//getchar();

	/*
	if (icode==(-1)) {
            // ILU �������������������:
            Real *U_val, *L_val;
	        int  *U_ind, *U_ptr, *L_ind, *L_ptr;
			IMatrix xO;
			
			initIMatrix(&xO, n); // �������������
             
            convertCRStoIMatrix(n, luval, ja, ia, uptr, &xO);
			delete luval; 
			delete uptr;
			convertIMatrixtoCSIR_ILU_ITL(&xO, U_val, U_ind, U_ptr, L_val, L_ind, L_ptr);
            freeIMatrix(&xO);
			// ���������� �� �����������
	        for (i = 0; i < n; i++) {
                QuickSortCSIR(U_ind, U_val, U_ptr[i], U_ptr[i+1]-1);
                QuickSortCSIR(L_ind, L_val, L_ptr[i], L_ptr[i+1]-1);
				L_val[L_ptr[i]]=1.0; // ������� �� ������� ���������.
	        }

			  // ���������� ������������ ������
	         
	         for (i=0; i<U_ptr[n]; i++) {
		         printf("%e ",U_val[i]);
	         }
	         printf("\n");
             for (i=0; i<U_ptr[n]; i++) {
		         printf("%d ",U_ind[i]);
	         }
	         printf("\n");
	         for (i=0; i<n+1; i++) {
		         printf("%d ",U_ptr[i]);
	         }
	         printf("\n");
	         getchar();

			 
	         for (i=0; i<L_ptr[n]; i++) {
		         printf("%e ",L_val[i]);
	         }
	         printf("\n");
             for (i=0; i<L_ptr[n]; i++) {
		         printf("%d ",L_ind[i]);
	         }
	         printf("\n");
	         for (i=0; i<n+1; i++) {
		         printf("%d ",L_ptr[i]);
	         }
	         printf("\n");
	         getchar();
	         

		}
	*/

} // ilu0_Saad

/* ����� ������������ ����������
* ��� �������� �������������� ������� � (val, col_ind, row_ptr).
* ����������������� �� ������ ��������, ������ : "������
* ������� ���� ������� �����������".
* dV - ������ ����� ����,
* x - ��������� ����������� � ������� ��� NULL.
* n - ����������� � nxn.
* ���������� �������� ���������� ��� ������� ����� 2000.
* ��� ����� � ��������� ��������� ����� 8000 ��������.
* ������������ ����� �������� ��������� � ���������� maxiter.
* �������� ������ �� ������� ������� � ���������� ���������:
*  dterminatedTResudual.
* ������ ����� ����������. ���� ������� ������ ������ r_tilda, �� 
* ������� ����� ����� ����������. ����������� �� ����� ������� r_tilda:
* ������� ����� ��������� ������������ Scal(r,r_tilda,n) != 0.0.
*/
void BiSoprGrad(IMatrix *xO, equation3D* &sl, equation3D_bon* &slb,
	            Real *dV, Real* &x, int maxelm, int maxbound,
				bool bSaad, Real alpharelax, int  maxiter)
{
	printf("\nBiConjugate Gradients Method...:\n");

    int i; // ������� ����� for
	int n=maxelm+maxbound;
	int iflag=1; // ����� ����������.

	// ����������� ������� ����
	// � CRS �������.
    Real *val;
    int* col_ind, *row_ptr;
	Real dbuf;

	// �������������� �� SIMPLESPARSE ������� � CRS ������ ��������.
	//simplesparsetoCRS(M, val, col_ind, row_ptr, n);
	equation3DtoCRS(sl, slb, val, col_ind, row_ptr, maxelm, maxbound, alpharelax);
	// ������� �����  ��������� ���� �� ������ ����,
	// �.�. ����������� ������� ����� ��������� � ����� 
	// ����������� �������� ����� ������� ������.
	Real *dax=new Real[n];
	Real *ri=new Real[n];
	MatrixCRSByVector(val,col_ind,row_ptr,x,dax,n);
	for (i=0; i<n; i++) ri[i]=dV[i]-dax[i];
	delete dax;
	if (fabs(NormaV(ri,n))<dterminatedTResudual) iflag=0;
	delete ri;
	 //if (iflag) Bi_CGStabCRS((maxelm+maxbound), val, col_ind, row_ptr, dV, x, 8000); // debug equation3DtoCRS
	 // printf("test equation3DtoCRS .../n");
	  // getchar();
	if (iflag==0) {
		delete val;
		delete col_ind;
		delete row_ptr;
	}

	if (iflag==1) {

	// ILU �������������������:
    Real *U_val, *L_val;
	int  *U_ind, *U_ptr, *L_ind, *L_ptr;

	if (!bSaad) {
		
		printf("Incoplete LU Decomposition begin...\n");
        convertIMatrixtoCSIR_ILU_ITL(xO, U_val, U_ind, U_ptr, L_val, L_ind, L_ptr);// ������������ ������
	    //printf("max memory BiSoprGrad...\n"); getchar(); // debug
	    // ������������ ����������� ������
	    freeIMatrix(xO);

    
	    /* // debug TODO 2
	    printf("TODO 2\n");
	    for (i=0; i<U_ptr[n]; i++) {
		     printf("%e ",U_val[i]);
	    }
	    printf("\n");
        for (i=0; i<U_ptr[n]; i++) {
		    printf("%d ",U_ind[i]);
	    }
	    printf("\n");
	    for (i=0; i<n+1; i++) {
		    printf("%d ",U_ptr[i]);
	    }
	    printf("\n");
	    getchar();
	    */

	    ILU0_Decomp_ITL(U_val, U_ind, U_ptr, L_val, L_ind, L_ptr, n);
	    printf("Incoplete LU Decomposition finish...\n");

	}
	else {
		// ILU(0) ���������� �� ����� �. �����
        printf("Incoplete LU Decomposition I.Saad begin...\n");
		freeIMatrix(xO);
		Real *luval;
		int *uptr;
		int icode=-1;
        ilu0_Saad(n, val, col_ind, row_ptr, luval, uptr, icode); // ILU(0) ����������
		if (icode==(-1)) {
			IMatrix xO1;
            initIMatrix(&xO1, n); // �������������

            convertCRStoIMatrix(n, luval, col_ind, row_ptr, uptr, &xO1);
			delete luval; 
			delete uptr;
			convertIMatrixtoCSIR_ILU_ITL(&xO1, U_val, U_ind, U_ptr, L_val, L_ind, L_ptr);
            freeIMatrix(&xO1);
			// ���������� �� �����������
	        for (i = 0; i < n; i++) {
                QuickSortCSIR(U_ind, U_val, U_ptr[i], U_ptr[i+1]-1);
                QuickSortCSIR(L_ind, L_val, L_ptr[i], L_ptr[i+1]-1);
				L_val[L_ptr[i]]=1.0; // ������� �� ������� ���������.
	        }

		}
		else {
			printf("Error!!! zero  diagonal elem in %d string matrix.\n",icode);
			getchar();
			exit(0); // ����� �� ���������.
		}

		printf("Incoplete LU Decomposition I.Saad finish...\n");

	}


	Real *r=new Real[n], *r_tilda=new Real[n];
	Real *p=new Real[n], *f=new Real[n], *p_tilda=new Real[n];
	Real nz; // �������
	Real *ap=new Real[n], *vcopy=new Real[n];
	Real a,b,dold, dnew;

	
	int k=0; // ����� ��������.

	// ��������� �����������:
    //X0==
	if (x==NULL) {
        x=new Real[n];
		for(i=0;i<n;i++) x[i] = 0.0;
	}

	// ��������� �������� �������
	Real e = dterminatedTResudual;

	MatrixCRSByVector(val,col_ind,row_ptr,x,ap,n);
	for (i=0; i<n; i++) {
		r[i]=dV[i]-ap[i];
		r_tilda[i]=r[i];
	}

	 // p==M^(-1)*r;
    for (i=0; i<n; i++) vcopy[i]=r[i];
    inverseL_ITL(vcopy, L_val, L_ind, L_ptr, p, n);
    for (i=0; i<n; i++) vcopy[i]=p[i];  
	inverseU_ITL(vcopy, U_val, U_ind, U_ptr, p, n);

    // p_tilda==M^(-T)*r_tilda;
	for (i=0; i<n; i++) vcopy[i]=r_tilda[i];
    inverseL_ITL(vcopy, U_val, U_ind, U_ptr, p_tilda, n);
    for (i=0; i<n; i++) vcopy[i]=p_tilda[i];  
	inverseU_ITL(vcopy, L_val, L_ind, L_ptr, p_tilda, n);
	   


	nz=NormaV(r,n); // ��������� �������� �������

	for (i=0; i<n; i++) vcopy[i]=r[i];
    inverseL_ITL(vcopy, L_val, L_ind, L_ptr, f, n);
    for (i=0; i<n; i++) vcopy[i]=f[i];  
	inverseU_ITL(vcopy, U_val, U_ind, U_ptr, f, n);
	// f==M^(-1)*r;
	dold=Scal(f,r_tilda,n); 

    while ((nz>e) && (k<maxiter)) { 
		MatrixCRSByVector(val,col_ind,row_ptr,p,ap,n);

		a=dold/Scal(ap,p_tilda,n);
        #pragma omp parallel for shared(n,x,r,p,ap,a) private(i) schedule (guided)
		for (i=0; i<n; i++) {
           x[i]+=a*p[i];
		   r[i]-=a*ap[i];
		}
		delete ap;
		ap=MatrixTransposeCRSByVector(val,col_ind,row_ptr,p_tilda,n);

        #pragma omp parallel for shared(n,r_tilda,ap,a) private(i) schedule (guided)
        for (i=0; i<n; i++) {
			r_tilda[i]-=a*ap[i];
		}

		#pragma omp parallel for shared(n,vcopy,r) private(i) schedule (guided)
        for (i=0; i<n; i++) vcopy[i]=r[i];
        inverseL_ITL(vcopy, L_val, L_ind, L_ptr, f, n);

        #pragma omp parallel for shared(n,vcopy,f) private(i) schedule (guided)
        for (i=0; i<n; i++) vcopy[i]=f[i];  
	    inverseU_ITL(vcopy, U_val, U_ind, U_ptr, f, n);

	    // f==M^(-1)*r;
		dnew=Scal(f,r_tilda,n);
		b=dnew/dold;
		dold=dnew;
		// ���������� �������.
        nz=NormaV(r,n);
		if (k%10==0) printf("iter residual\n");
		printf(" %d %e\n", k, nz);

		if ((fabs(b) < 1e-60) || (fabs(nz)>1e10)) {
			// ����� ������������ ���������� ������ ����������.
			printf("\nBiCG divergence detected...\n");
            getchar();
			exit(0); // ����� �� ����������.
			break; // ����� �� ����� while
		}

        #pragma omp parallel for shared(n,p,f,b) private(i) schedule (guided)
        for (i=0; i<n; i++) {
			p[i]=f[i]+b*p[i];
		}

		#pragma omp parallel for shared(n,vcopy,r_tilda) private(i) schedule (guided)
		for (i=0; i<n; i++) vcopy[i]=r_tilda[i];
        inverseL_ITL(vcopy, U_val, U_ind, U_ptr, f, n);
        #pragma omp parallel for shared(n,vcopy,f) private(i) schedule (guided)
        for (i=0; i<n; i++) vcopy[i]=f[i];  
	    inverseU_ITL(vcopy, L_val, L_ind, L_ptr, f, n);
	    // f==M^(-T)*r_tilda;
		#pragma omp parallel for shared(n,p_tilda,f,b,dbuf) private(i) schedule (guided)
        for (i=0; i<n; i++) {
			dbuf=p_tilda[i];
		    p_tilda[i]=f[i]+b*dbuf;
		}

		
		k++; // ������� � ��������� ��������.
	}

	// ������������ ������
	delete r; delete r_tilda;
	delete p; delete p_tilda;
	delete ap; delete f;
	delete vcopy;
	delete U_val; delete U_ind; delete U_ptr;
	delete L_val; delete L_ind; delete L_ptr;
	delete val; delete col_ind; delete row_ptr;
	

	} // if (iflag) end


} // BiSoprGrad

// �������� �.�. ����������� [1993]
// ��� �������� �������������� ������.
// ���������������� �� ����������
// "��������� ������ ������� ������ ���������" [2004]
// �������������� ���������������� ������������ ������������.
// �������� ILU0 �������������������. ����� ���� ����� ����� ILU0 ��������������������
// �� ����� �. ����� (bSaad==true) ��� ILU0 �������������������� �� ���������� ITL.
void SoloveichikAlg( IMatrix *xO, equation3D* &sl, equation3D_bon* &slb,// ����������� ������� ����
					     int maxelm, int maxbound, // ����� ���������� � ��������� ��
                         Real *dV,  // ������ ������ �����
                         Real* &dX0, // ������ ���������� �����������
                         bool bconsole_message, // �������� �� �������� ������� �� ������� ?
						 bool bSaad, // ���� bSaad==true �� ������������ ilu0 ���������� �� ����� �. ����� ����� ������������ ITL ilu0 ����������. 
						 int imaxiter,// ����������� ���������� ���-�� ��������
						 Real alpharelax) 
{
    
	int isize = xO->n;// ������ ���������� �������
	 // ����������� ������� ����
	 // � CRS �������.
     Real *val;
     int* col_ind, *row_ptr;

	 // �������������� �� SIMPLESPARSE ������� � CRS ������ ��������.
	 //simplesparsetoCRS(M, val, col_ind, row_ptr, isize);
	 equation3DtoCRS(sl, slb, val, col_ind, row_ptr, maxelm, maxbound, alpharelax);

	 // ILU �������������������:
     Real *U_val, *L_val;
	 int  *U_ind, *U_ptr, *L_ind, *L_ptr;

	 if (!bSaad) {
		
		printf("Incoplete LU Decomposition begin...\n");
        convertIMatrixtoCSIR_ILU_ITL(xO, U_val, U_ind, U_ptr, L_val, L_ind, L_ptr);// ������������ ������
	    //printf("max memory BiSoprGrad...\n"); getchar(); // debug
	    // ������������ ����������� ������
	    freeIMatrix(xO);

    
	    /* // debug TODO 2
	    printf("TODO 2\n");
	    for (i=0; i<U_ptr[n]; i++) {
		     printf("%e ",U_val[i]);
	    }
	    printf("\n");
        for (i=0; i<U_ptr[n]; i++) {
		    printf("%d ",U_ind[i]);
	    }
	    printf("\n");
	    for (i=0; i<n+1; i++) {
		    printf("%d ",U_ptr[i]);
	    }
	    printf("\n");
	    getchar();
	    */

	    ILU0_Decomp_ITL(U_val, U_ind, U_ptr, L_val, L_ind, L_ptr, isize);
	    printf("Incoplete LU Decomposition finish...\n");

	}
	else {
		// ILU(0) ���������� �� ����� �. �����
        printf("Incoplete LU Decomposition I.Saad begin...\n");
		freeIMatrix(xO);
		Real *luval;
		int *uptr;
		int icode=-1;
        ilu0_Saad(isize, val, col_ind, row_ptr, luval, uptr, icode); // ILU(0) ����������
		if (icode==(-1)) {
			IMatrix xO1;
            initIMatrix(&xO1, isize); // �������������

            convertCRStoIMatrix(isize, luval, col_ind, row_ptr, uptr, &xO1);
			delete luval; 
			delete uptr;
			convertIMatrixtoCSIR_ILU_ITL(&xO1, U_val, U_ind, U_ptr, L_val, L_ind, L_ptr);
            freeIMatrix(&xO1);
			// ���������� �� �����������
	        for (int i = 0; i < isize; i++) {
                QuickSortCSIR(U_ind, U_val, U_ptr[i], U_ptr[i+1]-1);
                QuickSortCSIR(L_ind, L_val, L_ptr[i], L_ptr[i+1]-1);
				L_val[L_ptr[i]]=1.0; // ������� �� ������� ���������.
	        }

		}
		else {
			printf("Error!!! zero  diagonal elem in %d string matrix.\n",icode);
			getchar();
			exit(0); // ����� �� ���������.
		}

		printf("Incoplete LU Decomposition I.Saad finish...\n");

	}


     int i,k; // �������� ����� for
     Real *dx, *dax, *dr, *dz, *dp, *dar1, *dres, *f, *vcopy;
     Real dar, dbr, dnz, dscalp;
	 Real kend=imaxiter; // ����������� �� ������������ ����� ��������
	 Real epsilon=dterminatedTResudual;  // �������� ����������
	 bool bweShouldContinue=true;


    // ��������� ������ ��� ������������ �������
    dx=new Real[isize]; dax=new Real[isize]; dr= new Real[isize];
    dar1=new Real[isize]; vcopy=new Real[isize];dp= new Real[isize];
	dres=new Real[isize]; f=new Real[isize]; dz=new Real[isize];// ������ ����������
   

   // ��������� �����������
   // X0 ==
   // ��� X0 ���������� ������ ���� ���������� � �������.
   if (dX0==NULL) {
	   dX0=new Real[isize];
	   for (i=0; i<isize; i++) {
		   dx[i]=0.0;
		   dX0[i]=0.0;
	   }
   }
   else {
	   for (i=0; i<isize; i++) dx[i]=dX0[i];
   }

   
   MatrixCRSByVector(val,col_ind,row_ptr,dx, dax, isize); // ��������� ������ �  dax
   for (i=0; i<isize; i++) dr[i]= dV[i] - dax[i];  // ��������� �������
   // dr=L^(-1)*(dV-A*dx);
   for (i=0; i<isize; i++) vcopy[i]=dr[i]; 
   inverseL_ITL(vcopy, L_val, L_ind, L_ptr, dr, isize);
   dnz=Scal(dr,dr,isize); // ��������� �������� �������
   // dz=U^(-1)*dr;
   for (i=0; i<isize; i++) vcopy[i]=dr[i];  // ������ ������ (���������� ����������� ������).
   inverseU_ITL(vcopy, U_val, U_ind, U_ptr, dz, isize);
   // dp=L^(-1)*A*dz;
   MatrixCRSByVector(val,col_ind,row_ptr,dz,dp, isize);// ��������� ������ � dp
   for (i=0; i<isize; i++) vcopy[i]=dp[i]; 
   inverseL_ITL(vcopy, L_val, L_ind, L_ptr, dp, isize);

   if (fabs(Scal( dp, dp, isize))>1e-270) 
   {
      k=1; // �������� ���������� ������ � 1
      // ��������� �������� ������� ��������� ����
      while ((bweShouldContinue) && (k <= kend) && (fabs(dnz) > epsilon))
	  {
         dscalp=1.0/Scal( dp, dp, isize);
         dar=Scal(dp, dr,isize)*dscalp;
         for (i=0; i<isize; i++)
		 {
            dx[i]=dx[i]+dar*dz[i];
            dr[i]=dr[i]-dar*dp[i];
		 }
         dnz=dnz-dar*dar/dscalp; // ����� �������
         
         if (bconsole_message) 
		 {
            // ������ ������� �� �������
            if ((k % 10) == 0)  printf("iter  residual\n");
            printf("%d %e \n",k,dnz);
		 } 
		 
         // f=U^(-1)*dr;
         for (i=0; i<isize; i++) vcopy[i]=dr[i];  
         inverseU_ITL(vcopy, U_val, U_ind, U_ptr, f, isize);
         for (i=0; i<isize; i++) vcopy[i]=f[i]; 
		 MatrixCRSByVector(val,col_ind,row_ptr,vcopy, dar1, isize);// ��������� ������ � dar1=A*U^(-1)*dr
         for (i=0; i<isize; i++) vcopy[i]=dar1[i]; 
		 // dar1=L^(-1)*A*U^(-1)*dr;
         inverseL_ITL(vcopy, L_val, L_ind, L_ptr, dar1, isize);

         dbr=-Scal(dp,dar1,isize)*dscalp;
         for (i=0; i<isize; i++)
		 {
            dz[i]=f[i]+dbr*dz[i];
            dp[i]=dar1[i]+dbr*dp[i];
		 }

         k++;
         // ���� ������� ���������� �� ��� ���� ����������
         if (dnz > 1e14) 
		 {
            // �������������� ���������� �����������
            for (i=0; i<isize; i++) if (dX0==NULL) dx[i]=0.0; else dx[i]=dX0[i];
            printf("\n divergence Soloveichik solver \n");
			//fprintf(fp_statistic_convergence, " divergence Soloveichik solver finish residual > 1e7 \n");
			fprintf(fp_statistic_convergence, "%e ",fabs(dnz));
            bweShouldContinue=false;
            break; // ����� �� ����� while
		 }
 
	  } // while

	  if (bweShouldContinue) {
		  //fprintf(fp_statistic_convergence, " Soloveichik solver finish residual=%e \n",dnz);
		  fprintf(fp_statistic_convergence, "%e ",fabs(dnz));
		  //fprintf(fp_statistic_convergence, " Soloveichik solver finish residual=%e \n",NormaChebyshev(dr, isize));  
	  }

      // ����������� ����������
      for (i=0; i<isize; i++) dres[i]=dx[i];
   }
   else
   {
      // ���������� ��������� �����������
	  for (i=0; i<isize; i++) dres[i]=dX0[i];
   }

   // ������������ ������ ���������� ��� ������������ �������
   delete dx; delete dax; delete dr; delete vcopy;
   delete dz; delete dp; delete dar1; delete f;
   delete U_val; delete U_ind; delete U_ptr;
   delete L_val; delete L_ind; delete L_ptr;
   delete val; delete col_ind; delete row_ptr;

   for (i=0; i<isize; i++) {
		 dX0[i]=dres[i];
	   }
   delete dres;  

} // SoloveichikAlg

// ����� ��� ��� ������ Bi-CGStabCRS
// �������� ��� �������� �������������� ������������ ������.
// �������������� ������� ���� ��������� � CRS �������
// A (val, col_ind, row_ptr).
// ����� �������� ����������� ������� BiCG � GMRES(1). 
void Bi_CGStabCRS(int n, Real *val, int* col_ind, int* row_ptr, Real *dV, Real* &dX0, int maxit)
{

	int iflag=1, icount=0;
	Real delta0, deltai;
	Real bet, roi;
	Real roim1=1.0, al=1.0, wi=1.0;
	Real *ri, *roc, *s, *t, *vi, *pi, *dx, *dax;
	Real epsilon=dterminatedTResudual;  // �������� ����������
	int i;

	ri=new Real[n]; roc=new Real[n]; s=new Real[n]; t=new Real[n];
	vi=new Real[n]; pi=new Real[n]; dx=new Real[n]; dax=new Real[n];

	for (i=0; i<n; i++) {
		s[i]=0.0;
		t[i]=0.0;
		vi[i]=0.0;
		pi[i]=0.0;
	}

    // ��������� �����������
    // X0 ==
    // ��� X0 ���������� ������ ���� ���������� � �������.
    if (dX0==NULL) {
	   dX0=new Real[n];
	   for (i=0; i<n; i++) {
		   dx[i]=0.0;
		   dX0[i]=0.0;
	   }
    }
    else {
	   for (i=0; i<n; i++) dx[i]=dX0[i];
    }

    MatrixCRSByVector(val,col_ind,row_ptr,dx,dax, n); // ��������� ������ �  dax
	for (i=0; i<n; i++) {
		ri[i]=dV[i]-dax[i];
		roc[i]=ri[i];
	}
	delta0=NormaV(ri,n);
	// ���� ������� ����� ������� �� �� �������:
	if (fabs(delta0)<epsilon) iflag=0; 

	while ( iflag != 0 && icount < maxit) {

		icount++;

		roi=Scal(roc,ri,n);
		bet=(roi/roim1)*(al/wi);
		for (i=0; i<n; i++) {
			pi[i]=ri[i]+(pi[i]-vi[i]*wi)*bet;
		}
	
		MatrixCRSByVector(val,col_ind,row_ptr,pi,vi, n);
		al=roi/Scal(roc,vi,n);
        for (i=0; i<n; i++) {
			s[i]=ri[i]-al*vi[i];
		}
		
        MatrixCRSByVector(val,col_ind,row_ptr,s,t, n);
		wi=Scal(t,s,n)/Scal(t,t,n);
		for (i=0; i<n; i++) {
			dx[i]+=al*pi[i]+wi*s[i];
			ri[i]=s[i]-wi*t[i];
		}
		deltai=NormaV(ri,n);
		// ������ ������� �� �������
        if ((icount % 10) == 0)  printf("iter  residual\n");
        printf("%d %e \n",icount,deltai);

		if (deltai <epsilon) iflag=0; // ����� ����������
		else roim1=roi;
	}

    // ������������ ������
	delete ri; delete roc; delete s; delete t;
	delete vi; delete pi; delete dax;

	for (i=0; i<n; i++) dX0[i]=dx[i];

	delete dx; 


} // Bi_CGStabCRS



// ����� ��� ��� ������ Bi-CGStab
// �������� ��� �������� �������������� ������������ ������.
// ������� ������������������� ILU(0).
// ����� �������� ����������� ������� BiCG � GMRES(1). 
void Bi_CGStab(IMatrix *xO, equation3D* &sl, equation3D_bon* &slb,
			   int maxelm, int maxbound,
			   Real *dV, Real* &dX0, int maxit, Real alpharelax)
{

     printf("Bi_CGStab preconditioning by ILU(0)...\n"); 

	 int i=0; // ������� ����� for 
	 int n = xO->n;// ������ ���������� �������
	 // ����������� ������� ����
	 // � CRS �������.
     Real *val;
     int* col_ind, *row_ptr;

	 // �������������� �� SIMPLESPARSE ������� � CRS ������ ��������.
	 //simplesparsetoCRS(M, val, col_ind, row_ptr, n);
	 equation3DtoCRS(sl, slb, val, col_ind, row_ptr, maxelm, maxbound, alpharelax);

	 // ILU �������������������:
     Real *U_val, *L_val;
	 int  *U_ind, *U_ptr, *L_ind, *L_ptr;

	 printf("Incoplete LU Decomposition begin...\n");
     convertIMatrixtoCSIR_ILU_ITL(xO, U_val, U_ind, U_ptr, L_val, L_ind, L_ptr);
	 // ������������ ����������� ������
	 freeIMatrix(xO);
	 ILU0_Decomp_ITL(U_val, U_ind, U_ptr, L_val, L_ind, L_ptr, n);
	 printf("Incoplete LU Decomposition finish...\n");

	 Real *dx=new Real[n], *dr=new Real[n], *dr0=new Real[n],
		  *dax=new Real[n], *vcopy=new Real[n], *p=new Real[n],
          *s=new Real[n], *dap=new Real[n];

	 Real alpha, omega, dnz, scal1, beta;
	 Real epsilon=dterminatedTResudual;  // �������� ����������
	 int icount;

	 // ��������� �����������
     // X0 ==
     // ��� X0 ���������� ������ ���� ���������� � �������.
     if (dX0==NULL) {
	     dX0=new Real[n];
	     for (i=0; i<n; i++) {
		     dx[i]=0.0;
		     dX0[i]=0.0;
	     }
     }
     else {
	     for (i=0; i<n; i++) dx[i]=10.0;//dX0[i];
     }

	 MatrixCRSByVector(val,col_ind,row_ptr,dx, dax, n); // ��������� ������ �  dax
     for (i=0; i<n; i++) dr[i]= dV[i] - dax[i];  // ��������� �������
     // dr=L^(-1)*(dV-A*dx);
     for (i=0; i<n; i++) vcopy[i]=dr[i]; 
     inverseL_ITL(vcopy, L_val, L_ind, L_ptr, dr, n);
      // dr=U^(-1)*dr;
     for (i=0; i<n; i++) vcopy[i]=dr[i];
     inverseU_ITL(vcopy, U_val, U_ind, U_ptr, dr, n);
	 // dr - ������� ���������� �����������.
	 for (i=0; i<n; i++) {
		 dr0[i]=dr[i];
		 p[i]=dr[i];
	 }

	 icount=0;
	 do {
		 icount++;
		 scal1=Scal(dr,dr0,n);

         MatrixCRSByVector(val,col_ind,row_ptr,p, vcopy, n); // ��������� ������ �  vcopy
         inverseL_ITL(vcopy, L_val, L_ind, L_ptr, dap, n);
         for (i=0; i<n; i++) vcopy[i]=dap[i];
         inverseU_ITL(vcopy, U_val, U_ind, U_ptr, dap, n);

		 alpha=scal1/Scal(dap,dr,n);

		 for (i=0; i<n; i++) {
			 s[i]=dr[i]-alpha*dap[i];
		 }

		 MatrixCRSByVector(val,col_ind,row_ptr,s, vcopy, n); // ��������� ������ �  vcopy
         inverseL_ITL(vcopy, L_val, L_ind, L_ptr, dap, n);
         for (i=0; i<n; i++) vcopy[i]=dap[i];
         inverseU_ITL(vcopy, U_val, U_ind, U_ptr, dap, n);

         omega=Scal(dap,s,n)/Scal(dap,dap,n);

         for (i=0; i<n; i++) {
			 dx[i]=dx[i]+alpha*p[i]+omega*s[i];
			 dr[i]=s[i]-omega*dap[i];
		 }

         dnz=NormaV(dr,n);
		 // ������ ������� �� �������
         if ((icount % 10) == 0)  printf("iter  residual\n");
         printf("%d %e \n",icount,dnz);

         beta=(Scal(dr,dr0,n)/scal1)*(alpha/omega);

         for (i=0; i<n; i++) {
			 p[i]=dr[i]+beta*(p[i]-omega*dap[i]);
		 }

	 } while ((fabs(dnz)> epsilon) && (icount<maxit));



	 delete vcopy; delete dr0; delete dr;
	 delete dax; delete s; delete dap; delete p;
	 delete U_val; delete L_val;
	 delete U_ind; delete U_ptr; delete L_ind; delete L_ptr;
	 delete val; delete col_ind; delete row_ptr;
	 
	 for (i=0; i<n; i++) dX0[i]=dx[i];
	 delete dx;

} // Bi_CGStab

// ���������� �������� �� ���� ����� �����.
int imax(int ia, int ib) {
	int ir=ia;
	if (ib>ia) ir=ib;
	return ir;
} // imax

// �.�.�����, �.�.������ 
// ��������� ������������� ������������� ������ � ���������������� �������.
// ������� �������� ���������������� ������������. ���������� � �������� �2(14) 2011���.
// �������� ������� �� ������ ��������� ���������� LR1 � Bi-CGStab P.
// LR1 - ������������ ����� ������������ ��� � ����� �. ��������� : ������
// ������� ������ �������� (�������� ������) � ������ ������-�������.
// Bi-CGStab P - �������� ��� ��� ������ � ������������������� : ������ Bi-CG � GMRES(1).
// ������ ���������, ������������ � ������������� � AliceFlow_v0_06 ���������� 
// 24 ������� 2011 ���� �� ������ ���������� ����������.
// ����� ��� ��� ������ Bi-CGStabCRS
// �������� ��� �������� �������������� ������������ ������.
// �������������� ������� ���� ��������� � CRS �������
// A (val, col_ind, row_ptr).
// ����� �������� ����������� ������� BiCG � GMRES(1).
//  begin 2 : 5 ��� 2012 ����. ������ ���������� ������� ������������.
// ������� ������������ ������� � �������������� ���������� �������� �������������������,
// ��� ������ ��������� ���� � �������� ������������������ ���� � ���������� ������������������.
// 
void LR1sK(FLOW &f, equation3D* &sl, equation3D_bon* &slb,
	       Real *val, int* col_ind, int* row_ptr,
		   int maxelm, int maxbound, int iVar,
		   Real *dV, Real* &dX0, int maxit, bool bprintmessage, bool bexporttecplot)
{
	
	// ���� ��������� ������ (������ ����������) �� ����� ��������� ������� � ��������� tecplot 360
	// ��� ����������� ������� �������.
	bexporttecplot=false; 
	
	//Real *val;
	//int* col_ind;
	//int* row_ptr;
	int n=maxelm+maxbound;
	// �������������� �� SIMPLESPARSE ������� � CRS ������ ��������.
	//equation3DtoCRS(sl, slb, val, col_ind, row_ptr, maxelm, maxbound);

	bool bnorelax=false; // ���� bnorelax==false ������ ������������������� LR1sk ����������� ������ ���������� �� ����������� ��������.
	if (iVar==PAM) bnorelax=true; // ��� �������� �������� ���������� �� �����������.

	// ����� ���� ��� ��������� ���� �������������� ������������ ������������� ������������������� �����.
	// ������� ����� ������ 3 ������ �������� 4 � ��� �������� 5.
	int imaxdubl=4; // 4 ��������� �������� imaxdubl.
	if (iVar==PAM) imaxdubl=5; // 5
	bool bprintf=false; // ���� bprintf==false �� �������� ������� ������ LR1sk �� ���������.
	int iflag=1, icount=0;
	Real delta0, deltai;
	Real bet, roi;
	Real roim1=1.0, al=1.0, wi=1.0;
	Real *ri, *roc, *s, *t, *vi, *pi, *dx, *dax;
	Real *y, *z; // ��������� ������������������
	Real epsilon=dterminatedTResudual;  // �������� ����������
	int i;

	ri=new Real[n]; roc=new Real[n]; s=new Real[n]; t=new Real[n];
	vi=new Real[n]; pi=new Real[n]; dx=new Real[n]; dax=new Real[n];
	y=new Real[n]; z=new Real[n]; // ��������� ����������� ������ ��� ����������� ������������������

    #pragma omp parallel for shared(s, t, vi, pi, y, z) private(i) schedule (guided)
	for (i=0; i<n; i++) {
		s[i]=0.0;
		t[i]=0.0;
		vi[i]=0.0;
		pi[i]=0.0;
		// ������������� �������� ��� ������������������
		y[i]=0.0;
		z[i]=0.0;
	}

    // ��������� �����������
    // X0 ==
    // ��� X0 ���������� ������ ���� ���������� � �������.
    if (dX0==NULL) {
	   dX0=new Real[n];

	   #pragma omp parallel for shared(dx, dX0) private(i) schedule (guided)
	   for (i=0; i<n; i++) {
		   dx[i]=0.0;
		   dX0[i]=0.0;
	   }
    }
    else {
       #pragma omp parallel for shared(dx, dX0) private(i) schedule (guided)
	   for (i=0; i<n; i++) dx[i]=dX0[i];
    }

    MatrixCRSByVector(val,col_ind,row_ptr,dx,dax, n); // ��������� ������ �  dax

	#pragma omp parallel for shared(ri, dV, dax, roc) private(i) schedule (guided)
	for (i=0; i<n; i++) {
		ri[i]=dV[i]-dax[i];
		roc[i]=ri[i];
	}
	delta0=NormaV(ri,n);

	// ������� � ��������� ���� ���������� � ��� ��� �������� ���������
	// ���������� �������� �� �������� � �������� ��������� SIMPLE.
	switch (iVar) {
	  case VX : fprintf(fp_statistic_convergence,"%+.16f ",delta0); break;
	  case VY : fprintf(fp_statistic_convergence,"%+.16f ",delta0); break;
	  case VZ : fprintf(fp_statistic_convergence,"%+.16f ",delta0); break;
      case PAM : fprintf(fp_statistic_convergence,"%+.16f ",delta0); break; // ��� �������� �������� ����� ����� ���� ����������� LR1sK ��������.
	}

	// ���� ������� ����� ������� �� �� �������:
	if (fabs(delta0)<dterminatedTResudual) iflag=0;

	
	if (iflag !=0) {
		// �������� ������� ������ �� �������� ������������� ������ ��������� �������.
	    epsilon*=delta0;
	    dterminatedTResudual=epsilon;
	}
	

	// ���� ��������� ������������.
	// ���� ����������� ��������� ������������������
	// ���������� ���������� �������� ����������� ������.
	// ����������� ���� �� ���� ���������������� ���������
	// ������� ������� ��������� ����� res[iter+1]/res[iter]<0.05 ��� �� 
	// ������� ������� ������� � ����� ��������� imaxdubl, ����
	// 0.05 <= res[iter+1]/res[iter] < 0.23 �� ���������� ��������
	// ���������� imaxdubl ������� �� ����. � ���� 
	// res[iter+1]/res[iter] >= 0.23 �� ����� ������� ������������������
	// �������� imaxdubl.

	Real* resiter=new Real[2];
	const int iNOW=1;
	const int iOLD=0;
	resiter[iOLD]=delta0; resiter[iNOW]=delta0;

	// ����� ��� ��������� ����� ��������� �������������� �������.
	// ����������� �������� ���� ���������� ����� ���������� �� 
	// ��������������� ������������.
	const Real LBAR=0.05;
	const Real RBAR=0.23;

	while ( iflag != 0 && icount < maxit) {

		icount++;

		roi=Scal(roc,ri,n);
		bet=(roi/roim1)*(al/wi);
		#pragma omp parallel for shared(pi, ri, vi, wi, bet) private(i) schedule (guided)
		for (i=0; i<n; i++) {
			Real rbufpi=ri[i]+(pi[i]-vi[i]*wi)*bet;
			pi[i]=rbufpi;
		}
	
		// Ky=pi
		// ����� ����� �������� � ���� ����� �� ����� ����������.
		#pragma omp parallel for shared(y) private(i) schedule (guided)
		for (i=0; i<n; i++) y[i]=0.0; // ���� �������� �� � ���� �� ������� ���������� ��� PAM !.
		solveLRn(y, pi, n, iVar, imaxdubl, bprintf, bnorelax, f.sosedi, f.maxelm, f.slau, f.slau_bon, f.iN, f.id, f.iWE, f.iSN, f.iBT, f.alpha, f.maxbound);

		MatrixCRSByVector(val,col_ind,row_ptr,y,vi, n);
		al=roi/Scal(roc,vi,n);
		#pragma omp parallel for shared(s,ri, al, vi) private(i) schedule (guided)
        for (i=0; i<n; i++) {
			s[i]=ri[i]-al*vi[i];
		}

		// Kz=s
		// ����� ����� �������� � ���� ����� �� ����� ����������.
		#pragma omp parallel for shared(z) private(i) schedule (guided)
		for (i=0; i<n; i++) z[i]=0.0; // ���� �������� �� � ���� �� ������� ���������� ��� PAM !.
        solveLRn(z, s, n, iVar, imaxdubl, bprintf, bnorelax, f.sosedi, f.maxelm, f.slau, f.slau_bon, f.iN, f.id, f.iWE, f.iSN, f.iBT, f.alpha, f.maxbound);
		
        MatrixCRSByVector(val,col_ind,row_ptr,z,t, n);
		wi=Scal(t,s,n)/Scal(t,t,n);
		#pragma omp parallel for shared(dx, al, y, wi, z, ri, s, t) private(i) schedule (guided)
		for (i=0; i<n; i++) {
			//dx[i]+=al*pi[i]+wi*s[i]; // ��� ���� ��� �������������������
			dx[i]+=al*y[i]+wi*z[i]; // ��� ����� � ��������������������
			ri[i]=s[i]-wi*t[i];
		}
		deltai=NormaV(ri,n);

		// ������������.
		resiter[iOLD]=resiter[iNOW]; resiter[iNOW]=deltai;
		Real dres=resiter[iNOW]/resiter[iOLD];
		if (fabs(dres)>RBAR) {
			imaxdubl++;
		}
		else if (fabs(dres)<LBAR) {
			imaxdubl=imax(1,imaxdubl-1);
		}

		if (bprintmessage) {
			// ������ ������� �� �������
            if ((icount % 10) == 0)  {
				printf("iter  residual imaxdubl\n");
				fprintf(fp_log,"iter  residual imaxdubl\n");
			}
            printf("%d %e %d\n",icount,deltai,imaxdubl);
            fprintf(fp_log,"%d %e %d\n",icount,deltai,imaxdubl);
		}

		if (deltai <epsilon) iflag=0; // ����� ����������
		else roim1=roi;

		// ������ ��������� ������� � �������� ����� �� ����� ���������.
		// ���������� ��� �������� ��� ������� ���������� �� ���������� 
		// ��������������� ���������������� �������. ��������� ���� ������ 
		// ��������� imaxdubl>=100; ������� ������ ����� �� ����� � ��.
		// �� ��� ��� ������� ����� ����������� ������ ������.
		// ������� � ������ ������� ��� AMG �� ��������. AMG -
		// �������������� �������������� ������, ������� ������� � 
		// ��������� ������� ��������� ���������� �� ������������������ ��������� �����.
		// � ������ ������� �������� ������� ������������������� 
		// ������ ��������� ������������ ������ �������, ��� 
		// � ����� ���� � ���� ��� ���������� ������ ����������� 
		// �� �������� ������ ����. ������ ��� ����� ������������������ ������
		// ����� ������� �������������������. ����� �����: ���������� ���������
		// ��������� BiCGStab � ����������� ������������������� ������ ����������� 
		// � ������ ���� ����� ������ ����� �������������� ��������.
		if (imaxdubl>=100) {
			// �������� ��������� 100 ����� ������ ���� ��������� �� ��������������� ������������.

			iflag=0; // ����� ����������
			printf("calculation can not cope with the stiffness of the problem...\n");
			fprintf(fp_log,"calculation can not cope with the stiffness of the problem...\n");
			printf("Please, press any key to continue calculation...\n");
			fprintf(fp_log,"Please, press any key to continue calculation...\n");
			bexporttecplot=true; // �������� �� ���������� => ������� �������� ��� ������� � ��������� tecplot.
			getchar();
		}

		// getchar(); // debug 1 iteration LR1sk
	}

	/* // ���������� ���.
	// ����� ������������ ������ � ������������ ��������� �������:
	Real maxerr=-1e30;
	int ierr=-1;
	for (i=0; i<n; i++) {
		if (fabs(ri[i])>maxerr) {
			ierr=i;
			maxerr=fabs(ri[i]);
		}
	}
	printf("node number max residual is %d, value residal is equal %e\n", ierr, maxerr);
	getchar();
	*/

    // ������������ ������
	delete ri; delete roc; delete s; delete t;
	delete vi; delete pi; delete dax;
	delete y; delete z;

	#pragma omp parallel for shared(dx, dX0) private(i) schedule (guided)
	for (i=0; i<n; i++) dX0[i]=dx[i];

	delete dx; 


} // LR1sK



// �.�.�����, �.�.������ 
// ��������� ����������� ������������� ������ � ���������������� �������.
// ������� �������� ���������������� ������������. ���������� � �������� �2(14) 2011���.
// �������� ������� �� ������ ��������� ���������� LR1 � Bi-CGStab P.
// LR1 - ���������� ����� ������������ ��� � ����� �. ��������� : ������
// ������� ������ �������� (�������� ������) � ������ ������-�������.
// Bi-CGStab P - �������� ��� ��� ������ � ������������������� : ������ Bi-CG � GMRES(1).
// ������ ���������, ������������ � ������������� � AliceFlow_v0_06 ���������� 
// 24 ������� 2011 ���� �� ������ ���������� ����������.
// ����� ��� ��� ������ Bi-CGStabCRS
// �������� ��� �������� �������������� ������������ ������.
// �������������� ������� ���� ��������� � CRS �������
// A (val, col_ind, row_ptr).
// ����� �������� ����������� ������� BiCG � GMRES(1).
//
// ��� ����� ���������������� � ������ ��������� �� ��������
// �� ����������� �� ���-���-����� �� ������������ �����.
// �������� ��-�� ���� ��� ���� ��� ����� ����� ����������� 
// ��� ��� ��������� ��������.
// �����. ����������� ��������� LR1sK ����� � ������ ���������������� 
// � ������ ���������.
// ������ ���������� 23 ������ 2011 ����.
//
void LR1sK_temp(TEMPER &tGlobal, equation3D* &sl, equation3D_bon* &slb,
	       Real *val, int* col_ind, int* row_ptr,
		   int maxelm, int maxbound, 
		   Real *dV, Real* &dX0, int maxit, int inumiter, bool bprintmessage, bool &bexporttecplot)
{

	// inumiter - ����� ���������� �������� (�������� ����� �������� � ������������ ��������� SIMPLE).
	// �������� inumiter - ����� ��� ���� ����� �������������� ��� �������, ����� ����� ����������
	// �������� ������� ���� �� ���������� �������� (��������� SIMPLE) � ������� ������� ��� inumiter.

	if (0) {
		if (inumiter>82) {
			printf("debug LR1sk for temperature solver...\n");
		}
	}

	bexporttecplot=false; // ������� � ������� �������� ���� � ������ ������� �� �����������.

	//Real *val;
	//int* col_ind;
	//int* row_ptr;
	int n=maxelm+maxbound;
	// �������������� �� SIMPLESPARSE ������� � CRS ������ ��������.
	//equation3DtoCRS(sl, slb, val, col_ind, row_ptr, maxelm, maxbound);

	bool bnorelax=true; // ��� ��������� ���������������� �� ������������ ����������.
	
	int imaxdubl=3; // ��������� ���������� �������� ����������� ������ 3
	
	bool bprintf=false; // ���� bprintf==false �� �������� ������� ������ LR1sk �� ���������.
	int iflag=1, icount=0;
	Real delta0, deltai;
	Real bet, roi;
	Real roim1=1.0, al=1.0, wi=1.0;
	Real *ri, *roc, *s, *t, *vi, *pi, *dx, *dax;
	Real *y, *z; // ��������� ������������������
	Real epsilon=dterminatedTResudual;  // �������� ����������
	int i;

	ri=new Real[n]; roc=new Real[n]; s=new Real[n]; t=new Real[n];
	vi=new Real[n]; pi=new Real[n]; dx=new Real[n]; dax=new Real[n];
	y=new Real[n]; z=new Real[n]; // ��������� ����������� ������ ��� ����������� ������������������

	#pragma omp parallel for shared(s,t,vi,pi,y,z,dax) private(i) schedule (guided)
	for (i=0; i<n; i++) {
		s[i]=0.0;
		t[i]=0.0;
		vi[i]=0.0;
		pi[i]=0.0;
		// ������������� �������� ��� ������������������
		y[i]=0.0;
		z[i]=0.0;
		// ��������� ��������� ������� �� ������.
		dax[i]=0.0;
	}

    // ��������� �����������
    // X0 ==
    // ��� X0 ���������� ������ ���� ���������� � �������.
    if (dX0==NULL) {
	   dX0=new Real[n];
	   #pragma omp parallel for shared(dx, dX0) private(i) schedule (guided)
	   for (i=0; i<n; i++) {
		   dx[i]=0.0;
		   dX0[i]=0.0;
	   }
    }
    else {
	   #pragma omp parallel for shared(dx, dX0) private(i) schedule (guided)
	   for (i=0; i<n; i++) dx[i]=dX0[i];
    }

	MatrixCRSByVector(val,col_ind,row_ptr,dx,dax, n); // ��������� ������ �  dax

	#pragma omp parallel for shared(ri,dV,dax,roc) private(i) schedule (guided)
	for (i=0; i<n; i++) {
		ri[i]=dV[i]-dax[i];
		roc[i]=ri[i];
	}
	delta0=NormaV(ri,n);
	
	//printf("debug %e\n",NormaV(dax,n)); // �������� �� ��������� ����������� ����
	//getchar();
	// ���� ������� ����� ������� �� �� �������:
	if (fabs(delta0)<dterminatedTResudual) iflag=0; 

	//printf("delta0=%e\n",delta0);
	//getchar();

	
	/*if (iflag != 0) {
       // �������� ������� ������ �� �������� ������������� ������ ��������� �������.
	   epsilon*=delta0; 
	   dterminatedTResudual=epsilon;
	}
	*/

	// ���� ��������� ������������.
	// ���� ����������� ��������� ������������������
	// ���������� ���������� �������� ����������� ������.
	// ����������� ���� �� ���� ���������������� ���������
	// ������� ������� ��������� ����� res[iter+1]/res[iter]<0.05 ��� �� 
	// ������� ������� ������� � ����� ��������� imaxdubl, ����
	// 0.05 <= res[iter+1]/res[iter] < 0.23 �� ���������� ��������
	// ���������� imaxdubl ������� �� ����. � ���� 
	// res[iter+1]/res[iter] >= 0.23 �� ����� ������� ������������������
	// �������� imaxdubl.

	Real* resiter=new Real[2];
	const int iNOW=1;
	const int iOLD=0;
	resiter[iOLD]=delta0; resiter[iNOW]=delta0;

	// ����� ��� ��������� ����� ��������� �������������� �������.
	// ����������� �������� ���� ���������� ����� ���������� �� 
	// ��������������� ������������.
	const Real LBAR=0.05;
	const Real RBAR=0.23;
	

	while ( iflag != 0 && icount < maxit) {

		icount++;

		roi=Scal(roc,ri,n);
		bet=(roi/roim1)*(al/wi);

		#pragma omp parallel for shared(pi,ri,vi,wi,bet) private(i) schedule (guided)
		for (i=0; i<n; i++) {
			Real pibuf=ri[i]+(pi[i]-vi[i]*wi)*bet;
			pi[i]=pibuf;
		}
	
		// Ky=pi
		// ����� ����� �������� � ���� ����� �� ����� ����������.
		#pragma omp parallel for shared(y) private(i) schedule (guided)
		for (i=0; i<n; i++) y[i]=0.0; // ���� �������� �� � ���� �� ������� ���������� ��� PAM !.
		solveLRn_temp(tGlobal, y, pi, n, imaxdubl, bprintf);

		MatrixCRSByVector(val,col_ind,row_ptr,y,vi, n);
		al=roi/Scal(roc,vi,n);

		#pragma omp parallel for shared(s,ri,al,vi) private(i) schedule (guided)
        for (i=0; i<n; i++) {
			s[i]=ri[i]-al*vi[i];
		}

		// Kz=s
		// ����� ����� �������� � ���� ����� �� ����� ����������.
		#pragma omp parallel for shared(z) private(i) schedule (guided)
		for (i=0; i<n; i++) z[i]=0.0; // ���� �������� �� � ���� �� ������� ���������� ��� PAM !.
        solveLRn_temp(tGlobal, z, s, n, imaxdubl, bprintf);
		
        MatrixCRSByVector(val,col_ind,row_ptr,z,t, n);
		wi=Scal(t,s,n)/Scal(t,t,n);

		#pragma omp parallel for shared(dx, al, y, wi, z, ri, s, t) private(i) schedule (guided)
		for (i=0; i<n; i++) {
			//dx[i]+=al*pi[i]+wi*s[i]; // ��� ���� ��� �������������������
			dx[i]+=al*y[i]+wi*z[i]; // ��� ����� � ��������������������
			ri[i]=s[i]-wi*t[i];
		}
		deltai=NormaV(ri,n);


		// ������������.
		resiter[iOLD]=resiter[iNOW]; resiter[iNOW]=deltai;
		Real dres=resiter[iNOW]/resiter[iOLD];
		if (fabs(dres)>RBAR) {
			imaxdubl++;
		}
		else if (fabs(dres)<LBAR) {
			imaxdubl=imax(1,imaxdubl-1);
		}

		// ������ ������� �� �������
		if (bprintmessage) {
            if ((icount % 10) == 0)  {
				printf("iter  residual imaxdubl\n");
				fprintf(fp_log,"iter  residual imaxdubl\n");
			}
            printf("%d %e %d\n",icount,deltai,imaxdubl);
			fprintf(fp_log,"%d %e %d\n",icount,deltai,imaxdubl);
		}

		if (deltai <epsilon) iflag=0; // ����� ����������
		else roim1=roi;

		// ������ ��������� ������� � �������� ����� �� ����� ���������.
		// ���������� ��� �������� ��� ������� ���������� �� ���������� 
		// ��������������� ���������������� �������. ��������� ���� ������ 
		// ��������� imaxdubl>=100; ������� ������ ����� �� ����� � ��.
		// �� ��� ��� ������� ����� ����������� ������ ������.
		// ������� � ������ ������� ��� AMG �� ��������. AMG -
		// �������������� �������������� ������, ������� ������� � 
		// ��������� ������� ��������� ���������� �� ������������������ ��������� �����.
		// � ������ ������� �������� ������� ������������������� 
		// ������ ��������� ������������ ������ �������, ��� 
		// � ����� ���� � ���� ��� ���������� ������ ����������� 
		// �� �������� ������ ����. ������ ��� ����� ������������������ ������
		// ����� ������� �������������������. ����� �����: ���������� ���������
		// ��������� BiCGStab � ����������� ������������������� ������ ����������� 
		// � ������ ���� ����� ������ ����� �������������� ��������.
		if (imaxdubl>=100) {
			// �������� ��������� 100 ����� ������ ���� ��������� �� ��������������� ������������.

			iflag=0; // ����� ����������
			printf("calculation can not cope with the stiffness of the problem...\n");
            fprintf(fp_log,"calculation can not cope with the stiffness of the problem...\n");
			printf("Please, press any key to continue calculation...\n");
			fprintf(fp_log,"Please, press any key to continue calculation...\n");
			bexporttecplot=true; // �������� �� ���������� => ������� �������� ��� ������� � ��������� tecplot.
			getchar();
		}

		//getchar(); // debug 1 iteration LR1sk
	}

    // ������������ ������
	delete ri; delete roc; delete s; delete t;
	delete vi; delete pi; delete dax;
	delete y; delete z;

	#pragma omp parallel for shared(dX0, dx) private(i) schedule (guided)
	for (i=0; i<n; i++) dX0[i]=dx[i];

	delete dx; 


} // LR1sK_temp