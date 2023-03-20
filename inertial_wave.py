  
from mpi4py import MPI
comm = MPI.COMM_WORLD
proc_id = comm.Get_rank()
n_procs = comm.Get_size()


import numpy as np
from scipy import integrate
from scipy.fftpack import fft
from numpy.lib.scimath import sqrt
import time
import cmath 

def find_nearest_idx(array, value):
       array = np.asarray(array)
       return (np.abs(array - value)).argmin()


#code_begin_time = start_time = time.time()
if proc_id==0:
    code_begin_time = start_time = time.time()


delta = 10.0; 
k0 = sqrt(3)/(1.0*delta); 
kx0 = k0/sqrt(3);
ky0 = k0/sqrt(3);
kz0 = k0/sqrt(3);
nu=1e-6;
Omega = 150;
Omega_t=15;
t =  Omega_t/(1.0*Omega); 

x1 = np.linspace(0,10,21);
z1 = np.linspace(0,30,151);

x = np.ravel(np.meshgrid(x1,z1, indexing='ij')[0]);
z = np.ravel(np.meshgrid(x1,z1, indexing='ij')[1]);
y=0;

kx_div = 151;
ky_div = 151;
kz_div = 701;

kx_max = 1;
ky_max = 1;
kz_max = 1;

kx_min = -1;
ky_min = -1;
kz_min = -1;


kx_inc = (kx_max-kx_min)/(1.0*kx_div) 
ky_inc = (ky_max-ky_min)/(1.0*ky_div) 
kz_inc = (kz_max-kz_min)/(1.0*kz_div) 



work_size = (kz_div) // n_procs
extra_work = (kz_div) % n_procs
my_work = work_size + (1 if proc_id<extra_work else 0)
l_start = work_size * proc_id + (proc_id if proc_id<extra_work else extra_work)
l_end = l_start + my_work

uzhat_f=0;
uz_f=0;
uzf=0;
uz_fr=0;
for l in range(l_start, l_end):
          
          if (l == 0 or l == kz_div):
              p = 1
          elif (l % 2 != 0):
              p = 4
          else:
              p = 2
  
          kz = kz_min + l * kz_inc
           
          for i in range(int(ky_div)):
              
              if (i == 0 or i == ky_div):
                  q = 1
              elif (i % 2 != 0):
                  q = 4
              else:
                  q = 2
      
              ky = ky_min + i * ky_inc
  
              for j in range(int(kx_div)):
              
                  if (j == 0 or j == kx_div):
                      r = 1
                  elif (j % 2 != 0):
                      r = 4
                  else:
                      r = 2
      
                  kx = kx_min + j * kx_inc

                  k=sqrt(kx**2+ky**2+kz**2);
                  
                  
                  wc= 2*Omega*kz/(1.0*k);
                  l1=wc;
                  l2=-wc;
                  
                  ux0 = 1j*(delta**5)*kz*np.exp(-(1.0/4)*(delta**2)*(k**2))/(4*sqrt(2));
                  uy0 =  0.0;
                  uz0 = -1j*(delta**5)*kx*np.exp(-(1.0/4)*(delta**2)*(k**2))/(4*sqrt(2));
                  
                  Az= 2*1j*kz*Omega*(kx*uy0-ky*ux0)/(1.0*(k**2)*(l1-l2))-l2*uz0/(1.0*(l1-l2));
                  Bz=2*1j*kz*Omega*(ky*ux0-kx*uy0)/(1.0*(k**2)*(l1-l2))+l1*uz0/(1.0*(l1-l2));

                  uzhat_f=Az*np.exp(1j*l1*t)+Bz*np.exp(1j*l2*t);

                  uz_f +=  uzhat_f*np.exp(1j*delta*(kz*z+kx*x+ky*y))*p*q*r;

uzf = comm.reduce(uz_f, op=MPI.SUM, root=0);
if proc_id == 0:
    uz_fr = uzf*kx_inc*ky_inc*kz_inc/(27.0);
    uz_fr  = uz_fr.reshape(len(x1),len(z1));
    np.savetxt('uzf_inertial_wt15.m',np.real(uz_fr), header='ufz=[', footer='];',comments=' ');
    print('total time elapsed: {:>5.2f} seconds'.format(time.time() - code_begin_time))


