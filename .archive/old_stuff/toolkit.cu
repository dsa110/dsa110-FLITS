// -*- c++ -*-
/*

Strategy is to operate on a single voltage file, and produce a heap of stuff. 
 - read in and send to GPU
 - simple promote
 - optionally calibrate voltages
 - correlate and write out
 - optionally remove delays from visibilities
 - optionally average visibilities in frequency
 - rotate visibilities to particular beam (later can be RA/DEC)
 - write out beamformed filterbank

*/

#include <iostream>
#include <algorithm>
using std::cout;
using std::cerr;
using std::endl;
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <string.h>
#include <unistd.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <time.h>
#include <syslog.h>
#include <pthread.h>

#include <mma.h>
#include <cuda.h>
#include "cuda_fp16.h"

#include <cuda_runtime_api.h>
using namespace nvcuda;

#define NANT 63
#define NCHAN 384
#define NT 30720
#define NBASE 2016
#define NPTR 8 // pols, small times, r/i
#define sep 1.0 // arcmin
#define AV 8
#define PI 3.141592653589793238
#define CVAC 299792458.0


// dedisperser
// dms are integer shifts
// run with NT*NANT blocks of NCHAN threads
__global__ void dedisperser(char *input, char *output, int *dms) {

  size_t bidx = blockIdx.x; 
  size_t ch = threadIdx.x;
  size_t iidx = bidx*NCHAN+ch;

  // time sample
  size_t tim = (size_t)(bidx / NANT);
  // antenna
  size_t ant = (size_t)(bidx % NANT);
  size_t oidx;

  // wrap
  if (tim < dms[ch]) {
    oidx = (NT-(dms[ch]-tim))*NANT*NCHAN + ant*NCHAN + ch;
    for (size_t i=0;i<4;i++) 
      output[4*oidx+i] = input[4*iidx+i];
  }

  // normal shift
  if (tim >= dms[ch]) {
    oidx = (tim-dms[ch])*NANT*NCHAN + ant*NCHAN + ch;
    for (size_t i=0;i<4;i++) 
      output[4*oidx+i] = input[4*iidx+i];
  }
 
}


// promoter to fp32
// run with NANT*NCHAN*NPTR/2/32 blocks of 32 threads
__global__ void promoter(char *input, float *output) {

  int bidx = blockIdx.x; // assume 16*48*NANT
  int tidx = threadIdx.x; // assume 32
  int iidx = bidx*32+tidx;
  
  output[2*iidx] = (float)((char)(((unsigned char)(input[iidx]) & (unsigned char)(15)) << 4) >> 4); //r
  output[2*iidx+1] = (float)((char)(((unsigned char)(input[iidx]) & (unsigned char)(240))) >> 4); //i

}

// correlator
// input is two packed time ints for all antennas
// also input antenna 1 and 2 indices for each baseline
// output is [2x time, baseline, freq, pols, r/i]
// run with NBASE*NCHAN/32 blocks of 32 threads
__global__ void correlator(float *input, float *output, int *a1, int *a2, float scfac, float *weights) {

  int bidx = blockIdx.x; // assume 16*48*NANT                                                              
  int tidx = threadIdx.x; // assume 32                                                                     
  int iidx = bidx*32+tidx;
  int basel = (int)(iidx/NCHAN); // baseline number
  int chgidx = (int)(bidx % (NCHAN/32)); // index of 32-channel group for this block
  int ch = (int)(iidx % NCHAN); // channel number
  
  // each block operates on 32 channels (one per thread)
  __shared__ float d1[32*NPTR];
  __shared__ float d2[32*NPTR];
  // start indices for each antenna from input
  int idx0_1 = a1[basel]*NCHAN*NPTR + chgidx*32*NPTR;
  int idx0_2 = a2[basel]*NCHAN*NPTR + chgidx*32*NPTR;

  // pull data into shared mem, for each antenna
  int ii = tidx*NPTR;
  for (int i=idx0_1+tidx*NPTR; i<idx0_1+(tidx+1)*NPTR; i++) {
    d1[ii] = input[i];
    ii++;
  }
  ii=tidx*NPTR;
  for (int i=idx0_2+tidx*NPTR; i<idx0_2+(tidx+1)*NPTR; i++) {
    d2[ii] = input[i];
    ii++;
  }


  // get weights for a1 and a2;
  float w_a1[4], w_a2[4];
  for (int i=0;i<4;i++) {
    w_a1[i] = weights[a1[basel]*192 + (int)(ch/8)*4 + i];
    w_a2[i] = weights[a2[basel]*192 + (int)(ch/8)*4 + i];
  }
  
  // now each thread can happily operate on a single channel
  // order is [time, pol, R/I]
  // make two separate arrays, each with [X*X / X*Y / Y*X / Y*Y, complexity]
  float output_tims[2][8], a1r, a1i, a2r, a2i;
  float w1r, w1i, w2r, w2i;
  // loop over times
  for (int ti=0;ti<2;ti++) {
    // loop over pols
    ii=0;
    for (int p1=0;p1<2;p1++) {
      for (int p2=0;p2<2;p2++) {
	
	a1r = d1[tidx*NPTR + ti*4 + p1*2];
	a1i = d1[tidx*NPTR + ti*4 + p1*2 + 1];
	a2r = d2[tidx*NPTR + ti*4 + p2*2];
	a2i = d2[tidx*NPTR + ti*4 + p2*2 + 1];

	w1r = a1r*w_a1[2*p1] - a1i*w_a1[2*p1+1];
	w2r = a2r*w_a2[2*p2] - a2i*w_a2[2*p2+1]; 
	w1i = a1r*w_a1[2*p1+1] + a1i*w_a1[2*p1];
	w2i = a2r*w_a2[2*p2+1] + a2i*w_a2[2*p2]; 
	/*w1r = a1r;
	w1i = a1i;
	w2r = a2r;
	w2i = a2i;*/
	
	output_tims[ti][2*ii] = w1r*w2r + w1i*w2i;
	output_tims[ti][2*ii+1] = w1r*w2i - w1i*w2r;

	ii++;
	
      }
    }
  }
  
  // write to output
  ii = basel*NCHAN*8 + ch*8;
  for (int i=0;i<8;i++) output[ii+i] += output_tims[0][i]*scfac;
  ii += NBASE*NCHAN*8;
  for (int i=0;i<8;i++) output[ii+i] += output_tims[1][i]*scfac;
  
  
}

// input has shape NBASE*NCHAN*8
// reduce to stokes I along NBASE axis using shared memory
// run with NCHAN blocks of 512 threads - will add 2016 baselines
__global__ void reduce_corrs(float *input, float *output, float scfac, int *a1, int *a2, int stokes, float *antpos, float minBase) {

  int bidx = blockIdx.x; // assume NCHAN
  int tidx = threadIdx.x; // assume 512                                                                  
  int iidx = bidx*512+tidx;

  volatile __shared__ float summer[512];

  // add into shared memory
  summer[tidx] = 0.;

  // stokes I
  if (stokes==0) {
    if (tidx<504) {
      if (a1[tidx]!=a2[tidx] && fabsf(antpos[a2[tidx]]-antpos[a1[tidx]])>minBase)
	summer[tidx] += input[tidx*NCHAN*8 + bidx*8] + input[tidx*NCHAN*8 + bidx*8 + 6];
      if (a1[tidx+504]!=a2[tidx+504] && fabsf(antpos[a2[tidx+504]]-antpos[a1[tidx+504]])>minBase)
	summer[tidx] += input[(tidx+1*504)*NCHAN*8 + bidx*8] + input[(tidx+1*504)*NCHAN*8 + bidx*8 + 6];
      if (a1[tidx+2*504]!=a2[tidx+2*504] && fabsf(antpos[a2[tidx+2*504]]-antpos[a1[tidx+2*504]])>minBase)
	summer[tidx] += input[(tidx+2*504)*NCHAN*8 + bidx*8] + input[(tidx+2*504)*NCHAN*8 + bidx*8 + 6];
      if (a1[tidx+3*504]!=a2[tidx+3*504] && fabsf(antpos[a2[tidx+3*504]]-antpos[a1[tidx+3*504]])>minBase)
	summer[tidx] += input[(tidx+3*504)*NCHAN*8 + bidx*8] + input[(tidx+3*504)*NCHAN*8 + bidx*8 + 6];
    }
  }
  // stokes Q
  if (stokes==1) {
    if (tidx<504) {
      if (a1[tidx]!=a2[tidx])
	summer[tidx] += input[tidx*NCHAN*8 + bidx*8] - input[tidx*NCHAN*8 + bidx*8 + 6];
      if (a1[tidx+504]!=a2[tidx+504])
	summer[tidx] += input[(tidx+1*504)*NCHAN*8 + bidx*8] - input[(tidx+1*504)*NCHAN*8 + bidx*8 + 6];
      if (a1[tidx+2*504]!=a2[tidx+2*504])
	summer[tidx] += input[(tidx+2*504)*NCHAN*8 + bidx*8] - input[(tidx+2*504)*NCHAN*8 + bidx*8 + 6];
      if (a1[tidx+3*504]!=a2[tidx+3*504])
	summer[tidx] += input[(tidx+3*504)*NCHAN*8 + bidx*8] - input[(tidx+3*504)*NCHAN*8 + bidx*8 + 6];
    }
  }
  // stokes U
  if (stokes==2) {
    if (tidx<504) {
      if (a1[tidx]!=a2[tidx])
	summer[tidx] += input[tidx*NCHAN*8 + bidx*8 + 2] + input[tidx*NCHAN*8 + bidx*8 + 4];
      if (a1[tidx+504]!=a2[tidx+504])
	summer[tidx] += input[(tidx+1*504)*NCHAN*8 + bidx*8 + 2] + input[(tidx+1*504)*NCHAN*8 + bidx*8 + 4];
      if (a1[tidx+2*504]!=a2[tidx+2*504])
	summer[tidx] += input[(tidx+2*504)*NCHAN*8 + bidx*8 + 2] + input[(tidx+2*504)*NCHAN*8 + bidx*8 + 4];
      if (a1[tidx+3*504]!=a2[tidx+3*504])
	summer[tidx] += input[(tidx+3*504)*NCHAN*8 + bidx*8 + 2] + input[(tidx+3*504)*NCHAN*8 + bidx*8 + 4];
    }
  }
  // stokes V
  if (stokes==3) {
    if (tidx<504) {
      if (a1[tidx]!=a2[tidx])
	summer[tidx] += input[tidx*NCHAN*8 + bidx*8 + 3] - input[tidx*NCHAN*8 + bidx*8 + 5];
      if (a1[tidx+504]!=a2[tidx+504])
	summer[tidx] += input[(tidx+1*504)*NCHAN*8 + bidx*8 + 3] - input[(tidx+1*504)*NCHAN*8 + bidx*8 + 5];
      if (a1[tidx+2*504]!=a2[tidx+2*504])
	summer[tidx] += input[(tidx+2*504)*NCHAN*8 + bidx*8 + 3] - input[(tidx+2*504)*NCHAN*8 + bidx*8 + 5];
      if (a1[tidx+3*504]!=a2[tidx+3*504])
	summer[tidx] += input[(tidx+3*504)*NCHAN*8 + bidx*8 + 3] - input[(tidx+3*504)*NCHAN*8 + bidx*8 + 5];
    }
  }

  // [X_i X_j*  X_i Y_j*  Y_i X_j* Y_i Y_j*]
  // Stokes I: 0.5*(Re(X_i X_j*) + Re(Y_i Y_j*))
  // Stokes Q: 0.5*(Re(X_i X_j*) - Re(Y_i Y_j*))
  // Stokes U: 0.5*(Re(X_i Y_j*) + Re(Y_i X_j*))
  // Stokes V: 0.5*(Im(X_i Y_j*) - Im(Y_i X_j*))
  
  __syncthreads();

  // now reduce in shared memory
  if (tidx<256) {
    summer[tidx] += summer[tidx+256];
    __syncthreads();
    summer[tidx] += summer[tidx+128];
    __syncthreads();
    summer[tidx] += summer[tidx+64];
    __syncthreads();
    summer[tidx] += summer[tidx+32];
    __syncthreads();
    summer[tidx] += summer[tidx+16];
    __syncthreads();
    summer[tidx] += summer[tidx+8];
    __syncthreads();
    summer[tidx] += summer[tidx+4];
    __syncthreads();
    summer[tidx] += summer[tidx+2];
    __syncthreads();
    summer[tidx] += summer[tidx+1];
  }

  __syncthreads();

  if (tidx==0) output[bidx] = (summer[0]*scfac);

}

// this kernel removes baseline delays by multiplying by exp(-2*pi*i*nu*tau)
// run with NBASE*NCHAN*4/32 blocks of 32 threads
// delays in ns
__global__ void delayer(float *input, float *freqs, float *delays) {

  int bidx = blockIdx.x; // assume 16*48*NANT
  int tidx = threadIdx.x; // assume 32
  int iidx = bidx*32+tidx;
  int bci = (int)(iidx/4);
  int basel = (int)(bci / NCHAN);
  int ch = (int)(bci % NCHAN);

  float vr, vi, arg=-2.*3.14159265359*freqs[ch]*delays[basel]*1e-9;
  vr = input[2*iidx]*cosf(arg) - input[2*iidx+1]*sinf(arg);
  __syncthreads();
  vi = input[2*iidx]*sinf(arg) + input[2*iidx+1]*cosf(arg);
  __syncthreads();

  input[2*iidx] = vr;
  input[2*iidx+1] = vi;
  
}

// kernel to enable frequency averaging of visibility output
// will only output XX and YY pols
// run with NBASE*NCHAN*4/AV/32 blocks of 32 threads
__global__ void fscrunch(float *input, float *output) {

  int bidx = blockIdx.x; 
  int tidx = threadIdx.x; // assume 32
  int iidx = bidx*32+tidx;
  int bcli = (int)(iidx/4);
  int poli = (int)(iidx % 4);
  int basel = (int)(bcli / (NCHAN/AV));
  int lch = (int)(bcli % (NCHAN/AV));

  int sumss[4];
  sumss[0] = 0;
  sumss[1] = 1;
  sumss[2] = 6;
  sumss[3] = 7;

  output[iidx] = 0.;
  for (int i=0;i<AV;i++) 
    output[iidx] += input[basel*NCHAN*8 + (AV*lch+i)*8 + sumss[poli]];
    
}

// really simple - adds the two times in correlator output
// run with NBASE*NCHAN*8/32 blocks of 32 threads
__global__ void adder(float *input, float *output) {

  int bidx = blockIdx.x; // assume 16*48*NANT
  int tidx = threadIdx.x; // assume 32
  int iidx = bidx*32+tidx;
  
  output[iidx] = input[iidx] + input[NBASE*NCHAN*8 + iidx];

}

// really simple - zeros correlator output
// run with 2*NBASE*NCHAN*8/32 blocks of 32 threads
__global__ void zeroer(float *input) {

  int bidx = blockIdx.x; // assume 16*48*NANT                                                              
  int tidx = threadIdx.x; // assume 32                                                                     
  int iidx = bidx*32+tidx;

  input[iidx] = 0.;

}


// CPU functions
int init_weights(char *wnam, float *antpos, float *weights, char *flagnam, int weight, int doflag);
// loads in weights
int init_weights(char * wnam, float *antpos, float *weights, char *flagnam, int weight, int doflag) {

  // assumes 64 antennas
  // antpos: takes only easting
  // weights: takes [ant, NW==48] 

  FILE *fin;
  FILE *fants;
  float wnorm;

  if (weight) {
    if (!(fin=fopen(wnam,"rb"))) {
      printf("Couldn't open weights file %s\n",wnam);
      return 1;
    }

    fread(antpos,64*sizeof(float),1,fin);
    fread(weights,64*48*2*2*sizeof(float),1,fin);

    for (int i=0;i<64*48*2;i++) {
      wnorm = sqrt(weights[2*i]*weights[2*i] + weights[2*i+1]*weights[2*i+1]);
      if (wnorm!=0.0) {
	weights[2*i] /= wnorm*wnorm;
	weights[2*i+1] /= wnorm*wnorm;
      }
    }           

    fclose(fin);
  }
  else {

    for (int i=0;i<64*48*2;i++) {
      weights[2*i] = 1.;
      weights[2*i+1] = 0.;
    }

    for (int i=0;i<64;i++) {
      antpos[i] = 0.;
    }

  }
 

  int ant;
  if (doflag) {
    if (!(fants=fopen(flagnam,"r"))) {
      printf("Couldn't open flag ants file %s\n",flagnam);
      return 1;
    }
    
    while (!feof(fants)) {
      fscanf(fants,"%d\n",&ant);
      for (int j=0;j<48*2*2;j++) {
	weights[ant*48*2*2+j] = 0.0;
      }
    }

    fclose(fants);
    
  }

  //for (int i=0;i<63;i++) 
  //  printf("%f\n",antpos[i]);
  
  printf("Loaded antenna positions and weights\n");
  return 0;

}

void calc_voltage_weights(float *antpos, float *weights, float *freqs, float *bfweights, int nBeamNum);
void calc_voltage_weights(float *antpos, float *weights, float *freqs, float *bfweights, int nBeamNum) {

  float theta, afac, twr, twi;
  theta = sep*(127.-(float)nBeamNum)*3.14159265358/10800.; // radians
  for(int nAnt=0;nAnt<64;nAnt++){
    for(int nChan=0;nChan<48;nChan++){
      for(int nPol=0;nPol<2;nPol++){
	afac = -2.*3.14159265358*freqs[nChan*8+4]*theta/CVAC; // factor for rotate
	twr = cos(afac*antpos[nAnt]);
	twi = sin(afac*antpos[nAnt]);

	bfweights[nAnt*(48*2*2)+nChan*2*2+nPol*2] = (twr*weights[(nAnt*(48*2)+nChan*2+nPol)*2] - twi*weights[(nAnt*(48*2)+nChan*2+nPol)*2+1]);
	bfweights[nAnt*(48*2*2)+nChan*2*2+nPol*2+1] = (twi*weights[(nAnt*(48*2)+nChan*2+nPol)*2] + twr*weights[(nAnt*(48*2)+nChan*2+nPol)*2+1]);
      }
    }
  }

}

// only does Stokes I for now
void sum_to_filterbank(float *corrout, unsigned char *filout) {

  float val;
  for (int iCh=0; iCh<NCHAN; iCh++) {
    val = 0.;
    for (int b=0; b<NBASE; b++) 
      val += (0.5*(corrout[b*NCHAN*8 + iCh*8] + corrout[b*NCHAN*8 + iCh*8 + 6]));
    filout[iCh]	= (unsigned char)(val);
  }

}

void usage()
{
  fprintf (stdout,
	   "toolkit [options]\n"
	   " -i input filename [no default]\n"
	   " -o output filename [no default - will not write if none given]\n"
	   " -t number of time integrations x 32.768us [default 8]\n"
	   " -w optional weights file\n"
	   " -f optional antenna flags file\n"
	   " -c set frequency of first channel in MHz [default 1530.0]\n"
	   " -b optionally set in 0-255 to rotate voltages to beam\n"
	   " -p coherent philterbank writing file [no default - will not write if none given]\n"
	   " -d file with NBASE delays to remove from baselines [optional - no default]\n"
	   " -a average visibilities by 8x in frequency\n"
	   " -m dedisperse [optional - no default]\n"
	   " -u unify all antennas (testing only)\n"
	   " -s output number of packets to be processed. 1 packet = 2 samples [default NT==30720]\n"
	   " -q offset from start in number of packets [default 0]\n"
	   " -g Stokes parameter to output to filterbank, from 0[I], 1[Q], 2[U], 3[V] [default 0]\n"
	   " -v minimum baseline length (E-W, in m) for input to beamformer [default 0]\n"
	   " -h print usage\n");
}


// MAIN

int main (int argc, char *argv[]) {

  // use cuda device 1
  printf("Using GPU 1\n");
  cudaSetDevice(1);

  // command line arguments
  int arg = 0;
  char * finnam;
  finnam=(char *)malloc(sizeof(char)*100);
  char * foutnam;
  foutnam=(char *)malloc(sizeof(char)*100);
  int writing=0;
  int tint = 8;
  char * wnam;
  wnam=(char *)malloc(sizeof(char)*100);
  char * flagnam;
  flagnam=(char *)malloc(sizeof(char)*100);
  int weight=0, doflag=0;
  int beamn = -1;
  float fch1 = 1530.;
  char * filnam;
  filnam = (char *)malloc(sizeof(char)*100);
  int philwriting = 0;
  char * delnam;
  delnam = (char *)malloc(sizeof(char)*100);
  int delaying = 0;
  int averaging = 0;
  int dedispersing = 0;
  float dm = 0.;
  int OUTNT=NT;
  int OFFT=0;
  int unify=0;
  int stokes=0;
  float minBase=-1.;

  while ((arg=getopt(argc,argv,"i:o:t:w:f:c:b:p:d:m:s:q:g:v:uah")) != -1)
    {
      switch (arg)
	{
	case 'i':
	  if (optarg)
	    {
	      strcpy(finnam,optarg);
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-i flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'o':
	  if (optarg)
	    {
	      strcpy(foutnam,optarg);
	      writing=1;
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-o flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'p':
	  if (optarg)
	    {
	      strcpy(filnam,optarg);
	      philwriting=1;
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-p flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'd':
	  if (optarg)
	    {
	      strcpy(delnam,optarg);
	      delaying=1;
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-d flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'v':
	  if (optarg)
	    {
	      minBase=atof(optarg);
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-v flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'g':
	  if (optarg)
	    {
	      stokes=atoi(optarg);
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-g flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'w':
	  if (optarg)
	    {
	      strcpy(wnam,optarg);
	      weight=1;
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-w flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 's':
	  if (optarg)
	    {
	      OUTNT = atoi(optarg);
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-s flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'm':
	  if (optarg)
	    {
	      dm = atof(optarg);
	      dedispersing=1;
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-m flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'q':
	  if (optarg)
	    {
	      OFFT = atoi(optarg);
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-q flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'f':
	  if (optarg)
	    {
	      strcpy(flagnam,optarg);
	      doflag=1;
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-f flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'c':
	  if (optarg)
	    {
	      fch1 = atof(optarg);
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-c flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 't':
	  if (optarg)
	    {
	      tint = atoi(optarg);
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-t flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'b':
	  if (optarg)
	    {
	      beamn = atoi(optarg);
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-b flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
 	case 'a':
	  averaging=1;
	  break;
 	case 'u':
	  unify=1;
	  break;
	case 'h':
	  usage();
	  return EXIT_SUCCESS;
	}
    }

  if (writing) printf("Reading from %s, writing to %s\n",finnam,foutnam);
  else printf("Reading from %s, no visibilities written\n",finnam);
  if (philwriting) {
    printf("Will write coherent filterbank to %s\n",filnam);
    if (stokes<0 || stokes>3) {
      printf("Cannot form Stokes parameter %d\n",stokes);
      return EXIT_FAILURE;
    }
    printf("Using Stokes parameter %d\n",stokes);
    printf("Minimum baseline (m): %g\n",minBase);
  }
  printf("Integrating by %d ints - check that this is power of 2\n",tint);
  printf("Assuming fch1 %f MHz\n",fch1);
  if (weight) printf("Will weight voltages using %s\n",wnam);
  if (doflag) printf("Will flag antennas using %s\n",flagnam);
  if (beamn>=0 && beamn <=255) printf("Will rotate voltages to beam %d\n",beamn);
  else
    printf("Not rotating voltages with beamn %d\n",beamn);
  if (averaging) printf("Will average visibilities by 8x in frequency\n");
  if (delaying) printf("Will apply baseline delays from %s\n",delnam);
  if (dedispersing) printf("Will dedisperse to DM %f, adding delay to 1530MHz\n",dm);

  // open input and output files
  FILE *fin, *fout, *flout;
  if (!(fin=fopen(finnam,"rb"))) 
    printf("could not open input file\n");
  if (writing) {
    if (!(fout=fopen(foutnam,"wb"))) 
      printf("could not open output file\n");
  }
  if (philwriting) {
    if (!(flout=fopen(filnam,"wb"))) 
      printf("could not open filterbank output file\n");
  }
  
  // read into memory and deal with dedispersion
  printf("initial memory allocation - please stay patient...\n");
  size_t asize = 2972712960;//NT*NANT*NCHAN*NPTR/((size_t)(2));
  size_t cpsize;
  char *indata = (char *)malloc(sizeof(char)*asize);
  char *d_alldata1, *d_alldata2;
  if (dedispersing)
    cudaMalloc((void **)&d_alldata1, asize*sizeof(char));
  cudaMalloc((void **)&d_alldata2, asize*sizeof(char));
  int *h_dms = (int *)malloc(sizeof(int)*NCHAN);
  int *d_dms;
  float myf;
  cudaMalloc((void **)&d_dms, sizeof(int)*NCHAN);
  if (dedispersing) {
    for (int i=0;i<NCHAN;i++) {
      myf = (fch1 - i*250./8192.)*1e-3;
      h_dms[i] = (int)(round(4.15*dm*(pow(myf,-2.)-pow(1.53,-2.))/(0.065536)));
      //printf("DM delays %f (MHz) %d (samples)\n",myf*1e3,h_dms[i]);
    }
    cudaMemcpy(d_dms, h_dms, NCHAN*sizeof(int), cudaMemcpyHostToDevice);
  }
  printf("Reading from file...\n");
  fread(indata, sizeof(char), asize, fin);  
  printf("Reading onto GPU...\n");
  if (!dedispersing)
    cudaMemcpy(d_alldata2, indata, asize*sizeof(char), cudaMemcpyHostToDevice);
  else {
    cudaMemcpy(d_alldata1, indata, asize*sizeof(char), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    printf("Dedispersing...\n");
    dedisperser<<<NT*NANT,NCHAN>>>(d_alldata1, d_alldata2, d_dms);
  }

  cudaDeviceSynchronize();
  free(h_dms);
  free(indata);
  if (dedispersing)
    cudaFree(d_alldata1);
  cudaFree(d_dms);
  
  // allocate all memory

  // CPU
  float *outdata = (float *)malloc(sizeof(float)*NBASE*NCHAN*8);
  float *filout = (float *)malloc(sizeof(float)*NCHAN);
  int *h_a1 = (int *)malloc(sizeof(int)*NBASE);
  int *h_a2 = (int *)malloc(sizeof(int)*NBASE);
  // GPU
  char *d_indata;
  float *d_promoted, *d_corrout, *d_finalout, *d_avout;
  int *d_a1, *d_a2;
  float *d_filout;
  cudaMalloc((void **)&d_indata, NANT*NCHAN*(NPTR/2)*sizeof(char));
  cudaMalloc((void **)&d_promoted, NANT*NCHAN*NPTR*sizeof(float));
  cudaMalloc((void **)&d_corrout, 2*NBASE*NCHAN*8*sizeof(float));
  cudaMalloc((void **)&d_finalout, NBASE*NCHAN*8*sizeof(float));
  cudaMalloc((void **)&d_avout, NBASE*(NCHAN/AV)*4*sizeof(float));
  cudaMalloc((void **)&d_a1, NBASE*sizeof(int));
  cudaMalloc((void **)&d_a2, NBASE*sizeof(int));
  cudaMalloc((void **)&d_filout, NCHAN*sizeof(float));

  // load in delays
  float * h_delays = (float *)malloc(sizeof(float)*NBASE);
  float * d_delays;
  cudaMalloc((void **)&d_delays, NBASE*sizeof(float));
  FILE *fdel;
  if (delaying) {
    if (!(fdel=fopen(delnam,"r"))) {
      printf("could not open delay file %s\n",delnam);
      return(1);
    }
    for (int i=0;i<NBASE;i++)
      fscanf(fdel,"%f\n",&h_delays[i]);
    fclose(fdel);
    cudaMemcpy(d_delays,h_delays,NBASE*sizeof(float),cudaMemcpyHostToDevice);
  }
  
  // load in weights and antpos
  float * antpos = (float *)malloc(sizeof(float)*64); // easting
  float * weights = (float *)malloc(sizeof(float)*64*48*2*2); // complex weights [ant, NW, pol, r/i]
  float * bfweights = (float *)malloc(sizeof(float)*64*48*2*2); // complex weights [ant, NW, pol, r/i]
  float * freqs = (float *)malloc(sizeof(float)*NCHAN); // freq
  float * d_freqs;
  cudaMalloc((void **)&d_freqs, NCHAN*sizeof(float));
  for (int i=0;i<NCHAN;i++) freqs[i] = (fch1 - i*250./8192.)*1e6;
  cudaMemcpy(d_freqs,freqs,NCHAN*sizeof(float),cudaMemcpyHostToDevice);
  init_weights(wnam,antpos,weights,flagnam,weight,doflag);
  if (beamn>=0 && beamn<=255)
    calc_voltage_weights(antpos,weights,freqs,bfweights,beamn);
  float *d_weights;
  cudaMalloc((void **)&d_weights, 64*48*2*2*sizeof(float));
  float *d_antpos;
  cudaMalloc((void **)&d_antpos, 64*sizeof(float));
  cudaMemcpy(d_antpos,antpos,64*sizeof(float),cudaMemcpyHostToDevice);
  if (beamn>=0 && beamn<=255)
    cudaMemcpy(d_weights,bfweights,64*48*2*2*sizeof(float),cudaMemcpyHostToDevice);
  else
    cudaMemcpy(d_weights,weights,64*48*2*2*sizeof(float),cudaMemcpyHostToDevice);
  
  // set up a1 and a2
  int ctr=0;
  for (int i=0;i<63;i++) {
    for (int j=i;j<63;j++) {
      h_a1[ctr] = i;
      h_a2[ctr] = j;
      ctr++;
    }
  }
  cudaMemcpy(d_a1,h_a1,NBASE*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_a2,h_a2,NBASE*sizeof(int),cudaMemcpyHostToDevice);

  // loop over input

  printf("starting loop\n");
  
  int timi=0;
  ctr = 0;
  for (int bigI=OFFT;bigI<OUTNT+OFFT;bigI++) {

    // read data, send to GPU, promote
    cpsize = bigI*NANT*NCHAN*NPTR/2;
    cudaMemcpy(d_indata, d_alldata2 + cpsize, (NANT*NCHAN*NPTR/2)*sizeof(char), cudaMemcpyDeviceToDevice);
    promoter<<<NANT*NCHAN*NPTR/2/32,32>>>(d_indata, d_promoted);    
    //promoter<<<NANT*NCHAN*NPTR/2/32,32>>>(d_alldata2 + cpsize, d_promoted);
    if (unify) {
      for (int i=1;i<NANT;i++)
	cudaMemcpy(d_promoted + i*NCHAN*NPTR, d_promoted, NCHAN*NPTR*sizeof(float), cudaMemcpyDeviceToDevice);
    }
    
    // deal with time integration
    if (timi==0) {
      zeroer<<<2*NBASE*NCHAN*8/32,32>>>(d_corrout);
      zeroer<<<NBASE*NCHAN*8/32,32>>>(d_finalout);
    }
    // correlate
    correlator<<<NBASE*NCHAN/32,32>>>(d_promoted,d_corrout,d_a1,d_a2,(1./(1.*tint)),d_weights);
    timi+=2;

    // deal with time integration
    if (timi>=tint) {

      // don't add up
      if (tint==1) {
		
	if (writing) {
	  if (delaying)
	    delayer<<<NBASE*NCHAN*4/32, 32>>>(d_corrout, d_freqs, d_delays);
	  if (averaging) {
	    fscrunch<<<NBASE*NCHAN*4/AV/32, 32>>>(d_corrout, d_avout);
	    cudaMemcpy(outdata, d_avout, NBASE*(NCHAN/AV)*4*sizeof(float), cudaMemcpyDeviceToHost);
	    fwrite(outdata,sizeof(float),NBASE*(NCHAN/AV)*4,fout);
	  }
	  else {
	    cudaMemcpy(outdata, d_corrout, NBASE*NCHAN*8*sizeof(float), cudaMemcpyDeviceToHost);
	    fwrite(outdata,sizeof(float),NBASE*NCHAN*8,fout);
	  }
	}
	if (philwriting) {
	  reduce_corrs<<<NCHAN,512>>>(d_corrout, d_filout, 0.25, d_a1, d_a2, stokes, d_antpos, minBase);
	  cudaMemcpy(filout, d_filout, NCHAN*sizeof(float), cudaMemcpyDeviceToHost);
	  fwrite(filout,sizeof(float),NCHAN,flout);
	}
		
	if (writing) {
	  if (delaying)
	    delayer<<<NBASE*NCHAN*4/32, 32>>>(d_corrout + NBASE*NCHAN*8, d_freqs, d_delays);
	  if (averaging) {
	    fscrunch<<<NBASE*NCHAN*4/AV/32, 32>>>(d_corrout + NBASE*NCHAN*8, d_avout);
	    cudaMemcpy(outdata, d_avout, NBASE*(NCHAN/AV)*4*sizeof(float), cudaMemcpyDeviceToHost);
	    fwrite(outdata,sizeof(float),NBASE*(NCHAN/AV)*4,fout);
	  }
	  else {
	    cudaMemcpy(outdata, d_corrout + NBASE*NCHAN*8, NBASE*NCHAN*8*sizeof(float), cudaMemcpyDeviceToHost);
	    fwrite(outdata,sizeof(float),NBASE*NCHAN*8,fout);
	  }
	}
	if (philwriting) {
	  reduce_corrs<<<NCHAN,512>>>(d_corrout + NBASE*NCHAN*8, d_filout, 0.25, d_a1, d_a2, stokes, d_antpos, minBase);
	  cudaMemcpy(filout, d_filout, NCHAN*sizeof(float), cudaMemcpyDeviceToHost);
	  fwrite(filout,sizeof(float),NCHAN,flout);
	}
	
      }

      // add up
      else {

	adder<<<NBASE*NCHAN*8/32,32>>>(d_corrout,d_finalout);	
	if (writing) {
	  if (delaying)
	    delayer<<<NBASE*NCHAN*4/32, 32>>>(d_finalout, d_freqs, d_delays);
	  if (averaging) {
	    fscrunch<<<NBASE*NCHAN*4/AV/32, 32>>>(d_finalout, d_avout);
	    cudaMemcpy(outdata, d_avout, NBASE*(NCHAN/AV)*4*sizeof(float), cudaMemcpyDeviceToHost);
	    fwrite(outdata,sizeof(float),NBASE*(NCHAN/AV)*4,fout);
	  }
	  else {
	    cudaMemcpy(outdata, d_finalout, NBASE*NCHAN*8*sizeof(float), cudaMemcpyDeviceToHost);
	    fwrite(outdata,sizeof(float),NBASE*NCHAN*8,fout);
	  }
	}
	if (philwriting) {
	  reduce_corrs<<<NCHAN,512>>>(d_finalout, d_filout, 4., d_a1, d_a2, stokes, d_antpos, minBase);
	  cudaMemcpy(filout, d_filout, NCHAN*sizeof(float), cudaMemcpyDeviceToHost);	  
	  fwrite(filout,sizeof(float),NCHAN,flout);
	}

      }

      ctr++;
      //printf("done with integration %d of %d\n",ctr,NT*2/tint);
      timi = 0;
    }
      

  }

  fclose(fin);
  if (writing) fclose(fout);
  if (philwriting) fclose(flout);

  cudaFree(d_alldata2);
  cudaFree(d_indata);
  cudaFree(d_corrout);
  cudaFree(d_finalout);
  cudaFree(d_promoted);
  cudaFree(d_a1);
  cudaFree(d_a2);
  cudaFree(d_weights);
  cudaFree(d_filout);
  cudaFree(d_avout);
  cudaFree(d_delays);
  cudaFree(d_freqs);
  cudaFree(d_antpos);
  free(filout);
  free(antpos);
  free(weights);
  free(freqs);
  free(wnam);
  free(flagnam);  
  free(outdata);
  free(h_a1);
  free(h_a2);
  free(finnam);
  free(foutnam);
  free(filnam);
  free(delnam);
  free(h_delays);

}
