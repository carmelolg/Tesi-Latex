#include "kernel.h"

//first elementary process kernel
__kernel void updateVentsEmission(MODEL_DEFINITION2D, __global Vent* vents, __global CALreal * emissionRates, __global CALreal * elapsed_time, int sizeVents, int sizeEmissionRate, Parameters parameters) {

 initThreads2D();

 CALreal emitted_lava = 0;
 int i = getX();
 int j = getY();

 for (unsigned int k = 0; k < sizeVents; k++) {
  int iVent = vents[k].y;
  int jVent = vents[k].x;
  if (i == iVent && j == jVent) {
   emitted_lava = thickness(*elapsed_time, parameters.Pclock, parameters.emission_time, parameters.Pac, emissionRates + k * sizeEmissionRate, sizeEmissionRate);
   if (emitted_lava > 0) {
    CALreal slt = calGet2Dr(MODEL2D, iVent, jVent, SLT) + emitted_lava;
    calSet2Dr(MODEL2D, slt, iVent, jVent, SLT);
    calSet2Dr(MODEL2D, parameters.PTvent, iVent, jVent, ST);
   }
  }
 }
}

//second elementary process kernel
__kernel void empiricalFlows(MODEL_DEFINITION2D, Parameters parameters) {

 initThreads2D();

 int i = getX();
 int j = getY();

 if (calGet2Dr(MODEL2D, i, j, SLT) > 0) {
  CALreal f[MOORE_NEIGHBORS];
  outflowsMin(MODEL2D, i, j, f, parameters);

  for (int k = 1; k < MOORE_NEIGHBORS; k++)
   if (f[k] > 0)
    calSet2Dr(MODEL2D, f[k], i, j, F(k - 1));
 }
}

//third elementary process kernel
__kernel void width_update(MODEL_DEFINITION2D) {

 initThreads2D();

 int i = getX();
 int j = getY();

 CALint outFlowsIndexes[NUMBER_OF_OUTFLOWS] = { 3, 2, 1, 0, 6, 7, 4, 5 };
 CALint n;
 CALreal initial_h = calGet2Dr(MODEL2D, i, j, SLT);
 CALreal initial_t = calGet2Dr(MODEL2D, i, j, ST);
 CALreal residualTemperature = initial_h * initial_t;
 CALreal residualLava = initial_h;
 CALreal h_next = initial_h;
 CALreal t_next;

 CALreal ht = 0;
 CALreal inSum = 0;
 CALreal outSum = 0;

 for (n = 1; n < get_neighborhoods_size(); n++) {
  CALreal inFlow = calGetX2Dr(MODEL2D, i, j, n, F(outFlowsIndexes[n - 1]));
  CALreal outFlow = calGet2Dr(MODEL2D, i, j, F(n - 1));
  CALreal neigh_t = calGetX2Dr(MODEL2D, i, j, n, ST);
  ht = fma(inFlow, neigh_t, ht);
  inSum += inFlow;
  outSum += outFlow;
 }
 h_next += inSum - outSum;
 calSet2Dr(MODEL2D, h_next, i, j, SLT);
 if (inSum > 0 || outSum > 0) {
  residualLava -= outSum;
  t_next = fma(residualLava, initial_t, ht) / (residualLava + inSum);
  calSet2Dr(MODEL2D, t_next, i, j, ST);
 }
}

//fourth elementary process kernel
__kernel void updateTemperature(MODEL_DEFINITION2D, __global CALbyte * Mb, __global CALreal * Msl, Parameters parameters) {

 initThreads2D();

 int i = getX();
 int j = getY();
 CALreal aus = 0;
 CALreal sh = calGet2Dr(MODEL2D, i, j, SLT);
 CALreal st = calGet2Dr(MODEL2D, i, j, ST);
 CALreal sz = calGet2Dr(MODEL2D, i, j, SZ);

 if (sh > 0 && calGetBufferElement2D(Mb, get_columns(), i, j) == CAL_FALSE) {
  aus = 1.0 + (3 * pow(st, 3.0) * parameters.Pepsilon * parameters.Psigma * parameters.Pclock * parameters.Pcool) / (parameters.Prho * parameters.Pcv * sh * parameters.Pac);
  st = st / pow(aus, (1.0 / 3.0));
  calSet2Dr(MODEL2D, st, i, j, ST);

  if (st <= parameters.PTsol) {
   calSet2Dr(MODEL2D, sz + sh, i, j, SZ);
   calSetBufferElement2D(Msl, get_columns(), i, j, calGetBufferElement2D(Msl, get_columns(), i, j) + sh);
   calSet2Dr(MODEL2D, 0, i, j, SLT);
   calSet2Dr(MODEL2D, parameters.PTsol, i, j, ST);
  }
 }
}