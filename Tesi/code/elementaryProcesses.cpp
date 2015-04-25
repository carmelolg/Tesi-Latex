//first elementary process
void updateVentsEmission(struct CALModel2D * model, int i, int j) {
 double emitted_lava = 0;
 for (unsigned int k = 0; k < sciara->vent.size(); k++) {
  int xVent = sciara->vent[k].x();
  int yVent = sciara->vent[k].y();
  if (i == yVent && j == xVent) {
   emitted_lava = sciara->vent[k].thickness(sciara->elapsed_time, sciara->Pclock, sciara->emission_time, sciara->Pac);
   if (emitted_lava > 0) {
    calSet2Dr(model, sciara->substates->Slt, yVent, xVent, calGet2Dr(sciara->model, sciara->substates->Slt, yVent, xVent) + emitted_lava);
    calSet2Dr(model, sciara->substates->St, yVent, xVent, sciara->PTvent);
   }
  }
 }
}

//second elementary process
void empiricalFlows(struct CALModel2D * model, int i, int j) {

 if (calGet2Dr(model, sciara->substates->Slt, i, j) > 0) {
  CALreal f[MOORE_NEIGHBORS];
  outflowsMin(model, i, j, f);

  for (int k = 1; k < MOORE_NEIGHBORS; k++)
   if (f[k] > 0) {
    calSet2Dr(model, sciara->substates->f[k - 1], i, j, f[k]);
    if (active)
     calAddActiveCellX2D(model, i, j, k);
   }
 }
}

//third elementary process
void width_update(struct CALModel2D* model, int i, int j) {
 CALint outFlowsIndexes[NUMBER_OF_OUTFLOWS] = { 3, 2, 1, 0, 6, 7, 4, 5 };
 CALint n;
 CALreal initial_h = calGet2Dr(model, sciara->substates->Slt, i, j);
 CALreal initial_t = calGet2Dr(model, sciara->substates->St, i, j);
 CALreal residualTemperature = initial_h * initial_t;
 CALreal residualLava = initial_h;
 CALreal h_next = initial_h;
 CALreal t_next;

 CALreal ht = 0;
 CALreal inSum = 0;
 CALreal outSum = 0;

 for (n = 1; n < model->sizeof_X; n++) {
  CALreal inFlow = calGetX2Dr(model, sciara->substates->f[outFlowsIndexes[n - 1]], i, j, n);
  CALreal outFlow = calGet2Dr(model, sciara->substates->f[n - 1], i, j);
  CALreal neigh_t = calGetX2Dr(model, sciara->substates->St, i, j, n);
  ht += inFlow * neigh_t;
  inSum += inFlow;
  outSum += outFlow;
 }
 h_next += inSum - outSum;
 calSet2Dr(model, sciara->substates->Slt, i, j, h_next);
 if (inSum > 0 || outSum > 0) {
  residualLava -= outSum;
  t_next = (residualLava * initial_t + ht) / (residualLava + inSum);
  calSet2Dr(model, sciara->substates->St, i, j, t_next);
 }
}

//fourth elementary process
void updateTemperature(struct CALModel2D* model, int i, int j) {
 CALreal nT, h, T, aus;
 CALreal sh = calGet2Dr(model, sciara->substates->Slt, i, j);
 CALreal st = calGet2Dr(model, sciara->substates->St, i, j);
 CALreal sz = calGet2Dr(model, sciara->substates->Sz, i, j);

 if (sh > 0 && !calGet2Db(model, sciara->substates->Mb, i, j)) {
  h = sh;
  T = st;
  if (h != 0) {
   nT = T;
   aus = 1.0 + (3 * pow(nT, 3.0) * sciara->Pepsilon * sciara->Psigma * sciara->Pclock * sciara->Pcool) / (sciara->Prho * sciara->Pcv * h * sciara->Pac);
   st = nT / pow(aus, 1.0 / 3.0);
   calSet2Dr(model, sciara->substates->St, i, j, st);

  }
  if (st <= sciara->PTsol && sh > 0) {
   calSet2Dr(model, sciara->substates->Sz, i, j, sz + sh);
   calSetCurrent2Dr(model, sciara->substates->Msl, i, j, calGet2Dr(model, sciara->substates->Msl, i, j) + sh);
   calSet2Dr(model, sciara->substates->Slt, i, j, 0);
   calSet2Dr(model, sciara->substates->St, i, j, sciara->PTsol);
  } else
   calSet2Dr(model, sciara->substates->Sz, i, j, sz);
 }
}