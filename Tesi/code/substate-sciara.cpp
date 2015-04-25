 for (n = 1; n < model->sizeof_X; n++) {
  CALreal inFlow = calGetX2Dr(model, sciara->substates->f[outFlowsIndexes[n - 1]], i, j, n);
  CALreal outFlow = calGet2Dr(model, sciara->substates->f[n - 1], i, j);
  CALreal neigh_t = calGetX2Dr(model, sciara->substates->St, i, j, n);
  ht += inFlow * neigh_t;
  inSum += inFlow;
  outSum += outFlow;
 }
