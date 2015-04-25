void initSciara(char * demPath, int steps) {

 sciara = new Sciara;
 
 /*some initializations*/

 //model definition
 sciara->model = calCADef2D(sciara->rows, sciara->cols, CAL_MOORE_NEIGHBORHOOD_2D, CAL_SPACE_FLAT, CAL_NO_OPT);

 //substates definitions
 sciara->substates = new SciaraSubstates();

 sciara->substates->Sz = calAddSubstate2Dr(sciara->model);
 sciara->substates->Slt = calAddSubstate2Dr(sciara->model);
 sciara->substates->St = calAddSubstate2Dr(sciara->model);

 sciara->substates->Mb = calAddSingleLayerSubstate2Db(sciara->model);
 sciara->substates->Mv = calAddSingleLayerSubstate2Di(sciara->model);
 sciara->substates->Msl = calAddSingleLayerSubstate2Dr(sciara->model);
 sciara->substates->Sz_t0 = calAddSingleLayerSubstate2Dr(sciara->model);

 //substates initializations
 calInitSubstate2Dr(sciara->model, sciara->substates->Sz, 0);
 calInitSubstate2Dr(sciara->model, sciara->substates->Slt, 0);
 calInitSubstate2Dr(sciara->model, sciara->substates->St, 0);

 for (int i = 0; i < sciara->rows * sciara->cols; ++i) {
  sciara->substates->Mb->current[i] = CAL_FALSE;
  sciara->substates->Mv->current[i] = 0;
  sciara->substates->Msl->current[i] = 0;
  sciara->substates->Sz_t0->current[i] = 0;
 }

 for (int i = 0; i < NUMBER_OF_OUTFLOWS; ++i) {
  sciara->substates->f[i] = calAddSubstate2Dr(sciara->model);
  calInitSubstate2Dr(sciara->model, sciara->substates->f[i], 0);
 }

 //elementary processes definition
 calAddElementaryProcess2D(sciara->model, updateVentsEmission);
 calAddElementaryProcess2D(sciara->model, empiricalFlows);
 calAddElementaryProcess2D(sciara->model, width_update);
 calAddElementaryProcess2D(sciara->model, updateTemperature);

 //run definition
 sciara->run = calRunDef2D(sciara->model, 1, steps, CAL_UPDATE_IMPLICIT);

 calRunAddInitFunc2D(sciara->run, simulationInitialize);
 calRunAddSteeringFunc2D(sciara->run, steering);
 calRunAddStopConditionFunc2D(sciara->run, stopCondition);

}