 /*...*/

 //create a model 
 struct CALModel2D * model = calCADef2D(ROWS, COLS, CAL_MOORE_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);
 
 //add an integer substate to the model
 CALSubstate2Di substate_i = calAddSubstate2Di(model);

 //add a real substate to the model
 CALSubstate2Dr substate_r = calAddSubstate2Dr(model);
 
 //init substates
 calInitSubstate2Di(model,substate_i,1);
 calInitSubstate2Dr(model,substate_r,1.5);

 //create an integer single layer substate
 CALSubstate2Di single_layer_substate_i = calAddSingleLayerSubstate2Di(model);

 /*...*/
