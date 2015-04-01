/* ... */

//add transition function's elementary processes
calAddElementaryProcess2D(model, transition_function);

//Add init function
calRunAddInitFunc2D( simulation, init_function);
calRunAddStopConditionFunc2D( simulation, stop_condition_function);
calRunAddSteeringFunc2D( simulation, steering_function);

//Start simulation
calRun2D( simulation);

//saving configuration
calSaveSubstate2Db( model, substate, PATH_FINAL);

//finalizations
calRunFinalize2D( simulation);
calFinalize2D( model);

/* ... */
