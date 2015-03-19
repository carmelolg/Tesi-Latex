CALbyte calRunCAStep2D(struct CALRun2D* simulation)
{
 //execute user transition function if defined
 if (simulation->globalTransition)
 {
	simulation->globalTransition(simulation->ca2D);
	if (simulation->UPDATE_MODE == CAL_UPDATE_IMPLICIT)
		calUpdate2D(simulation->ca2D);
 }
 else
 //execute all elementary processes defined by the user
 calGlobalTransitionFunction2D(simulation->ca2D);
 
 //execute steering function if defined
 if (simulation->steering)
 {
	simulation->steering(simulation->ca2D);
	if (simulation->UPDATE_MODE == CAL_UPDATE_IMPLICIT)
		calUpdate2D(simulation->ca2D);
 }

 //check stop condition if defined
 if (simulation->stopCondition)
	if (simulation->stopCondition(simulation->ca2D)) 
		return CAL_FALSE;
 
 return CAL_TRUE;
}