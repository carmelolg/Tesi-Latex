int main()
{

	/* ...	*/

	//add substates
	calCudaAddSubstate2Dr(model, NUMBER_OF_SUBSTATES_REAL);
	calCudaAddSubstate2Db(model, NUMBER_OF_SUBSTATES_BYTE);

	//load configuration
	calCudaLoadSubstate2Dr(model, SUBSTATE1_PATH, SUBSTATE1_INDEX);
	calCudaLoadSubstate2Dr(model, SUBSTATE2_PATH, SUBSTATE2_INDEX);
	calCudaLoadSubstate2Db(model, SUBSTATE3_PATH, SUBSTATE3_INDEX);

	/* ... */

}

