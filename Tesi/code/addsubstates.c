int main()
{

	/* ...	*/

	// add substates
	substate1 = calAddSubstate2Db(model);
	substate2 = calAddSubstate2Di(model);
	substate3 = calAddSubstate2Dr(model);

	// load substate from file
	calLoadSubstate2Db(model, substate1, PATH_substate1);
	calLoadSubstate2Dr(model, substate3, PATH_substate3);

	/* ... */

}
