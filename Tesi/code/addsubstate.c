CALSubstate2Db* gol_substate;

int main()
{

	/* ...	*/

	//Model
	CALModel2D* GameOfLife = calCADef2D(ROWS, COLUMNS, CAL_MOORE_NEIGHBORHOOD_2D, CAL_SPACE_FLAT, CAL_NO_OPT);

	// add substates
	gol_substate = calAddSubstate2Db(GameOfLife);

	// load substate from file
	calLoadSubstate2Db(GameOfLife, gol_substate, PATH);

	/* ... */

	return 0;
}
