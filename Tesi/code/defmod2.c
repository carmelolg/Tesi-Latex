/* ...	*/

//MODEL
CALModel2D* model = calCADef2D(ROWS, COLUMNS, CAL_CUSTOM_NEIGHBORHOOD_2D,
		CAL_SPACE_TOROIDAL, CAL_NO_OPT);

// Neighborhood definition
calAddNeighbor2D (model , 0, 0);
calAddNeighbor2D (model , - 1, 0);
calAddNeighbor2D (model , 0, - 1);
calAddNeighbor2D (model , 0, + 1);
calAddNeighbor2D (model , + 1, 0);

/* ...	*/
