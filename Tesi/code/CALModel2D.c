struct CALModel2D {

	//!< Number of rows of the 2D cellular space.
	int rows;

	//!< Number of columns of the 2D cellular space.
	int columns;

	//!< Type of cellular space: toroidal or non-toroidal.
	enum CALSpaceBoundaryCondition T;

	//!< Type of optimization used. It can be CAL_NO_OPT or CAL_OPT_ACTIVE_CELLS.
	enum CALOptimization OPTIMIZATION;

	//!< Computational Active cells object. if A.actives==NULL no optimization is applied.
	struct CALActiveCells2D A;

	//!< Array of cell coordinates defining the cellular automaton neighbourhood relation.
	struct CALCell2D* X;

	//!< Number of cells belonging to the neighbourhood. Note that predefined neighbourhoods include the central cell.
	int sizeof_X;

	//!< Neighbourhood relation's id.
	enum CALNeighborhood2D X_id;

	//!< Array of pointers to 2D substates of type byte
	struct CALSubstate2Db** pQb_array;

	//!< Array of pointers to 2D substates of type int
	struct CALSubstate2Di** pQi_array;

	//!< Array of pointers to 2D substates of type real (floating point)
	struct CALSubstate2Dr** pQr_array;

	//!< Number of substates of type byte.
	int sizeof_pQb_array;

	//!< Number of substates of type int.
	int sizeof_pQi_array;

	//!< Number of substates of type real (floating point).
	int sizeof_pQr_array;

	//!< Array of function pointers to the transition function's elementary processes callback functions. Note that a substates' update must be performed after each elementary process has been applied to each cell of the cellular space (see calGlobalTransitionFunction2D).
	void (**elementary_processes)(struct CALModel2D* ca2D, int i, int j);

	//!< Number of function pointers to the transition functions's elementary processes callbacks.
	int num_of_elementary_processes;
};
