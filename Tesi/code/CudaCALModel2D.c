struct CudaCALModel2D {

	//!< Number of rows of the 2D cellular space.
	int rows;

	//!< Number of columns of the 2D cellular space.
	int columns;

	//!< Type of cellular space: toroidal or non-toroidal.
	enum CALSpaceBoundaryCondition T;

	//!< Type of optimization used. It can be CAL_NO_OPT or CAL_OPT_ACTIVE_CELLS.
	enum CALOptimization OPTIMIZATION;

	//!< Array of flags having the substates' dimension: flag is CAL_TRUE if the corresponding cell is active, CAL_FALSE otherwise.
	CALbyte* activecell_flags;

	//!< Number of CAL_TRUE flags.
	int activecell_size_next;

	//!< i-Array of computational active cells.
	int *i_activecell;

	//!< j-Array of computational active cells.
	int *j_activecell;

	//!< Number of active cells in the current step.
	int activecell_size_current;


	//!< Array of cell coordinates defining the cellular automaton neighbourhood relation.
	int *i;
	int *j;

	//!< Number of cells belonging to the neighbourhood. Note that predefined neighbourhoods include the central cell.
	int sizeof_X;

	//!< Neighbourhood relation's id.
	enum CALNeighborhood2D X_id;

	//!< Current linearised matrix of the substate, used for reading purposes.
	CALbyte* pQb_array_current;
	//!< Next linearised matrix of the substate, used for writing purposes.
	CALbyte* pQb_array_next;

	//!< Current linearised matrix of the substate, used for reading purposes.
	CALint* pQi_array_current;
	//!< Next linearised matrix of the substate, used for writing purposes.
	CALint* pQi_array_next;

	//!< Current linearised matrix of the substate, used for reading purposes.
	CALreal* pQr_array_current;
	//!< Next linearised matrix of the substate, used for writing purposes.
	CALreal* pQr_array_next;

	//!< Number of substates of type byte.
	int sizeof_pQb_array;
	//!< Number of substates of type int.
	int sizeof_pQi_array;
	//!< Number of substates of type real (floating point).
	int sizeof_pQr_array;

	//!< Array of function pointers to the transition function's elementary processes callback functions. Note that a substates' update must be performed after each elementary process has been applied to each cell of the cellular space (see calGlobalTransitionFunction2D).
	void (**elementary_processes)(struct CudaCALModel2D* ca2D);
	//!< Number of function pointers to the transition functions's elementary processes callbacks.
	int num_of_elementary_processes;
};
