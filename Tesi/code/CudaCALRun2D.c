struct CudaCALRun2D
{
	//!< Pointer to the cellular automaton structure.
	struct CudaCALModel2D* ca2D;

	//!< Pointer to the cellular automaton structure on device.
	struct CudaCALModel2D* device_ca2D;

	//!< Pointer to the cellular automaton structure for data passing.
	struct CudaCALModel2D* h_device_ca2D;

	//!< Stream compaction data structure
	unsigned int * device_array_of_index_dim;

	//!< Current simulation step.
	int step;

	//!< Initial simulation step.
	int initial_step;

	//!< Final simulation step; if 0 the simulation becomes a loop.
	int final_step;

	//!< Callbacks substates' update mode; it can be CAL_UPDATE_EXPLICIT or CAL_UPDATE_IMPLICIT.
	enum CALUpdateMode UPDATE_MODE;

	//!< Simulation's initialization callback function.
	void (*init)(struct CudaCALModel2D*);

	//!< CA's globalTransition callback function. If defined, it is executed instead of cal2D.c::calGlobalTransitionFunction2D.
	void (*globalTransition)(struct CudaCALModel2D*);

	//!< Simulation's steering callback function.
	void (*steering)(struct CudaCALModel2D*);

	//!< Simulation's stopCondition callback function.
	void (*stopCondition)(struct CudaCALModel2D*);

	//!< Simulation's finalize callback function.
	void (*finalize)(struct CudaCALModel2D*);
};
