struct CALRun2D
{
	//!< Pointer to the cellular automaton structure.
	struct CALModel2D* ca2D;

	//!< Current simulation step.
	int step;

	//!< Initial simulation step.
	int initial_step;

	//!< Final simulation step; if 0 the simulation becomes a loop.
	int final_step;

	//!< Callbacks substates' update mode; it can be CAL_UPDATE_EXPLICIT or CAL_UPDATE_IMPLICIT.
	enum CALUpdateMode UPDATE_MODE;

	//!< Simulation's initialization callback function.
	void (*init)(struct CALModel2D*);

	//!< CA's globalTransition callback function. If defined, it is executed instead of cal2D.c::calGlobalTransitionFunction2D.
	void (*globalTransition)(struct CALModel2D*);

	//!< Simulation's steering callback function.
	void (*steering)(struct CALModel2D*);

	//!< Simulation's stopCondition callback function.
	CALbyte (*stopCondition)(struct CALModel2D*);

	//!< Simulation's finalize callback function.
	void (*finalize)(struct CALModel2D*);
};
