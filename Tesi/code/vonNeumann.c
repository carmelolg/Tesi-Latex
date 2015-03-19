/* ... */

/*
       | 1 |  
    ---|---|---
     2 | 0 | 3
    ---|---|---
       | 4 |  
*/
struct CALModel2D * model = calCADef2D(ROWS, COLS, CAL_CUSTOM_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);

//neighborhood definition
calAddNeighbor2D(ca2D,   0,   0);
calAddNeighbor2D(ca2D, - 1,   0);
calAddNeighbor2D(ca2D,   0, - 1);
calAddNeighbor2D(ca2D,   0, + 1);
calAddNeighbor2D(ca2D, + 1,   0);

/* ... */
