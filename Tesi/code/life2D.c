#include <cal2D.h>
#include <cal2DRun.h>
#include <cal2DIO.h>

#define ROWS 100
#define COLS 100
#define STEPS 1000
#define PATH gol_result

//gol substate
struct CALSubstate2Di * gol_substate;

//gol transition function
void gol_transition_function(struct CALModel2D* gol, int i, int j)
{
 int sum = 0, n;
 for (n=1; n<gol->sizeof_X; n++)
  sum += calGetX2Di(gol, gol_substate, i, j, n);

 if ((sum == 3) || (sum == 2 && calGet2Di(gol, gol_substate, i, j) == 1))
  calSet2Di(gol, gol_substate, i, j, 1);
 else
  calSet2Di(gol, gol_substate, i, j, 0);
}

//initialization function
void init(struct CALModel2D* gol){
 calInitSubstate2Di(gol,gol_substate,0);
 calInit2Di(gol, gol_substate, 0, 2, 1);
 calInit2Di(gol, gol_substate, 1, 0, 1);
 calInit2Di(gol, gol_substate, 1, 2, 1);
 calInit2Di(gol, gol_substate, 2, 1, 1);
 calInit2Di(gol, gol_substate, 2, 2, 1);
}

int main() {

 //gol model definition
 struct CALModel2D * model = calCADef2D(ROWS, COLS, CAL_MOORE_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);
 gol_substate = calAddSubstate2Di(model);
 calAddElementaryProcess2D(model,gol_transition_function);

 //gol execution definition
 struct CALRun2D * run = calRunDef2D(model, 1, STEPS, CAL_UPDATE_IMPLICIT);
 calRunAddInitFunc2D(run,init);
 calRun2D(run);
 
 //save computation result on a file
 calSaveSubstate2Di(model, gol_substate, PATH);
 return 0;
}
