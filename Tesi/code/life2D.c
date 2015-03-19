#include <cal2D.h>
#include <cal2DRun.h>
#include <cal2DIO.h>

#define ROWS 300
#define COLS 300
#define STEPS 1000
#define SAVE_PATH life_result

//life substate
struct CALSubstate2Di * lifeSubstate;

//life transition function
void life_transition_function(struct CALModel2D* life, int i, int j)
{
 int sum = 0, n;
 for (n=1; n<life->sizeof_X; n++)
  sum += calGetX2Di(life, lifeSubstate, i, j, n);

 if ((sum == 3) || (sum == 2 && calGet2Di(life, lifeSubstate, i, j) == 1))
  calSet2Di(life, lifeSubstate, i, j, 1);
 else
  calSet2Di(life, lifeSubstate, i, j, 0);
}

//initialization function
void init(struct CALModel2D* life){
 calInitSubstate2Di(life,lifeSubstate,0);
 calInit2Di(life, lifeSubstate, 0, 2, 1);
 calInit2Di(life, lifeSubstate, 1, 0, 1);
 calInit2Di(life, lifeSubstate, 1, 2, 1);
 calInit2Di(life, lifeSubstate, 2, 1, 1);
 calInit2Di(life, lifeSubstate, 2, 2, 1);
}

int main() {

 //life model definition
 struct CALModel2D * model = calCADef2D(ROWS, COLS, CAL_MOORE_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);
 lifeSubstate = calAddSubstate2Di(model);
 calAddElementaryProcess2D(model,life_transition_function);

 //life execution definition
 struct CALRun2D * run = calRunDef2D(model, 1, STEPS, CAL_UPDATE_IMPLICIT);
 calRunAddInitFunc2D(run,init);
 calRun2D(run);
 
 //save computation result on a file
 calSaveSubstate2Di(model, lifeSubstate, SAVE_PATH);
 return 0;
}
