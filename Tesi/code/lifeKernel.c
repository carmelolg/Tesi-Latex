#include <cal2DActive.h>

__kernel void transitionFunction(MODEL_DEFINITION2D) {

	//initialize work items
	initThreads2D();

	//get work item id on the first dimension
	int i = getX();
	//get work item id on the second dimension
	int j = getY();

	//game of life transition function
	int sum = 0, n;
	for (n = 1; n < get_neighborhoods_size(); n++)
		sum += calGetX2Di(MODEL2D, i, j, n, 0);
	if ((sum == 3) || (sum == 2 && calGet2Di(MODEL2D, i, j, 0) == 1))
		calSet2Di(MODEL2D, 1, i, j, 0);
	else
		calSet2Di(MODEL2D, 0, i, j, 0);
}