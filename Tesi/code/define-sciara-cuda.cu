//-----------------------------------------------------------
//   			THE Sciara-FV2 CELLULAR AUTOMATON			-
//-----------------------------------------------------------
#define maximum_steps 				0		       
#define stopping_threshold			0.00
#define refreshing_step            	0
#define thickness_visual_threshold 	0.00
#define Pclock                     	60.00
#define PTsol                      	1143.00
#define PTvent                     	1360.00
#define Pr_Tsol                 	0.0750
#define Pr_Tvent	              	0.90
#define Phc_Tsol	             	60.00
#define Phc_Tvent	            	0.4
#define Pcool                      	9.0
#define Prho                       	2600.00
#define Pepsilon                  	0.90
#define Psigma                     	5.68E-08
#define Pcv                        	1150
#define algorithm                  	MIN       
#define layers						40
#define rows						378
#define cols						517
#define cell_size					10.000000
#define nodata_value				0
#define num_emission_rate			15
#define num_total_emission_rates	2
#define xllcorner					499547.500000
#define yllcorner					4174982.500000
#define rad2						1.41421356237

__device__ CALreal a,b,c,d;

//Number of steps
#define STEPS 3200

//Files path of event.
#define DEM_PATH "data/2006/2006_000000000000_Morphology.txt"
#define VENTS_PATH "data/2006/2006_000000000000_Vents.txt"
#define EMISSION_RATE_PATH "data/2006/2006_000000000000_EmissionRate.txt"
#define TEMPERATURE_PATH "data/2006/2006_000000000000_Temperature.txt"
#define THICKNESS_PATH "data/2006/2006_000000000000_Thickness.txt"
#define SOLIDIFIED_LAVA_THICKNESS_PATH "data/2006/2006_000000000000_SolidifiedLavaThickness.txt"
#define REAL_EVENT_THICKNESS_PATH "data/2006/2006_000000000000_RealEvent.txt"

#define O_DEM_PATH "data/2006_SAVE/2006_000000000000_Morphology.txt"
#define O_VENTS_PATH "data/2006_SAVE/2006_000000000000_Vents.txt"
#define O_EMISSION_RATE_PATH "data/2006_SAVE/2006_000000000000_EmissionRate.txt"
#define O_TEMPERATURE_PATH "data/2006_SAVE/2006_000000000000_Temperature.txt"
#define O_THICKNESS_PATH "data/2006_SAVE/2006_000000000000_Thickness.txt"
#define O_SOLIDIFIED_LAVA_THICKNESS_PATH "data/2006_SAVE/2006_000000000000_SolidifiedLavaThickness.txt"

//Use active_cells optimization, comment for not use.
#define ACTIVE_CELLS

//Define values of outflows, dimension of neighbors and substates
#define NUMBER_OF_OUTFLOWS 8
#define MOORE_NEIGHBORS 9
#define VON_NEUMANN_NEIGHBORS 5

#define NUMBER_OF_SUBSTATES_REAL 13
#define NUMBER_OF_SUBSTATES_INT 1
#define NUMBER_OF_SUBSTATES_BYTE 1

//Enumerative for increase readability of code.  
enum SUBSTATES_NAMES_REAL{
	ALTITUDE=0,THICKNESS,TEMPERATURE,PRE_EVENT_TOPOGRAPHY, SOLIDIFIED, FLOWN,FLOWO,FLOWE,FLOWS, FLOWNO, FLOWSO, FLOWSE,FLOWNE
};
enum SUBSTATES_NAMES_INT{
	VENTS=0,
};
enum SUBSTATES_NAMES_BYTE{
	TOPOGRAPHY_BOUND=0,
};

//Grids and blocks configuration
CALint N = 21;
CALint M = 47;
dim3 block(N,M);
dim3 grid(cols/block.x, rows/block.y);
