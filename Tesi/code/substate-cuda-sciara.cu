enum SUBSTATES_NAMES_REAL{
	ALTITUDE=0,THICKNESS,TEMPERATURE,PRE_EVENT_TOPOGRAPHY, SOLIDIFIED, FLOWN,FLOWO,FLOWE,FLOWS, FLOWNO, FLOWSO, FLOWSE,FLOWNE
};
enum SUBSTATES_NAMES_INT{
	VENTS=0,
};
enum SUBSTATES_NAMES_BYTE{
	TOPOGRAPHY_BOUND=0,
};

int main(){
	/* ... */
	
	calCudaGet2Dr(model, offset, THICKNESS);
	calCudaSet2Dr(model, offset, PTvent, TEMPERATURE);
	calCudaSet2Dr(model, offset, calCudaGet2Dr(model, offset, THICKNESS) + emitted_lava, THICKNESS);
	
	/* ... */
}