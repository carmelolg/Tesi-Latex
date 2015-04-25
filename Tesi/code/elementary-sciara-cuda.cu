//Elementary processes and support functions

__global__
	void updateVentsEmission(struct CudaCALModel2D * model) {

		CALint offset = calCudaGetIndex(model), i = calCudaGetIndexRow(model, offset), j= calCudaGetIndexColumn(model, offset);
		CALreal emitted_lava;

		if(calCudaGet2Di(model,offset,VENTS) == 1)
		{
			emitted_lava = 1.806732;
			if (emitted_lava > 0) {
				calCudaSet2Dr(model, offset, calCudaGet2Dr(model, offset, THICKNESS) + emitted_lava, THICKNESS);
				calCudaSet2Dr(model, offset, PTvent, TEMPERATURE);
			}
		}

		if(calCudaGet2Di(model,offset,VENTS) == 2)
		{
			emitted_lava = 1.806732;
			if (emitted_lava > 0) {
				calCudaSet2Dr(model, offset, calCudaGet2Dr(model, offset, THICKNESS) + emitted_lava, THICKNESS);
				calCudaSet2Dr(model, offset, PTvent, TEMPERATURE);
			}
		}
}

__device__ 
	double powerLaw(double k1, double k2, double T) {
		double log_value = k1 + k2 * T;
		return pow(10.0, log_value);
}

__device__
	void outflowsMin(struct CudaCALModel2D * model, int offset, CALreal *f) {

		bool n_eliminated[MOORE_NEIGHBORS];
		double z[MOORE_NEIGHBORS];
		double h[MOORE_NEIGHBORS];
		double H[MOORE_NEIGHBORS];
		double theta[MOORE_NEIGHBORS];
		double w[MOORE_NEIGHBORS]; 		//Distances between central and adjecent cells
		double Pr_[MOORE_NEIGHBORS];	//Relaiation rate arraj
		bool loop;
		int counter;
		double avg, _w, _Pr, hc, sum, sumZ;

		CALreal t = calCudaGet2Dr(model, offset, TEMPERATURE);

		_w = cell_size;
		_Pr = powerLaw(a, b, t);
		hc = powerLaw(c, d, t);

		for (int k = 0; k < MOORE_NEIGHBORS; k++) {

			h[k] = calCudaGetX2Dr(model, offset, k, THICKNESS);
			H[k] = f[k] = theta[k] = 0;
			w[k] = _w;
			Pr_[k] = _Pr;
			CALreal sz = calCudaGetX2Dr(model, offset, k, ALTITUDE);
			CALreal sz0 = calCudaGet2Dr(model, offset, ALTITUDE);
			if (k < VON_NEUMANN_NEIGHBORS)
				z[k] = calCudaGetX2Dr(model, offset, k, ALTITUDE);
			else
				z[k] = sz0 - (sz0 - sz) / rad2;
		}

		H[0] = z[0];
		n_eliminated[0] = true;

		for (int k = 1; k < MOORE_NEIGHBORS; k++)
			if (z[0] + h[0] > z[k] + h[k]) {
				H[k] = z[k] + h[k];
				theta[k] = atan(((z[0] + h[0]) - (z[k] + h[k])) / w[k]);

				n_eliminated[k] = true;
			} else
				n_eliminated[k] = false;

			do {
				loop = false;
				avg = h[0];
				counter = 0;
				for (int k = 0; k < MOORE_NEIGHBORS; k++)
					if (n_eliminated[k]) {
						avg += H[k];
						counter++;
					}
					if (counter != 0)
						avg = avg / double(counter);
					for (int k = 0; k < MOORE_NEIGHBORS; k++)
						if (n_eliminated[k] && avg <= H[k]) {
							n_eliminated[k] = false;
							loop = true;
						}
			} while (loop);

			for (int k = 1; k < MOORE_NEIGHBORS; k++) {
				if (n_eliminated[k] && h[0] > hc * cos(theta[k])) {
					f[k] = Pr_[k] * (avg - H[k]);
				}
			}
}

__global__
	void empiricalFlows(struct CudaCALModel2D * model) {

		CALint offset = calCudaGetIndex(model);

		if (calCudaGet2Dr(model, offset, THICKNESS) > 0) {
			CALreal f[MOORE_NEIGHBORS];
			outflowsMin(model, offset, f);

			if (f[1] > 0) {
				calCudaSet2Dr(model, offset, f[1],  FLOWN);
#ifdef ACTIVE_CELLS
				calCudaAddActiveCellX2D(model, offset, 1);
#endif
			}

			if (f[2] > 0) {
				calCudaSet2Dr(model, offset, f[2],  FLOWO);
#ifdef ACTIVE_CELLS
				calCudaAddActiveCellX2D(model, offset, 2);
#endif
			}

			if (f[3] > 0) {
				calCudaSet2Dr(model, offset, f[3],  FLOWE);
#ifdef ACTIVE_CELLS
				calCudaAddActiveCellX2D(model, offset, 3);
#endif
			}

			if (f[4] > 0) {
				calCudaSet2Dr(model, offset, f[4],  FLOWS);
#ifdef ACTIVE_CELLS
				calCudaAddActiveCellX2D(model, offset, 4);
#endif
			}

			if (f[5] > 0) {
				calCudaSet2Dr(model, offset, f[5],  FLOWNO);
#ifdef ACTIVE_CELLS
				calCudaAddActiveCellX2D(model, offset, 5);
#endif
			}

			if (f[6] > 0) {
				calCudaSet2Dr(model, offset, f[6],  FLOWSO);
#ifdef ACTIVE_CELLS
				calCudaAddActiveCellX2D(model, offset, 6);
#endif
			}

			if (f[7] > 0) {
				calCudaSet2Dr(model, offset, f[7],  FLOWSE);
#ifdef ACTIVE_CELLS
				calCudaAddActiveCellX2D(model, offset, 7);
#endif
			}

			if (f[8] > 0) {
				calCudaSet2Dr(model, offset, f[8],  FLOWNE);
#ifdef ACTIVE_CELLS
				calCudaAddActiveCellX2D(model, offset, 8);
#endif
			}

		}
}

__global__
	void width_update(struct CudaCALModel2D* model) {
		CALint outFlowsIndexes[NUMBER_OF_OUTFLOWS] = { 3, 2, 1, 0, 6, 7, 4, 5 };
		CALint n;
		CALint offset = calCudaGetIndex(model);
		CALreal initial_h = calCudaGet2Dr(model, offset, THICKNESS);
		CALreal initial_t = calCudaGet2Dr(model, offset, TEMPERATURE);
		CALreal residualTemperature = initial_h * initial_t;
		CALreal residualLava = initial_h;
		CALreal h_next = initial_h;
		CALreal t_next;

		CALreal ht = 0;
		CALreal inSum = 0;
		CALreal outSum = 0;

		CALreal inFlow = 0, outFlow = 0, neigh_t = 0;

		// n = 1
		inFlow = calCudaGetX2Dr(model, offset, 1, FLOWS);
		outFlow = calCudaGet2Dr(model, offset, FLOWN);
		neigh_t = calCudaGetX2Dr(model, offset, 1, TEMPERATURE);
		ht += inFlow * neigh_t;
		inSum += inFlow;
		outSum += outFlow;

		// n = 2
		inFlow = calCudaGetX2Dr(model, offset, 2, FLOWE);
		outFlow = calCudaGet2Dr(model, offset, FLOWO);
		neigh_t = calCudaGetX2Dr(model, offset, 2, TEMPERATURE);
		ht += inFlow * neigh_t;
		inSum += inFlow;
		outSum += outFlow;

		// n = 3
		inFlow = calCudaGetX2Dr(model, offset, 3, FLOWO);
		outFlow = calCudaGet2Dr(model, offset, FLOWE);
		neigh_t = calCudaGetX2Dr(model, offset, 3, TEMPERATURE);
		ht += inFlow * neigh_t;
		inSum += inFlow;
		outSum += outFlow;

		// n = 4
		inFlow = calCudaGetX2Dr(model, offset, 4, FLOWN);
		outFlow = calCudaGet2Dr(model, offset, FLOWS);
		neigh_t = calCudaGetX2Dr(model, offset, 4, TEMPERATURE);
		ht += inFlow * neigh_t;
		inSum += inFlow;
		outSum += outFlow;

		// n = 5
		inFlow = calCudaGetX2Dr(model, offset, 5, FLOWSE);
		outFlow = calCudaGet2Dr(model, offset, FLOWNO);
		neigh_t = calCudaGetX2Dr(model, offset, 5, TEMPERATURE);
		ht += inFlow * neigh_t;
		inSum += inFlow;
		outSum += outFlow;

		// n = 6
		inFlow = calCudaGetX2Dr(model, offset, 6, FLOWNE);
		outFlow = calCudaGet2Dr(model, offset, FLOWSO);
		neigh_t = calCudaGetX2Dr(model, offset, 6, TEMPERATURE);
		ht += inFlow * neigh_t;
		inSum += inFlow;
		outSum += outFlow;

		// n = 7
		inFlow = calCudaGetX2Dr(model, offset, 7, FLOWNO);
		outFlow = calCudaGet2Dr(model, offset, FLOWSE);
		neigh_t = calCudaGetX2Dr(model, offset, 7, TEMPERATURE);
		ht += inFlow * neigh_t;
		inSum += inFlow;
		outSum += outFlow;

		// n = 8
		inFlow = calCudaGetX2Dr(model, offset, 8, FLOWSO);
		outFlow = calCudaGet2Dr(model, offset, FLOWNE);
		neigh_t = calCudaGetX2Dr(model, offset, 8, TEMPERATURE);
		ht += inFlow * neigh_t;
		inSum += inFlow;
		outSum += outFlow;

		h_next += inSum - outSum;
		calCudaSet2Dr(model, offset, h_next, THICKNESS);
		if (inSum > 0 || outSum > 0) {
			residualLava -= outSum;
			t_next = (residualLava * initial_t + ht) / (residualLava + inSum);
			calCudaSet2Dr(model, offset, t_next, TEMPERATURE);
		}
}

__global__
	void updateTemperature(struct CudaCALModel2D* model) {
		CALreal nT, h, T, aus;
		CALint offset = calCudaGetIndex(model);
		CALreal sh = calCudaGet2Dr(model, offset, THICKNESS);
		CALreal st = calCudaGet2Dr(model, offset, TEMPERATURE);
		CALreal sz = calCudaGet2Dr(model, offset, ALTITUDE);

		if (sh > 0 && !calCudaGet2Db(model, offset, TOPOGRAPHY_BOUND)) {
			h = sh;
			T = st;
			if (h != 0) {
				nT = T;

				aus = 1.0 + (3 * pow(nT, 3.0) * Pepsilon * Psigma * Pclock * Pcool) / (Prho * Pcv * h * cell_size * cell_size);
				st = nT / pow(aus, 1.0 / 3.0);
				calCudaSet2Dr(model, offset, st, TEMPERATURE);

			}

			//solidification
			if (st <= PTsol && sh > 0) {
				calCudaSet2Dr(model, offset, sz + sh, ALTITUDE);
				calCudaSetCurrent2Dr(model, offset, calCudaGet2Dr(model, offset, SOLIDIFIED) + sh, SOLIDIFIED);
				calCudaSet2Dr(model, offset, 0, THICKNESS);
				calCudaSet2Dr(model, offset, PTsol, TEMPERATURE);

			} else
				calCudaSet2Dr(model, offset, sz, ALTITUDE);
		}
}

__global__
	void removeActiveCells(struct CudaCALModel2D* model) {
		CALint offset = calCudaGetIndex(model);
		CALreal st = calCudaGet2Dr(model, offset, TEMPERATURE);
		if (st <= PTsol && !calCudaGet2Db(model, offset, TOPOGRAPHY_BOUND))
			calCudaRemoveActiveCell2D(model, offset);
}

__global__
	void steering(struct CudaCALModel2D* model) {

		CALint offset = calCudaGetIndex(model);

		calCudaInit2Dr(model, offset, 0, FLOWN);
		calCudaInit2Dr(model, offset, 0, FLOWO);
		calCudaInit2Dr(model, offset, 0, FLOWE);
		calCudaInit2Dr(model, offset, 0, FLOWS);
		calCudaInit2Dr(model, offset, 0, FLOWNO);
		calCudaInit2Dr(model, offset, 0, FLOWSO);
		calCudaInit2Dr(model, offset, 0, FLOWSE);
		calCudaInit2Dr(model, offset, 0, FLOWNE);

		if (calCudaGet2Db(model, offset, TOPOGRAPHY_BOUND) == CAL_TRUE) {
			calCudaSet2Dr(model, offset, 0, THICKNESS);
			calCudaSet2Dr(model, offset, 0, TEMPERATURE);
		}
}

__device__
	void evaluatePowerLawParams(CALreal value_sol, CALreal value_vent, CALreal &k1, CALreal &k2) {
		k2 = (log10(value_vent) - log10(value_sol)) / (PTvent - PTsol);
		k1 = log10(value_sol) - k2 * (PTsol);
}

__device__
	void MakeBorder(CudaCALModel2D* model) {

		CALint i, j, offset = calCudaGetIndex(model);

		i = calCudaGetIndexRow(model, offset);
		j = calCudaGetIndexColumn(model, offset);

		//prima riga
		if(i == 0){
			if (calCudaGet2Dr(model, offset, ALTITUDE) >= 0) {
				calCudaSetCurrent2Db(model, offset, CAL_TRUE, TOPOGRAPHY_BOUND);
#ifdef ACTIVE_CELLS
				calCudaAddActiveCell2D(model, offset);
#endif
			}
		}

		//ultima riga
		if(i == rows - 1){

			if (calCudaGet2Dr(model, offset, ALTITUDE) >= 0) {
				calCudaSetCurrent2Db(model, offset, CAL_TRUE, TOPOGRAPHY_BOUND);
#ifdef ACTIVE_CELLS
				calCudaAddActiveCell2D(model, offset);
#endif
			}


		}


		//prima colonna
		if( j == 0 ){
			if (calCudaGet2Dr(model, offset, ALTITUDE) >= 0) {
				calCudaSetCurrent2Db(model, offset, CAL_TRUE, TOPOGRAPHY_BOUND);
#ifdef ACTIVE_CELLS
				calCudaAddActiveCell2D(model, offset);
#endif
			}
		}


		//ultima colonna
		if(j == cols - 1){
			if (calCudaGet2Dr(model, offset, ALTITUDE) >= 0) {
				calCudaSetCurrent2Db(model, offset, CAL_TRUE, TOPOGRAPHY_BOUND);
#ifdef ACTIVE_CELLS
				calCudaAddActiveCell2D(model, offset);
#endif
			}

		}

		if( i > 0 && j > 0 && i < rows - 1 && j < cols - 1){
			if (calCudaGet2Dr(model, offset, ALTITUDE) >= 0) {
				for (int k = 1; k < model->sizeof_X; k++)
					if (calCudaGetX2Dr(model, offset, k, ALTITUDE) < 0) {
						calCudaSetCurrent2Db(model, offset, CAL_TRUE, TOPOGRAPHY_BOUND);
#ifdef ACTIVE_CELLS
						calCudaAddActiveCell2D(model, offset);
#endif
						break;
					}
			}
		}

}

__global__
	void simulationInitialize(struct CudaCALModel2D* model) {

		//dichiarazioni
		unsigned int maximum_number_of_emissions = 0;

		CALint offset = calCudaGetSimpleOffset();

		calCudaInit2Dr(model, offset, 0.000000, THICKNESS);

		//TODO single layer initialization
		calCudaInit2Db(model, offset, CAL_FALSE, TOPOGRAPHY_BOUND);
		
		calCudaInit2Dr(model, offset, 0, FLOWN);
		calCudaInit2Dr(model, offset, 0, FLOWO);
		calCudaInit2Dr(model, offset, 0, FLOWE);
		calCudaInit2Dr(model, offset, 0, FLOWS);
		calCudaInit2Dr(model, offset, 0, FLOWNO);
		calCudaInit2Dr(model, offset, 0, FLOWSO);
		calCudaInit2Dr(model, offset, 0, FLOWSE);
		calCudaInit2Dr(model, offset, 0, FLOWNE);

		//definisce il bordo della morfologia
		MakeBorder(model);

		//calcolo a b (parametri viscosità) c d (parametri resistenza al taglio)
		evaluatePowerLawParams(Pr_Tsol, Pr_Tvent, a, b);
		evaluatePowerLawParams(Phc_Tsol, Phc_Tvent, c, d);
		
#ifdef ACTIVE_CELLS

		if(calCudaGet2Di(model,offset,VENTS) == 1){
			calCudaAddActiveCell2D(model, offset);
		}
		if(calCudaGet2Di(model,offset,VENTS) == 2){
			calCudaAddActiveCell2D(model, offset);
		}

#endif


}