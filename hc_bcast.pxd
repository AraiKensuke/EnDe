#cdef void CIFatFxdMks_nogil(double *p_x, double* p_fxdMk, double* p_l0_i2pidcovs, double* p_us, double* p_iSgs, double* p_f, double *p_iq2, double* p_zs, double* p_qdr_mk, double* p_qdr_sp, long M, long Nx, long mdim, double dt) nogil
cdef void CIFatFxdMks_nogil(double* p_fxdMk, double* p_l0_i2pidcovs, double* p_us, double* p_iSgs, double* p_zs, double* p_qdr_mk, double* p_qdr_sp, long M, long Nx, long mdim, double dt) nogil

cdef void CIFatFxdMks_kde_nogil(double* p_fxdMk, double* p_l0_i2pidcovs, double* p_us, double iBm2, double* p_zs, double* p_qdr_mk, double* p_qdr_sp, double* p_i_spc_occ_dt, long M, long Nx, long mdim, double dt) nogil

cdef void CIFatFxdMks_kde_nogil_closeonly(double* p_fxdMk, double* p_l0_i2pidcovs, double* p_us, double iBm2, double* p_CIF, double* p_qdr_mk, double* p_qdr_sp, double* p_i_spc_occ_dt, long M, long Nx, long mdim, long* p_theseClose, long Nclose, double dt) nogil
