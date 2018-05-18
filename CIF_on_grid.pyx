cdef void CIF_on_grid_KDE(int mdim, double *p_O, char *p_O01, long* p_pos_hstgrm_t0t1, p_us, p_dm, covs, mrngs, g_Ms) with nogil:
    cdef int tt, i0, i1, i2, i3, ii0, ii1, ii2, ii3, u0, u1, u2, u3, w0, w1, w2, w3

    if mdim > 1:
        #  since for kde we're make a list of closest spikes to this one
        Nclose = 0
        for coi in xrange(M):
            coiK = coi*mdim

            if (((p_us[coiK] - p_us[cK])*(p_us[coiK] - p_us[cK])) + \
                ((p_us[coiK+1] - p_us[cK+1])*(p_us[coiK+1] - p_us[cK+1])) + \
                ((p_us[coiK+2] - p_us[cK+2])*(p_us[coiK+2] - p_us[cK+2])) + \
                ((p_us[coiK+3] - p_us[cK+3])*(p_us[coiK+3] - p_us[cK+3])) < close):
                p_theseClose[Nclose] = coi
                Nclose += 1

        printf("doing cluster %d     close %ld\n", c, Nclose)

    if mdim == 2:
        u0 = <int>((p_us[cK]   - LLcrnr0) / p_dm[0])
        u1 = <int>((p_us[cK+1] - LLcrnr1) / p_dm[1])

        idl_grd_sz0 = p_dm[0]
        idl_grd_sz1 = p_dm[1]

        w0 = <int>((sqrt(covs[c, 0, 0]) / idl_grd_sz0)*sds_to_use)
        w1 = <int>((sqrt(covs[c, 1, 1]) / idl_grd_sz1)*sds_to_use)

        #  sample near peak densely, no approx
        #  away from peak, sample more sparsely
        for i0 from u0 - w0 <= i0 < u0 + w0 + 1:
            u_g0 = p_mrngs[i0]  #  mrngs[0, i0]  mrngs[0, 0:g_Ms[0]]
            for 0 <= ic < Nclose:
                c = p_theseClose[ic]
                tmp = 0
                cmdim  = c*mdim
                p_qdr_mk[c, 0] = (p_fxdMk[0]-p_us[cmdim]) * (p_fxdMk[0]-p_us[cmdim])
            for i1 from u1 - w1 <= i1 < u1 + w1 + 1:

                ii = i0*g_M1+ i1
                if (not ((i0 > g_Ms[0]) or (i0 < 0) or \
                         (i1 > g_Ms[1]) or (i1 < 0))) and \
                    (p_O01[ii] == 0):
                    u_g1 = p_mrngs[g_M + i1]
                    for 0 <= ic < Nclose:
                        c = p_theseClose[ic]
                        tmp = 0
                        cmdim  = c*mdim
                        p_qdr_mk_comps[c, 1] = (u_g1-p_us[cmdim+1]) * (u_g1-p_us[cmdim+1])
                        p_qdr_mk[c] = 0
                        for j in xrange(mdim):
                            p_qdr_mk[c] += p_qdr_mk_comps[c, j]
                        p_qdr_mk[c] *= iBm2

                    for 0 <= ix < Nx:  #  the mark contribution constant, modulating it by spatial contribution
                        tmp = 0
                        for 0 <= ic < Nclose:
                            c = p_theseClose[ic]
                            arg = p_qdr_sp[c*Nx + ix] +p_qdr_mk[c]
                            tmp += p_l0dt_i2pidcovs[c]*exp(-0.5*arg)
                        p_CIF[ix] = tmp * p_i_spc_occ_dt[ix]


                    #if p_O01[ii] == 0:
                    p_O01[ii] = 1

                    icnt += 1
                    #  mrngs   # mdim x g_M

                    for nn in xrange(ooNx):
                        p_O[ii] += p_pos_hstgrm_t0t1[nn]*p_CIF_at_grid_mks[nn]
                    p_O[ii] *= ddt
