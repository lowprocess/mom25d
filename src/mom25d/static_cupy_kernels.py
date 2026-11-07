"""Fused float64 static pair integration, shared by CUDA and HIP through CuPy.

One thread owns one pair; near observation reductions retain their order.
There are no device-side dense matrices or pair-by-quadrature temporaries.
"""
SOURCE = r'''
extern "C" __global__ void static_pairs(
    const double* V, const double* L, const double* tau, const double* un,
    const double* area, const double* P3,
    const double* P0, const double* W0, int n0,
    const double* P1, const double* W1, int n1,
    const double* P2, const double* W2, int n2,
    const int* pp, const int* qq, const double* dd, const int* kk,
    int n, int scalar, double* out) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    int p = pp[i], q = qq[i], k = kk[i], width = scalar ? 1 : 6;
    double d = dd[i], ad = fabs(d), m[6] = {};
    if (k < 0) {
        double w = area[p] * area[q] / 9.;
        for (int ip=0; ip<3; ++ip) {
            double x=P3[6*(size_t)p+2*ip], y=P3[6*(size_t)p+2*ip+1];
            for (int iq=0; iq<3; ++iq) {
                double sx=P3[6*(size_t)q+2*iq], sy=P3[6*(size_t)q+2*iq+1];
                double dx=x-sx, dy=y-sy;
                double g=w/sqrt(dx*dx+dy*dy+d*d);
                m[0]+=g;
                if (!scalar) {
                    m[1]+=g*x; m[2]+=g*y; m[3]+=g*sx; m[4]+=g*sy; m[5]+=g*(x*sx+y*sy);
                }
            }
        }
    } else {
        const double* points = k==0 ? P0 : (k==1 ? P1 : P2);
        const double* weights = k==0 ? W0 : (k==1 ? W1 : W2);
        int nq = k==0 ? n0 : (k==1 ? n1 : n2);
        for (int j=0; j<nq; ++j) {
            double x=points[2*((size_t)p*nq+j)], y=points[2*((size_t)p*nq+j)+1];
            double inner=0., irx=0., iry=0.;
            for (int e=0; e<3; ++e) {
                size_t ve=6*(size_t)q+2*e;
                double vx=V[ve]-x, vy=V[ve+1]-y;
                double sm=vx*tau[ve]+vy*tau[ve+1], sp=sm+L[3*(size_t)q+e];
                double t0=vx*un[ve]+vy*un[ve+1];
                double r02=t0*t0+d*d, r0=sqrt(r02);
                double rm=sqrt(r02+sm*sm), rp=sqrt(r02+sp*sp);
                double lg=r0>1e-14*L[3*(size_t)q+e] ? asinh(sp/r0)-asinh(sm/r0) : 0.;
                double term=t0*lg;
                if (ad>0.) term-=ad*(atan2(t0*sp,r02+ad*rp)-atan2(t0*sm,r02+ad*rm));
                inner+=term;
                if (!scalar) {
                    double edge=.5*(r02*lg+sp*rp-sm*rm);
                    irx+=edge*un[ve]; iry+=edge*un[ve+1];
                }
            }
            double w=area[p]*weights[j];
            m[0]+=w*inner;
            if (!scalar) {
                double sx=irx+x*inner, sy=iry+y*inner;
                m[1]+=w*x*inner; m[2]+=w*y*inner;
                m[3]+=w*sx; m[4]+=w*sy; m[5]+=w*(x*sx+y*sy);
            }
        }
    }
    for (int j=0; j<width; ++j) out[(size_t)i*width+j]=m[j];
}
'''
