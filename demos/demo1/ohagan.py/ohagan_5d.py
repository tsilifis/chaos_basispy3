import numpy as np 
import scipy.stats as st 
import matplotlib.pyplot as plt 
import chaos_basispy as cb

def ohagan(xx):
	assert xx.shape == (6, 1)
	a1 = np.array([0.0118, 0.0456, 0.2297, 0.0393, 0.1177, 0.3865])
	a2 = np.array([0.4341, 0.0887, 0.0512, 0.3233, 0.1489, 1.0360])
	a3 = np.array([0.1044, 0.2057, 0.0774, 0.2730, 0.1253, 0.7526])


	M = np.array([[-0.022482886,  -0.18501666,  0.13418263,   0.36867264,   0.17172785,   0.13651143],
       			  [ 0.25659630,  0.053792287,  0.25800381,   0.23795905,  -0.59125756, -0.081627077],
     			  [-0.055999811,   0.19542252, 0.095529005,  -0.28626530,  -0.14441303,   0.22369356 ],
       			  [ 0.66448103,   0.43069872,  0.29924645,  -0.16202441,  -0.31479544,  -0.39026802],
      			  [-0.12127800,   0.12463327,  0.10656519,  0.046562296,  -0.21678617,   0.19492172],
      			  [-0.28376230,  -0.32820154, -0.10496068,  -0.22073452,  -0.13708154,  -0.14426375]])

	term1 = np.dot(a1, xx)
	term2 = np.dot(a2, np.sin(xx))
	term3 = np.dot(a3, np.cos(xx))
	term4 = np.dot(xx.T, np.dot(M, xx))
	return term1 + term2 + term3 + term4


xx = st.norm.rvs(size = (100000, 6))
f = np.zeros(xx.shape[0])
for i in range(xx.shape[0]):
	f[i] = ohagan(xx[i,:].reshape(6,1))

[xx_quad, ww] = cb.QuadratureRule().get_rule(6, 5)
print(xx_quad.shape)
f_quad = np.zeros(xx_quad.shape[0])
for i in range(f_quad.shape[0]):
  f_quad[i] = ohagan(xx_quad[i,:].reshape(6,1))
coeffs = cb.PolyChaos().comp_coeffs(xx_quad, f_quad, ww, degree = 5)
print(coeffs.shape)

pol = cb.PolyBasis(6, 5, 'H')
P = pol(xx)
f_pc = np.dot(P, coeffs)

plt.hist(f, bins = 100, density = True)
plt.hist(f_pc, bins = 100, density = True)
plt.show()

plt.plot(f[:1000], 'x')
plt.plot(f_pc[:1000], '+')
plt.show()
