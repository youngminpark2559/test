# ======================================================================
# 3 Piecewise Image Flattening 

# c E_l: local flattening
E_l

# c E_g: global sparsity
E_g

# c E_a: image approximation
E_a

# c alpha: weight for global sparsity
alpha

# c beta: weight for image approximation
beta

# c first_t: first term
first_t=E_l

# c second_t: second term
second_t=alpha*E_g

# c third_t: third term
third_t=beta*E_a

# c min_val: minimum value when minimizing first_t+second_t+third_t
min_val=minimize(first_t+second_t+third_t)


# ======================================================================
# Local Flattening 

# Eq 2

# c I_i: RGB vector of input image at pixel p_i
# (3,) 1D array
I_i

# c con_I_i: CIELab vector at pixel p_i, converted from I_i
# (3,) 1D array, elements are within [0,1]
con_I_i=CIELab_color_space(I_i)

# c x: RGB (piecewise flatten) transformed image,
# (3,) 1D array
x


# Let's assume (h=3) 3x3 neighborhood 
# (8 neighborhood pixels around center pixel) 

# sum_i[w_{i,1}||x_i-x_1||_1+...+w_{i,8}||x_i-x_8||_1]

# = sum_i[w_{i,1}||x_i-x_1||_1]+...+sum_i[w_{i,8}||x_i-x_8||_1]

# = [w_{1,1}||x_1-x_1||_1+...+w_{n,1}||x_n-x_1||_1]
#   +...
#   +[w_{1,8}||x_1-x_8||_1+...+w_{n,8}||x_n-x_8||_1]


# ---------------
# Eq 3

# c first_f: first f term
first_f=con_I_i of input image but kappa is multiplied by l

# c second_f: second f term
second_f=con_I_i of neighbors but kappa is multiplied by l

# w_{11}=exp(-||first_f_1-second_f_1||_2^2/2*sigma^2)
# ...
# w_{18}=exp(-||first_f_1-second_f_8||_2^2/2*sigma^2)


# ---------------
# Eq 4

# c z_r: all r values from transformed image
z_r

# c z_g: all g values from transformed image
z_g

# c z_b: all b values from transformed image
z_b

# c z: r, g, b vectors into array
# 3*n dimensional array
# Each vector from z_r,z_g,z_b is n dismensional array
z=[z_r,z_g,z_b]


# R channel of input image
# np1_p1_r: neighbor pixel 1 of input image pixel 1 in r channel
# np8_p1_r: neighbor pixel 8 of input image pixel 1 in r channel
M_r=[[np1_p1_r,np1_p1_r,...,np8_p1_r],
     [np1_p2_r,np1_p2_r,...,np8_p2_r],
     ...
     [np1_pn_r,np1_pn_r,...,np8_pn_r]]

# G channel of input image
M_g=[[np1_p1_g,np1_p1_g,...,np8_p1_g],
     [np1_p2_g,np1_p2_g,...,np8_p2_g],
     ...
     [np1_pn_g,np1_pn_g,...,np8_pn_g]]

# B channel of input image
M_b=[[np1_p1_b,np1_p1_b,...,np8_p1_b],
     [np1_p2_b,np1_p2_b,...,np8_p2_b],
     ...
     [np1_pn_b,np1_pn_b,...,np8_pn_b]]


# M_r
#    M_g
#       M_b

# Each matrix (M_r,M_g,M_b) is (n,8) matrix
L_mat=
[
    M_r  0 0 0 0 0 0 0 0  0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0  M_g  0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  M_b
]



L_mat: (3*n,3*8)
L_mat.T: (3*8,3*n)
z: (3*n,1)
E_l=l1_norm(mat_mul(L_mat.T,z))


# ======================================================================
# Global Sparsity

# Eq 5

# c fnsp: fixed number (this number is denoted as n_s) of superpixels 
# using algorithm in [Felzenszwalb and Huttenlocher 2004]. 
fnsp=algo_by_Felzenszwalb_Huttenlocher(input_img)

# c rep_p: representative pixels
# This forms set named S_r
# which is subset of pixels in original image. 
rep_p=find_pixel_closest_color_of_superpixel(fnsp)


# c glo_spar_t: global sparsity energy term
glo_spar_t

for o_i_p in representative image pixels:
    for o_n_p in representative neighbor pixels:
        w_ij=weight[o_i_p][o_n_p]
        x_i=tfed_img[o_i_p]
        x_j=tfed_img[o_n_p]
        L1_norm=L1_norm(x_i-x_j)
        glo_spar_t+=w_ij*L1_norm


# ======================================================================
# Image Approximation 

# Eq 7

# c z: concatenation of all pixel colors from input image
z
# c z^{in}: concatenation of all pixel colors from transformed image
z^{in}

# c E_a: image approximation term
E_a=square(L2_norm(z-z^{in}))


# ======================================================================
# L1 Solver 

# intermediate variables introduced by Split Bregman method
b_1^k / b_2^k / d_1^k / d_2^k
# c eps: epsilon, controls termination of iteration
eps
# c lamb: lambda, controls relative weight of L1 energy term
lamb


fir_t=beta*E_a
sec_t=lamb*||d_1^k-E_l-b_1^k||_2^2
thi_t=lamb*||d_2^k-alpha*E_g-b_2^k||_2^2
sum_eq_8=fir_t+sec_t+thi_t

z=z_when_minimum_from_following(sum_eq_8)

# ======================================================================
# Algorithm 1 
# Split-Bregman method for piecewise image flattening

# Initialization
z^0=z^{in}
d_1^0,b_1^0,d_2^0,b_2^0=0

while square(L2_norm(z^k-z^{k-1}))>eps:
    # Get A
    A=beta*I_{3nx3n}+
      lamb*L.t*L+
      lamb*alpha^2*G.t*G
    # Get v
    v=beta*z^{in}+
      lamb*L.t*(d_1^k-b_1^k)+
      lamb*alpha*G.t(d_2^k-b_2^k)

    Update z^{k+1} by solving Az^{k+1}=v

    # Get intermediate variables at k+1
    d_1^{k+1}=Shrink(Lz^{k+1}+b_1^k,1/lamb)
    d_2^{k+1}=Shrink(alpha*Gz^{k+1}+b_2^k,1/lamb)
    b_1^{k+1}=b_1^k+Lz^{k+1}-d_1^{k+1}
    b_2^{k+1}=b_2^k+alpha*Gz^{k+1}-d_2^{k+1}

    # Increment k by 1
    k=k+1

# After end of while, return z^k
return z^k

def Shrink(y,gamma):
    fir_t=y/||y||
    sec_t=max(||y||-gamma,0)
    sum_t=fir_t*sec_t
    return sum_t


# ======================================================================
# 3.1 Edge-Preserving Smoothing 

# c hat_g_ij: maximum gradient magnitude
# along the line segment between p_i and p_j pixels
# which is calculated by Sobel operator in case of this paper
hat_g_ij

# c eta: weighting coefficient for hat_g_ij
eta

# The following parameter setting is used 
# for all edge-preserving smoothing results presented in this paper: 
beta=2.5
kappa=0.3
eta=0.4
sigma=1.0
h=11
lamb==5.0
epsilon=0.001

fir_t=||f_i-f_j||_2^2
sec_t=eta*sq(hat_g_ij)

# c w_ij_epsm: w_ij when edge_preserving smoothing is performed
w_ij_epsm=exp(-max(fir_t,sec_t)/2*sigma^2)


# ======================================================================
# 4.1 Clustering 

# c pw_f_img: piecewise flattened image
pw_f_img

# c CL_c_f_img: CIELab color space flattened image
CL_c_f_img=convert(pw_f_img)

# c c_res: clustering result
c_res=[]
for i in range(iter_clustering):
    global c_a_DPGMM
    if i==0:
        # Clustering is performed on following pixelwise vectors/ 
        # f_c = [kappa*l',a',b'].T
        # [l',a',b'] are CIELab color channels computed from piecewise flattened image
        clustered=clustering(CL_c_f_img)

        # self-adaptive probabilistic clustering method 
        # to automatically determine proper number of pixel clusters
        # DPGMM is infinite mixture model with Dirichlet Process 
        # as prior distribution over number of clusters
        # c_a_DPGMM: cluster after DPGMM
        c_a_DPGMM=Dirichlet_Process_Gaussian_Mixture_Model(clustered)
    else:
        clustered=clustering(c_a_DPGMM)
        c_a_DPGMM=Dirichlet_Process_Gaussian_Mixture_Model(clustered)

    c_res.append(c_a_DPGMM)

# ---------------
# Probabilistic boosting tree (PBT)

# We sample m_t pixels (from image (c_res) created from DPGMM? Not sure of this) 
# (in our implementation, m_t is always set to 30,000) from image 
sam_pix=sampe_30000_pixels(c_res)

# c CIELab_c_img: CIELab color channle image
# fp = [l^2,a^2,b^2,l∗a,l∗b,a∗b,l,a,b]^T is form of converted image (from sam_pix) 
# for train and test on PBT
CIELab_c_img=convert(sam_pix)

# c PBT: probabilistic boosting tree
P_{i,k}=P(C_k|f_{p_i})=PBT(CIELab_c_img)

# ======================================================================
# 4.2 Reflectance Labeling

# C_k: cluster
# f_{p_i}: sample
# P_{i,k}=P(C_k|f_{p_i})

# Eq 11

# c phi: reflectance label of pixel p_i
phi

# c e_u_temp: energy unary temp
e_u_temp=0

for one_p in all_pixels:
    e_u+=log{P(C_{phi[one_p]}|f_{one_p})}

# c e_u: energy unary
e_u=-e_u_temp

# Eq 12, Eq 13, Eq 14
e_pair=0
for o_pixel in all_pixels:
    for o_neighbor in all_neighbors:
        if phi[o_pixel]!=phi[o_neighbor]:
            tau=1
        else:
            tau=0
        f[o_pixel]=[k*l,a[o_pixel],b[o_pixel]].t
        f[o_neighbor]=[k*l,a[o_neighbor],b[o_neighbor]].t
        upper_t=sq(L2_norm(f[o_pixel]-f[o_neighbor]))
        lower_t=2*sq(sigma)
        w[o_pixel][o_neighbor]=exp(-upper_t/lower_t)
        e_pair+=w[o_pixel][o_neighbor]*tau

crf_e=e_u+gamma*e_pair

# ======================================================================
# 4.3 Reflectance and Shading Estimation 

# Eq 15

# piecewise flattening image (edge preserving smoothing) ->
# clustering ->
# DPGMM ->
# clustering ->
# DPGMM ->
# ... ->
# PBT ->
# superpixel
# c n_prime_s: superpixels
# All subpixels of each superpixel have same label
# And reflectances have minor differences across each superpixel
n_prime_s

# c I_bar_k: average intensity of pixels within superpixel q_k
I_bar_k

# c q_k: superpixel
q_k

# c R_bar_k: scalar reflectance of q_k
R_bar_k

# c S_bar_k: "average" scalar shading over q_k
S_bar_k=I_bar_k/S_bar_k

# In our algorithm, 
# scalar reflectance of superpixel is unknown, 
# and these unknowns are solved by imposing shading smoothness prior 
# across neighboring superpixels 
# and minimizing following cost function

# c k: kth superpixel
k=q_k

# c l: kth superpixel's neighbors
l=q_l

# c xi: constant
xi

R_hat_k=log(R_bar_k)
S_hat_k=log(I_bar_k)-R_hat_k

# image composed of superpixel
img_su

first_t=0
for o_su_p in all_superpixels:
    for o_n_su_p in all_neighbors_to_one_superpixel:
        first_t+=sq(S_hat_k[o_su_p]-S_hat_k[o_n_su_p])

second_t=0
for o_su_p in all_superpixels:
    for o_n_su_p in all_neighbors_to_one_superpixel:
        second_t+=sq(R_hat_k[o_su_p]-R_hat_k[o_n_su_p])

final_cost_f_eq15=first_t+xi*second_t

# ======================================================================
# Eq 16

# c I_i: color vector at pixel i from original image
# (3,) 1D array
I_i

# c R_i: scalar reflectance from Eq 15
R_i

s_final_s_i=0
for o_c in all_channel:
    s_final_s_i+=I_i[o_c]

final_R_i=3*R_i/s_final_s_i*I_i
