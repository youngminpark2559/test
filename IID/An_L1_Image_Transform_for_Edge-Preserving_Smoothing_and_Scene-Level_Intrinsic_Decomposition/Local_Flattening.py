# ======================================================================
# Local Flattening 

# Eq 2

# c_I_i: RGB color vector of input image at pixel p_i
# (3,) 1D array
I_i

# c con_I_i: converted I_i into CIELab color vector at pixel p_i
# (3,) 1D array, elements are within [0,1]
con_I_i=CIELab_color_space(I_i)

# c x_i: RGB color vector of transformed image, 
# (3,) 1D array
x_i


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

# Suppose there are n pixels in image and m_l neighboring pixel pairs in (2). 
# Let z_r be vector concatenating R channel at all pixels of transformed image. 
# z_g and z_b are defined similarly, and are concatenations of G and B channels at all pixels. 
# Let z be concatenation of z_r, z_g, z_b. 
# That is, z ([z_r^T, z_g^T, z_b^T ]^T ) is 3D vector. 
# Let M = {M_{ij}} be m_l x n matrix, 
# where 
# M_{ki} = w_{ij}
# M_{kj} = -w_{ij} 
# if p_i and p_j are neighboring pixels that form k-th pixel pair. 
# The energy term in (2) can be rewritten in matrix-vector form as follows. 

# El = ||Lz||1, where L3ml×3n =   M M M   (4) 

# c z_r: all r values from transformed image
z_r
# c z_g: all g values from transformed image
z_g
# c z_b: all b values from transformed image
z_b

# c z: concatenated z with r, g, b vectors
# 3D array
z=[z_r,z_g,z_b]


# Let M = {M_{ij}} be m_l x n matrix, 
# where 
# M_{ki} = w_{ij}
# M_{kj} = -w_{ij} 
# if p_i and p_j are neighboring pixels that form k-th pixel pair. 
# The energy term in (2) can be rewritten in matrix-vector form as follows. 

# El = ||Lz||1, where L3ml×3n =   M M M   (4) 


# c nnp: number of neighborh pixels
nnp
# c npi: number of pixel of image
npi
