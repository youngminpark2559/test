# ======================================================================
# Local Flattening 

# Eq 2

# c I[i]: RGB color vector of input image at pixel p_i
# (3,) 1D array
I[i]

# c con_I[i]: converted I[i] into CIELab color vector at pixel p_i
# (3,) 1D array, elements are within [0,1]
con_I[i]=CIELab_color_space_converter(I[i])

# c x_i: RGB color vector of (piecewise flatten) transformed image (which is result of Eq 1), 
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
first_f=elements are con_I[i] ([l_i,a_i,b_i]^T) but kappa is multiplied by l ([\kappa*l_i,a_i,b_i]^T)

# c second_f: second f term
second_f=con_I[i] of neighbors but kappa is multiplied by l
For example, 
[\kappa*l_neighbor1_of_pixel_i,a_i,b_i]^T
...
[\kappa*l_neighbor8_of_pixel_i,a_i,b_i]^T

# In conclusion, 
# w_{11}=exp(-||first_f_1-second_f_1||_2^2/2*sigma^2)
# ...
# w_{18}=exp(-||first_f_1-second_f_8||_2^2/2*sigma^2)


# ---------------
# Eq 4

# c z_r: all r values from transformed image x, n dimensional vector
z_r

# c z_g: all g values from transformed image x, n dimensional vector
z_g

# c z_b: all b values from transformed image x, n dimensional vector
z_b

# c z: concatenated z with r, g, b vectors
# 3*n dimensional vector
z=[z_r,z_g,z_b]
