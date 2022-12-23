#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cmath>
#include <iostream>
#include <string>
#include <math.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <cstdio>
#include <mkl.h>
#include <assert.h>

// used to initialize numpy, otherwise it fails.
int init_(){
    mkl_set_dynamic(0);
    mkl_set_num_threads(1);
    import_array(); // PyError if not successful
    return 0;
}


// the relevant vairables are I_x=I_y, Q_z, xy grid (number of lattice sites and grid size), and the quadratic potential coefficient k
// I_z is not relevant, because I_z*Q_z is a constant in the Hamiltonian, and therefore has no effect
const static int x_max = X_MAX;
const static int x_n = 2*x_max+1;
const static double I = I_XY, grid_size = GRID_SIZE, k = V_K, Q_z = Q_Z; 

const static int moment_order = MOMENT;
const static int error_boundary_length = 8;

const static int Legendre_p = 500; ////////////////////////////////


const static double k_div_2 = k/2., Q_z_2_div_4_div_2I = Q_z*Q_z/4./(2.*I), Q_z_2_div_2I = Q_z*Q_z/(2.*I);
const static double E_z = sqrt(k_div_2); 
static double displacement_x, displacement_y, E_x, E_y;

static VSLStreamStatePtr stream;
static MKL_Complex16** allocate_state_memory();
static double** allocate_real_matrix(); 
static void __attribute__((hot)) set_potential_data(double displacement_x_, double displacement_y_);
static int factorial(int i);
static int combination_w_repetition(int item, int choices);
const int number_of_variables = 4;

static int x_start = 4, x_end = x_n-4, y_start = 4, y_end = x_n-4;
const static double angle_threshold_to_neglect = 60./180.*M_PI; // 60 degrees

const static int num_of_matrices_used = 40;
class Set_World{
public:
    double x[x_n], x_2[x_n]; 
    double **measure, **sin_2beta_div_beta, **sin_beta_div_beta_2;
    double **px_factor, **py_factor;
    double **g_div_beta, **tan_half_beta_div_beta;
    double **two_g_div_beta_plus_g_2, **tan_half_beta_2, **one_plus_g_beta_mul_2tan_half_beta_div_beta, **g_div_beta_plus_tan_half_beta_div_beta_plus_g_2;
    MKL_INT error_boundary_mask[x_n][x_n];
    double** real_matrix_storage[2];
    MKL_Complex16** complex_matrix_storage[num_of_matrices_used];
    double Legendre_p_coefficients[Legendre_p+1];

    int total_number_of_moments;
    int matrix_index_for_order_i_moment[moment_order];
    int indices_of_matrix_for_computing_moments[150][number_of_variables];
    ///////////////////////////// *** initializer *** ///////////////////////////////////////////

    Set_World(){
    // compute the basic  x, x^2, V=\lambda*x^4  arrays
    for(int i=0;i < x_n;i++){x[i]=grid_size*((double)(i-x_max));}
    for(int i=0;i < x_n;i++){x_2[i]=x[i]*x[i];}
    for(int i=0;i < x_n;i++){for(int j=0;j < x_n;j++){
        if((i<error_boundary_length || i >= x_n-error_boundary_length || j<error_boundary_length || j>=x_n-error_boundary_length) && (i>=4 && j>=4 && i<x_n-4 && j<x_n-4)){
            error_boundary_mask[i][j]=1;
        }else{
            error_boundary_mask[i][j]=0;
        }
    }}
    for(int i=0;i<num_of_matrices_used;i++){complex_matrix_storage[i]=allocate_state_memory();}
    for(int i=0;i<2;i++){real_matrix_storage[i]=allocate_real_matrix();}
    measure=allocate_real_matrix();
    sin_2beta_div_beta=allocate_real_matrix();
    sin_beta_div_beta_2=allocate_real_matrix();
    px_factor=allocate_real_matrix();
    py_factor=allocate_real_matrix();
    g_div_beta=allocate_real_matrix();
    tan_half_beta_div_beta=allocate_real_matrix();
    two_g_div_beta_plus_g_2=allocate_real_matrix();tan_half_beta_2=allocate_real_matrix();one_plus_g_beta_mul_2tan_half_beta_div_beta=allocate_real_matrix();g_div_beta_plus_tan_half_beta_div_beta_plus_g_2=allocate_real_matrix();
    // put data into matrices
    double x_, y_, beta, beta_2, temp;
    double d_beta_measure, tan_half_beta;
    for(int i=0;i < x_n;i++){
        y_ = x[i];
        for(int j=0;j < x_n;j++){
            x_ = x[j];
            beta = sqrt(x_*x_+y_*y_);
            temp = sin(beta)/beta;
            sin_2beta_div_beta[i][j]=sin(2.*beta)/beta;
            measure[i][j]=temp;
            sin_beta_div_beta_2[i][j]=temp*temp;
            d_beta_measure = (beta*cos(beta)-sin(beta))/(beta*beta);
            px_factor[i][j] = 0.5 * x_/beta * d_beta_measure;
            py_factor[i][j] = 0.5 * y_/beta * d_beta_measure;
            tan_half_beta = tan(beta/2.);
            tan_half_beta_div_beta[i][j] = tan_half_beta/beta;
            g_div_beta[i][j] = (1./(2.*tan_half_beta) - 1./beta + tan_half_beta/2.)/beta;
            }
        }
    measure[x_max][x_max] = 1.;
    sin_beta_div_beta_2[x_max][x_max] = 1.;
    sin_2beta_div_beta[x_max][x_max] = 2.;
    px_factor[x_max][x_max] = 0.;
    py_factor[x_max][x_max] = 0.;
    tan_half_beta_div_beta[x_max][x_max] = 0.5;
    g_div_beta[x_max][x_max] = 1./6.;
    double g_2;
    for(int i=0;i < x_n;i++){
        y_ = x[i];
        for(int j=0;j < x_n;j++){
            x_ = x[j];
            beta_2 = x_*x_+y_*y_;
            beta = sqrt(beta_2);
            g_2 = g_div_beta[i][j]*g_div_beta[i][j]*beta_2;
            two_g_div_beta_plus_g_2[i][j] = 2.*g_div_beta[i][j] + g_2;
            tan_half_beta = tan(beta/2.);
            tan_half_beta_2[i][j] = tan_half_beta*tan_half_beta;
            one_plus_g_beta_mul_2tan_half_beta_div_beta[i][j] = (1.+g_div_beta[i][j]*beta_2)*2.*tan_half_beta_div_beta[i][j];
            g_div_beta_plus_tan_half_beta_div_beta_plus_g_2[i][j] = g_div_beta[i][j] + tan_half_beta_div_beta[i][j] + g_2;
        }
    }
    for(int i=1; i<=Legendre_p; i++){
        Legendre_p_coefficients[i] = 1./sqrt(4.*(double)i*(double)i - 1.);
    }
    Legendre_p_coefficients[0] = 0.;
    

    total_number_of_moments = 0;
    for(int i=1; i<=(moment_order-2); i++){
        total_number_of_moments += combination_w_repetition(number_of_variables, i);
        matrix_index_for_order_i_moment[i-1] = total_number_of_moments;
        //printf("%d combinations", total_number_of_moments);
    }
    assert(total_number_of_moments<= num_of_matrices_used);
    for(int i=moment_order-1; i<=moment_order; i++){
        total_number_of_moments += combination_w_repetition(number_of_variables, i);
        matrix_index_for_order_i_moment[i-1] = total_number_of_moments;
        //printf("%d combinations", total_number_of_moments);
    }

    int current_combination[number_of_variables] = {};
    bool combination_valid;
    int combination_index = 0;
    for(int o=1; o<=(moment_order); o++){
        current_combination[0] = o; 
        for(int i=1; i<number_of_variables; i++) current_combination[i] = 0;
        combination_valid = true;
        while(combination_valid){
            //printf("(%d, %d, %d, %d)\n", current_combination[0], current_combination[1],current_combination[2],current_combination[3]);
            for(int i=0; i<number_of_variables; i++) indices_of_matrix_for_computing_moments[combination_index][i] = current_combination[i];
            combination_index += 1;
            combination_valid = find_next_combination(current_combination);
        }
    }
    assert(combination_index==total_number_of_moments);
    //printf("%d combinations\n", combination_index);
    }
    bool find_next_combination(int* indices){
        int temp;
        for(int i=0; i< number_of_variables-1; i++){
            if(indices[i]!=0){
                temp = indices[i];
                indices[i] = 0;
                indices[0] = temp - 1;
                indices[i+1] += 1;
                return true;
            }
        }
        return false;
    }
    ~Set_World(){
        for(int i=0;i<num_of_matrices_used;i++){free(complex_matrix_storage[i]);}
        for(int i=0;i<2;i++){free(real_matrix_storage[i]);}
        free(measure);
        free(sin_2beta_div_beta);
        free(sin_beta_div_beta_2);
        free(px_factor);
        free(py_factor);
        free(g_div_beta); free(tan_half_beta_div_beta);
        free(two_g_div_beta_plus_g_2); free(tan_half_beta_2); free(one_plus_g_beta_mul_2tan_half_beta_div_beta); free(g_div_beta_plus_tan_half_beta_div_beta_plus_g_2);
    }
};

const static Set_World world;


static int check_type(PyArrayObject* state);
static int check_type_real(PyArrayObject* state);

static MKL_Complex16** allocate_state_memory(){
    MKL_Complex16 *ptr, **state;
    state = (MKL_Complex16 **)malloc( sizeof(MKL_Complex16 *) * x_n + sizeof(MKL_Complex16) * x_n * x_n );
    // ptr is now pointing to the first element in of 2D array
    ptr = (MKL_Complex16 *)(state + x_n);
    for(int i = 0; i < x_n; i++)
        state[i] = (ptr + x_n * i);
    for(int i = 0; i < x_n; i++){
        for(int j = 0; j < x_n; j++){
            state[i][j].real=0.; state[i][j].imag=0.;
        }
    }
    return state;
}
static double** allocate_real_matrix(){
    double *ptr, **matrix;
    matrix = (double **)malloc( sizeof(double *) * x_n + sizeof(double) * x_n * x_n );
    // ptr is now pointing to the first element in of 2D array
    ptr = (double *)(matrix + x_n);
    for(int i = 0; i < x_n; i++)
        matrix[i] = (ptr + x_n * i);
    for(int i = 0; i < x_n; i++){
        for(int j = 0; j < x_n; j++){
            matrix[i][j]=0.;
        }
    }
    return matrix;
}

static void __attribute__((hot)) compute_probability_distribution(MKL_Complex16** psi, double** prob_distribution){ 
    //double measure, x_2, y_2, beta_2;
    double a, b;
    for(int i=4;i< x_n-4;i++){
        //y_2 = world.x_2[i];
    for(int j=4;j < x_n-4;j++){
        //x_2 = world.x_2[j];
        //beta_2 = y_2 + x_2 ;
        //measure = 1. - beta_2/6. + beta_2*beta_2/120.; // two lowest order approximation. The next term is - (x*x + y*y)*(x*x + y*y)*(x*x + y*y)/5040
        a = psi[i][j].real; b = psi[i][j].imag;
        prob_distribution[i][j]= (world.measure[i][j]*(a*a + b*b));
        }}
}
static PyObject* Py_prob(PyObject *self, PyObject *args){
    PyObject* temp1, *temp2;
    PyArrayObject* state, *out;
    if (!PyArg_ParseTuple(args, "OO", &temp1, &temp2)){
        PyErr_SetString(PyExc_TypeError, "The input does not match the required input signature (state (numpy array), result (numpy array))");
        return NULL;
    }
    PyArray_OutputConverter(temp1, &state);
    if (!state) {PyErr_SetString(PyExc_TypeError, "The input state cannot be identified as a Numpy array"); return NULL;}
    PyArray_OutputConverter(temp2, &out);
    if (!state) {PyErr_SetString(PyExc_TypeError, "The result argument cannot be identified as a Numpy array"); return NULL;}
    if (check_type(state)!=0) return NULL;
    if (check_type_real(out)!=0) return NULL;
    // get the c array from numpy (complex128 is stored as successive float64 two by two)
    MKL_Complex16* psi[x_n];
    for(int i=0; i<x_n; i++){
        psi[i] =  (MKL_Complex16*) PyArray_GETPTR2(state, i, 0);
    }
    double* result[x_n];
    for(int i=0; i<x_n; i++){
        result[i] =  (double*) PyArray_GETPTR2(out, i, 0);
    }
    
    compute_probability_distribution(psi,result);
    Py_RETURN_NONE;
}
double norm(const MKL_Complex16*const* psi){
    double _norm;
    double prob_tot=0.;
    double a, b;
    for(int i=0;i< x_n;i++){
    for(int j=0;j< x_n;j++){
        a = psi[i][j].real; b = psi[i][j].imag;
        prob_tot += (world.measure[i][j]*(a*a + b*b));
        }}
    prob_tot *= (grid_size*grid_size);
    _norm = sqrt(prob_tot);
    return _norm;
}
static PyObject* Py_norm(PyObject *self, PyObject *args){
    PyObject* temp;
    PyArrayObject* state;
    if (!PyArg_ParseTuple(args, "O", &temp)){
        PyErr_SetString(PyExc_TypeError, "The input does not match the required input signature (state (numpy array))");
        return NULL;
    }
    PyArray_OutputConverter(temp, &state);
    if (!state) {PyErr_SetString(PyExc_TypeError, "The input state cannot be identified as a Numpy array"); return NULL;}
    if (check_type(state)!=0) return NULL;
    // get the c array from numpy (complex128 is stored as successive float64 two by two)
    MKL_Complex16* psi[x_n];
    for(int i=0; i<x_n; i++){
        psi[i] =  (MKL_Complex16*) PyArray_GETPTR2(state, i, 0);
    }
    return Py_BuildValue("d", norm(psi));
}
static void x_y_expect(const MKL_Complex16*const* psi, double* result_x, double* result_y){
    *result_x=0.; *result_y=0.;
    double x, y;
    double a,b,p;
    for(int i=4;i< x_n-4;i++){
        y = world.x[i];
    for(int j=4;j < x_n-4;j++){
        x = world.x[j];
        a = psi[i][j].real; b = psi[i][j].imag;
        p = world.measure[i][j]*(a*a+b*b);
        *result_x = (*result_x)+x*p;
        *result_y = (*result_y)+y*p;
        }}
    *result_x = (*result_x)*grid_size*grid_size;
    *result_y = (*result_y)*grid_size*grid_size;
    return;
}

static PyObject* Py_x_y_expect(PyObject *self, PyObject *args){
    PyObject* temp;
    PyArrayObject* state;
    if (!PyArg_ParseTuple(args, "O", &temp)){
        PyErr_SetString(PyExc_TypeError, "The input does not match the required input signature (state (numpy array))");
        return NULL;
    }
    PyArray_OutputConverter(temp, &state);
    if (!state) {PyErr_SetString(PyExc_TypeError, "The input state cannot be identified as a Numpy array"); return NULL;}
    if (check_type(state)!=0) return NULL;
    // get the c array from numpy (complex128 is stored as successive float64 two by two)
    MKL_Complex16* psi[x_n];
    for(int i=0; i<x_n; i++){
        psi[i] =  (MKL_Complex16*) PyArray_GETPTR2(state, i, 0);
    }
    double x_mean=0., y_mean=0.;
    x_y_expect(psi, &x_mean, &y_mean);
    return Py_BuildValue("dd", x_mean, y_mean);
}


static double normalize(MKL_Complex16** psi){
    double _norm;
    double prob_tot=0.;
    double a, b;
    for(int i=4;i< x_n-4;i++){
    for(int j=4;j < x_n-4;j++){
        a = psi[i][j].real; b = psi[i][j].imag;
        prob_tot += (world.measure[i][j]*(a*a + b*b));
        }}
    prob_tot *= (grid_size*grid_size);
    _norm = sqrt(prob_tot);
    if(_norm==0.){printf("norm is zero");return _norm;}
    if(_norm!=0.){cblas_dscal (2*x_n*(x_n-8), 1./_norm, &psi[4][0].real, 1);}
    return _norm;
}

static double x_relative[x_n];
static double y_relative[x_n];
static void set_x_y_relative_data(const MKL_Complex16*const* psi){
    double x_mean, y_mean;
    x_y_expect(psi, &x_mean, &y_mean);
    for(int i=0;i< x_n;i++){
    x_relative[i] = std::max(std::min(world.x[i]-x_mean, angle_threshold_to_neglect), -angle_threshold_to_neglect);
    y_relative[i] = std::max(std::min(world.x[i]-y_mean, angle_threshold_to_neglect), -angle_threshold_to_neglect);
}
}

// the following functions assume that "set_x_y_relative_data" is already called
static void x_hat_relative(const MKL_Complex16*const* psi, MKL_Complex16** result, double factor){
    double x_multiply;
    for(int i=4;i< x_n-4;i++){for(int j=4;j < x_n-4;j++){x_multiply=x_relative[j]*factor; result[i][j].real=x_multiply*psi[i][j].real; result[i][j].imag=x_multiply*psi[i][j].imag;}}
}
static void y_hat_relative(const MKL_Complex16*const* psi, MKL_Complex16** result, double factor){
    double y_multiply;
    for(int i=4;i< x_n-4;i++){y_multiply=y_relative[i]*factor;for(int j=4;j < x_n-4;j++){ result[i][j].real=y_multiply*psi[i][j].real; result[i][j].imag=y_multiply*psi[i][j].imag;}}
}
/*
static void __attribute__((hot)) x_hat_relative_square(const double total_coefficient, MKL_Complex16** psi, MKL_Complex16** result){
   double x_multiply = 0.;
    if(total_coefficient==1.){
        for(int i=4;i< x_n-4;i++){for(int j=4;j < x_n-4;j++){x_multiply=x_relative[j]*x_relative[j]; result[i][j].real=psi[i][j].real*x_multiply; result[i][j].imag=psi[i][j].imag*x_multiply;}}
    }else{
        for(int i=4;i< x_n-4;i++){for(int j=4;j < x_n-4;j++){x_multiply=total_coefficient*x_relative[j]*x_relative[j]; result[i][j].real=psi[i][j].real*x_multiply; result[i][j].imag=psi[i][j].imag*x_multiply;}}
    }
}
static void __attribute__((hot)) y_hat_relative_square(const double total_coefficient, MKL_Complex16** psi, MKL_Complex16** result){
   double y_multiply = 0.;
    if(total_coefficient==1.){
        for(int i=4;i< x_n-4;i++){y_multiply=y_relative[i]*y_relative[i]; for(int j=4;j < x_n-4;j++){result[i][j].real=psi[i][j].real*y_multiply; result[i][j].imag=psi[i][j].imag*y_multiply;}}
    }else{
        for(int i=4;i< x_n-4;i++){y_multiply=total_coefficient*y_relative[i]*y_relative[i]; for(int j=4;j < x_n-4;j++){result[i][j].real=psi[i][j].real*y_multiply; result[i][j].imag=psi[i][j].imag*y_multiply;}}
    }
}*/
static double x_relative_2_cache[x_n];
static void __attribute__((hot)) xy_squeezing_term__add_into_result(MKL_Complex16** psi, const double &gamma, MKL_Complex16** result){
    double x_multiply, y_multiply, factor;
    double total_coefficient = -gamma/4.;
    {
        int i=y_start;
        y_multiply = total_coefficient*y_relative[i]*y_relative[i];
        for(int j=x_start;j < x_end;j++){
            x_multiply = total_coefficient*x_relative[j]*x_relative[j]; 
            x_relative_2_cache[j] = x_multiply;
            factor = x_multiply + y_multiply;
            result[i][j].real = psi[i][j].real*factor + result[i][j].real; result[i][j].imag = psi[i][j].imag*factor + result[i][j].imag;
        }
    }
    for(int i=y_start+1;i< y_end;i++){
        y_multiply=total_coefficient*y_relative[i]*y_relative[i];
        for(int j=x_start;j < x_end;j++){
            x_multiply=x_relative_2_cache[j]; 
            factor = x_multiply + y_multiply;
            result[i][j].real = psi[i][j].real*factor + result[i][j].real; result[i][j].imag = psi[i][j].imag*factor + result[i][j].imag;
        }
    }
}
static double x_relative_factor_cache[x_n];
static void __attribute__((hot)) xy_squeezing_term__add_into_result__and_get_DX_DY(MKL_Complex16** psi, const double &gamma,  MKL_Complex16** result, MKL_Complex16** result_x, MKL_Complex16** result_y){
    double x_multiply, y_multiply, factor;
    double DX_multiply, DY_multiply;
    double state_real, state_imag;
    double total_coefficient = -gamma/4.;
    double xy_coefficient = sqrt(gamma*0.5);
    {
        int i=y_start;
        y_multiply = total_coefficient*y_relative[i]*y_relative[i];
        DY_multiply = xy_coefficient*y_relative[i];
        for(int j=x_start;j < x_end;j++){
            x_multiply = total_coefficient*x_relative[j]*x_relative[j]; 
            x_relative_2_cache[j] = x_multiply;

            DX_multiply = xy_coefficient*x_relative[j];
            x_relative_factor_cache[j] = DX_multiply;

            state_real = psi[i][j].real; state_imag = psi[i][j].imag; 
            factor = x_multiply + y_multiply;
            result[i][j].real = state_real*factor + result[i][j].real; result[i][j].imag = state_imag*factor + result[i][j].imag;
            result_x[i][j].real = state_real*DX_multiply; result_x[i][j].imag = state_imag*DX_multiply;
            result_y[i][j].real = state_real*DY_multiply; result_y[i][j].imag = state_imag*DY_multiply;
        }
    }
    for(int i=y_start+1;i< y_end;i++){
        y_multiply=total_coefficient*y_relative[i]*y_relative[i];
        DY_multiply = xy_coefficient*y_relative[i];
        for(int j=x_start;j < x_end;j++){
            x_multiply=x_relative_2_cache[j]; 

            DX_multiply = x_relative_factor_cache[j];

            state_real = psi[i][j].real; state_imag = psi[i][j].imag; 
            factor = x_multiply + y_multiply;
            result[i][j].real = state_real*factor + result[i][j].real; result[i][j].imag = state_imag*factor + result[i][j].imag;
            result_x[i][j].real = state_real*DX_multiply; result_x[i][j].imag = state_imag*DX_multiply;
            result_y[i][j].real = state_real*DY_multiply; result_y[i][j].imag = state_imag*DY_multiply;
        }
    }
}
static void __attribute__((hot)) DX_DY(MKL_Complex16** state, const double &gamma, MKL_Complex16** result_x, MKL_Complex16** result_y){
    double DX_multiply, DY_multiply;
    double state_real, state_imag;
    double xy_coefficient = sqrt(gamma*0.5);
    {
        int i=y_start;
        DY_multiply = xy_coefficient*y_relative[i];
        for(int j=x_start;j < x_end;j++){
            DX_multiply = xy_coefficient*x_relative[j];
            x_relative_factor_cache[j] = DX_multiply;

            state_real = state[i][j].real; state_imag = state[i][j].imag; 
            result_x[i][j].real = state_real*DX_multiply; result_x[i][j].imag = state_imag*DX_multiply;
            result_y[i][j].real = state_real*DY_multiply; result_y[i][j].imag = state_imag*DY_multiply;
        }
    }
    for(int i=y_start+1;i< y_end;i++){
        DY_multiply = xy_coefficient*y_relative[i];
        for(int j=x_start;j < x_end;j++){
            DX_multiply = x_relative_factor_cache[j];

            state_real = state[i][j].real; state_imag = state[i][j].imag; 
            result_x[i][j].real = state_real*DX_multiply; result_x[i][j].imag = state_imag*DX_multiply;
            result_y[i][j].real = state_real*DY_multiply; result_y[i][j].imag = state_imag*DY_multiply;
        }
    }
    
}
///////////////////////////////////

const double finite_diff_dx_m4=3./840./grid_size,
 finite_diff_dx_m3=-32./840./grid_size,
 finite_diff_dx_m2=168./840./grid_size,
 finite_diff_dx_m1=-672./840./grid_size,
 finite_diff_dx_0=0.,
 finite_diff_dx_p1=672./840./grid_size,
 finite_diff_dx_p2=-168./840./grid_size,
 finite_diff_dx_p3=32./840./grid_size,
 finite_diff_dx_p4=-3./840./grid_size;

#define DY_MID_FACTOR(X)        psi[i-4][j].X*finite_diff_dx_m4 + psi[i-3][j].X*finite_diff_dx_m3 + psi[i-2][j].X*finite_diff_dx_m2 + psi[i-1][j].X*finite_diff_dx_m1 + psi[i+1][j].X*finite_diff_dx_p1 + psi[i+2][j].X*finite_diff_dx_p2 + psi[i+3][j].X*finite_diff_dx_p3 + psi[i+4][j].X*finite_diff_dx_p4
#define DX_MID_FACTOR(X)        psi[i][j-4].X*finite_diff_dx_m4 + psi[i][j-3].X*finite_diff_dx_m3 + psi[i][j-2].X*finite_diff_dx_m2 + psi[i][j-1].X*finite_diff_dx_m1 + psi[i][j+1].X*finite_diff_dx_p1 + psi[i][j+2].X*finite_diff_dx_p2 + psi[i][j+3].X*finite_diff_dx_p3 + psi[i][j+4].X*finite_diff_dx_p4

#define DX_MID_FACTOR_temp(X)   temp[i][j-4].X*finite_diff_dx_m4 + temp[i][j-3].X*finite_diff_dx_m3 + temp[i][j-2].X*finite_diff_dx_m2 + temp[i][j-1].X*finite_diff_dx_m1 + temp[i][j+1].X*finite_diff_dx_p1 + temp[i][j+2].X*finite_diff_dx_p2 + temp[i][j+3].X*finite_diff_dx_p3 + temp[i][j+4].X*finite_diff_dx_p4
/*
static void px_hat(MKL_Complex16** psi, MKL_Complex16** result){
    for(int i=4;i< x_n-4;i++){
        for(int j=4;j < x_n-4;j++){
            result[i][j].imag=-(DX_MID_FACTOR(real)+px_factor[i][j]*psi[i][j].real);
            result[i][j].real=(DX_MID_FACTOR(imag)+px_factor[i][j]*psi[i][j].imag);
        }
    }
}
static void py_hat(MKL_Complex16** psi, MKL_Complex16** result){
    for(int i=4;i< x_n-4;i++){
        for(int j=4;j < x_n-4;j++){
            result[i][j].imag=-(DY_MID_FACTOR(real)+py_factor[i][j]*psi[i][j].real);
            result[i][j].real=(DY_MID_FACTOR(imag)+py_factor[i][j]*psi[i][j].imag);
        }
    }
}
*/
static void __attribute__((hot)) px_hat_relative(const MKL_Complex16*const* psi, MKL_Complex16** result, double px_expect, double factor){
    double shift = -px_expect;
        for(int i=4;i< x_n-4;i++){
            for(int j=4;j < x_n-4;j++){
                result[i][j].imag=(-(DX_MID_FACTOR(real)+world.px_factor[i][j]*psi[i][j].real) + shift*psi[i][j].imag)*factor;
                result[i][j].real=((DX_MID_FACTOR(imag)+world.px_factor[i][j]*psi[i][j].imag) + shift*psi[i][j].real)*factor;
            }
            }
}
static void __attribute__((hot)) py_hat_relative(const MKL_Complex16*const* psi, MKL_Complex16** result, double py_expect, double factor){
    double shift = -py_expect;
        for(int i=4;i< x_n-4;i++){
            for(int j=4;j< x_n-4;j++){
                result[i][j].imag=(-(DY_MID_FACTOR(real)+world.py_factor[i][j]*psi[i][j].real) + shift*psi[i][j].imag)*factor;
                result[i][j].real=((DY_MID_FACTOR(imag)+world.py_factor[i][j]*psi[i][j].imag) + shift*psi[i][j].real)*factor;
            }
        }
}

static void __attribute__((hot)) px_py_expect(const MKL_Complex16*const* psi, double* px_mean, double* py_mean){
    *px_mean = 0.;
    *py_mean = 0.;
    double px_real, px_imag, py_real, py_imag;
    for(int i=4;i< x_n-4;i++){
        for(int j=4;j < x_n-4;j++){
            px_real = (DX_MID_FACTOR(imag)+world.px_factor[i][j]*psi[i][j].imag);
            px_imag = -(DX_MID_FACTOR(real)+world.px_factor[i][j]*psi[i][j].real);
            *px_mean = (*px_mean)+(psi[i][j].real * px_real + psi[i][j].imag * px_imag) * world.measure[i][j];

            py_real = (DY_MID_FACTOR(imag)+world.py_factor[i][j]*psi[i][j].imag);
            py_imag = -(DY_MID_FACTOR(real)+world.py_factor[i][j]*psi[i][j].real);
            *py_mean = (*py_mean)+(psi[i][j].real * py_real + psi[i][j].imag * py_imag) * world.measure[i][j];
        }
    }
    *px_mean = (*px_mean)*grid_size*grid_size;
    *py_mean = (*py_mean)*grid_size*grid_size;
}
static PyObject* Py_px_py_expect(PyObject *self, PyObject *args){
    PyObject* temp;
    PyArrayObject* state;
    if (!PyArg_ParseTuple(args, "O", &temp)){
        PyErr_SetString(PyExc_TypeError, "The input does not match the required input signature (state (numpy array))");
        return NULL;
    }
    PyArray_OutputConverter(temp, &state);
    if (!state) {PyErr_SetString(PyExc_TypeError, "The input state cannot be identified as a Numpy array"); return NULL;}
    if (check_type(state)!=0) return NULL;
    // get the c array from numpy (complex128 is stored as successive float64 two by two)
    const MKL_Complex16* psi[x_n];
    for(int i=0; i<x_n; i++){
        psi[i] =  (MKL_Complex16*) PyArray_GETPTR2(state, i, 0);
    }
    double px_mean, py_mean;
    px_py_expect(psi, &px_mean, &py_mean);
    return Py_BuildValue("dd", px_mean, py_mean);
}


static inline void compute_x_or_p_relative_state(const MKL_Complex16*const* psi, char x_or_p, double px_mean, double py_mean, MKL_Complex16** result){
    if(x_or_p=='x'){x_hat_relative(psi,result,sqrt(k/2.));}else{
        if(x_or_p=='y'){y_hat_relative(psi,result,sqrt(k/2.));}else{
            if(x_or_p=='p'){px_hat_relative(psi,result,px_mean,1./sqrt(2.*I));}else{
                if(x_or_p=='q'){py_hat_relative(psi,result,py_mean,1./sqrt(2.*I));}else{assert(false);}
            }
        }
    }
}

static int check_type(PyArrayObject* state){
    if(PyArray_NDIM(state)!=2){
        PyErr_SetString(PyExc_ValueError, "The state array is not two-dimensional");
        return -1;
    }else{
    if(PyArray_SHAPE(state)[0]!=x_n || PyArray_SHAPE(state)[1]!=x_n){
        PyErr_SetString(PyExc_ValueError, ("The state array does not match the required size "+std::to_string(x_n)).c_str());
        return -1;
    }else{
        PyArray_Descr* descr = PyArray_DESCR(state);
        if(descr->type_num!=NPY_COMPLEX128){
            PyErr_SetString(PyExc_ValueError, "The state array does not match the required datatype: Complex128");
            return -1;
        }        
        return 0;
    }
    }
}
static int check_type_real(PyArrayObject* state){
    if(PyArray_NDIM(state)!=2){
        PyErr_SetString(PyExc_ValueError, "The state array is not two-dimensional");
        return -1;
    }else{
    if(PyArray_SHAPE(state)[0]!=x_n || PyArray_SHAPE(state)[1]!=x_n){
        PyErr_SetString(PyExc_ValueError, ("The state array does not match the required size "+std::to_string(x_n)).c_str());
        return -1;
    }else{
        PyArray_Descr* descr = PyArray_DESCR(state);
        if(descr->type_num!=NPY_FLOAT64){
            PyErr_SetString(PyExc_ValueError, "The state array does not match the required datatype: Float64");
            return -1;
        }        
        return 0;
    }
    }
}

static int check_type_real_vector(PyArrayObject* state, int length){
    if(PyArray_NDIM(state)!=1){
        PyErr_SetString(PyExc_ValueError, "The output vector is not one-dimensional");
        return -1;
    }else{
    if(PyArray_SHAPE(state)[0]!=length){
        PyErr_SetString(PyExc_ValueError, ("The output vector does not match the required size "+std::to_string(length)).c_str());
        return -1;
    }else{
        PyArray_Descr* descr = PyArray_DESCR(state);
        if(descr->type_num!=NPY_FLOAT64){
            PyErr_SetString(PyExc_ValueError, "The output vector does not match the required datatype: Float64");
            return -1;
        }        
        return 0;
    }
    }
}

static double __attribute__((hot)) inner_product_real(const MKL_Complex16*const* psi1, const MKL_Complex16*const* psi2){ // computing the real part of the inner product between psi1 and psi2
    double result =0.;
    for(int i=4;i<x_n-4;i++){
    for(int j=4;j<x_n-4;j++){
        result += world.measure[i][j]*(psi1[i][j].real*psi2[i][j].real + psi1[i][j].imag*psi2[i][j].imag);
        }}
    result *=(grid_size*grid_size);
    return result;
}
static int factorial(int i){
    if(i<1){
        return 1;
    }else{
        int result = 1;
        for(int j=1;j<=i;j++){
            result *= j;
        }
        return result;
    }
}
static int combination_w_repetition(int items, int choices){
    return factorial(items+choices-1)/(factorial(choices)*factorial(items-1));
}
static int Hamming_dist(const int* array1, const int* array2){
    int result = 0;
    for(int i=0;i<number_of_variables;i++) result += abs(array1[i]-array2[i]);
    return result;
}
static void find_difference_and_compute_one_more_order(int index1, int index2, double px_mean, double py_mean){
    for(int i=0; i<number_of_variables; i++){
        if(world.indices_of_matrix_for_computing_moments[index1][i] == world.indices_of_matrix_for_computing_moments[index2][i]-1){
            if(i==0){
                compute_x_or_p_relative_state(world.complex_matrix_storage[index1], 'x', px_mean, py_mean, world.complex_matrix_storage[index2]);
            }else{if(i==1){
                compute_x_or_p_relative_state(world.complex_matrix_storage[index1], 'y', px_mean, py_mean, world.complex_matrix_storage[index2]);
            }else{if(i==2){
                compute_x_or_p_relative_state(world.complex_matrix_storage[index1], 'p', px_mean, py_mean, world.complex_matrix_storage[index2]);
            }else{
                compute_x_or_p_relative_state(world.complex_matrix_storage[index1], 'q', px_mean, py_mean, world.complex_matrix_storage[index2]);
            }}}break;
        }
    }
}
static bool indices_match(int index1, int index2, int target_combination_index){
    bool success = (world.indices_of_matrix_for_computing_moments[index1][0]+world.indices_of_matrix_for_computing_moments[index2][0] == world.indices_of_matrix_for_computing_moments[target_combination_index][0]);
    for(int i=1; i< number_of_variables; i++){
        success = (success && (world.indices_of_matrix_for_computing_moments[index1][i]+world.indices_of_matrix_for_computing_moments[index2][i] == world.indices_of_matrix_for_computing_moments[target_combination_index][i]));
    }
    return success;
}
static int compute_statistics(const MKL_Complex16*const* psi, double* data){
    // data are arranged in the order of:
    // <x>, <p>
    // centered moments -- <xx>, Re<xp>, <pp>, <xxx>, Re<xxp>, Re<xpp>, etc.

    // calculate <x>, <p> and prepare (x_hat-<x>) and (p_hat-<p>)
    set_x_y_relative_data(psi);
    double x_mean, y_mean;
    x_y_expect(psi, &x_mean, &y_mean);
    data[0] = x_mean*sqrt(k/2.), data[1] = y_mean*sqrt(k/2.);
    double px_mean, py_mean;
    px_py_expect(psi, &px_mean, &py_mean);
    data[2] = px_mean/sqrt(2.*I), data[3] = py_mean/sqrt(2.*I);
    compute_x_or_p_relative_state(psi, 'x', px_mean, py_mean, world.complex_matrix_storage[0]);
    compute_x_or_p_relative_state(psi, 'y', px_mean, py_mean, world.complex_matrix_storage[1]);
    compute_x_or_p_relative_state(psi, 'p', px_mean, py_mean, world.complex_matrix_storage[2]);
    compute_x_or_p_relative_state(psi, 'q', px_mean, py_mean, world.complex_matrix_storage[3]);
    int previous_order_start_index = 0;
    bool found;
    int current_order_start_index;
    for(int o = 2; o<= moment_order-2; o++){
        current_order_start_index = world.matrix_index_for_order_i_moment[o-2];
        for(int index2=current_order_start_index; index2<world.matrix_index_for_order_i_moment[o-1]; index2++){
            found = false;
            for(int index1=previous_order_start_index; index1<current_order_start_index; index1++){
                if(Hamming_dist(world.indices_of_matrix_for_computing_moments[index1], world.indices_of_matrix_for_computing_moments[index2])==1){
                    find_difference_and_compute_one_more_order(index1, index2, px_mean, py_mean);
                    found = true;
                    break;
                }
            }assert(found);
        }
        previous_order_start_index = current_order_start_index;
    }
    // put moments below 3 into the data array
    for(int i = number_of_variables; i<world.matrix_index_for_order_i_moment[moment_order-3]; i++){
        data[i] = inner_product_real(psi, world.complex_matrix_storage[i]);
    }
    // put 4th moments into the data array
    bool found2, found1;
    for(int combination_index=world.matrix_index_for_order_i_moment[moment_order-3]; combination_index<world.matrix_index_for_order_i_moment[moment_order-2]; combination_index++){
        found2 = false;
        for(int index2=world.matrix_index_for_order_i_moment[moment_order-4]; index2<world.matrix_index_for_order_i_moment[moment_order-3]; index2++){
            if(Hamming_dist(world.indices_of_matrix_for_computing_moments[index2], world.indices_of_matrix_for_computing_moments[combination_index])==1){
                found2 = true;
                found1 = false;
                for(int index1=0; index1<world.matrix_index_for_order_i_moment[moment_order-5]; index1++){
                    if(indices_match(index2, index1, combination_index)){
                        data[combination_index] = inner_product_real(world.complex_matrix_storage[index1], world.complex_matrix_storage[index2]);
                        found1 = true;
                        break;
                    }
                }assert(found1);
                break;
            }
        }assert(found2);
    }
    // put 5th moments into the data array
    for(int combination_index=world.matrix_index_for_order_i_moment[moment_order-2]; combination_index<world.matrix_index_for_order_i_moment[moment_order-1]; combination_index++){
        found2 = false;
        for(int index2=world.matrix_index_for_order_i_moment[moment_order-4]; index2<world.matrix_index_for_order_i_moment[moment_order-3]; index2++){
            if(Hamming_dist(world.indices_of_matrix_for_computing_moments[index2], world.indices_of_matrix_for_computing_moments[combination_index])==2){
                found2 = true;
                found1 = false;
                for(int index1=world.matrix_index_for_order_i_moment[moment_order-5]; index1<world.matrix_index_for_order_i_moment[moment_order-4]; index1++){
                    if(indices_match(index2, index1, combination_index)){
                        data[combination_index] = inner_product_real(world.complex_matrix_storage[index1], world.complex_matrix_storage[index2]);
                        found1 = true;
                        break;
                    }
                }assert(found1);
                break;
            }
        }assert(found2);
    }
    // clear the cache (the function "go_one_step" only uses matrix cache up to the 24th matrix)
    for(int m=0;m<24;m++){for(int i=4;i<x_n-4;i++){for(int j=4;j<x_n-4;j++){world.complex_matrix_storage[m][i][j].real=0.;world.complex_matrix_storage[m][i][j].imag=0.;}}}
    // post-processing
    double sigma_x=sqrt(data[4]), sigma_y=sqrt(data[6]), sigma_px=sqrt(data[9]), sigma_py=sqrt(data[13]);
    for(int combination_index=4; combination_index<world.matrix_index_for_order_i_moment[moment_order-1]; combination_index++){
        if(combination_index<14 && (combination_index==4 || combination_index==6 || combination_index==9 || combination_index==13)) continue;
        for(int i=0; i<number_of_variables; i++){
            for(int o=0; o<world.indices_of_matrix_for_computing_moments[combination_index][i]; o++){
                data[combination_index] = i==0? data[combination_index]/sigma_x : (i==1? data[combination_index]/sigma_y : (i==2? data[combination_index]/sigma_px : data[combination_index]/sigma_py));
            }
        }
    }
    // change the kurtosis into excess kurtosis
    data[34] -= 3.; data[38] -= 3.; data[48] -= 3.; data[68] -= 3.;
    // remove some constant factors in something like Gaussian (x^2*y^2)
    data[36] -= 1.; data[45] -= 1.; data[59] -= 1.; data[64] -= 1.;
    // remove the apparent non-hermiticity of p_x^2 * x^2
    data[43] += 1.; data[61] += 1.;
    return 0;
}
static PyObject* get_moments(PyObject *self, PyObject *args){
    PyObject* temp1, *temp2;
    PyArrayObject* state, *data_np;
    if (!PyArg_ParseTuple(args, "OO", &temp1, &temp2)){
        PyErr_SetString(PyExc_TypeError, "The input does not match the required signature (state array (complex), moment data array (float))");
        return NULL;
    }
    PyArray_OutputConverter(temp1, &state);
    if (!state) {
        PyErr_SetString(PyExc_TypeError, "The input state (argument 1) cannot be identified as a Numpy array");
        return NULL;
    }
    PyArray_OutputConverter(temp2, &data_np);
    if (!data_np) {
        PyErr_SetString(PyExc_TypeError, "The input moment data array (argument 2) cannot be identified as a Numpy array");
        return NULL;
    }
    if (check_type(state)!=0) return NULL;
    if (check_type_real_vector(data_np, world.matrix_index_for_order_i_moment[moment_order-1])!=0) return NULL;
    const MKL_Complex16* psi[x_n];
    for(int i=0; i<x_n; i++){
        psi[i] =  (MKL_Complex16*) PyArray_GETPTR2(state, i, 0);
    }
    double* data;
    data = (double*) PyArray_GETPTR1(data_np, 0);
    if (compute_statistics(psi, data)!=0) return NULL;
    Py_RETURN_NONE;
}



// 7 points estimation
/*
const double finite_diff_dx2_m3=2./180./grid_size/grid_size,
 finite_diff_dx2_m2=-27./180./grid_size/grid_size,
 finite_diff_dx2_m1=270./180./grid_size/grid_size,
 finite_diff_dx2_0=-490./180./grid_size/grid_size,
 finite_diff_dx2_p1=270./180./grid_size/grid_size,
 finite_diff_dx2_p2=-27./180./grid_size/grid_size,
 finite_diff_dx2_p3=2./180./grid_size/grid_size;

*/

// 9 points estimation
const double finite_diff_dx2_m4=-9./5040./grid_size/grid_size,
 finite_diff_dx2_m3=128./5040./grid_size/grid_size,
 finite_diff_dx2_m2=-1008./5040./grid_size/grid_size,
 finite_diff_dx2_m1=8064./5040./grid_size/grid_size,
 finite_diff_dx2_0=-14350./5040./grid_size/grid_size,
 finite_diff_dx2_p1=8064./5040./grid_size/grid_size,
 finite_diff_dx2_p2=-1008./5040./grid_size/grid_size,
 finite_diff_dx2_p3=128./5040./grid_size/grid_size,
 finite_diff_dx2_p4=-9./5040./grid_size/grid_size;


#define DY_2_MID_FACTOR(X)        psi[i-4][j].X*finite_diff_dx2_m4 + psi[i-3][j].X*finite_diff_dx2_m3 + psi[i-2][j].X*finite_diff_dx2_m2 + psi[i-1][j].X*finite_diff_dx2_m1 + psi[i][j].X*finite_diff_dx2_0 + psi[i+1][j].X*finite_diff_dx2_p1 + psi[i+2][j].X*finite_diff_dx2_p2 + psi[i+3][j].X*finite_diff_dx2_p3 + psi[i+4][j].X*finite_diff_dx2_p4
#define DX_2_MID_FACTOR(X)        psi[i][j-4].X*finite_diff_dx2_m4 + psi[i][j-3].X*finite_diff_dx2_m3 + psi[i][j-2].X*finite_diff_dx2_m2 + psi[i][j-1].X*finite_diff_dx2_m1 + psi[i][j].X*finite_diff_dx2_0 + psi[i][j+1].X*finite_diff_dx2_p1 + psi[i][j+2].X*finite_diff_dx2_p2 + psi[i][j+3].X*finite_diff_dx2_p3 + psi[i][j+4].X*finite_diff_dx2_p4

double potential[x_n][x_n];

static void __attribute__((hot)) Hamiltonian_hat(MKL_Complex16** psi, MKL_Complex16** result, MKL_Complex16** temp, bool to_multiply_minus_i){
    // caches
    double x, y, y_2, x_2, y_mul_Q_z;
    double dx2_factor, dy2_factor, dx_factor, dy_factor;
    double cache, potential_cache;
    const double inertia_factor = 1./(2.*I);
    MKL_Complex16 dx, dy, dx2, dy2;

    int i=0, j=0;
    // assuming that the data is constantly zero for i,j <= 3 and i,j >= x_n-4
    if (!to_multiply_minus_i){
        for(i=y_start;i<y_end;i++){
            y = world.x[i];
            //y_mul_2_div_3 = 2./3.*y;
            y_2 = y*y;
            //y_2_div_3_plus_1 = 1.+1./3.*y_2;
            y_mul_Q_z = y*Q_z;
            for(j=x_start;j<x_end;j++){
                x = world.x[j];
                x_2 = x*x;
                dx.real = DX_MID_FACTOR(real); dx.imag = DX_MID_FACTOR(imag);
                dx2.real = DX_2_MID_FACTOR(real); dx2.imag = DX_2_MID_FACTOR(imag);
                dy.real = DY_MID_FACTOR(real); dy.imag = DY_MID_FACTOR(imag);
                dy2.real = DY_2_MID_FACTOR(real); dy2.imag = DY_2_MID_FACTOR(imag);
                // take the cache of dy
                temp[i][j].real = dy.real; temp[i][j].imag = dy.imag;
                // all terms in ther result except -(-2xy/3 dx dy)
                // potential first, kinetics second. Note that Q_z = -i d\theta, and d\theta = i Q_z
                cache = world.two_g_div_beta_plus_g_2[i][j];
                dx2_factor = 1. + y_2*cache; dy2_factor = 1. + x_2*cache;
                cache = world.g_div_beta_plus_tan_half_beta_div_beta_plus_g_2[i][j];
                dx_factor = -cache*x; dy_factor = -cache*y;
                cache = world.one_plus_g_beta_mul_2tan_half_beta_div_beta[i][j];
                potential_cache = potential[i][j];
                result[i][j].real = potential_cache*psi[i][j].real - inertia_factor*(dx2_factor*dx2.real + dy2_factor*dy2.real + y_mul_Q_z*cache*dx.imag - x*Q_z*cache*dy.imag + dx_factor*dx.real + dy_factor*dy.real);
                result[i][j].imag = potential_cache*psi[i][j].imag - inertia_factor*(dx2_factor*dx2.imag + dy2_factor*dy2.imag - y_mul_Q_z*cache*dx.real + x*Q_z*cache*dy.real + dx_factor*dx.imag + dy_factor*dy.imag);
            }
            }
        double y_factor;
        for(i=y_start;i<y_end;i++){
            y = world.x[i];
            y_factor = inertia_factor*2.*y;
            for(j=x_start;j<x_end;j++){
                x = world.x[j];
                // the term -(-2xy/3 dx dy)
                cache = world.two_g_div_beta_plus_g_2[i][j];
                result[i][j].real += x*y_factor*cache*(DX_MID_FACTOR_temp(real));
                result[i][j].imag += x*y_factor*cache*(DX_MID_FACTOR_temp(imag));
            }
            }
            
    }else{
        // here we simultanesouly multiply it by a (-i) factor
        for(i=y_start;i<y_end;i++){
            y = world.x[i];
            //y_mul_2_div_3 = 2./3.*y;
            y_2 = y*y;
            //y_2_div_3_plus_1 = 1.+1./3.*y_2;
            y_mul_Q_z = y*Q_z;
            for(j=x_start;j<x_end;j++){
                x = world.x[j];
                x_2 = x*x;
                dx.real = DX_MID_FACTOR(real); dx.imag = DX_MID_FACTOR(imag);
                dx2.real = DX_2_MID_FACTOR(real); dx2.imag = DX_2_MID_FACTOR(imag);
                dy.real = DY_MID_FACTOR(real); dy.imag = DY_MID_FACTOR(imag);
                dy2.real = DY_2_MID_FACTOR(real); dy2.imag = DY_2_MID_FACTOR(imag);
                // take the cache of dy
                temp[i][j].real = dy.real; temp[i][j].imag = dy.imag;
                // all terms in ther result except -(-2xy/3 dx dy)
                // potential first, kinetics second. Note that Q_z = -i d\theta, and d\theta = i Q_z
                cache = world.two_g_div_beta_plus_g_2[i][j];
                dx2_factor = 1. + y_2*cache; dy2_factor = 1. + x_2*cache;
                cache = world.g_div_beta_plus_tan_half_beta_div_beta_plus_g_2[i][j];
                dx_factor = -cache*x; dy_factor = -cache*y;
                cache = world.one_plus_g_beta_mul_2tan_half_beta_div_beta[i][j];
                potential_cache = potential[i][j];
                result[i][j].real = potential[i][j]*psi[i][j].imag - inertia_factor*(dx2_factor*dx2.imag + dy2_factor*dy2.imag - y_mul_Q_z*cache*dx.real + x*Q_z*cache*dy.real + dx_factor*dx.imag + dy_factor*dy.imag);
                result[i][j].imag = -potential[i][j]*psi[i][j].real + inertia_factor*(dx2_factor*dx2.real + dy2_factor*dy2.real + y_mul_Q_z*cache*dx.imag - x*Q_z*cache*dy.imag + dx_factor*dx.real + dy_factor*dy.real);
            }
            }
        double y_factor;
        for(i=y_start;i<y_end;i++){
            y = world.x[i];
            y_factor = inertia_factor*2.*y;
            for(j=x_start;j<x_end;j++){
                x = world.x[j];
                // the term -(-2xy/3 dx dy)
                cache = world.two_g_div_beta_plus_g_2[i][j];
                result[i][j].real += (x*y_factor*cache*(DX_MID_FACTOR_temp(imag)));
                result[i][j].imag -= (x*y_factor*cache*(DX_MID_FACTOR_temp(real)));
            }
            }
            
    }
            
}
static void __attribute__((hot)) set_potential_data(double displacement_x_, double displacement_y_){
    double x, y;
    double E_x_=displacement_x_*E_z, E_y_=displacement_y_*E_z;

    double x_2, y_2, beta_2, beta_4;
    //double sin_beta_div_beta_2;
    //double beta, sin_beta, cos_beta, sin_beta_div_beta;
    double y_E_y, x_E_x_plus_y_E_y;

    displacement_x = displacement_x_; displacement_y = displacement_y_;
    E_x = E_x_; E_y = E_y_;

    for(int i=0; i<x_n;i++){
        y = world.x[i];
        y_2 = y*y;
        y_E_y = y * E_y_;
        for(int j=0; j<x_n;j++){
            x = world.x[j]; 
            x_2 = x*x;
            x_E_x_plus_y_E_y = x * E_x_ + y_E_y;
            beta_2 = x_2 + y_2;
            beta_4 = beta_2 * beta_2;
            // we calculate (\frac{\sin\beta}{\beta})^2 up to the 4-th order term //(1. - 1./3.*beta_2 + 2./45.*beta_4);
            //if (beta_2 != 0.){
            //    sin_beta_div_beta_2 = 1. - 1./3.*beta_2 + 2./45.*beta_4; 
            //}else{
            //    sin_beta_div_beta_2 = 1.;
            //}
            
            potential[i][j]= world.sin_beta_div_beta_2[i][j]*( k_div_2 *  beta_2 - x_E_x_plus_y_E_y*x_E_x_plus_y_E_y) - world.sin_2beta_div_beta[i][j]*E_z*x_E_x_plus_y_E_y \
                + Q_z_2_div_2I* world.tan_half_beta_2[i][j]; // the second last term is the main force, comparable to the trapping potential
            // the coefficient of the second last term is 2\sin\beta\cos\beta / \beta = \sin 2\beta / \beta //(2.-4./3.*beta_2 + 4./15.*beta_4)
    }
    }
}

static PyObject* Py_copy_matrix(PyObject *self, PyObject *args){
    PyObject* temp;
    PyArrayObject* array;
    if (!PyArg_ParseTuple(args, "O", &temp)){
        PyErr_SetString(PyExc_TypeError, "The input does not match the required input signature (state (numpy array))");
        return NULL;
    }
    PyArray_OutputConverter(temp, &array);
    if (!array) {PyErr_SetString(PyExc_TypeError, "The input state cannot be identified as a Numpy array"); return NULL;}
    if (check_type_real(array)!=0) return NULL;
    // get the c array from numpy (complex128 is stored as successive float64 two by two)
    double* data[x_n];
    for(int i=0; i<x_n; i++){
        data[i] =  (double*) PyArray_GETPTR2(array, i, 0);
    }
    for(int i=0; i<x_n; i++){
        for(int j=0; j<x_n; j++){
            data[i][j]=world.px_factor[i][j];
        }
    }
    Py_RETURN_NONE;
}

static double __attribute__((hot)) energy_expect(MKL_Complex16** state, MKL_Complex16** temp1, MKL_Complex16** temp2){
    set_potential_data(0., 0.);
    Hamiltonian_hat(state, temp1, temp2, false);
    return inner_product_real(state, temp1);
}
static PyObject* Py_energy_expect(PyObject *self, PyObject *args){
    PyObject* temp;
    PyArrayObject* state;
    if (!PyArg_ParseTuple(args, "O", &temp)){
        PyErr_SetString(PyExc_TypeError, "The input does not match the required input signature (state (numpy array))");
        return NULL;
    }
    PyArray_OutputConverter(temp, &state);
    if (!state) {PyErr_SetString(PyExc_TypeError, "The input state cannot be identified as a Numpy array"); return NULL;}
    if (check_type(state)!=0) return NULL;
    // get the c array from numpy (complex128 is stored as successive float64 two by two)
    MKL_Complex16* psi[x_n];
    for(int i=0; i<x_n; i++){
        psi[i] =  (MKL_Complex16*) PyArray_GETPTR2(state, i, 0);
    }
    return Py_BuildValue("d", energy_expect(psi, world.complex_matrix_storage[0], world.complex_matrix_storage[1]));
}


// carry out one step calculation
static double __attribute__((hot)) go_one_step(MKL_Complex16** psi, double dt, double _gamma, MKL_Complex16**const* allocated_memory);
static double check_boundary_error(MKL_Complex16** psi, int* Fail, double**const* real_matrix_storage);

static PyObject* step(PyObject *self, PyObject *args){
    double dt, control_x, control_y, _gamma;
    PyObject* temp;
    PyArrayObject* state;
    if (!PyArg_ParseTuple(args, "Odddd", &temp, &dt, &control_x, &control_y, &_gamma)){
        PyErr_SetString(PyExc_TypeError, "The input does not match the required input signature (state (numpy array), dt (double), control_x (double), control_y (double), \\gamma (double)");
        return NULL;
    }
    PyArray_OutputConverter(temp, &state);
    if (!state) {
        PyErr_SetString(PyExc_TypeError, "The input object cannot be identified as a Numpy array");
        return NULL;
    }
    // check the shape and datatype of the array, so that an error will be raised if the shape is not consistent with this c module, avoiding a segmentation fault
    if (check_type(state)==-1) return NULL;

    // get the c array from numpy (complex128 is stored as successive float64 two by two)
    MKL_Complex16* psi[x_n];
    for(int i=0; i<x_n; i++){psi[i] = (MKL_Complex16*) PyArray_GETPTR2(state, i, 0);}
    // update the cache of dt and force
    if ((control_x != displacement_x) || (control_y != displacement_y)){
        set_potential_data(control_x, control_y);
    }
    double norm;
    int Fail_norm=0;
    norm=go_one_step(psi, dt, _gamma, world.complex_matrix_storage);
    if(norm==0.)Fail_norm=1;
    
    int Fail=0; // not necessarily used
    double boundary_prob = check_boundary_error(psi, &Fail, world.real_matrix_storage);//check_boundary_error(psi, &Fail, world.real_matrix_storage);
    return Py_BuildValue("id", Fail_norm, boundary_prob); 
}
static PyObject* simulate_n_steps(PyObject *self, PyObject *args){
    double dt, control_x, control_y, _gamma;
    int n_step;
    PyObject* temp;
    PyArrayObject* state;
    if (!PyArg_ParseTuple(args, "Oddddi", &temp, &dt, &control_x, &control_y, &_gamma, &n_step)){
        PyErr_SetString(PyExc_TypeError, "The input does not match the required input signature (state (numpy array), dt (double), control_x (double), control_y (double), \\gamma (double), number of steps (int)");
        return NULL;
    }
    PyArray_OutputConverter(temp, &state);
    if (!state) {
        PyErr_SetString(PyExc_TypeError, "The input object cannot be identified as a Numpy array");
        return NULL;
    }
    // check the shape and datatype of the array, so that an error will be raised if the shape is not consistent with this c module, avoiding a segmentation fault
    if (check_type(state)==-1) return NULL;

    // get the c array from numpy (complex128 is stored as successive float64 two by two)
    MKL_Complex16* psi[x_n];
    for(int i=0; i<x_n; i++){psi[i] = (MKL_Complex16*) PyArray_GETPTR2(state, i, 0);}
    // update the cache of dt and force
    if ((control_x != displacement_x) || (control_y != displacement_y)){
        set_potential_data(control_x, control_y);
    }
    double norm;
    int Fail_norm=0;
    for(int i=0; i<n_step; i++){
        norm = go_one_step(psi, dt, _gamma, world.complex_matrix_storage);
        if(norm==0.)Fail_norm=1;
    }

    int Fail=0; // not necessarily used
    double boundary_prob = check_boundary_error(psi, &Fail, world.real_matrix_storage);
    return Py_BuildValue("id", Fail_norm, boundary_prob); 
}


static void __attribute__((hot)) D1(MKL_Complex16** state, const double &gamma, MKL_Complex16** result, MKL_Complex16** temp){
    // compute the deterministic Hamiltonian term, stored in result
    Hamiltonian_hat(state, result, temp, true);
    // add the deterministic squeezing term
    xy_squeezing_term__add_into_result(state, gamma, result);
}
static void __attribute__((hot)) D1_DX_DY(MKL_Complex16** state, const double &gamma, MKL_Complex16** result, MKL_Complex16** result_x, MKL_Complex16** result_y, MKL_Complex16** temp){
    // compute the deterministic Hamiltonian term, stored in result
    Hamiltonian_hat(state, result, temp, true);
    // add the deterministic squeezing term
    xy_squeezing_term__add_into_result__and_get_DX_DY(state, gamma, result, result_x, result_y);
}
static void add(MKL_Complex16** m1, const double a, MKL_Complex16** m2, MKL_Complex16** out){
    for(int i=y_start;i<y_end;i++){
        for(int j=x_start;j < x_end;j++){
            out[i][j].real = m1[i][j].real + a * m2[i][j].real;
            out[i][j].imag = m1[i][j].imag + a * m2[i][j].imag;
        }
    }
}
static void __attribute__((hot)) compute_Runge_Kutta_terms(MKL_Complex16** state, const double dt, const double gamma, MKL_Complex16** Y1, MKL_Complex16** k2, MKL_Complex16** k3, MKL_Complex16** k4, MKL_Complex16** temp){
    set_x_y_relative_data(Y1);
    D1(Y1, gamma, k2, temp);
    add(state, dt/2., k2, Y1);
    MKL_Complex16** Y2=Y1;

    set_x_y_relative_data(Y2);
    D1(Y2, gamma, k3, temp);
    add(state, dt, k3, Y2);
    MKL_Complex16** Y3=Y2;

    set_x_y_relative_data(Y3);
    D1(Y3, gamma, k4, temp);
}
static void __attribute__((hot)) sum_Runge_Kutta(MKL_Complex16** state, const double dt, MKL_Complex16** k1, MKL_Complex16** k2, MKL_Complex16** k3, MKL_Complex16** k4){
    for(int i=y_start;i<y_end;i++){
        for(int j=x_start;j < x_end;j++){
            state[i][j].real += 1./6.*dt * (k1[i][j].real+2.*k2[i][j].real+2.*k3[i][j].real+k4[i][j].real);
            state[i][j].imag += 1./6.*dt * (k1[i][j].imag+2.*k2[i][j].imag+2.*k3[i][j].imag+k4[i][j].imag);
        }
    }   
}
static void __attribute__((hot)) keep_orders_above_2(MKL_Complex16** state, const double dt, MKL_Complex16** k1){
    for(int i=y_start;i<y_end;i++){
        for(int j=x_start;j < x_end;j++){
            k1[i][j].real = state[i][j].real - dt*k1[i][j].real;
            k1[i][j].imag = state[i][j].imag - dt*k1[i][j].imag;
        }
    }   
}
static void compute_Y_XY_plus_minus_and_Y_half(MKL_Complex16** state, const double dt, MKL_Complex16** D1_state, MKL_Complex16** DX_state, MKL_Complex16** DY_state, MKL_Complex16** Y_X_plus, MKL_Complex16** Y_X_minus \
 , MKL_Complex16** Y_Y_plus, MKL_Complex16** Y_Y_minus, MKL_Complex16** Y_half){
    double temp1, temp2;
    const double sqrt_dt = sqrt(dt);
    const double dt_div_2 = dt*0.5;
    for(int i=y_start;i<y_end;i++){
        for(int j=x_start;j < x_end;j++){
            temp1 = state[i][j].real + dt_div_2 * D1_state[i][j].real;
            temp2 = state[i][j].imag + dt_div_2 * D1_state[i][j].imag;
            Y_half[i][j].real = temp1;
            Y_half[i][j].imag = temp2;

            Y_X_plus[i][j].real = temp1 + sqrt_dt * DX_state[i][j].real;
            Y_X_plus[i][j].imag = temp2 + sqrt_dt * DX_state[i][j].imag;

            Y_X_minus[i][j].real = temp1 - sqrt_dt * DX_state[i][j].real;
            Y_X_minus[i][j].imag = temp2 - sqrt_dt * DX_state[i][j].imag;

            Y_Y_plus[i][j].real = temp1 + sqrt_dt * DY_state[i][j].real;
            Y_Y_plus[i][j].imag = temp2 + sqrt_dt * DY_state[i][j].imag;

            Y_Y_minus[i][j].real = temp1 - sqrt_dt * DY_state[i][j].real;
            Y_Y_minus[i][j].imag = temp2 - sqrt_dt * DY_state[i][j].imag;
        }
    }   
}
/*
static void change_Y_plus_minus_into_Phi_plus_minus(MKL_Complex16** Y_plus, const double dt, MKL_Complex16** D2_Y_plus, MKL_Complex16** Y_minus){
    //double temp1, temp2;
    const double sqrt_dt = sqrt(dt);
    for(int i=4;i<x_n-4;i++){
        for(int j=4;j<x_n-4;j++){
            Y_minus[i][j].real = Y_plus[i][j].real - sqrt_dt * D2_Y_plus[i][j].real;
            Y_minus[i][j].imag = Y_plus[i][j].imag - sqrt_dt * D2_Y_plus[i][j].imag;

            Y_plus[i][j].real  = Y_plus[i][j].real + sqrt_dt * D2_Y_plus[i][j].real;
            Y_plus[i][j].imag  = Y_plus[i][j].imag + sqrt_dt * D2_Y_plus[i][j].imag;
        }
    }   
}*/
static void simple_sum_up(MKL_Complex16** state, double dt, MKL_Complex16** DX_state, MKL_Complex16** DY_state, MKL_Complex16** D1_Y_half, \
     MKL_Complex16** DX_state_w_higher_orders, MKL_Complex16** DY_state_w_higher_orders, MKL_Complex16** D1_Y_X_plus, MKL_Complex16** D1_Y_X_minus, MKL_Complex16** D1_Y_Y_plus, MKL_Complex16** D1_Y_Y_minus, \
     MKL_Complex16** DX_Y_X_plus, MKL_Complex16** DX_Y_X_minus, MKL_Complex16** DX_Y_Y_plus, MKL_Complex16** DX_Y_Y_minus, MKL_Complex16** DY_Y_X_plus, MKL_Complex16** DY_Y_X_minus, MKL_Complex16** DY_Y_Y_plus, MKL_Complex16** DY_Y_Y_minus \
     //, MKL_Complex16** DX_Phi_XX_plus, MKL_Complex16** DX_Phi_XX_minus, MKL_Complex16** DY_Phi_YY_plus, MKL_Complex16** DY_Phi_YY_minus 
     );

static void __attribute__((hot)) sample_Ito_integrals(double* r, double dt);
double random_buffer[(1+Legendre_p)*2];
double r[12];

static int int_round(double x){return (int)(x < 0 ? (x - 0.5) : (x + 0.5));}

static double __attribute__((hot)) go_one_step(MKL_Complex16** psi, double dt, double _gamma, MKL_Complex16**const* allocated_memory){ 
    // sample random variables
    sample_Ito_integrals(r, dt); 

    // start calculation
    MKL_Complex16** D1_state = allocated_memory[0], **DX_state=allocated_memory[1], **DY_state=allocated_memory[2];
    MKL_Complex16 ** temp = allocated_memory[23];
    double x_mean, y_mean;
    x_y_expect(psi, &x_mean, &y_mean);
    x_start = std::max(4, int_round((x_mean - angle_threshold_to_neglect)/grid_size)+x_max);
    y_start = std::max(4, int_round((y_mean - angle_threshold_to_neglect)/grid_size)+x_max);
    x_end = std::min(x_n-4, int_round((x_mean + angle_threshold_to_neglect)/grid_size)+x_max+1);
    y_end = std::min(x_n-4, int_round((y_mean + angle_threshold_to_neglect)/grid_size)+x_max+1);
    if (x_start>=x_n-4 || y_start>=x_n-4 || y_end<=4 || x_end<=4){printf("x_start %d\nx_end %d\ny_start %d\ny_end %d\nx_mean %f\ny_mean %f\nnorm %f\n norm %f\n", x_start,x_end,y_start,y_end,x_mean,y_mean,norm(psi), normalize(psi));printf("we have problems\n");}
    
    set_x_y_relative_data(psi);
    D1_DX_DY(psi, _gamma, D1_state, DX_state, DY_state, temp);

    // initialize Y as a 1st order step forward from psi
    MKL_Complex16 **Y_X_plus=allocated_memory[3], **Y_X_minus=allocated_memory[4];
    MKL_Complex16 **Y_Y_plus=allocated_memory[5], **Y_Y_minus=allocated_memory[6];
    MKL_Complex16 **Y_half=allocated_memory[19];
    compute_Y_XY_plus_minus_and_Y_half(psi, dt, D1_state, DX_state, DY_state, Y_X_plus, Y_X_minus, Y_Y_plus, Y_Y_minus, Y_half);
    // prepare the Runge-Kutta terms for deterministic evolution 
    MKL_Complex16 **k2=allocated_memory[20], **k3=allocated_memory[21], **k4=allocated_memory[22];
    compute_Runge_Kutta_terms(psi, dt, _gamma, Y_half, k2, k3, k4, temp); 
    MKL_Complex16 **D1_Y_half = k2; // D1_state is k_1 in Runge Kutta 4
    // "Y_half" is no longer used starting from here

    MKL_Complex16 **D1_Y_X_plus=allocated_memory[7], **DX_Y_X_plus=allocated_memory[8], **DY_Y_X_plus=allocated_memory[9];
    set_x_y_relative_data(Y_X_plus);
    D1_DX_DY(Y_X_plus, _gamma, D1_Y_X_plus, DX_Y_X_plus, DY_Y_X_plus, temp);
    MKL_Complex16 **D1_Y_X_minus=allocated_memory[10], **DX_Y_X_minus=allocated_memory[11], **DY_Y_X_minus=allocated_memory[12];
    set_x_y_relative_data(Y_X_minus);
    D1_DX_DY(Y_X_minus, _gamma, D1_Y_X_minus, DX_Y_X_minus, DY_Y_X_minus, temp);
    MKL_Complex16 **D1_Y_Y_plus=allocated_memory[13], **DX_Y_Y_plus=allocated_memory[14], **DY_Y_Y_plus=allocated_memory[15];
    set_x_y_relative_data(Y_Y_plus);
    D1_DX_DY(Y_Y_plus, _gamma, D1_Y_Y_plus, DX_Y_Y_plus, DY_Y_Y_plus, temp);
    MKL_Complex16 **D1_Y_Y_minus=allocated_memory[16], **DX_Y_Y_minus=allocated_memory[17], **DY_Y_Y_minus=allocated_memory[18];
    set_x_y_relative_data(Y_Y_minus);
    D1_DX_DY(Y_Y_minus, _gamma, D1_Y_Y_minus, DX_Y_Y_minus, DY_Y_Y_minus, temp);
    /*
    //////////////////////////////////////////////////////////
    change_Y_plus_minus_into_Phi_plus_minus(Y_X_plus, dt, DX_Y_X_plus, Y_X_minus);
    MKL_Complex16** Phi_XX_plus=Y_X_plus, **Phi_XX_minus=Y_X_minus;
    change_Y_plus_minus_into_Phi_plus_minus(Y_Y_plus, dt, DY_Y_Y_plus, Y_Y_minus);
    MKL_Complex16** Phi_YY_plus=Y_Y_plus, **Phi_YY_minus=Y_Y_minus;

    MKL_Complex16 **DX_Phi_XX_plus=allocated_memory[24], **DX_Phi_XX_minus=allocated_memory[25];
    MKL_Complex16 **DY_Phi_YY_plus=allocated_memory[26], **DY_Phi_YY_minus=allocated_memory[27];
    //MKL_Complex16 **null1 = Y_half, **null2 = temp;
    set_x_y_relative_data(Phi_XX_plus);
    DX_DY(Phi_XX_plus, _gamma, DX_Phi_XX_plus, temp);
    set_x_y_relative_data(Phi_XX_minus);
    DX_DY(Phi_XX_minus, _gamma, DX_Phi_XX_minus, temp);
    set_x_y_relative_data(Phi_YY_plus);
    DX_DY(Phi_YY_plus, _gamma, temp, DY_Phi_YY_plus);
    set_x_y_relative_data(Phi_YY_minus);
    DX_DY(Phi_YY_minus, _gamma, temp, DY_Phi_YY_minus);                           
    ////////////////////////////////////////////////////////////
    */
    
    // sum Runge Kutta terms for deterministic time evolution (except for the succeeding terms behind the first term of L_0)
    sum_Runge_Kutta(psi, dt, D1_state, k2, k3, k4);
    // compute the noise terms adjusted by the higher order time evolutions
    // first we remove the first order in the result of the deterministic time evolution
    keep_orders_above_2(psi, dt, D1_state);
    MKL_Complex16 **adjusted_state = D1_state;
    // then we get the adjusted noise terms, which involve (deterministic time evolution order above 2)*b'
    set_x_y_relative_data(adjusted_state);
    DX_DY(adjusted_state, _gamma, k3, k4);
    MKL_Complex16 **DX_state_w_higher_orders = k3, **DY_state_w_higher_orders = k4;
    // the result is consistent with the algorithm because the numerical scheme only guarantees accuracy up to the order (deterministic time evolution order 1)*noise
    // however, it does not compute the real time evolution, because the noise does not commute with the Hamiltonian, 
    // so the commutators (H^(n-1)*p/m) are ignored (they are much smaller as can be confirmed numerically), and it assmues the noise always appears after the deterministic time evolution



    // sum them all
    simple_sum_up(psi, dt, DX_state, DY_state, D1_Y_half, DX_state_w_higher_orders, DY_state_w_higher_orders, D1_Y_X_plus, D1_Y_X_minus, D1_Y_Y_plus, D1_Y_Y_minus, DX_Y_X_plus, DX_Y_X_minus, DX_Y_Y_plus, DX_Y_Y_minus, \
     DY_Y_X_plus, DY_Y_X_minus, DY_Y_Y_plus, DY_Y_Y_minus \
     //,DX_Phi_XX_plus, DX_Phi_XX_minus, DY_Phi_YY_plus, DY_Phi_YY_minus 
     );
    
    if(x_start > 4){
        //printf("x_start %d\n", x_start);
        for(int i=4;i<x_n-4;i++){
        for(int j=4;j<x_start;j++){
            psi[i][j].real = 0.; psi[i][j].imag = 0.;
        }
    }  
    }
    if(x_end < x_n-4){
        //printf("x_end %d\n", x_end);
        for(int i=4;i<x_n-4;i++){
        for(int j=x_end;j<x_n-4;j++){
            psi[i][j].real = 0.; psi[i][j].imag = 0.;
        }
    }  
    }
    if(y_start > 4){
        //printf("y_start %d\n", y_start);
        for(int i=4;i<y_start;i++){
        for(int j=4;j<x_n-4;j++){
            psi[i][j].real = 0.; psi[i][j].imag = 0.;
        }
    }  
    }
    if(y_end < x_n-4){
        //printf("y_end %d\n", y_end);
        for(int i=y_end;i<x_n-4;i++){
        for(int j=4;j<x_n-4;j++){
            psi[i][j].real = 0.; psi[i][j].imag = 0.;
        }
    }  
    }   
    double _norm = normalize(psi);
    return _norm;
}
// pp.379
static void __attribute__((hot)) simple_sum_up(MKL_Complex16** state, double dt, MKL_Complex16** DX_state, MKL_Complex16** DY_state, MKL_Complex16** D1_Y_half, \
     MKL_Complex16** DX_state_w_higher_orders, MKL_Complex16** DY_state_w_higher_orders, MKL_Complex16** D1_Y_X_plus, MKL_Complex16** D1_Y_X_minus, MKL_Complex16** D1_Y_Y_plus, MKL_Complex16** D1_Y_Y_minus, \
     MKL_Complex16** DX_Y_X_plus, MKL_Complex16** DX_Y_X_minus, MKL_Complex16** DX_Y_Y_plus, MKL_Complex16** DX_Y_Y_minus, MKL_Complex16** DY_Y_X_plus, MKL_Complex16** DY_Y_X_minus, MKL_Complex16** DY_Y_Y_plus, MKL_Complex16** DY_Y_Y_minus \
     //, MKL_Complex16** DX_Phi_XX_plus, MKL_Complex16** DX_Phi_XX_minus, MKL_Complex16** DY_Phi_YY_plus, MKL_Complex16** DY_Phi_YY_minus 
     ){

    /*r[0]=I1;  r[1]=I2;
    r[2]=I01; r[3]=I02;
    r[4]=I10; r[5]=I20;
    r[6]=I11; r[7]=I22;
    r[8]=I12; r[9]=I21;
    r[10]=I111;r[11]=I222;*/
    double I1 = r[0], I2 = r[1], I01 = r[2], I02 = r[3], I10 = r[4], I20 = r[5], I11 = r[6], I22 = r[7], I12 = r[8], I21 = r[9];
    // double I111 = r[10], I222 = r[11];
    for(int i=y_start;i<y_end;i++){
        for(int j=x_start;j < x_end;j++){
            state[i][j].real += 0.25*dt*(D1_Y_X_plus[i][j].real+D1_Y_X_minus[i][j].real+D1_Y_Y_plus[i][j].real+D1_Y_Y_minus[i][j].real-4.*D1_Y_half[i][j].real) + DX_state_w_higher_orders[i][j].real*I1 + DY_state_w_higher_orders[i][j].real*I2 + \
                I10*(D1_Y_X_plus[i][j].real - D1_Y_X_minus[i][j].real) + I20*(D1_Y_Y_plus[i][j].real - D1_Y_Y_minus[i][j].real) + \
                I11*(DX_Y_X_plus[i][j].real - DX_Y_X_minus[i][j].real) + I22*(DY_Y_Y_plus[i][j].real - DY_Y_Y_minus[i][j].real) + \
                I12*(DY_Y_X_plus[i][j].real - DY_Y_X_minus[i][j].real) + I21*(DX_Y_Y_plus[i][j].real - DX_Y_Y_minus[i][j].real) + \
                I01*(DX_Y_X_plus[i][j].real + DX_Y_X_minus[i][j].real + DX_Y_Y_plus[i][j].real + DX_Y_Y_minus[i][j].real - 4.*DX_state[i][j].real) + \
                I02*(DY_Y_X_plus[i][j].real + DY_Y_X_minus[i][j].real + DY_Y_Y_plus[i][j].real + DY_Y_Y_minus[i][j].real - 4.*DY_state[i][j].real) \
                //+ I111*(DX_Phi_XX_plus[i][j].real - DX_Phi_XX_minus[i][j].real - DX_Y_X_plus[i][j].real + DX_Y_X_minus[i][j].real) 
                //+ I222*(DY_Phi_YY_plus[i][j].real - DY_Phi_YY_minus[i][j].real - DY_Y_Y_plus[i][j].real + DY_Y_Y_minus[i][j].real)
                ;

            state[i][j].imag += 0.25*dt*(D1_Y_X_plus[i][j].imag+D1_Y_X_minus[i][j].imag+D1_Y_Y_plus[i][j].imag+D1_Y_Y_minus[i][j].imag-4.*D1_Y_half[i][j].imag) + DX_state_w_higher_orders[i][j].imag*I1 + DY_state_w_higher_orders[i][j].imag*I2 + \
                I10*(D1_Y_X_plus[i][j].imag - D1_Y_X_minus[i][j].imag) + I20*(D1_Y_Y_plus[i][j].imag - D1_Y_Y_minus[i][j].imag) + \
                I11*(DX_Y_X_plus[i][j].imag - DX_Y_X_minus[i][j].imag) + I22*(DY_Y_Y_plus[i][j].imag - DY_Y_Y_minus[i][j].imag) + \
                I12*(DY_Y_X_plus[i][j].imag - DY_Y_X_minus[i][j].imag) + I21*(DX_Y_Y_plus[i][j].imag - DX_Y_Y_minus[i][j].imag) + \
                I01*(DX_Y_X_plus[i][j].imag + DX_Y_X_minus[i][j].imag + DX_Y_Y_plus[i][j].imag + DX_Y_Y_minus[i][j].imag - 4.*DX_state[i][j].imag) + \
                I02*(DY_Y_X_plus[i][j].imag + DY_Y_X_minus[i][j].imag + DY_Y_Y_plus[i][j].imag + DY_Y_Y_minus[i][j].imag - 4.*DY_state[i][j].imag) \
                //+ I111*(DX_Phi_XX_plus[i][j].imag - DX_Phi_XX_minus[i][j].imag - DX_Y_X_plus[i][j].imag + DX_Y_X_minus[i][j].imag) 
                //+ I222*(DY_Phi_YY_plus[i][j].imag - DY_Phi_YY_minus[i][j].imag - DY_Y_Y_plus[i][j].imag + DY_Y_Y_minus[i][j].imag)
                ;
              //+ dterm7[i];
        }
    }        
}

static double check_boundary_error(MKL_Complex16** psi, int* Fail, double**const* real_matrix_storage){
    double threshold = 3.e-3;
    double **prob_distribution = real_matrix_storage[0]; 
    double **temp = real_matrix_storage[1];

    compute_probability_distribution(psi, prob_distribution);
    vdPackM(x_n*x_n, prob_distribution[0], world.error_boundary_mask[0], temp[0]);
    //double boundary_prob = cblas_dasum (x_n*x_n-((x_n-2*error_boundary_length)*(x_n-2*error_boundary_length)), temp[0], 1)*grid_size*grid_size;
    double boundary_prob = cblas_dasum (x_n*x_n, temp[0], 1)*grid_size*grid_size;
    if ( boundary_prob>threshold){
        *Fail = 1; 
        }
    return boundary_prob;
}


double* const x_xi = &random_buffer[0];
double* const y_xi = &random_buffer[1+Legendre_p];
static void __attribute__((hot)) sample_Ito_integrals(double* r, double dt){
    vdRngGaussian( VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, (1+Legendre_p)*2, random_buffer, 0., 1. );
    double sqrt_dt = sqrt(dt);
    double I1  = sqrt_dt*x_xi[0];
    double I01 = dt*sqrt_dt*0.5*(x_xi[0]+x_xi[1]/sqrt(3.));
    double I10 = dt*I1-I01;
    double I11 = 0.5*(I1*I1-dt);
    double I111= I1/2.* (I1*I1/3.-dt);
    double I2  = sqrt_dt*y_xi[0];
    double I02 = dt*sqrt_dt*0.5*(y_xi[0]+y_xi[1]/sqrt(3.));
    double I20 = dt*I2-I02;
    double I22 = 0.5*(I2*I2-dt);
    double I222= I2/2.* (I2*I2/3.-dt);

    double I12 = x_xi[0]*y_xi[0];
    for(int i=1; i<=Legendre_p; i++){
        I12 += (x_xi[i-1]*y_xi[i]-x_xi[i]*y_xi[i-1])*world.Legendre_p_coefficients[i];
    }
    I12 = I12*dt/2.;
    double I21 = I1*I2 - I12;

    r[0]=I1;  r[1]=I2;
    r[2]=I01; r[3]=I02;
    r[4]=I10; r[5]=I20;
    r[6]=I11; r[7]=I22;
    r[8]=I12; r[9]=I21;
    r[10]=I111;r[11]=I222;
    return;
}

static PyObject* Py_sample_Ito(PyObject *self, PyObject *args){
    PyObject* temp1;
    PyArrayObject* array;
    if (!PyArg_ParseTuple(args, "O", &temp1)){
        PyErr_SetString(PyExc_TypeError, "The input does not match the required signature (output data array (float))");
        return NULL;
    }
    PyArray_OutputConverter(temp1, &array);
    if (!array) {
        PyErr_SetString(PyExc_TypeError, "The input (argument 1) cannot be identified as a Numpy array");
        return NULL;
    }
    if (check_type_real_vector(array, 12)!=0) return NULL;
    double* data;
    data = (double*) PyArray_GETPTR1(array, 0);
    sample_Ito_integrals(data, 1.);
    Py_RETURN_NONE;
}

static PyObject* set_seed(PyObject* self, PyObject *args){
    int seed;
    if (!PyArg_ParseTuple(args, "i", &seed)){printf("Parse fail.\n"); return NULL;}
    vslNewStream( &stream, VSL_BRNG_MT19937, seed );
    set_potential_data(0.,0.);
    Py_RETURN_NONE;
}


static PyObject* check_settings(PyObject* self, PyObject *args){
    return Py_BuildValue("(iddddii)", x_n, grid_size, k, I, Q_z, moment_order, error_boundary_length);
}

static PyMethodDef methods[] = {
    {"step", (PyCFunction)step, METH_VARARGS,"Do one simulation step."},
    {"simulate_n_steps", (PyCFunction)simulate_n_steps, METH_VARARGS,"Do 10 simulation steps."},
    {"prob", (PyCFunction)Py_prob,METH_VARARGS,("compute probability distribution (with measure)")},
    {"set_seed", (PyCFunction)set_seed, METH_VARARGS,
     "Initialize the random number generator with a seed."},
    {"check_settings", (PyCFunction)check_settings, METH_VARARGS,
     "test whether the imported C module responds and return (x_n,grid_size,\\lambda,mass)."},
    {"get_moments", (PyCFunction)get_moments,METH_VARARGS,("get distribution moments up to the "+std::to_string(MOMENT)+"th").c_str()},
    {"x_y_expect", (PyCFunction)Py_x_y_expect,METH_VARARGS,("compute <psi| hat_x |psi> and <psi| hat_y |psi>")}, 
    {"px_py_expect", (PyCFunction)Py_px_py_expect,METH_VARARGS,("compute <psi| hat_p_x |psi> and <psi| hat_p_y |psi>")}, 
    {"energy", (PyCFunction)Py_energy_expect,METH_VARARGS,("compute <psi| hat_H |psi>")},
    {"norm", (PyCFunction)Py_norm,METH_VARARGS,("compute | |psi> |")},
    {"sample_Ito", (PyCFunction)Py_sample_Ito, METH_VARARGS,"sample 2D Ito integrals"}, /////////////////////////////////////////////////////////////// do a performance test to decide "Legendre_p"
    {"copy", (PyCFunction)Py_copy_matrix, METH_VARARGS,"copy some data"},
    {NULL, NULL, 0, NULL}   /* sentinel */
};

static struct PyModuleDef simulationmodule = {
    PyModuleDef_HEAD_INIT,
    "simulation",
    NULL,
    -1,
    methods
};

const int init = init_();

extern "C"{

PyMODINIT_FUNC
__attribute__((externally_visible)) PyInit_simulation(void)
{
    //mkl_set_memory_limit (MKL_MEM_MCDRAM, 256); // mbytes
    //init_();
    return PyModule_Create(&simulationmodule);
}

}
