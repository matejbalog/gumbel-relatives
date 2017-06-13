/* To compile this file, put it in the examples folder in your libdai installation location and
 * run the following command from the libdai root directory:
 * 
 * Ubuntu:
 * g++ -Iinclude -Wno-deprecated -Wall -W -Wextra -fpic -O3 -g -DDAI_DEBUG  -Llib -oexamples/spin_glass examples/spin_glass.cpp -ldai -lgmpxx -lgmp
 *
 * Mac:
 * g++ -Iinclude -I/opt/local/include -Wno-deprecated -Wall -W -Wextra -fPIC -DMACOSX -arch x86_64 -O3 -g -DDAI_DEBUG  -Llib -L/opt/local/lib -o examples/spin_glass examples/spin_glass.cpp -ldai -lgmpxx -lgmp -arch x86_64
 */

#include <dai/factorgraph.h>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>

#include <dai/alldai.h>  // Include main libDAI header file
#include <dai/jtree.h>

using namespace std;
using namespace dai;

const double EULER = 0.5772156649;

// Random seed
int seed;

// Parameters (global variables set by main method)
string topology;        // one of {chain, cycle, grid}
int n;                  // standard notation for number of variables
int A, B;               // for grid: n=AxB variables V = [0, 1, ..., A-1] x [0, 1, ...,B-1]
int R;                  // variables take values in [0, 1, ..., R-1]
float f;                // unary potentials drawn randomly from [-f, +f]
float c;                // edge potentials drawn randomly from [0, c] or [-c, +c]
string potentials_type; // one of {fixed, attractive_random, mixed}

// Global variables
double logZ_exact;      // exact value

// Variables and Factors
vector<Var> VAR;
vector<Factor> F;

/*
 * Utility functions
 */
// Random number uniformly distributed in the interval [a, b]
double randu(double a, double b) {
    if (a == b)
        return a;
    else
        return a + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(b-a)));
}
// Random number with Gumbel distribution of mean 0
double rand_gumbel() {
    return -std::log((double)-std::log((double)randu(0.0, 1.0))) - EULER;
}
// Sets sampling bounds [theta_e_min, theta_e_max] for edge potentials based on model type
void set_bounds(string potentials_type, double& theta_e_min, double& theta_e_max) {
    if (potentials_type == "fixed") {
        theta_e_min = c;
        theta_e_max = c;
    }
    else if (potentials_type == "attractive") {
        theta_e_min = 0.0;
        theta_e_max = c;
    }
    else if (potentials_type == "mixed") {
        theta_e_min = -c;
        theta_e_max = +c;
    }
    else throw invalid_argument("Invalid argument potentials_type");
}
// Creates an edge between variables n1 and n2 of potential J
Factor create_factor_Ising( const Var &n1, const Var &n2, Real J ) {
    DAI_ASSERT( n1 != n2 );
    Real buf[R*R];
    int i = 0;
    for (int i2 = 0; i2 < R; ++i2)
        for (int i1 = 0; i1 < R; ++i1)
            buf[i++] = (i1 == i2) ? std::exp(J) : std::exp(-J);
    return Factor( VarSet(n1, n2), &buf[0] );
}


/*
 * Model generation
 * These functions set the global variables VAR (model variables) and F (model factors).
 */
void generate_model_chain(bool verbose = false) {
    if (verbose) cout << "Generating Chain model variables and factors..." << endl;

    // Variables
    VAR.clear();
    for (int i = 0; i < n; ++i)
        VAR.push_back( Var(i, R) );
    
    // Range of values for edge potentials
    double theta_e_min = 0.0, theta_e_max = 0.0;
    set_bounds(potentials_type, theta_e_min, theta_e_max);
        
    // Factors with potentials
    F.clear();
    for (int i = 0; i < n; ++i) {
        // unary potential
        Real buf[R];
        for (int r = 0; r < R; ++r)
            buf[r] = std::exp(randu(-f, +f));
        F.push_back( Factor(VAR[i], &buf[0]) );
        // edge potential
        if(i >= 1) F.push_back( create_factor_Ising(VAR[i], VAR[i-1], randu(theta_e_min, theta_e_max)) );
    }
}
void generate_model_cycle(bool verbose = false) {
    if (verbose) cout << "Generating Cycle model variables and factors..." << endl;

    // Cycle is just a Chain with one more Factor
    generate_model_chain();
    if (n == 1) return;
    
    // Range of values for edge potentials
    double theta_e_min = 0.0, theta_e_max = 0.0;
    set_bounds(potentials_type, theta_e_min, theta_e_max);
    F.push_back( create_factor_Ising(VAR[0], VAR[n-1], randu(theta_e_min, theta_e_max)) );
}
void generate_model_grid(bool verbose = false) {
    if (verbose) cout << "Generating Spin Glass model..." << endl;
    //DAI_ASSERT( R == 2);
    DAI_ASSERT(n = A * B);
    
    // Variables
    VAR.clear();
    for (int a = 0; a < A; ++a)
        for (int b = 0; b < B; ++b)
            VAR.push_back( Var(a*B+b, R) );
    
    // Range of values for edge potentials
    double theta_e_min = 0.0, theta_e_max = 0.0;
    set_bounds(potentials_type, theta_e_min, theta_e_max);
    
    // Factors with potentials
    F.clear();
    for (int a = 0; a < A; ++a)
        for (int b = 0; b < B; ++b) {
            // unary potential
            if (R == 2)
                F.push_back( createFactorIsing(VAR[a*B+b], randu(-f, +f)) );
            else {
                Real buf[R];
                for (int r = 0; r < R; ++r)
                    buf[r] = std::exp(randu(-f, +f));
                F.push_back( Factor(VAR[a*B+b], &buf[0]) );
            }
            // edge potential
            if(a >= 1) F.push_back( create_factor_Ising(VAR[a*B+b], VAR[(a-1)*B+b], randu(theta_e_min, theta_e_max)) );
            if(b >= 1) F.push_back( create_factor_Ising(VAR[a*B+b], VAR[a*B+(b-1)], randu(theta_e_min, theta_e_max)) );
        }
}


/* 
 * Perturbations
 * These functions *copy* the global variable F (model factors), so VAR and F are not modified.
 */
FactorGraph perturb_none(bool verbose = false) {
    if (verbose) cout << "Constructing FactorGraph with no perturbations..." << endl;
    
    // Factor graph
    FactorGraph SpinGlassModel(F);
    return SpinGlassModel;
}
FactorGraph perturb_unary(bool verbose = false) {
    if (verbose) cout << "Constructing FactorGraph with unary perturbations..." << endl;
    
    // Perturbation factors
    vector<Factor> factors(F);
    for(int i = 0; i < n; ++i) {
        Factor local( (VarSet(VAR[i])) );
        for (int r = 0; r < R; ++r)
            local.set(r, std::exp(rand_gumbel()));
        factors.push_back(local);
    }

    // Factor graph
    FactorGraph SpinGlassModel(factors);
    return SpinGlassModel;
}


/*
 * libDAI calls
 */
double logZ(FactorGraph fg, bool verbose = false) {
    if (verbose) cout << "Computing logZ on FactorGraph..." << endl;
    
    // Set some constants
    size_t maxiter = 10000;
    Real   tol = 1e-9;
    size_t verb = 0;

    // Store the constants in a PropertySet object
    PropertySet opts;
    opts.set("maxiter",maxiter);  // Maximum number of iterations
    opts.set("tol",tol);          // Tolerance for convergence
    opts.set("verbose",verb);     // Verbosity (amount of output generated)

    JTree jt;
    vector<size_t> jtmapstate;
    // Construct a JTree (junction tree) object from the FactorGraph fg
    // using the parameters specified by opts and an additional property
    // that specifies the type of updates the JTree algorithm should perform
    jt = JTree( fg, opts("updates",string("HUGIN")) );
    // Initialize junction tree algorithm
    jt.init();
    // Run junction tree algorithm
    jt.run();
    // Report log partition sum (normalizing constant) of fg, calculated by the junction tree algorithm
    return jt.logZ();
}
double MAP_JT(FactorGraph fg, bool verbose = false) {
    if (verbose) cout << "Computing MAP on FactorGraph..." << endl;
    
    // Set some constants
    size_t maxiter = 100000;
    Real   tol = 1e-10;
    size_t verb = 0;

    // Store the constants in a PropertySet object
    PropertySet opts;
    opts.set("maxiter",maxiter);  // Maximum number of iterations
    opts.set("tol",tol);          // Tolerance for convergence
    opts.set("verbose",verb);     // Verbosity (amount of output generated)

    JTree jtmap;
    vector<size_t> jtmapstate;
    // Construct another JTree (junction tree) object that is used to calculate
    // the joint configuration of variables that has maximum probability (MAP state)
    //jtmap = JTree( fg, opts("updates",string("HUGIN"))("inference",string("MAXPROD")) );
    jtmap = JTree( fg, opts("updates",string("SHSH"))("logdomain",false)("heuristic",string("MINWEIGHT"))("inference",string("MAXPROD")) );
    // Initialize junction tree algorithm
    jtmap.init();
    // Run junction tree algorithm
    jtmap.run();
    // Calculate joint state of all variables that has maximum probability
    jtmapstate = jtmap.findMaximum();
    return fg.logScore( jtmapstate );
}
double MAP_BP(FactorGraph fg, bool verbose = false) {
    if (verbose) cout << "Computing MAP on FactorGraph using max-product BP..." << endl;
    
    // Set some constants
    size_t maxiter = 10000;
    Real   tol = 1e-9;
    size_t verb = 0;

    // Store the constants in a PropertySet object
    PropertySet opts;
    opts.set("maxiter",maxiter);  // Maximum number of iterations
    opts.set("tol",tol);          // Tolerance for convergence
    opts.set("verbose",verb);     // Verbosity (amount of output generated)

    // Construct a BP (belief propagation) object from the FactorGraph fg
    // using the parameters specified by opts and two additional properties,
    // specifying the type of updates the BP algorithm should perform and
    // whether they should be done in the real or in the logdomain
    BP mp(fg, opts("updates",string("SEQRND"))("logdomain",false)("inference",string("MAXPROD"))("damping",string("0.5")));
    // Initialize max-product algorithm
    mp.init();
    // Run max-product algorithm
    mp.run();
    // Calculate joint state of all variables that has maximum probability
    // based on the max-product result
    vector<size_t> mpstate = mp.findMaximum();
    return fg.logScore( mpstate );
}

void json_header() {
    cout << "{" << endl;
    cout << "   \"topology\" : \"" << topology << "\"," << endl;
    cout << "   \"n\" : " << n << "," << endl;
    if (topology == "grid") {
        cout << "   \"A\" : " << A << "," << endl;
        cout << "   \"B\" : " << B << "," << endl;
    }
    cout << "   \"K\" : " << R << "," << endl;
    cout << "   \"f\" : " << f << "," << endl;
    cout << "   \"c\" : " << c << "," << endl;
    cout << "   \"potentials_type\" : \"" << potentials_type << "\"," << endl;
    cout << "   \"lnZ\" : " << logZ_exact << "," << endl;
    cout << "   \"seed\" : " << seed << "," << endl;
}

void json_MAPs_unary_JT(int M) {
    cout << "   \"MAPs_unary_JT\" : [";
    for (int m = 0; m < M - 1; ++m)
        cout << MAP_JT(perturb_unary()) << ", ";
    cout << MAP_JT(perturb_unary()) << "]" << endl;
    cout << "}" << endl;
}
void json_MAPs_unary_BP(int M) {
    cout << "   \"MAPs_unary_BP\" : [";
    for (int m = 0; m < M - 1; ++m)
        cout << MAP_BP(perturb_unary()) << ", ";
    cout << MAP_BP(perturb_unary()) << "]" << endl;
    cout << "}" << endl;
}
void json_MAPs_unary_JT_BP(int M) {
    // Compute MAP values
    vector<double> MAPs_JT;
    vector<double> MAPs_BP;
    for (int m = 0; m < M; ++m) {
        FactorGraph fg = perturb_unary();
        MAPs_JT.push_back(MAP_JT(fg));
        MAPs_BP.push_back(MAP_BP(fg));
    }

    // Print JT
    cout << "   \"MAPs_unary_JT\" : [";
    for (int m = 0; m < M - 1; ++m)
        cout << MAPs_JT[m] << ", ";
    cout << MAPs_JT[M - 1] << "]," << endl;

    // Print BP
    cout << "   \"MAPs_unary_BP\" : [";
    for (int m = 0; m < M - 1; ++m)
        cout << MAPs_BP[m] << ", ";
    cout << MAPs_BP[M - 1] << "]" << endl;

    // Print footer
    cout << "}" << endl;
}

/*
Command line arguments:
- topology: {chain, cycle, grid}
- n: int >= 1
- A: grid height, only relevant for grid
- K: int >= 2
- f: float
- c: float
- potentials_type: string
- M: int
- MAP_solver: string
- seed: int
*/
int main(int argc, char* argv[]) {
    // parse arguments
    if (argc < 11)
        cerr << "Not enough arguments provided" << endl;
    topology = argv[1];

    // set global variables
    n = atoi(argv[2]);
    A = atoi(argv[3]);
    B = n / A;
    if ((topology == "grid") && (A*B != n))
        throw invalid_argument("Grid width A does not divide n");
    R = atoi(argv[4]);
    f = atof(argv[5]);
    c = atof(argv[6]);
    potentials_type = argv[7];

    // set computation parameters
    int M = atoi(argv[8]);
    string MAP_solver = argv[9];

    // seed randomness
    seed = atoi(argv[10]);
    srand (static_cast <unsigned> (seed));
    
    // construct appropriate model
    if (topology == "chain") generate_model_chain();
    else if (topology == "cycle") generate_model_cycle();
    else if (topology == "grid") generate_model_grid();
    else throw invalid_argument("Invalid model topology provided");

    // compute logZ and requested number M perturb-and-MAP samples
    logZ_exact = logZ(perturb_none());
    json_header();
    if (MAP_solver == "JT") json_MAPs_unary_JT(M);
    else if (MAP_solver == "BP") json_MAPs_unary_BP(M);
    else if (MAP_solver == "JT_BP") json_MAPs_unary_JT_BP(M);
    else throw invalid_argument("Invalid MAP solver argument provided");

    return 0;
}
