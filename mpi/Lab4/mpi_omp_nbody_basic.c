#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#include <mpi.h>
#include <omp.h>

#define GET_TIME(now) { \
   struct timeval t; \
   gettimeofday(&t, NULL); \
   now = t.tv_sec + t.tv_usec/1000000.0; \
}

#define DIM 2  
#define X 0    
#define Y 1    

const double G = 6.673e-11;

void Usage(char* prog_name);
void Get_args(int argc, char* argv[], int* n_p, int* n_steps_p, 
      double* delta_t_p, int* output_freq_p, char* g_i_p);
void Get_init_cond(double masses[], double positions[], double velocities[], int n);
void Gen_init_cond(double masses[], double positions[], double velocities[], int n);
void Output_state(double time, double positions[], double velocities[], int n);
void Compute_force(int part, double forces[], double masses[], double positions[], int n);
void Update_part(int part, double forces[], double masses[], double positions[], double velocities[], int n, double delta_t);

int main(int argc, char* argv[]) {
   int n;                      
   int n_steps;                
   int step;                   
   int part;                   
   int output_freq;            
   double delta_t;             
   double t;                   
   double* masses;             
   double* positions;          
   double* velocities;         
   double* forces;             
   char g_i;                   

   MPI_Init(&argc, &argv);
   int rank, comm;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &comm);


   Get_args(argc, argv, &n, &n_steps, &delta_t, &output_freq, &g_i);
   masses = malloc(n * sizeof(double));
   positions = malloc(n * DIM * sizeof(double));
   velocities = malloc(n * DIM * sizeof(double));
   forces = malloc(n * DIM * sizeof(double));
   if (g_i == 'i')
      Get_init_cond(masses, positions, velocities, n);
   else
      Gen_init_cond(masses, positions, velocities, n);

   double start, finish;       
   GET_TIME(start);
   
   int n_start, n_end, n_pre;
   n_pre = n / comm;
   if(n % comm > 0) n_pre++;
   n_start = n_pre * rank;
   n_end = n_pre * (rank + 1);
   if (n_end > n) n_end = n;

   if (rank == 0)
      Output_state(0, positions, velocities, n);
   for (step = 1; step <= n_steps; step++) {
      t = step * delta_t;
      
      # pragma omp parallel for schedule(dynamic)
      for (part = n_start; part < n_end; part++) {
         // Compute_force(part, forces, masses, positions, n);

         int k;
         double mg; 
         double f_part_k[DIM];
         double len, len_3, fact;
         double f_part_k_X, f_part_k_Y;
         double sum_X, sum_Y;
         double positions_X, positions_Y, positions_m;

         sum_X = sum_Y = 0.0;
         positions_X = positions[part * DIM + X];
         positions_Y = positions[part * DIM + Y];
         positions_m = masses[part];
         for (k = 0; k < n; k++) {
            if (k != part) {
               f_part_k_X = positions_X - positions[k * DIM + X];
               f_part_k_Y = positions_Y - positions[k * DIM + Y];
               len = sqrt(f_part_k_X*f_part_k_X + f_part_k_Y*f_part_k_Y);
               len_3 = len*len*len;
               mg = -G*positions_m*masses[k];
               fact = mg/len_3;
         
               sum_X += f_part_k_X * fact;
               sum_Y += f_part_k_Y * fact;
            }
         }
         forces[part * DIM + X] = sum_X;
         forces[part * DIM + Y] = sum_Y;
      }


      for (part = n_start; part < n_end; part++)
         Update_part(part, forces, masses, positions, velocities, n, delta_t);
      
      MPI_Allgather(MPI_IN_PLACE, (n_end-n_start)*DIM, MPI_DOUBLE, positions, (n_end-n_start)*DIM, MPI_DOUBLE, MPI_COMM_WORLD);
      
      MPI_Allgather(MPI_IN_PLACE, (n_end-n_start)*DIM, MPI_DOUBLE, velocities, (n_end-n_start)*DIM, MPI_DOUBLE, MPI_COMM_WORLD);
      
      if (step % output_freq == 0 && rank == 0)
         Output_state(t, positions, velocities, n);
   }
   
   GET_TIME(finish);
   if (rank == 0)
      printf("Elapsed time = %e seconds\n", finish - start);

   free(masses);
   free(positions);
   free(velocities);
   free(forces);

   MPI_Finalize();
   return 0;
} 

void Usage(char* prog_name) {
   fprintf(stderr, "usage: %s <number of particles> <number of timesteps>\n",
         prog_name);
   fprintf(stderr, "   <size of timestep> <output frequency>\n");
   fprintf(stderr, "   <g|i>\n");
   fprintf(stderr, "   'g': program should generate init conds\n");
   fprintf(stderr, "   'i': program should get init conds from stdin\n");
    
   exit(0);
}  
void Get_args(int argc, char* argv[], int* n_p, int* n_steps_p, 
      double* delta_t_p, int* output_freq_p, char* g_i_p) {
   if (argc != 6) Usage(argv[0]);
   *n_p = strtol(argv[1], NULL, 10);
   *n_steps_p = strtol(argv[2], NULL, 10);
   *delta_t_p = strtod(argv[3], NULL);
   *output_freq_p = strtol(argv[4], NULL, 10);
   *g_i_p = argv[5][0];

   if (*n_p <= 0 || *n_steps_p < 0 || *delta_t_p <= 0) Usage(argv[0]);
   if (*g_i_p != 'g' && *g_i_p != 'i') Usage(argv[0]);
}  
void Get_init_cond(double masses[], double positions[], double velocities[], int n) {
   int part;

   printf("For each particle, enter (in order):\n");
   printf("   its mass, its x-coord, its y-coord, ");
   printf("its x-velocity, its y-velocity\n");
   for (part = 0; part < n; part++) {
      scanf("%lf", &masses[part]);
      scanf("%lf", &positions[part * DIM + X]);
      scanf("%lf", &positions[part * DIM + Y]);
      scanf("%lf", &velocities[part * DIM + X]);
      scanf("%lf", &velocities[part * DIM + Y]);
   }
} 

void Gen_init_cond(double masses[], double positions[], double velocities[], int n) {
   int part;
   double mass = 5.0e24;
   double gap = 1.0e5;
   double speed = 3.0e4;

   srandom(1);
   for (part = 0; part < n; part++) {
      masses[part] = mass;
      positions[part * DIM + X] = part * gap;
      positions[part * DIM + Y] = 0.0;
      velocities[part * DIM + X] = 0.0;
      if (part % 2 == 0)
         velocities[part * DIM + Y] = speed;
      else
         velocities[part * DIM + Y] = -speed;
   }
} 

void Output_state(double time, double positions[], double velocities[], int n) {
   int part;
   printf("%.2f\n", time);
   for (part = 0; part < n; part++) {
      printf("%3d %10.3e ", part, positions[part * DIM + X]);
      printf("  %10.3e ", positions[part * DIM + Y]);
      printf("  %10.3e ", velocities[part * DIM + X]);
      printf("  %10.3e\n", velocities[part * DIM + Y]);
   }
   printf("\n");
} 

void Compute_force(int part, double forces[], double masses[], double positions[], int n) {
   int k;
   double mg; 
   double f_part_k[DIM];
   double len, len_3, fact;

   forces[part * DIM + X] = forces[part * DIM + Y] = 0.0;
   for (k = 0; k < n; k++) {
      if (k != part) {
         f_part_k[X] = positions[part * DIM + X] - positions[k * DIM + X];
         f_part_k[Y] = positions[part * DIM + Y] - positions[k * DIM + Y];
         len = sqrt(f_part_k[X]*f_part_k[X] + f_part_k[Y]*f_part_k[Y]);
         len_3 = len*len*len;
         mg = -G*masses[part]*masses[k];
         fact = mg/len_3;
         f_part_k[X] *= fact;
         f_part_k[Y] *= fact;
   
         forces[part * DIM + X] += f_part_k[X];
         forces[part * DIM + Y] += f_part_k[Y];
      }
   }
}  

void Update_part(int part, double forces[], double masses[], double positions[], double velocities[], int n, double delta_t) {
   double fact = delta_t/masses[part];

   positions[part * DIM + X] += delta_t * velocities[part * DIM + X];
   positions[part * DIM + Y] += delta_t * velocities[part * DIM + Y];
   velocities[part * DIM + X] += fact * forces[part * DIM + X];
   velocities[part * DIM + Y] += fact * forces[part * DIM + Y];
}



