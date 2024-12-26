#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <pthread.h>

#define GET_TIME(now) { \
   struct timeval t; \
   gettimeofday(&t, NULL); \
   now = t.tv_sec + t.tv_usec/1000000.0; \
}

#define DIM 2  /* Two-dimensional system */
#define X 0    /* x-coordinate subscript */
#define Y 1    /* y-coordinate subscript */
#define NUM_THREADS 4

pthread_mutex_t mutexsum;


const double G = 6.673e-11;  /* Gravitational constant. */
                             /* Units are m^3/(kg*s^2)  */

typedef double vect_t[DIM];  /* Vector type for position, etc. */

struct particle_s {
   double m;  /* Mass     */
   vect_t s;  /* Position */
   vect_t v;  /* Velocity */
};

struct thread_data {
   int thread_id;
   int start;
   int end;
   int pre;
   int n;
   struct particle_s* curr;
   vect_t* forces;
   vect_t* buffer_forces;
};

struct thread_data thread_data_array[NUM_THREADS];


void Usage(char* prog_name);
void Get_args(int argc, char* argv[], int* n_p, int* n_steps_p, 
      double* delta_t_p, int* output_freq_p, char* g_i_p);
void Get_init_cond(struct particle_s curr[], int n);
void Gen_init_cond(struct particle_s curr[], int n);
void Output_state(double time, struct particle_s curr[], int n);
void Compute_force(int part, vect_t forces[], struct particle_s curr[], 
      int n);
void *Compute_force_pth(void *threadarg);
void Update_part(int part, vect_t forces[], struct particle_s curr[], 
      int n, double delta_t);
void *Gather_force(void *threadarg);

/*--------------------------------------------------------------------*/
int main(int argc, char* argv[]) {
   int n;                      /* Number of particles        */
   int n_steps;                /* Number of timesteps        */
   int step;                   /* Current step               */
   int part;                   /* Current particle           */
   int output_freq;            /* Frequency of output        */
   double delta_t;             /* Size of timestep           */
   double t;                   /* Current Time               */
   struct particle_s* curr;    /* Current state of system    */
   vect_t* forces;             /* Forces on each particle    */
   char g_i;                   /* _G_en or _i_nput init conds */
   double start, finish;       /* For timings                */

   Get_args(argc, argv, &n, &n_steps, &delta_t, &output_freq, &g_i);
   curr = malloc(n*sizeof(struct particle_s));
   forces = malloc(n*sizeof(vect_t));
   if (g_i == 'i')
      Get_init_cond(curr, n);
   else
      Gen_init_cond(curr, n);

   GET_TIME(start);
   #ifndef NO_OUTPUT
   Output_state(0, curr, n);
   #endif

   pthread_t threads[NUM_THREADS];
   pthread_t gather_threads[NUM_THREADS];
   int rc; 
   pthread_mutex_init(&mutexsum, NULL);
   vect_t* buffer_forces = malloc(NUM_THREADS*n*sizeof(vect_t));


   for (int i = 0; i < NUM_THREADS; i++) 
   {
      int n_start, n_end, n_pre;
      n_pre = n / NUM_THREADS;
      if(n % NUM_THREADS > 0) n_pre++;
      n_start = n_pre * i;
      n_end = n_pre * (i + 1);
      if (n_end > n) n_end = n - 1;

      thread_data_array[i].thread_id = i;
      thread_data_array[i].start = n_start;
      thread_data_array[i].end = n_end;
      thread_data_array[i].pre = n_pre;
      thread_data_array[i].n = n;
      thread_data_array[i].curr = curr;
      thread_data_array[i].forces = buffer_forces;
      thread_data_array[i].buffer_forces = buffer_forces + i * n;
      // rc = pthread_create(&threads[i], NULL, Compute_force_pth, (void *) &thread_data_array[i]);
      // if (rc)
      // {
      //    printf("ERROR; return code from pthread_create() is %d\n", rc);
      //    exit(-1);
      // }
   }
   for (step = 1; step <= n_steps; step++) {
      t = step*delta_t;
      // memset(forces, 0, n*sizeof(vect_t));
      memset(buffer_forces, 0, NUM_THREADS*n*sizeof(vect_t));
      // for (part = 0; part < n-1; part++)
      //    Compute_force(part, forces, curr, n);

      for (int i = 0; i < NUM_THREADS; i++) 
      {
         rc = pthread_create(&threads[i], NULL, Compute_force_pth, (void *) &thread_data_array[i]);
         if (rc)
         {
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
         }
      }

      for (int i = 0; i < NUM_THREADS; i++) 
      {
         pthread_join(threads[i], NULL);
      }


      // for (int i = 0; i < NUM_THREADS; i++) 
      // {
      //    rc = pthread_create(&gather_threads[i], NULL, Gather_force, (void *) &thread_data_array[i]);
      //    if (rc)
      //    {
      //       printf("ERROR; return code from pthread_create() is %d\n", rc);
      //       exit(-1);
      //    }
      // }

      // for (int i = 0; i < NUM_THREADS; i++) 
      // {
      //    pthread_join(gather_threads[i], NULL);
      // }

      for (int i = 1; i < NUM_THREADS; i++) 
      {
         for (int k = 0; k < n; k++) 
         {
            buffer_forces[k][X] += buffer_forces[i*n+k][X];
            buffer_forces[k][Y] += buffer_forces[i*n+k][Y];
         }
      }

      for (part = 0; part < n; part++)
         Update_part(part, buffer_forces, curr, n, delta_t);
      #ifndef NO_OUTPUT
      if (step % output_freq == 0)
         Output_state(t, curr, n);
      #endif
   }
   
   GET_TIME(finish);
   printf("Elapsed time = %e seconds\n", finish-start);

   free(curr);
   free(forces);
   free(buffer_forces);
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

void Get_init_cond(struct particle_s curr[], int n) {
   int part;

   printf("For each particle, enter (in order):\n");
   printf("   its mass, its x-coord, its y-coord, ");
   printf("its x-velocity, its y-velocity\n");
   for (part = 0; part < n; part++) {
      scanf("%lf", &curr[part].m);
      scanf("%lf", &curr[part].s[X]);
      scanf("%lf", &curr[part].s[Y]);
      scanf("%lf", &curr[part].v[X]);
      scanf("%lf", &curr[part].v[Y]);
   }
}  

void Gen_init_cond(struct particle_s curr[], int n) {
   int part;
   double mass = 5.0e24;
   double gap = 1.0e5;
   double speed = 3.0e4;

   srandom(1);
   for (part = 0; part < n; part++) {
      curr[part].m = mass;
      curr[part].s[X] = part*gap;
      curr[part].s[Y] = 0.0;
      curr[part].v[X] = 0.0;
      if (part % 2 == 0)
         curr[part].v[Y] = speed;
      else
         curr[part].v[Y] = -speed;
   }
}  

void Output_state(double time, struct particle_s curr[], int n) {
   int part;
   printf("%.2f\n", time);
   for (part = 0; part < n; part++) {
      printf("%3d %10.3e ", part, curr[part].s[X]);
      printf("  %10.3e ", curr[part].s[Y]);
      printf("  %10.3e ", curr[part].v[X]);
      printf("  %10.3e\n", curr[part].v[Y]);
   }
   printf("\n");
} 

void Compute_force(int part, vect_t forces[], struct particle_s curr[], 
      int n) {
   int k;
   double mg; 
   vect_t f_part_k;
   double len, len_3, fact;

   for (k = part+1; k < n; k++) {
      f_part_k[X] = curr[part].s[X] - curr[k].s[X];
      f_part_k[Y] = curr[part].s[Y] - curr[k].s[Y];
      len = sqrt(f_part_k[X]*f_part_k[X] + f_part_k[Y]*f_part_k[Y]);
      len_3 = len*len*len;
      mg = -G*curr[part].m*curr[k].m;
      fact = mg/len_3;
      f_part_k[X] *= fact;
      f_part_k[Y] *= fact;

      forces[part][X] += f_part_k[X];
      forces[part][Y] += f_part_k[Y];
      forces[k][X] -= f_part_k[X];
      forces[k][Y] -= f_part_k[Y];
   }
}  


void *Compute_force_pth(void *threadarg) {

   int start;
   int end;
   int n;
   struct particle_s* curr;
   vect_t* forces;
   vect_t* buffer_forces;
   struct thread_data *my_data;

   my_data = (struct thread_data *) threadarg;
   start = my_data->start;
   end = my_data->end;
   n = my_data->n;
   curr = my_data->curr;
   forces = my_data->forces;
   buffer_forces = my_data->buffer_forces;

   int k;
   double mg; 
   vect_t f_part_k;
   double len, len_3, fact;

   for (int part = start; part < end; part++) {
      for (k = part+1; k < n; k++) {
         f_part_k[X] = curr[part].s[X] - curr[k].s[X];
         f_part_k[Y] = curr[part].s[Y] - curr[k].s[Y];
         len = sqrt(f_part_k[X]*f_part_k[X] + f_part_k[Y]*f_part_k[Y]);
         len_3 = len*len*len;
         mg = -G*curr[part].m*curr[k].m;
         fact = mg/len_3;
         f_part_k[X] *= fact;
         f_part_k[Y] *= fact;

         // pthread_mutex_lock(&mutexsum);
         buffer_forces[part][X] += f_part_k[X];
         buffer_forces[part][Y] += f_part_k[Y];
         buffer_forces[k][X] -= f_part_k[X];
         buffer_forces[k][Y] -= f_part_k[Y];
         // pthread_mutex_unlock(&mutexsum);
      }
   }
}  

void Update_part(int part, vect_t forces[], struct particle_s curr[], 
      int n, double delta_t) {
   double fact = delta_t/curr[part].m;

   curr[part].s[X] += delta_t * curr[part].v[X];
   curr[part].s[Y] += delta_t * curr[part].v[Y];
   curr[part].v[X] += fact * forces[part][X];
   curr[part].v[Y] += fact * forces[part][Y];
}  


void *Gather_force(void *threadarg) {

   int thread_id;
   int start;
   int end;
   int pre;
   int n;
   struct particle_s* curr;
   vect_t* forces;
   vect_t* buffer_forces;
   struct thread_data *my_data;

   my_data = (struct thread_data *) threadarg;
   thread_id = my_data->thread_id;
   start = my_data->start;
   end = my_data->end;
   pre = my_data->pre;
   n = my_data->n;
   curr = my_data->curr;
   forces = my_data->forces;
   buffer_forces = my_data->buffer_forces;

   for (int add_rank = 1, recv_rank = 2; add_rank * 2 <= NUM_THREADS; add_rank *= 2, recv_rank*=2) {
      if (thread_id % recv_rank ==  0) {
         // MPI_Recv(buffer_mpi+(thread_id + add_rank)*n_pre*DIM, DIM*(n-(thread_id + add_rank)*n_pre), MPI_DOUBLE, (thread_id + add_rank), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
         for (int k = (thread_id + add_rank)*pre; k < n; k++)
         {
            forces[k][X] += forces[k+(thread_id + add_rank)*n][X];
            forces[k][Y] += forces[k+(thread_id + add_rank)*n][Y];
         }
         // printf("recv: thread_id = %d, (thread_id + add_rank) = %d\n", thread_id, (thread_id+add_rank));
      } 
      // else if (thread_id % recv_rank == add_rank && thread_id - add_rank >= 0) {
      //    // MPI_Send(forces+n_start*DIM, DIM*(n-n_start), MPI_DOUBLE, (thread_id - add_rank), 0, MPI_COMM_WORLD);
      //    // printf("send: thread_id = %d, (thread_id - add_rank) = %d\n", thread_id, (thread_id-add_rank));
      // }
   }
}


