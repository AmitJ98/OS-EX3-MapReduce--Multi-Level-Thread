#include "MapReduceFramework.h"
#include "Barrier.h"
#include <pthread.h>
#include <algorithm>
#include <iostream>
#include <atomic>

#define SYSTEM_ERROR "system error: "
#define STAGE_LOCATION << 62
#define TOTAL_LOCATION << 31
#define GET_STAGE_BITS >> 62
#define GET_TOTAL_BITS >> 31 & (0x7fffffff)
#define GET_COUNT_BITS & (0x7fffffff)

// exit(1)
// emit2 and emit3
//% calculation


struct Job {
    const MapReduceClient& client;
    const InputVec& inputVec;
    OutputVec& outputVec;
    int ThreadLevel;
    JobState* State;
    pthread_t *all_threads;
    std::atomic<uint64_t> atomic_state{};
    std::atomic<int> atomic_next_to_process;
    std::vector<IntermediateVec>* beforeShuffle;
    std::vector<IntermediateVec*>* afterShuffle;
    pthread_mutex_t mutex;
    Barrier* barrier;
    bool thread_join;

    Job(const MapReduceClient& client, const InputVec& inputVec, OutputVec& outputVec,
        int ThreadLevel, JobState* State, pthread_t* all_threads,
        std::vector<IntermediateVec>* beforeShuffle, std::vector<IntermediateVec*>* afterShuffle,
        Barrier* barrier):
        client(client), inputVec(inputVec), outputVec(outputVec), ThreadLevel(ThreadLevel), State(State),
        all_threads(all_threads), atomic_next_to_process(0),
        beforeShuffle(beforeShuffle), afterShuffle(afterShuffle),barrier(barrier),
        thread_join(false)
    {}
};

typedef struct Job Job;

typedef struct {
    const int id;
    Job* job;
    IntermediateVec* intermediateVec;
}ThreadContext;


typedef struct {
    ThreadContext** thread_context;
    Job* job;
}JobContext;



void lock_mutex(pthread_mutex_t *mutex)
{
  if (pthread_mutex_lock (mutex)!=0)
  {
    std::cout<< SYSTEM_ERROR "mutex lock failed" <<std::endl;
    exit(1);
  }
}


void unlock_mutex(pthread_mutex_t *mutex)
{
  if (pthread_mutex_unlock (mutex)!=0)
  {
    std::cout<< SYSTEM_ERROR "mutex lock failed" <<std::endl;
    exit(1);
  }
}


void reset_counter(Job *job, stage_t stage, uint64_t total_elements)
{
  job->atomic_state = ((uint64_t) stage STAGE_LOCATION) | (total_elements TOTAL_LOCATION);
  job->atomic_next_to_process = 0;
}


bool sort_helper(const IntermediatePair &a, const IntermediatePair &b)
{
  return *a.first < *b.first;
}


void sort_phase(ThreadContext *thread_context)
{
  IntermediateVec* curr_intermediate = thread_context->intermediateVec;
  std::sort (curr_intermediate->begin (), curr_intermediate->end (), sort_helper);

  lock_mutex (&thread_context->job->mutex);

  thread_context->job->beforeShuffle->push_back (*curr_intermediate);

  unlock_mutex (&thread_context->job->mutex);
}


void map_phase(ThreadContext *thread_context)
{
  Job *j = thread_context->job;
  uint64_t x = j->atomic_state.load();
  while ((x GET_COUNT_BITS) < (x GET_TOTAL_BITS))
  {
    long unsigned int old_val = j->atomic_next_to_process.fetch_add(1);
    if (old_val >= j->inputVec.size())
    {
      break;
    }
    std::pair<K1*, V1*> pair = j->inputVec.at (old_val);
    j->client.map (pair.first, pair.second, thread_context);
    x = thread_context->job->atomic_state.fetch_add(1);
  }
  sort_phase(thread_context);
}


K2* max_key(Job* j)
{
  K2* key = nullptr;
  for (auto& vec : *j->beforeShuffle)
  {
    if (!vec.empty())
    {
      key = vec.back().first;
      break;
    }
  }
  if (key == nullptr)
  {
    return key;
  }

  for (auto& vec : *j->beforeShuffle)
  {
    if (vec.empty())
    {
      continue;
    }
    if (*key < *vec.back().first )
    {
      key = vec.back().first;
    }
  }
  return key;
}


void shuffle_phase(Job* job)
{
  IntermediatePair curr;
  K2* maxKey = max_key(job);
  while(maxKey)
  {
    IntermediateVec* temp = new IntermediateVec;
    for (auto& vec : *job->beforeShuffle)
    {
      if (!vec.empty())
      {
        curr = vec.back();
      }
      while (!vec.empty() && !(*curr.first < *maxKey || *maxKey < *curr.first))
      {
        temp->push_back(vec.back());
        vec.pop_back();
        job->atomic_state.fetch_add (1);

        if (!vec.empty())
        {
          curr = vec.back();
        }
      }
    }
    job->afterShuffle->push_back(temp);
    maxKey = max_key(job);
  }
}


void reduce_phase(ThreadContext *thread_context)
{
  Job *j = thread_context->job;
  uint64_t x = j->atomic_state.load();
  while ((x GET_COUNT_BITS) < (x GET_TOTAL_BITS))
  {
    long unsigned int old_val = j->atomic_next_to_process.fetch_add(1);
    if (old_val >= j->afterShuffle->size())
    {
      break;
    }
    IntermediateVec* pair = j->afterShuffle->at(old_val);
    j->client.reduce(pair, thread_context);
    x = thread_context->job->atomic_state.fetch_add(1);
  }
}


size_t count_elements(Job* j, stage_t state)
{
  size_t count = 0;
  if(state == SHUFFLE_STAGE)
  {
    for (auto& vec : *j->beforeShuffle)
    {
      count += vec.size();
    }
  }

  if(state == REDUCE_STAGE)
  {
    count = j->afterShuffle->size();
  }
  return count;
}


void* thread_flow(void* arguments)
{
  auto *thread_context = (ThreadContext *) arguments;
  map_phase (thread_context);

  thread_context->job->barrier->barrier();

  if(thread_context->id == 0)
  {
    reset_counter (thread_context->job, SHUFFLE_STAGE,
                   count_elements (thread_context->job,SHUFFLE_STAGE));

    shuffle_phase(thread_context->job);

    reset_counter (thread_context->job, REDUCE_STAGE,
                   count_elements (thread_context->job,REDUCE_STAGE));
  }


  thread_context->job->barrier->barrier();

  reduce_phase(thread_context);

  return (void *) thread_context;
}


void emit2 (K2* key, V2* value, void* context)
{
  ThreadContext* thread_context = (ThreadContext*)context;
  thread_context->intermediateVec->push_back({key,value});
}


void emit3 (K3* key, V3* value, void* context)
{
  ThreadContext* thread_context = (ThreadContext*)context;
  Job *job = thread_context->job;
  lock_mutex (&job->mutex);
  job->outputVec.push_back({key,value});
  unlock_mutex (&job->mutex);

}


JobContext* create_job(const MapReduceClient& client,const InputVec& inputVec,
                       OutputVec& outputVec,int multiThreadLevel)
{

  Job* job = new Job(
      client,
      inputVec,
      outputVec,
      multiThreadLevel,
      new JobState{UNDEFINED_STAGE, 0},
      new pthread_t[multiThreadLevel],
      new std::vector<IntermediateVec>,
      new std::vector<IntermediateVec*>,
      new Barrier(multiThreadLevel)
  );

  if (pthread_mutex_init (&job->mutex, nullptr)!=0)
  {
    std::cout << SYSTEM_ERROR "mutex creation failed" << std::endl;
    exit (1);
  }

  ThreadContext** thread_contexts= new ThreadContext*[multiThreadLevel];
  reset_counter (job, MAP_STAGE, job->inputVec.size ());


  for(int i = 0; i < multiThreadLevel ; i++)
  {
    auto* contextThread = new ThreadContext{i,job,new IntermediateVec};
    thread_contexts[i] = contextThread;
    if(pthread_create(job->all_threads + i, nullptr, thread_flow,contextThread))
    {
      std::cout << SYSTEM_ERROR "create thread failed\n"<< std::endl;
      exit(1);
    }
  }

  JobContext* job_context = new JobContext {thread_contexts,job};
  return job_context;
}


JobHandle startMapReduceJob(const MapReduceClient& client,const InputVec& inputVec,
                            OutputVec& outputVec,int multiThreadLevel)
{
  return create_job(client,inputVec,outputVec,multiThreadLevel);
}


void getJobState(JobHandle job, JobState* state)
{
  JobContext* job_Context = ((JobContext*) job);
  Job  *j = job_Context->job;

  uint64_t x = j->atomic_state.load();
  auto finished = x GET_COUNT_BITS;
  auto total_elements = x GET_TOTAL_BITS;
  auto new_stage = x GET_STAGE_BITS;

  j->State->percentage = (float)finished / (float)total_elements * 100.0f;
  j -> State ->stage = static_cast<stage_t>(new_stage);

  *state = *(j->State);
}


void waitForJob(JobHandle job)
{
  JobContext* job_Context = ((JobContext*) job);
  if (job_Context->job->thread_join == true)
  {
    return;
  }
  job_Context->job->thread_join = true;
  for (int i=0;i<job_Context->job->ThreadLevel;i++)
  {
    if(pthread_join(job_Context->job->all_threads[i], nullptr))
    {
      std::cout << SYSTEM_ERROR "mutex join failed"<< std::endl;
      exit(1);
    }
  }
}


void closeJobHandle(JobHandle job)
{
  waitForJob(job);
  JobContext* job_Context = ((JobContext*) job);
  Job* j = job_Context->job;

  if(pthread_mutex_destroy(&j->mutex)!=0)
  {
    std::cout <<SYSTEM_ERROR "mutex failed to destroy" << std::endl;
    exit(1);
  }

  if (job_Context->thread_context != nullptr)
  {
    for (int i = 0; i < j->ThreadLevel; i++)
    {
      if (job_Context->thread_context[i] != nullptr)
      {
        delete job_Context->thread_context[i]->intermediateVec;
        delete job_Context->thread_context[i];
      }
    }
    delete[] job_Context->thread_context;
  }


  if (j->beforeShuffle != nullptr)
  {
    for (auto& vec : *j->beforeShuffle)
    {
      vec.clear();
    }
    j->beforeShuffle->clear();
    delete j->beforeShuffle;
  }


  if (j->afterShuffle != nullptr)
  {
    for (auto& vec : *j->afterShuffle)
    {
      vec->clear();
      delete vec;
    }
    j->afterShuffle->clear();
    delete j->afterShuffle;
  }

  delete j->State;
  delete j->barrier;
  delete[] j->all_threads;

  delete j;
  delete job_Context;
}
