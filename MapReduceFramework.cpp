#include "MapReduceFramework.h"
#include <pthread.h>
#include <algorithm>
#include <iostream>

#define SYSTEM_ERROR "system error: "


typedef struct {
    const MapReduceClient& client;
    const InputVec& inputVec;
    OutputVec& outputVec;
    int ThreadLevel;
    JobState* State;
    pthread_t *all_threads;
    std::atomic<int> atomic_total_elements;
    std::atomic<int> atomic_next_to_process;
    std::atomic<int> atomic_stage;
    std::atomic<int> atomic_finished_counter;
    std::vector<IntermediateVec>* beforeShuffle;
    std::vector<IntermediateVec>* afterShuffle;
    pthread_mutex_t mutex;
    Barrier* barrier;
    bool thread_join = false;
} Job;


typedef struct {
    const int id;
    Job* job;
    IntermediateVec* intermediateVec;
}ThreadContext;


typedef struct {
    ThreadContext** thread_context;
    Job* job;
}JobContext;


bool sort_helper(const IntermediatePair &a, const IntermediatePair &b)
{
  return *a.first < *b.first;
}


void sort_phase(ThreadContext *thread_context)
{
  IntermediateVec* curr_intermediate = thread_context->intermediateVec;
  std::sort (curr_intermediate->begin(), curr_intermediate->end(),sort_helper);
  if(pthread_mutex_lock (&thread_context->job->mutex))
  {
    std::cerr << SYSTEM_ERROR "mutex lock failed\n"<< std::endl;
    exit(1);
  }
  thread_context->job->beforeShuffle->push_back(*curr_intermediate);
  if(pthread_mutex_unlock (&thread_context->job->mutex))
  {
    std::cerr << SYSTEM_ERROR "mutex unlock failed\n"<< std::endl;
    exit(1);
  }

}


void map_phase(ThreadContext *thread_context)
{
  Job *j = thread_context->job;
  while (j ->atomic_next_to_process.load () < j->atomic_total_elements.load ())
  {
    int old_val = j->atomic_next_to_process.fetch_add (1);
    std::pair<K1*,V1*> pair =  j->inputVec.at(old_val);
    j->client.map(pair.first, pair.second, thread_context);
    j->atomic_finished_counter++;
  }
  sort_phase (thread_context);
}


K2* max_key(Job* j) {
  K2* key = nullptr;
  for (auto& vec : *j->beforeShuffle) {
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
  K2* maxKey = max_key(job);
  while(maxKey)
  {
    IntermediateVec* temp = new IntermediateVec;
    for (auto& vec : *job->beforeShuffle)
    {
      IntermediatePair curr = vec.back();
      while (!vec.empty() && !(*curr.first < *maxKey || *maxKey < *curr.first))
      {
        temp->push_back(vec.back());
        vec.pop_back();
        job->atomic_finished_counter++;
        curr = vec.back();
      }
    }
    job->afterShuffle->push_back(*temp);
    maxKey = max_key(job);
  }
}


void reduce_phase(ThreadContext *thread_context){
  Job *j = thread_context->job;
  while (j ->atomic_next_to_process.load () < j->atomic_total_elements.load ())
  {
    int old_val = j->atomic_next_to_process.fetch_add(1);
    IntermediateVec pair = j->afterShuffle->at(old_val);
    j->client.reduce(&pair, thread_context);
    j->atomic_finished_counter++;
  }
}


size_t count_elements(Job* j, stage_t state){
  size_t count = 0;
  if(state == SHUFFLE_STAGE)
  {
    for (auto& vec : *j->beforeShuffle)
    {
      count+=vec.size();
    }
  }

  if(state == REDUCE_STAGE){
    for (auto& vec : *j->afterShuffle)
    {
      count+=1;
    }
  }

  return count;
}


void reset_counters(Job *job)
{
  if(job->atomic_stage == MAP_STAGE)
  {
    job->atomic_stage = SHUFFLE_STAGE;
    job->atomic_next_to_process.store(0);
    job->atomic_finished_counter.store(0);
    job->atomic_total_elements.store(count_elements(job, SHUFFLE_STAGE));
  }

  else if(job->atomic_stage == SHUFFLE_STAGE)
  {
    job->atomic_stage = REDUCE_STAGE;
    job->atomic_next_to_process.store(0);
    job->atomic_finished_counter.store(0);
    job->atomic_total_elements.store(count_elements(job, REDUCE_STAGE));
  }
}


void* thread_flow(void* arguments)
{
  auto *thread_context = (ThreadContext *) arguments;
  thread_context->job->atomic_stage = MAP_STAGE;
  map_phase (thread_context);

  thread_context->job->barrier->barrier();

  //shuflle only thread 0
  if(thread_context->id == 0)
  {
    reset_counters (thread_context->job);
    shuffle_phase(thread_context->job);
    reset_counters (thread_context->job);
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
  if(pthread_mutex_lock (&job->mutex))
  {
    std::cerr << SYSTEM_ERROR "mutex lock failed\n"<< std::endl;
    exit(1);
  }
  job->outputVec.push_back({key,value});
  if(pthread_mutex_unlock (&job->mutex))
  {
    std::cerr << SYSTEM_ERROR "mutex unlock failed\n"<< std::endl;
    exit(1);
  }

}


JobContext* create_job(const MapReduceClient& client,const InputVec& inputVec,
                        OutputVec& outputVec,int multiThreadLevel)
{
  Job *job = new Job{
        .client =  client,
        .inputVec =  inputVec,
        .outputVec =  outputVec,
        .ThreadLevel = multiThreadLevel,
        .State = new JobState{UNDEFINED_STAGE,0},
        .all_threads = new pthread_t[multiThreadLevel],
        .beforeShuffle = new std::vector<IntermediateVec>,
        .afterShuffle = new std::vector<IntermediateVec>
      };

  job->barrier = new Barrier (multiThreadLevel);
  job->atomic_total_elements = job->inputVec.size();
  job->atomic_next_to_process = 0;
  if(pthread_mutex_init(&job->mutex, nullptr)){
    std::cerr << SYSTEM_ERROR "create mutex failed\n"<< std::endl;
    exit(1);
  };
  ThreadContext** thread_contexts= new ThreadContext*[multiThreadLevel];

  for(int i = 0; i < multiThreadLevel ; i++)
  {
    auto* contextThread = new ThreadContext{i,job,new IntermediateVec};
    thread_contexts[i] = contextThread;
    if(pthread_create(job->all_threads + i, nullptr, thread_flow,contextThread)){
      std::cerr << SYSTEM_ERROR "create thread failed\n"<< std::endl;
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
  j -> State ->percentage = (j->atomic_finished_counter.load() / j->atomic_total_elements.load()) * 100.0f;
  j -> State ->stage = static_cast<stage_t> (j->atomic_stage.load());
  if (j->State->stage == UNDEFINED_STAGE)
  {
    j->State->percentage = 0;
  }

  *state = *(j->State);
}



void waitForJob(JobHandle job)
{
  JobContext* job_Context = ((JobContext*) job);
  if (job_Context->job->thread_join == true)
  {
    return; // cant activate pthread_join twice
  }
  for (int i=0;i<job_Context->job->ThreadLevel;i++)
  {
    if(pthread_join(job_Context->job->all_threads[i], nullptr))
    {
      std::cerr << SYSTEM_ERROR "mutex join falied\n"<< std::endl;
      exit(1);
    }
  }
  job_Context->job->thread_join = true;
}


void closeJobHandle(JobHandle job)
{
  waitForJob(job);
  JobContext* job_Context = ((JobContext*) job);
  Job* j = job_Context->job;

  // Delete thread contexts and their intermediate vectors
  if (job_Context->thread_context != nullptr) {
    for (int i = 0; i < j->ThreadLevel; i++) {
      if (job_Context->thread_context[i] != nullptr) {
        delete job_Context->thread_context[i]->intermediateVec;
        delete job_Context->thread_context[i];
      }
    }
    delete[] job_Context->thread_context;
  }

  // Delete JobState
  delete j->State;

  // Delete beforeShuffle
  delete j->beforeShuffle;

  // Clear and delete afterShuffle
  if (j->afterShuffle != nullptr) {
    for (auto& vec : *j->afterShuffle) {
      vec.clear(); // Clear contents if necessary
    }
    delete j->afterShuffle;
  }

  // Delete barrier
  delete j->barrier;

  // Destroy mutex
  if (pthread_mutex_destroy(&j->mutex)) {
    std::cerr << SYSTEM_ERROR "destroy mutex failed\n" << std::endl;
    exit(1);
  }

  // Delete all_threads array
  delete[] j->all_threads;

  // Delete the Job struct itself
  delete j;
}














