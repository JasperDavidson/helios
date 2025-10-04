The idea of a thread pool is to introduce efficiency when computing parallel data across threads
- It's **inefficient** to launch and destruct a new thread if you know there will be a large number of threads created regardless (stack allocation, kernel creation, etc.)
- Instead, thread pools create a large batch of threads at the start --> this amortizes the cost of thread creation across the entire operation
- Interesting questions arise, like how many threads should be started based on tasks the pool is likely to get (too many will cause further inefficiency because they're dead threads, too few will render the pool relatively meaningless)

Thread Pool class diagram:
- Variables
	- Vector of worker threads, this is the pool itself
	- Queue of tasks to work through assigned to by the main thread
	- Mutex to block multiple threads from trying to assign themselves the next task from the queue (i.e. multiple dead threads waiting for a task)
	- Condition variable - this works with the mutex to block a task *until* it receives a signal from a thread
		- In this case it makes threads wait until they receive a signal from the main thread that there is a new task to grab
		- Prevents threads from constantly running while idle
- Methods
	- Constructor should create a certain number of threads that all run the core worker_loop() method
	- Destructor should lock the queue mutex, set stop to true so threads no longer fetch, then notify all threads to wakeup and join them all
	- Worker loop - the method constantly running that assigns incoming tasks to empty threads/places them in the queue + ensures unused threads are idle until needed
